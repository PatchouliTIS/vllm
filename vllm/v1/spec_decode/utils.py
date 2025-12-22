# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton

_SAMPLING_EPS = 1e-5


def is_spec_decode_unsupported(sampling_params: SamplingParams) -> bool:
    """True if request is incompatible with speculative decoding"""
    return (
        sampling_params.frequency_penalty != 0.0
        or sampling_params.presence_penalty != 0.0
        # or sampling_params.repetition_penalty != 1.0
        or sampling_params.min_p > _SAMPLING_EPS
        or sampling_params.logprobs is not None
    )


@triton.jit
def eagle_prepare_inputs_padded_kernel(
    cu_num_draft_tokens_ptr,  # [num_reqs]
    valid_sampled_tokens_count_ptr,  # [num_reqs]
    query_start_loc_gpu_ptr,  # [num_reqs + 1]
    token_indices_to_sample_ptr,  # [num_reqs] (output)
    num_reqs,  # tl.int32
):
    """
    Fused kernel for Eagle prepare_input_padded. This kernel computes the
    token index to sample for each request, taking into account the number
    of draft tokens and the number of valid sampled tokens (which is one more than
    the number of accepted tokens).
    """
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Calculate num_draft_tokens from cu_num_draft_tokens, which is an inclusive
    # cumulative sum (first entry is the first value, not zero).
    cu_draft_curr = tl.load(cu_num_draft_tokens_ptr + req_idx)

    num_draft_tokens = 0
    if req_idx == 0:
        num_draft_tokens = cu_draft_curr
    else:
        cu_draft_prev = tl.load(cu_num_draft_tokens_ptr + req_idx - 1)
        num_draft_tokens = cu_draft_curr - cu_draft_prev

    valid_count = tl.load(valid_sampled_tokens_count_ptr + req_idx)
    num_rejected_tokens = num_draft_tokens + 1 - valid_count
    num_rejected_tokens = tl.where(num_draft_tokens > 0, num_rejected_tokens, 0)

    # query_start_loc[req_idx + 1] is the start position of the next request,
    # which is one past the last token of this request.
    q_last_tok_idx = tl.load(query_start_loc_gpu_ptr + req_idx + 1) - 1

    index_to_sample = q_last_tok_idx - num_rejected_tokens
    tl.store(token_indices_to_sample_ptr + req_idx, index_to_sample)


@triton.jit
def eagle_prepare_next_token_padded_kernel(
    sampled_token_ids_ptr,  # [num_reqs, num_sampled_tokens_per_req]
    discard_request_mask_ptr,  # [num_reqs]
    backup_next_token_ids_ptr,  # [num_reqs]
    next_token_ids_ptr,  # [num_reqs] (output)
    valid_sampled_tokens_count_ptr,  # [num_reqs] (output)
    vocab_size,  # tl.int32
    num_sampled_tokens_per_req,  # tl.int32 (num_spec_tokens + 1)
    num_reqs,  # tl.int32
    stride_sampled_token_ids,  # tl.int32 (stride for dim 0)
    BLOCK_SIZE_TOKENS: tl.constexpr,  # Power-of-2 >= num_sampled_tokens_per_req
):
    """
    Fused kernel for Eagle prepare_next_token_ids_padded. This kernel computes the
    number of valid (1 + accepted) tokens for each request, and the corresponding
    "next" token id to sample from during speculative decoding. This is the
    "last accepted token" from the sampled tokens, or the backup token if no
    tokens were accepted or if the request is marked as discarded.
    """
    req_idx = tl.program_id(axis=0)
    if req_idx >= num_reqs:
        return

    # Check if this request is discarded.
    is_discarded = tl.load(discard_request_mask_ptr + req_idx)

    if is_discarded:
        backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
        valid_count = tl.full((), 0, dtype=tl.uint32)
        tl.store(next_token_ids_ptr + req_idx, backup_token)
        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)
    else:
        # Count the number of valid tokens among the sampled tokens.
        token_offs = tl.arange(0, BLOCK_SIZE_TOKENS)
        token_mask = token_offs < num_sampled_tokens_per_req

        row_ptr = sampled_token_ids_ptr + req_idx * stride_sampled_token_ids
        token_ids = tl.load(row_ptr + token_offs, mask=token_mask, other=-1)

        # Rejected tokens are -1, valid tokens are in [0, vocab_size)
        is_valid_mask = (token_ids != -1) & (token_ids < vocab_size) & token_mask
        valid_count = tl.sum(is_valid_mask)

        if valid_count > 0:
            # Guaranteed to be well-defined since
            # valid_count > 0 implies is_valid_mask is not empty
            last_valid_index = tl.max(tl.where(is_valid_mask, token_offs, -1))

            # Select the token at that index, using a sum trick since
            # we don't want to load again to access token_ids[last_valid_index].
            last_valid_token = tl.sum(
                tl.where(token_offs == last_valid_index, token_ids, 0)
            )
            tl.store(next_token_ids_ptr + req_idx, last_valid_token)
        else:
            # No valid tokens found, use backup token
            backup_token = tl.load(backup_next_token_ids_ptr + req_idx)
            tl.store(next_token_ids_ptr + req_idx, backup_token)

        tl.store(valid_sampled_tokens_count_ptr + req_idx, valid_count)


# =============================================================================
# N-gram Proposer Triton Kernels
# =============================================================================


@triton.jit
def ngram_find_matches_kernel(
    token_ids_ptr,  # [batch_size, max_seq_len]
    seq_lengths_ptr,  # [batch_size]
    first_match_positions_ptr,  # [batch_size, num_ngram_sizes] output
    max_seq_len: tl.constexpr,
    stride_tokens: tl.constexpr,
    min_ngram_len: tl.constexpr,
    max_ngram_len: tl.constexpr,
    num_ngram_sizes: tl.constexpr,
    BLOCK_SIZE_POS: tl.constexpr,
    MAX_NGRAM_LEN: tl.constexpr,
):
    """
    Find the first n-gram match position for each (batch, ngram_len) pair.

    Grid: (batch_size, num_ngram_sizes)

    For each sequence and n-gram length, this kernel:
    1. Extracts the suffix (last ngram_len tokens) from the sequence
    2. Searches for the earliest occurrence of this suffix earlier in the sequence
    3. Stores the match position (or -1 if no match found)

    Args:
        token_ids_ptr: Token IDs [batch_size, max_seq_len]
        seq_lengths_ptr: Actual length of each sequence [batch_size]
        first_match_positions_ptr: Output match positions [batch_size, num_ngram_sizes]
        max_seq_len: Maximum sequence length (constexpr)
        stride_tokens: Stride for token_ids dim 0
        min_ngram_len: Minimum n-gram length to search
        max_ngram_len: Maximum n-gram length to search
        num_ngram_sizes: Number of different n-gram sizes (max - min + 1)
        BLOCK_SIZE_POS: Block size for position iteration
        MAX_NGRAM_LEN: Maximum n-gram length (constexpr, for static loop)
    """
    batch_idx = tl.program_id(0)
    ngram_idx = tl.program_id(1)

    # Current n-gram length for this program
    ngram_len = min_ngram_len + ngram_idx

    # Load sequence length for this batch
    seq_len = tl.load(seq_lengths_ptr + batch_idx)

    # Row pointer for this sequence's tokens
    row_ptr = token_ids_ptr + batch_idx * stride_tokens

    # Output pointer for this (batch, ngram_idx)
    out_ptr = first_match_positions_ptr + batch_idx * num_ngram_sizes + ngram_idx

    # Check if sequence is long enough for this n-gram
    if seq_len < ngram_len + 1:
        # Need at least ngram_len + 1 tokens (ngram + at least 1 draft token)
        tl.store(out_ptr, -1)
        return

    # Suffix starts at position (seq_len - ngram_len)
    suffix_start = seq_len - ngram_len

    # Load suffix tokens into registers
    # We use a static loop with MAX_NGRAM_LEN and mask out unused positions
    suffix_offs = tl.arange(0, MAX_NGRAM_LEN)
    suffix_mask = suffix_offs < ngram_len
    suffix_tokens = tl.load(
        row_ptr + suffix_start + suffix_offs,
        mask=suffix_mask & (suffix_start + suffix_offs < max_seq_len),
        other=-1,
    )

    # Maximum valid search position: must leave room for at least 1 draft token
    # Match at position p means tokens[p:p+ngram_len] == suffix
    # Draft tokens start at position (p + ngram_len)
    # We need (p + ngram_len) < seq_len, so p < seq_len - ngram_len
    # Also p must be strictly before suffix_start, so p <= suffix_start - 1
    # Combined: p <= min(seq_len - ngram_len - 1,
    #           suffix_start - 1) = seq_len - ngram_len - 1
    max_search_pos = seq_len - ngram_len - 1

    # Initialize first_match to "not found"
    # Use max_seq_len as sentinel (any valid match will be smaller)
    first_match = max_seq_len

    # Search through all positions in blocks
    # Note: Triton does not support `break` statements, so we process all blocks
    # and use tl.minimum to track the earliest match position.
    for block_start in tl.range(0, max_seq_len, BLOCK_SIZE_POS):
        # Positions in this block
        pos_offs = tl.arange(0, BLOCK_SIZE_POS)
        positions = block_start + pos_offs

        # Mask for valid search positions
        pos_mask = positions <= max_search_pos

        # For each position, check if all ngram_len tokens match the suffix
        # We accumulate match counts using a static loop
        match_counts = tl.zeros((BLOCK_SIZE_POS,), dtype=tl.int32)

        for k in tl.static_range(0, MAX_NGRAM_LEN):
            if k < ngram_len:
                # Load window token at position (positions + k)
                window_tokens = tl.load(
                    row_ptr + positions + k,
                    mask=pos_mask & (positions + k < max_seq_len),
                    other=-2,  # Use different sentinel to ensure no false match
                )
                # Get suffix token at position k (broadcast across positions)
                # Use sum trick to extract scalar from masked vector
                suffix_k = tl.sum(tl.where(suffix_offs == k, suffix_tokens, 0))
                # Increment count where tokens match
                match_counts += (window_tokens == suffix_k).to(tl.int32)

        # Full match = all ngram_len tokens matched
        is_match = (match_counts == ngram_len) & pos_mask

        # Find the minimum position that matches in this block
        # Use where to set non-matches to max_seq_len
        match_positions = tl.where(is_match, positions, max_seq_len)
        block_min = tl.min(match_positions)

        # Update global first match
        first_match = tl.minimum(first_match, block_min)

    # Convert sentinel to -1 for "no match"
    result = tl.where(first_match < max_seq_len, first_match, -1)
    tl.store(out_ptr, result)


@triton.jit
def ngram_select_and_extract_kernel(
    token_ids_ptr,  # [batch_size, max_seq_len]
    seq_lengths_ptr,  # [batch_size]
    first_match_positions_ptr,  # [batch_size, num_ngram_sizes]
    draft_tokens_ptr,  # [batch_size, num_draft_tokens] output
    max_seq_len: tl.constexpr,
    stride_tokens: tl.constexpr,
    min_ngram_len: tl.constexpr,
    num_ngram_sizes: tl.constexpr,
    num_draft_tokens: tl.constexpr,
    BLOCK_SIZE_NGRAM: tl.constexpr,
    BLOCK_SIZE_DRAFT: tl.constexpr,
):
    """
    Select the longest matching n-gram and extract draft tokens.

    Grid: (batch_size,)

    For each sequence, this kernel:
    1. Finds the longest n-gram that has a valid match
    2. Extracts num_draft_tokens tokens following the match position
    3. Masks invalid positions with -1

    Args:
        token_ids_ptr: Token IDs [batch_size, max_seq_len]
        seq_lengths_ptr: Actual length of each sequence [batch_size]
        first_match_positions_ptr: Match positions from ngram_find_matches_kernel
        draft_tokens_ptr: Output draft tokens [batch_size, num_draft_tokens]
        max_seq_len: Maximum sequence length
        stride_tokens: Stride for token_ids dim 0
        min_ngram_len: Minimum n-gram length
        num_ngram_sizes: Number of n-gram sizes
        num_draft_tokens: Number of draft tokens to extract (k)
        BLOCK_SIZE_NGRAM: Block size for n-gram iteration (>= num_ngram_sizes)
        BLOCK_SIZE_DRAFT: Block size for draft token iteration (>= num_draft_tokens)
    """
    batch_idx = tl.program_id(0)

    # Load sequence length
    seq_len = tl.load(seq_lengths_ptr + batch_idx)

    # Row pointers
    token_row_ptr = token_ids_ptr + batch_idx * stride_tokens
    match_row_ptr = first_match_positions_ptr + batch_idx * num_ngram_sizes
    draft_row_ptr = draft_tokens_ptr + batch_idx * num_draft_tokens

    # Load all match positions for this sequence
    ngram_offs = tl.arange(0, BLOCK_SIZE_NGRAM)
    ngram_mask = ngram_offs < num_ngram_sizes
    match_positions = tl.load(match_row_ptr + ngram_offs, mask=ngram_mask, other=-1)

    # Find the longest n-gram with a valid match (match_position >= 0)
    # Longer n-grams have higher indices, so we want the maximum index
    # where match_position >= 0
    has_match = match_positions >= 0

    # Create indices where we have matches, -1 otherwise
    valid_indices = tl.where(has_match & ngram_mask, ngram_offs, -1)
    best_ngram_idx = tl.max(valid_indices)

    # Check if we found any match
    has_any_match = best_ngram_idx >= 0

    if has_any_match:
        # Get the match position for the best n-gram
        # Use sum trick to extract the value at best_ngram_idx
        best_match_pos = tl.sum(
            tl.where(ngram_offs == best_ngram_idx, match_positions, 0)
        )

        # Calculate n-gram length: min_ngram_len + best_ngram_idx
        best_ngram_len = min_ngram_len + best_ngram_idx

        # Draft tokens start right after the matched n-gram
        draft_start = best_match_pos + best_ngram_len

        # Number of tokens available for drafting
        tokens_available = seq_len - draft_start

        # Extract draft tokens
        draft_offs = tl.arange(0, BLOCK_SIZE_DRAFT)
        draft_mask = draft_offs < num_draft_tokens

        # Compute read positions (clamp to valid range)
        read_positions = draft_start + draft_offs
        read_positions = tl.minimum(read_positions, max_seq_len - 1)
        read_positions = tl.maximum(read_positions, 0)

        # Load tokens
        draft_tokens = tl.load(
            token_row_ptr + read_positions, mask=draft_mask, other=-1
        )

        # Mask out positions beyond available tokens
        valid_draft_mask = (draft_offs < tokens_available) & draft_mask
        draft_tokens = tl.where(valid_draft_mask, draft_tokens, -1)

        # Store results
        tl.store(draft_row_ptr + draft_offs, draft_tokens, mask=draft_mask)
    else:
        # No match found, fill with -1
        draft_offs = tl.arange(0, BLOCK_SIZE_DRAFT)
        draft_mask = draft_offs < num_draft_tokens
        tl.store(
            draft_row_ptr + draft_offs,
            tl.full((BLOCK_SIZE_DRAFT,), -1, dtype=tl.int32),
            mask=draft_mask,
        )


def ngram_propose_triton(
    token_ids: torch.Tensor,
    seq_lengths: torch.Tensor,
    min_ngram_len: int,
    max_ngram_len: int,
    num_draft_tokens: int,
) -> torch.Tensor:
    """
    GPU-accelerated n-gram proposal using Triton kernels.

    This function finds n-gram matches in token sequences and extracts
    draft tokens following the matches. It tries multiple n-gram lengths
    and selects the longest match for each sequence.

    Args:
        token_ids: Token IDs for each sequence [batch_size, max_seq_len]
        seq_lengths: Actual length of each sequence [batch_size]
        min_ngram_len: Minimum n-gram size to search for (e.g., 2)
        max_ngram_len: Maximum n-gram size to search for (e.g., 5)
        num_draft_tokens: Number of draft tokens to extract (k)

    Returns:
        Draft token predictions [batch_size, num_draft_tokens]
        -1 indicates invalid/no-match positions
    """
    batch_size = token_ids.shape[0]
    max_seq_len = token_ids.shape[1]
    device = token_ids.device
    num_ngram_sizes = max_ngram_len - min_ngram_len + 1

    # Allocate intermediate buffer for match positions
    first_match_positions = torch.full(
        (batch_size, num_ngram_sizes), -1, dtype=torch.int32, device=device
    )

    # Allocate output buffer
    draft_tokens = torch.full(
        (batch_size, num_draft_tokens), -1, dtype=torch.int32, device=device
    )

    # Kernel configuration
    BLOCK_SIZE_POS = 256  # Positions per block for searching
    MAX_NGRAM_LEN = triton.next_power_of_2(max_ngram_len)
    BLOCK_SIZE_NGRAM = triton.next_power_of_2(num_ngram_sizes)
    BLOCK_SIZE_DRAFT = triton.next_power_of_2(num_draft_tokens)

    # Ensure seq_lengths is int32 for Triton
    if seq_lengths.dtype != torch.int32:
        seq_lengths = seq_lengths.to(torch.int32)

    # Launch kernel 1: Find matches for each (batch, ngram_len)
    grid_find = (batch_size, num_ngram_sizes)
    ngram_find_matches_kernel[grid_find](
        token_ids,
        seq_lengths,
        first_match_positions,
        max_seq_len,
        token_ids.stride(0),
        min_ngram_len,
        max_ngram_len,
        num_ngram_sizes,
        BLOCK_SIZE_POS=BLOCK_SIZE_POS,
        MAX_NGRAM_LEN=MAX_NGRAM_LEN,
    )

    # Launch kernel 2: Select best match and extract draft tokens
    grid_extract = (batch_size,)
    ngram_select_and_extract_kernel[grid_extract](
        token_ids,
        seq_lengths,
        first_match_positions,
        draft_tokens,
        max_seq_len,
        token_ids.stride(0),
        min_ngram_len,
        num_ngram_sizes,
        num_draft_tokens,
        BLOCK_SIZE_NGRAM=BLOCK_SIZE_NGRAM,
        BLOCK_SIZE_DRAFT=BLOCK_SIZE_DRAFT,
    )

    return draft_tokens
