# FireRedASR vs FireRedASR2S 代码差异对比

## 快速对比表

| 维度 | FireRedASR (1代) | FireRedASR2S |
|------|-----------------|--------------|
| **版权声明** | 无 | Copyright 2026 Xiaohongshu |
| **导入方式** | 绝对导入 | 相对导入 |
| **数据类型** | float16 | bfloat16 |
| **generate 参数** | 无 attention_mask | 有 attention_mask |
| **top_p 参数** | 1.0 | 移除 |
| **speech_lens 支持** | 强制 None | 完整支持 |
| **padding masking** | 只清零嵌入 | 同时清零嵌入和 mask |
| **错误检查** | 基础 | 详细 |

## 详细代码差异

### 1. 数据类型差异

**FireRedASR (1代)**:
```python
if args.use_fp16:
    torch_dtype = torch.float16
else:
    torch_dtype = torch.float32
```

**FireRedASR2S**:
```python
# Training use torch.bfloat16
if args.use_fp16:
    #torch_dtype = torch.float16
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float32
```

**影响**：
- float16：数值范围 ±65504，容易溢出
- bfloat16：数值范围与 float32 相同，训练更稳定

---

### 2. generate 方法参数

**FireRedASR (1代)**:
```python
generated_ids = self.llm.generate(
    inputs_embeds=inputs_embeds,
    # ❌ 缺少 attention_mask
    max_new_tokens=max_new_tokens,
    num_beams=beam_size,
    do_sample=False,
    min_length=decode_min_len,
    top_p=1.0,  # ← 多余参数
    repetition_penalty=repetition_penalty,
    length_penalty=llm_length_penalty,
    temperature=temperature,
    bos_token_id=self.llm.config.bos_token_id,
    eos_token_id=self.llm.config.eos_token_id,
    pad_token_id=self.llm.config.pad_token_id,
)
```

**FireRedASR2S**:
```python
generated_ids = self.llm.generate(
    inputs_embeds=inputs_embeds,
    attention_mask=attention_mask,  # ✅ 添加
    max_new_tokens=max_new_tokens,
    num_beams=beam_size,
    do_sample=False,
    min_length=decode_min_len,
    # top_p 移除（do_sample=False 时无效）
    repetition_penalty=repetition_penalty,
    length_penalty=llm_length_penalty,
    temperature=temperature,
    bos_token_id=self.llm.config.bos_token_id,
    eos_token_id=self.llm.config.eos_token_id,
    pad_token_id=self.llm.config.pad_token_id,
)
```

**Bug 修复**：添加 attention_mask 确保模型不会关注 padding token。

---

### 3. _merge_input_ids_with_speech_features 核心差异

#### 差异 A：speech_lens 处理

**FireRedASR (1代)**:
```python
def _merge_input_ids_with_speech_features(
    self, speech_features, inputs_embeds, input_ids, attention_mask, labels=None,
    speech_lens=None
):
    speech_lens = None  # ← 强制设为 None，忽略动态长度

    # ... 省略中间代码 ...

    # 简单填充，不考虑实际长度
    if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
        raise ValueError(...)

    final_embedding[speech_to_overwrite] = (
        speech_features.contiguous().reshape(-1, embed_dim).to(target_device)
    )

    if speech_lens is not None:  # 永远不会执行
        speech_to_overwrite &= speech_pad_position
    final_attention_mask |= speech_to_overwrite
```

**FireRedASR2S**:
```python
def _merge_input_ids_with_speech_features(
    self, speech_features, inputs_embeds, input_ids, attention_mask, labels=None,
    speech_lens=None
):
    # speech_lens 不再强制为 None

    # ... 省略中间代码 ...

    if speech_lens is not None:
        # 1. 验证长度合法性
        if torch.any(speech_lens > speech_len):
            raise ValueError(
                f"speech_lens contains values ({speech_lens.max()}) larger than "
                f"speech_len ({speech_len})"
            )

        # 2. 计算每个音频特征的位置索引
        speech_cumsum = speech_to_overwrite.long().cumsum(-1)
        speech_position_counter = torch.where(speech_to_overwrite, speech_cumsum - 1, 0)

        # 3. 只填充有效位置
        valid_speech_positions = speech_position_counter < speech_lens[:, None].to(target_device)
        speech_to_overwrite &= valid_speech_positions

        # 4. 验证总帧数匹配
        if speech_to_overwrite.sum().item() != int(speech_lens.sum().item()):
            raise ValueError(
                f"speech_lens and speech token distribution mismatch: "
                f"expected total speech frames {speech_lens.sum().item()}, "
                f"but got {speech_to_overwrite.sum().item()} positions."
            )

        # 5. 精确填充每个音频特征
        batch_idx, seq_idx = torch.where(speech_to_overwrite)
        speech_feature_idx = speech_position_counter[speech_to_overwrite]
        final_embedding[batch_idx, seq_idx] = speech_features[batch_idx, speech_feature_idx].to(target_device)
    else:
        # 回退到简单模式
        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(...)
        final_embedding[speech_to_overwrite] = (
            speech_features.contiguous().reshape(-1, embed_dim)[:speech_to_overwrite.sum()].to(target_device)
        )

    final_attention_mask[speech_to_overwrite] = 1
```

**关键改进**：
- ✅ 支持动态音频长度（每个样本可以有不同的帧数）
- ✅ 精确索引（使用 speech_position_counter）
- ✅ 严格验证（检查长度合法性和总帧数匹配）

---

#### 差异 B：padding 处理

**FireRedASR (1代)**:
```python
# 6. Mask out the embedding at padding positions
batch_indices, pad_indices = torch.where(
    input_ids == self.llm.config.pad_token_id
)
indices_to_mask = new_token_positions[batch_indices, pad_indices]

final_embedding[batch_indices, indices_to_mask] = 0
# ⚠️ Bug: 未更新 final_attention_mask！
```

**FireRedASR2S**:
```python
# 6. Mask out the embedding at padding positions
batch_indices_pad, pad_indices = torch.where(
    input_ids == self.llm.config.pad_token_id
)
if len(batch_indices_pad) > 0:  # ← 添加空检查
    indices_to_mask = new_token_positions[batch_indices_pad, pad_indices]
    final_embedding[batch_indices_pad, indices_to_mask] = 0
    final_attention_mask[batch_indices_pad, indices_to_mask] = 0  # ✅ 修复！
```

**Bug 修复**：
- ✅ 同时清零嵌入和 attention mask
- ✅ 添加空检查避免不必要的操作

---

#### 差异 C：attention_mask 赋值

**FireRedASR (1代)**:
```python
final_attention_mask |= speech_to_overwrite  # 使用位或运算
```

**FireRedASR2S**:
```python
final_attention_mask[speech_to_overwrite] = 1  # 直接赋值
```

**说明**：功能相同，但 FireRedASR2S 的写法更清晰。

---

## vLLM 适配差异

### FireRedASR (vLLM 版本)

```python
# 文件：vllm/vllm/model_executor/models/fireredasr.py

class FireRedASRForConditionalGeneration(nn.Module, ...):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.dtype = vllm_config.model_config.dtype  # 从配置读取

        self.model = FireRedASRModel(...)

    def embed_multimodal(self, **kwargs: object) -> MultiModalEmbeddings:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        speech = audio_input["input_features"]
        speech_lengths = audio_input["speech_lengths"].to(torch.int32)

        # vLLM 使用简化的合并逻辑
        enc_output = self.model.get_encoder_outputs(
            speech=speech, speech_lengths=speech_lengths
        )
        return enc_output
```

### FireRedASR2 (vLLM 版本)

```python
# 文件：vllm/vllm/model_executor/models/fireredasr2.py

class FireRedASR2ForConditionalGeneration(nn.Module, ...):
    # 结构与 FireRedASR 相同
    # 主要差异在于：
    # 1. 类名不同
    # 2. 使用不同的 processor (FireRedASR2FeatureExtractor)
    # 3. 配置中可能指定 bfloat16
```

**注意**：vLLM 版本都使用了简化的 `_merge_multimodal_embeddings`，而不是原始的 `_merge_input_ids_with_speech_features`。

---

## 使用建议

### 选择 FireRedASR (1代) 的场景：
- ✅ 已有 FireRedASR 1代的预训练权重
- ✅ 需要与原始实现完全兼容
- ✅ 硬件不支持 bfloat16

### 选择 FireRedASR2S 的场景：
- ✅ 从头训练或微调
- ✅ 需要更稳定的训练
- ✅ 需要动态音频长度支持
- ✅ 需要更好的推理质量（修复了 Bug）

---

## 迁移指南

### 从 FireRedASR 1代迁移到 FireRedASR2S

1. **更新配置**：
```json
{
  "architectures": ["FireRedASR2ForConditionalGeneration"],  // 改名
  "torch_dtype": "bfloat16",  // 改数据类型
  ...
}
```

2. **转换权重**（如果需要）：
```python
import torch

# 加载 1代权重
checkpoint = torch.load("fireredasr_v1.pt")

# 转换数据类型
for key in checkpoint:
    if checkpoint[key].dtype == torch.float16:
        checkpoint[key] = checkpoint[key].to(torch.bfloat16)

# 保存为 2S 权重
torch.save(checkpoint, "fireredasr2s.pt")
```

3. **更新代码**：
- 将 `FireRedASRForConditionalGeneration` 改为 `FireRedASR2ForConditionalGeneration`
- 确保传递 `attention_mask` 参数

---

## 性能对比

| 指标 | FireRedASR (1代) | FireRedASR2S |
|------|-----------------|--------------|
| **训练稳定性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **推理质量** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **内存占用** | 相同 | 相同 |
| **推理速度** | 相同 | 相同 |
| **硬件兼容性** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

---

## 总结

FireRedASR2S 是 FireRedASR 1代的改进版本，主要改进包括：
1. ✅ 使用 bfloat16 提升训练稳定性
2. ✅ 修复 attention_mask 相关 Bug
3. ✅ 支持动态音频长度
4. ✅ 增强错误检查

**推荐使用 FireRedASR2S**，除非有特殊的兼容性需求。
