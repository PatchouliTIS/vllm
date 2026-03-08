# FireRedASR 1代 vLLM 适配说明

## 概述

本适配基于 FireRedASR 1代（非 2S 版本）实现，保持了原始版本的特性和行为。

## 主要差异

### 与 FireRedASR2S 的差异

| 特性 | FireRedASR (1代) | FireRedASR2S |
|------|-----------------|--------------|
| **数据类型** | float16 | bfloat16 |
| **speech_lens 处理** | 简化（设为 None） | 完整支持动态长度 |
| **attention_mask** | 基础实现 | 增强实现 |
| **Bug 修复** | 保留原有行为 | 修复了多个 Bug |

## 文件结构

```
vllm/
├── vllm/model_executor/models/
│   ├── fireredasr.py          # FireRedASR 1代模型实现
│   └── fireredasr2.py         # FireRedASR2S 模型实现
├── vllm/transformers_utils/processors/
│   ├── fireredasr.py          # FireRedASR 1代 processor
│   └── fireredasr2.py         # FireRedASR2S processor
└── vllm/model_executor/models/registry.py  # 模型注册
```

## 使用方法

### 1. 配置文件

在 HuggingFace 模型的 `config.json` 中设置：

```json
{
  "architectures": ["FireRedASRForConditionalGeneration"],
  "torch_dtype": "float16",
  "audio_encoder_conf": {
    "idim": 80,
    "n_layers_enc": 12,
    "n_head": 8,
    "d_model": 512,
    "kernel_size": 33,
    "pe_maxlen": 5000
  },
  "encoder_downsample_rate": 2,
  "hidden_size": 1536,
  "vocab_size": 151936
}
```

### 2. 启动 vLLM 服务

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/fireredasr-model \
    --dtype float16 \
    --trust-remote-code
```

### 3. 推理示例

```python
from vllm import LLM, SamplingParams

# 初始化模型
llm = LLM(
    model="/path/to/fireredasr-model",
    dtype="float16",
    trust_remote_code=True
)

# 准备音频输入
prompt = {
    "prompt": "<|im_start|>user\n<|AUDIO|>请转写音频为文字<|im_end|>\n<|im_start|>assistant\n",
    "multi_modal_data": {
        "audio": ("audio.wav", 16000)
    }
}

# 生成
sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=256
)

outputs = llm.generate(prompt, sampling_params)
print(outputs[0].outputs[0].text)
```

## 核心组件

### 1. FireRedASREncoder
- 基于 Conformer 架构的音频编码器
- 包含 Conv2dSubsampling、RelPositionalEncoding 等模块

### 2. FireRedASRAdapter
- 将编码器输出映射到 LLM 维度
- 默认下采样率为 2（在编码器 4x 基础上）

### 3. FireRedASRForConditionalGeneration
- 主模型类
- 集成编码器、适配器和 Qwen2 解码器

## 权重映射

从 HuggingFace 权重到 vLLM 的映射规则：

```python
{
    "llm.": "model.decoder.",
    "encoder.": "model.encoder.audio_encoder.",
    "encoder_projector.": "model.encoder_projector.",
    "net.0": "pre_layer_norm",
    "net.1": "linear_expand",
    "net.4": "linear_project",
}
```

## 注意事项

### 1. 数据类型
- FireRedASR 1代使用 **float16**
- 如果遇到数值不稳定，考虑使用 FireRedASR2S（bfloat16）

### 2. speech_lens 处理
- 当前实现简化了 speech_lens 处理
- 如果需要精确的动态长度控制，使用 FireRedASR2S

### 3. 已知限制
- 不支持 CTC 时间戳对齐（原始 AED 模型的 CTC 层未包含）
- padding 处理相对简化

## 与原始实现的对应关系

| 原始 FireRedASR | vLLM 适配 |
|----------------|----------|
| `FireRedAsrLlm` | `FireRedASRForConditionalGeneration` |
| `ConformerEncoder` | `FireRedASREncoder.audio_encoder` |
| `Adapter` | `FireRedASRAdapter` |
| `AutoModelForCausalLM` | `Qwen2ForCausalLM` |
| `transcribe()` | `forward()` + `embed_multimodal()` |

## 性能优化

vLLM 提供的优化：
- ✅ PagedAttention：高效的 KV Cache 管理
- ✅ 连续批处理：动态批处理优化
- ✅ 张量并行：多 GPU 推理支持
- ✅ 量化支持：INT8/FP8 量化（需配置）

## 故障排查

### 问题 1：模型加载失败
```
解决方案：检查 config.json 中的 architectures 是否为 "FireRedASRForConditionalGeneration"
```

### 问题 2：音频处理错误
```
解决方案：确保音频采样率为 16000 Hz，格式为单声道
```

### 问题 3：生成质量差
```
解决方案：
1. 检查音频质量
2. 调整 temperature 和 repetition_penalty
3. 考虑使用 FireRedASR2S（修复了 attention_mask Bug）
```

## 开发者信息

- 基于 FireRedASR 1代适配
- 参考实现：`/home/patchouli/Workspace/CUDA-Agent/FireRedASR/fireredasr/models/fireredasr_llm.py`
- vLLM 版本：最新版本
- 适配日期：2026-03-07

## 许可证

- SPDX-License-Identifier: Apache-2.0
- SPDX-FileCopyrightText: Copyright contributors to the vLLM project
