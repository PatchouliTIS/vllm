# FireRedASR vLLM 适配快速开始

## 🚀 快速开始

### 1. 文件清单

适配完成后，新增/修改的文件：

```
vllm/
├── vllm/model_executor/models/
│   ├── fireredasr.py                    # ✅ 新增：FireRedASR 1代模型
│   ├── fireredasr2.py                   # 已存在：FireRedASR2S 模型
│   └── registry.py                      # ✅ 修改：添加 FireRedASR 注册
├── vllm/transformers_utils/processors/
│   ├── fireredasr.py                    # ✅ 新增：FireRedASR 1代 processor
│   └── fireredasr2.py                   # 已存在：FireRedASR2S processor
├── FIREREDASR_ADAPTATION.md             # ✅ 新增：适配说明文档
├── FIREREDASR_COMPARISON.md             # ✅ 新增：版本对比文档
└── FIREREDASR_QUICKSTART.md             # ✅ 新增：本文档
```

### 2. 验证安装

```bash
# 检查文件是否存在
ls -l vllm/vllm/model_executor/models/fireredasr.py
ls -l vllm/vllm/transformers_utils/processors/fireredasr.py

# 检查语法
python -m py_compile vllm/vllm/model_executor/models/fireredasr.py
python -m py_compile vllm/vllm/transformers_utils/processors/fireredasr.py

# 检查注册
grep "FireRedASRForConditionalGeneration" vllm/vllm/model_executor/models/registry.py
```

### 3. 准备模型配置

创建 `config.json`：

```json
{
  "architectures": ["FireRedASRForConditionalGeneration"],
  "model_type": "fireredasr",
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
  "vocab_size": 151936,

  "bos_token_id": 151644,
  "eos_token_id": 151645,
  "pad_token_id": 151643
}
```

### 4. 准备 processor 配置

创建 `preprocessor_config.json`：

```json
{
  "feature_extractor_type": "FireRedASRFeatureExtractor",
  "processor_class": "FireRedASRProcessor",
  "feature_size": 80,
  "sampling_rate": 16000,
  "chunk_length": 30,
  "padding_value": 0.0,
  "dither": 0.0,
  "hop_length": 160,
  "num_mel_bins": 80,
  "frame_length": 25,
  "frame_shift": 10
}
```

### 5. 测试推理

#### 方法 A：使用 Python API

```python
from vllm import LLM, SamplingParams
import numpy as np

# 初始化模型
llm = LLM(
    model="/path/to/fireredasr-model",
    dtype="float16",
    trust_remote_code=True,
    gpu_memory_utilization=0.9
)

# 准备音频（示例：生成随机音频）
audio = np.random.randn(16000 * 5).astype(np.float32)  # 5秒音频

# 构建 prompt
prompt = {
    "prompt": "<|im_start|>user\n<|AUDIO|>请转写音频为文字<|im_end|>\n<|im_start|>assistant\n",
    "multi_modal_data": {
        "audio": (audio, 16000)
    }
}

# 生成参数
sampling_params = SamplingParams(
    temperature=1.0,
    max_tokens=256,
    repetition_penalty=1.0
)

# 推理
outputs = llm.generate(prompt, sampling_params)
print("转写结果:", outputs[0].outputs[0].text)
```

#### 方法 B：使用 OpenAI API 服务器

```bash
# 启动服务器
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/fireredasr-model \
    --dtype float16 \
    --trust-remote-code \
    --port 8000

# 使用 curl 测试
curl http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F file="@audio.wav" \
  -F model="fireredasr" \
  -F language="zh"
```

### 6. 权重转换（如果需要）

如果你有原始 FireRedASR 的 PyTorch checkpoint：

```python
import torch

# 加载原始权重
checkpoint = torch.load("fireredasr_original.pt", map_location="cpu")

# 提取模型权重
if "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
else:
    state_dict = checkpoint

# 转换权重名称（根据 hf_to_vllm_mapper）
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key
    # 应用映射规则
    if key.startswith("llm."):
        new_key = key.replace("llm.", "model.decoder.")
    elif key.startswith("encoder."):
        new_key = key.replace("encoder.", "model.encoder.audio_encoder.")
    elif key.startswith("encoder_projector."):
        new_key = key.replace("encoder_projector.", "model.encoder_projector.")

    # 特殊映射
    new_key = new_key.replace("net.0", "pre_layer_norm")
    new_key = new_key.replace("net.1", "linear_expand")
    new_key = new_key.replace("net.4", "linear_project")

    new_state_dict[new_key] = value

# 保存为 HuggingFace 格式
torch.save(new_state_dict, "pytorch_model.bin")
print(f"转换完成！共 {len(new_state_dict)} 个参数")
```

### 7. 常见问题

#### Q1: 模型加载失败，提示找不到 FireRedASRForConditionalGeneration

**解决方案**：
```bash
# 检查注册是否成功
python -c "from vllm.model_executor.models.registry import ModelRegistry; print('FireRedASRForConditionalGeneration' in ModelRegistry._get_supported_archs())"
```

#### Q2: 音频处理错误

**解决方案**：
- 确保音频采样率为 16000 Hz
- 确保音频为单声道
- 检查音频长度不超过 30 秒（默认 chunk_length）

```python
import librosa

# 加载并重采样音频
audio, sr = librosa.load("audio.wav", sr=16000, mono=True)
print(f"音频长度: {len(audio)/sr:.2f} 秒")
```

#### Q3: 生成质量差

**解决方案**：
1. 检查音频质量（信噪比、清晰度）
2. 调整生成参数：
```python
sampling_params = SamplingParams(
    temperature=0.8,        # 降低随机性
    repetition_penalty=1.2, # 增加重复惩罚
    max_tokens=512          # 增加最大长度
)
```
3. 考虑使用 FireRedASR2S（修复了 Bug）

#### Q4: 内存不足

**解决方案**：
```python
llm = LLM(
    model="/path/to/fireredasr-model",
    dtype="float16",
    gpu_memory_utilization=0.7,  # 降低 GPU 内存使用
    max_num_seqs=1,              # 减少并发序列数
)
```

### 8. 性能优化

#### 启用张量并行（多 GPU）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/fireredasr-model \
    --dtype float16 \
    --tensor-parallel-size 2 \
    --trust-remote-code
```

#### 启用量化（降低内存）

```bash
python -m vllm.entrypoints.openai.api_server \
    --model /path/to/fireredasr-model \
    --quantization awq \
    --trust-remote-code
```

### 9. 与 FireRedASR2S 对比

| 特性 | FireRedASR (1代) | FireRedASR2S |
|------|-----------------|--------------|
| **配置** | `FireRedASRForConditionalGeneration` | `FireRedASR2ForConditionalGeneration` |
| **数据类型** | float16 | bfloat16 |
| **稳定性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **推理质量** | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **兼容性** | 与原始 1代兼容 | 改进版本 |

**推荐**：如果没有特殊需求，建议使用 FireRedASR2S。

### 10. 下一步

- 📖 阅读 [FIREREDASR_ADAPTATION.md](./FIREREDASR_ADAPTATION.md) 了解详细适配说明
- 📊 阅读 [FIREREDASR_COMPARISON.md](./FIREREDASR_COMPARISON.md) 了解版本差异
- 🔧 根据实际需求调整配置和参数
- 🚀 部署到生产环境

### 11. 支持

如果遇到问题：
1. 检查日志输出
2. 验证配置文件格式
3. 确认权重文件完整性
4. 参考文档中的故障排查部分

---

**适配完成日期**: 2026-03-07
**基于版本**: FireRedASR 1代
**vLLM 版本**: 最新版本
