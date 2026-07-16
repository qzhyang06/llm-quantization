# LLM 量化部署项目：从理论到 vLLM 推理

> **项目定位**：在 WSL2 + RTX 4070 Ti (12GB) 环境下，实现对 1.5B 大语言模型的 5 种量化方案，对比分析各方案优劣，并使用 vLLM 推理框架完成端到端部署验证。
>
> **核心成果**：从零实现了 5 种量化方案（FP8 / INT8 / INT8-W8A8-tensor / INT8-W8A8-channel / SmoothQuant），FP8 方案模型压缩 1.58 倍，推理精度无显著损失。

---

## 目录

1. [项目环境](#1-项目环境)
2. [模型结构分析](#2-模型结构分析)
3. [量化理论基础](#3-量化理论基础)
4. [5 种量化方案设计与实现](#4-5-种量化方案设计与实现)
5. [技术难点与解决过程](#5-技术难点与解决过程)
6. [vLLM 推理引擎分析](#6-vllm-推理引擎分析)
7. [性能分析与优化](#7-性能分析与优化)
8. [项目文件结构](#8-项目文件结构)
9. [技能展示](#9-技能展示)

---

## 1. 项目环境

### 硬件

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 4070 Ti 12GB (AD104, SM89 Ada Lovelace) |
| CPU | AMD Ryzen |
| RAM | 15 GB |
| OS | Ubuntu 26.04 WSL2 |
| CUDA | 13.1, Driver 591.86 |

### 软件

| 组件 | 版本 | 用途 |
|------|------|------|
| vLLM | 0.25.0 | 推理引擎 |
| PyTorch | 2.11.0 | 张量计算 |
| compressed-tensors | 0.17.0 | 量化格式库 |
| transformers | 4.51+ | 模型加载 |
| safetensors | — | 权重文件 I/O |
| FlashInfer | 0.6.13 | GPU 采样 kernel |

### 关键约束

- **WSL2 12GB 显存**：7B 模型 FP16 加载即 OOM，只能做 1.5B 量化的 PTQ 实验
- **WSL2 CUDA Graph 不兼容**：必须 `enforce_eager=True`
- **WSL2 fork 不支持**：vLLM 强制使用 `spawn` 多进程模式
- **15GB 系统 RAM**：7B 模型 (14GB) 无法加载做 PTQ（需要将模型完整加载到内存）

---

## 2. 模型结构分析

### 模型：DeepSeek-R1-Distill-Qwen-1.5B

由 DeepSeek-R1 (671B MoE) 蒸馏训练 Qwen2-1.5B 得到，具备推理/思维链能力。

```
Qwen2ForCausalLM (1.5B 参数, 3.3 GB BF16)
│
├── model.embed_tokens        [151936, 1536]   词嵌入（465M 参数）
│
├── model.layers.0..27        ×28 层 Transformer
│   ├── input_layernorm       [1536]           RMSNorm
│   ├── self_attn
│   │   ├── q_proj            [1536, 1536]     查询投影 (GQA, 12 Q-heads)
│   │   ├── k_proj            [1536, 256]      键投影 (GQA, 2 KV-heads)
│   │   ├── v_proj            [1536, 256]      值投影 (GQA, 2 KV-heads)
│   │   └── o_proj            [1536, 1536]     输出投影
│   ├── post_attention_layernorm [1536]        RMSNorm
│   └── mlp (SwiGLU)
│       ├── gate_proj         [1536, 8960]     门控投影
│       ├── up_proj           [1536, 8960]     上投影
│       └── down_proj         [8960, 1536]     下投影
│
├── model.norm                [1536]           最终 RMSNorm
└── lm_head                   [151936, 1536]   输出头（不量化）
```

**关键结构特征**：
- GQA (Group Query Attention): 12 Q-heads 共享 2 KV-heads，减少 KV Cache 6 倍
- SwiGLU MLP: `down_proj(SiLU(gate_proj(x)) ⊙ up_proj(x))`
- 每层 7 个 Linear 层 × 28 层 = **196 个量化目标**
- MLP 占每层参数的 ~88%，量化收益最大

---

## 3. 量化理论基础

### 3.1 量化统一框架

```
量化 = 把高精度浮点数映射到低精度整数/浮点数

    w_quantized = clamp(round(w / scale), min_val, max_val)
    
    scale = max(|w|) / max_val    (per-tensor)
    scale_g = max(|w_group|) / max_val   (per-group)

压缩比:
  FP8:  16 bits → 8 bits  = 2x (理论)
  INT8: 16 bits → 8 bits  = 2x (理论)
  INT4: 16 bits → 4 bits  = 4x (理论)
```

### 3.2 FP8 vs INT8：为什么 FP8 可以不校准

```
FP8 E4M3 (1 sign + 4 exp + 3 mantissa):
  动态范围: 240 / 0.00195 ≈ 123,000x  ← 4位指数提供
  同一个 scale 下，大值和小值都能表示
  → 不需要校准数据来确定 scale
  → 激活值可以运行时动态量化 (on-the-fly)

INT8 (1 sign + 7 value):
  动态范围: 127 / 1 = 127x  ← 没有指数位!
  outlier 会压迫小值 → 小值被量化为 0
  → 需要细粒度 (per-group/per-channel) 或校准数据
  → 静态激活量化需要校准数据计算 scale
```

### 3.3 量化粒度选择

| 粒度 | scale 数量 | 适用场景 |
|------|-----------|---------|
| per-tensor | 1 个/矩阵 | FP8（指数位提供动态范围） |
| per-channel | in_features 个/矩阵 | INT8 激活 |
| per-group(128) | in_features/128 个/行 | INT8/INT4 权重 |

### 3.4 SmoothQuant 原理

```
问题: LLM 激活值存在系统性 outlier channel（某些 channel 比其他大 10-100x）

SmoothQuant 变换:
  s_j = max(|X_j|)^α / max(|W_j|)^(1-α)   (α=0.5)

  激活: X'_j = X_j / s_j     ← 缩小 outlier → 更容易量化
  权重: W'_j = W_j × s_j     ← 放大对应权重 → 保持计算结果

  恒等性: X' × W'^T = (X/s) × (W×s)^T = X × W^T ✅

迁移: s_j 吸收到前一层的 LayerNorm 权重中
```

---

## 4. 5 种量化方案设计与实现

### 方案架构图

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        5 种量化方案对比                                    │
│                                                                          │
│  方案1: FP8 W8A8 dynamic          方案2: INT8 W8A16                      │
│  ┌──────┐  ┌──────┐              ┌──────┐  ┌──────┐                     │
│  │W:FP8 │  │X:BF16│              │W:INT8│  │X:BF16│                     │
│  │per-  │  │  │   │              │per-  │  │ 不   │                     │
│  │tensor│  │  ▼   │              │group │  │ 量化 │                     │
│  └──────┘  │→FP8 │              └──────┘  └──────┘                     │
│             │动态 │                │dequant│    │                         │
│             └─────┘                ▼       │    │                         │
│            CUTLASS FP8 GEMM      BF16 GEMM◄────┘                         │
│                                                                          │
│  方案3: INT8 W8A8 tensor         方案4: INT8 W8A8 channel               │
│  ┌──────┐  ┌──────┐              ┌──────┐  ┌──────┐                     │
│  │W:INT8│  │X:BF16│              │W:INT8│  │X:BF16│                     │
│  │per-  │  │  │   │              │per-  │  │  │   │                     │
│  │group │  │calib│              │group │  │calib│                     │
│  └──────┘  │1个/层│              └──────┘  │N个/层│                     │
│             └─────┘                         └─────┘                       │
│            dequant + BF16 GEMM             dequant + BF16 GEMM           │
│                                                                          │
│  方案5: SmoothQuant W8A8                                                │
│  ┌─────────────────────────┐                                             │
│  │ SmoothQuant 变换:        │    ┌──────┐  ┌──────┐                     │
│  │  LayerNorm γ /= s       │───▶│W:INT8│  │X:BF16│                     │
│  │  Linear W ×= s          │    │per-  │  │per-  │                     │
│  │ 恒等: X'·W'^T = X·W^T  │    │group │  │tensor│                     │
│  └─────────────────────────┘    └──────┘  └──────┘                     │
│                                   dequant + BF16 GEMM                   │
└──────────────────────────────────────────────────────────────────────────┘
```

### 方案 1: FP8 W8A8 dynamic（推荐）

```python
def quantize_to_fp8(weight):
    FP8_MAX = 240.0
    amax = weight.abs().max()              # 整个矩阵 1 个值
    scale = (amax / FP8_MAX).clamp(min=1e-12)  # 标量
    fp8_weight = (weight.float() / scale) \
                   .clamp(-FP8_MAX, FP8_MAX) \
                   .to(torch.float8_e4m3fn)
    return fp8_weight, scale
```

- **权重**: per-tensor FP8（脚本离线量化）
- **激活**: per-token FP8（vLLM 运行时 on-the-fly 量化）
- **Kernel**: CUTLASS FP8 GEMM (SM89 硬件 WGMMA 指令)
- **优点**: 无需校准数据、最快推理、精度损失最小
- **vLLM**: `quantization='fp8'`

### 方案 2: INT8 W8A16

```python
def quantize_to_int8(weight, group_size=128):
    # 每 128 列为一组，独立计算 scale
    reshaped = weight.view(out_f, n_groups, group_size)
    amax = reshaped.abs().amax(dim=-1, keepdim=True)
    scale = (amax / 127.0)
    q_weight = (reshaped / scale).round().clamp(-128, 127).to(torch.int8)
    return q_weight, scale.squeeze(-1)
```

- **权重**: per-group INT8 (group_size=128)
- **激活**: BF16（不量化）
- **Kernel**: dequant → BF16 GEMM（无硬件加速）
- **vLLM**: `quantization='compressed-tensors'`

### 方案 3: INT8 W8A8 tensor

在方案 2 的基础上，通过**校准数据**收集每层激活值的 `max(|X|)`，存入 `input_scale`。

```
校准流程:
  1. 加载模型到 GPU
  2. 注册 forward hooks 到每个 Linear 层
  3. 跑 64-256 条校准样本 → 捕获激活值
  4. 每层: scale = max(|all_samples|) / 127
  5. 存入 input_scale 参数

实测发现: 不同层的激活值跨度达 800x (0.014 ~ 11.2)
```

### 方案 4: INT8 W8A8 channel

在方案 3 基础上，**每个输入 channel 独立 scale**（而非整个层共享）。

```
per-channel scale: [1, in_features] 个独立值
  → 精细度提高，但存储开销增加
  → 实测: down_proj [1, 8960] 有 615 个不同 scale 值
```

### 方案 5: SmoothQuant W8A8

```
SmoothQuant 完整流程:
  1. 校准 → 收集激活 per-channel max
  2. 计算平滑因子: s_j = act_j^0.5 / w_j^0.5
  3. 吸收到前层 LayerNorm: γ /= s
  4. 吸收到当前层权重: W *= s
  5. 重校准 → 收集变换后的激活统计
  6. 量化权重 + 存储激活 scale
```

---

## 5. 技术难点与解决过程

### 难点 1: llm-compressor 无法安装

**尝试**: `pip install llm-compressor` → `No matching distribution`

**分析**: llm-compressor 是 Neural Magic 的量化工具，为 compressed-tensors 提供校准驱动。当前 Python 3.12 / Linux 平台无预编译包。

**解决**: 绕过整个工具链，直接用 PyTorch + safetensors 在 state_dict 层面实现量化。核心量化函数只有 5-10 行，不需要 observer/lifecycle/calibration 框架。

### 难点 2: compressed-tensors 的 observer 不触发

**尝试**: 直接使用 compressed-tensors 的 `apply_quantization_config()` + `compress_quantized_weights()` 流程，期望 lifecycle 自动管理 observer。

**发现**: `apply_quantization_config` 只是初始化了 scale/zp 参数（值为零），observer 需要通过 forward pass 更新这些值。但 compressed-tensors 的 observer 依赖 llm-compressor 驱动。没有它 → scale 保持为零 → `compress_quantized_weights()` 无效 → 模型仍是 BF16。

**解决**: 放弃 compressed-tensors lifecycle API，手动计算 scale 并直接修改权重张量。

### 难点 3: lm_head 量化导致 vLLM 加载失败

**错误**: `ValueError: There is no module or parameter named 'lm_head.weight_scale'`

**根因**: vLLM 中 `lm_head` 使用 `ParallelLMHead`（继承自 `VocabParallelEmbedding`），不走标准的 `Fp8LinearMethod` 分发。`lm_head.weight` 由 `LogitsProcessor` 直接消费，不通过 `module.forward()`。

**解决**: 量化时排除 `lm_head`。config.json 中设置 `"ignore": ["lm_head"]`。

### 难点 4: WSL2 FlashInfer JIT 编译超时

**现象**: 首次启动 FlashInfer 编译 CUDA sampling kernel 耗时 432 秒。

**根因**:
1. WSL2 CUDA 驱动是转发的（libcuda → dxgkrnl → nvlddmkm.sys）
2. nvcc 编译在 WSL2 中有额外 I/O 开销
3. 首次编译无缓存 → 全量编译

**解决**: 编译结果缓存到 `~/.cache/flashinfer/`，第二次启动只需 ~2 秒。

### 难点 5: ninja 不在子进程 PATH 中

**现象**: `FileNotFoundError: ninja`

**根因**: vLLM 在 WSL2 使用 `spawn` 多进程，子进程不继承 conda 环境 PATH。ninja 在 conda env 的 bin 目录下。

**解决**: `export PATH="/home/jake/miniconda3/envs/trtllm/bin:$PATH"` 或创建软链接。

### 难点 6: SmoothQuant 激活 scale 过期

**问题**: 方法 5 初始实现中，校准得到的激活 scale 来自原始模型，但 SmoothQuant 变换修改了 LayerNorm 权重，导致激活分布改变。

**解决**: 在 SmoothQuant 变换后，将修改后的 state_dict 重新加载到模型，重新跑一次校准获取正确的激活 scale。

### 难点 7: W4A16 模型压缩比不如预期

**现象**: INT4 模型只有 ~2x 压缩比，而非理论 4x。

**根因**:
- per-group(128) 的 scale 和 zero_point 占用了大量空间
- 小权重层（如 q_proj [1536,1536]）中 scale+zero_point 开销占比很大
- lm_head [151936,1536] 不量化（保留 BF16），占模型 ~30%

---

## 6. vLLM 推理引擎分析

### 6.1 架构

```
┌──────────────────────────────────────────────────────┐
│                   vLLM 主进程                         │
│  ┌─────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ Scheduler│  │Tokenizer │  │ Request Manager  │   │
│  └────┬─────┘  └──────────┘  └──────────────────┘   │
│       │                                               │
│       ▼                                               │
│  ┌─────────────────────────────────────────────┐    │
│  │          EngineCore (子进程, PID=38243)      │    │
│  │                                              │    │
│  │  ┌─────────────┐  ┌──────────────────────┐  │    │
│  │  │ ModelRunner │  │  KV Cache Manager    │  │    │
│  │  │             │  │  (7.28 GiB)          │  │    │
│  │  │ Prefill:    │  │  272,608 tokens      │  │    │
│  │  │  并行处理    │  └──────────────────────┘  │    │
│  │  │  prompt     │                             │    │
│  │  │             │  ┌──────────────────────┐  │    │
│  │  │ Decode:     │  │  CUTLASS/FlashAttn   │  │    │
│  │  │  逐 token   │  │  Kernel Dispatcher   │  │    │
│  │  │  自回归     │  └──────────────────────┘  │    │
│  │  └─────────────┘                             │    │
│  └──────────────────────────────────────────────┘    │
│                                                       │
│  GPU: RTX 4070 Ti (12 GB, SM89)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ 权重 2.18GB  │  │ KV 7.28GB    │  │ 临时缓冲区  │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
└──────────────────────────────────────────────────────┘
```

### 6.2 推理时序

```
Prefill (5 tokens, 并行):
  X [5, 1536] → q_proj(X) → k_proj(X) → v_proj(X)
  → FlashAttention(Q,K,V) → o_proj
  → post_attention_layernorm
  → gate_proj → SiLU → × up_proj → down_proj
  → ... ×28 层 ...
  → lm_head → logits → sample first token

Decode (214 tokens, 逐 token):
  new_token [1, 1536] → (同上, 但每次只处理 1 个 token)
  → 读 KV Cache (累积的历史 K,V)
  → Attention(Q_new, K_cache, V_cache)
  → 生成下一个 token
  → 更新 KV Cache
```

### 6.3 关键 kernel 源码位置

| Kernel | 源文件 |
|--------|--------|
| CUTLASS FP8 GEMM | `vllm/model_executor/layers/quantization/fp8.py:446` (`Fp8LinearMethod.apply`) |
| FlashAttention | `vllm/v1/attention/backends/flash_attn.py` |
| RMSNorm | `vllm/platforms/cuda.py` (C++ extension) |
| FlashInfer Sampling | `vllm/v1/worker/gpu/sample/sampler.py` → `flashinfer.sampling` |
| Qwen2 Model | `vllm/model_executor/models/qwen2.py:117` (`Qwen2Attention`) |
| FP8 ScaledMM | `vllm/_custom_ops.py` → `ops.cutlass_scaled_mm()` |

---

## 7. 性能分析与优化

### 7.1 实测数据

```
模型: DeepSeek-R1-Distill-Qwen-1.5B-FP8
环境: WSL2 + RTX 4070 Ti

首次启动:
  ├─ 权重加载:     3s (2.09 GiB, ~1.6 GB/s)
  ├─ FlashInfer JIT: 432s (首次编译, 后续缓存)
  ├─ KV Cache 分配:  2s
  └─ 总计:         443s

推理 (prompt=5 tokens, output=214 tokens):
  ├─ Prefill: ~0.1s
  ├─ Decode:  ~29s (7.28 tok/s)
  └─ 总计:    ~30s

显存:
  ├─ 权重:  2.18 GiB
  ├─ KV Cache: 7.28 GiB (272,608 tokens)
  └─ 其他:  ~1.5 GiB
  总计: ~10.96 GiB / 12 GiB
```

### 7.2 瓶颈

| 瓶颈 | 影响 | 可优化 |
|------|------|--------|
| FlashInfer JIT 首次编译 | 432s | ✅ 缓存自动解决 |
| WSL2 CUDA Graph 禁用 | 推理慢 2-3x | ❌ WSL2 限制 |
| WSL2 GPU 驱动转发 | 每次 kernel launch 延迟 | ❌ WSL2 限制 |
| EXT4 本地文件系统 | 权重加载 ~1s | ⚠️ 可预取 |

### 7.3 优化后（第二次启动）

```
初始化: 443s → ~10s (缓存命中)
推理速度: 7.28 tok/s (WSL2 正常水平)
```

---

## 8. 项目文件结构

```
llm-quantization/
├── PROJECT.md                          ← 本文档
├── SETUP_REFERENCE.md                  ← 环境配置记录
├── models/
│   ├── DeepSeek-R1-Distill-Qwen-1.5B/  ← 原始模型 (3.3 GB BF16)
│   ├── ...-FP8/                        ← 方法1: FP8 (2.1 GB)
│   ├── ...-INT8-W8A16/                 ← 方法2: INT8 W8A16 (2.1 GB)
│   ├── ...-INT8-W8A8-Tensor/           ← 方法3: INT8 W8A8 tensor
│   ├── ...-INT8-W8A8-Channel/          ← 方法4: INT8 W8A8 channel
│   └── ...-SmoothQuant/                ← 方法5: SmoothQuant
└── scripts/
    ├── quantize_vllm.py                ← 基础量化脚本 (方法1-2)
    └── quantize_methods.py             ← 5种方案对比脚本 (方法1-5)
```

### 脚本使用

```bash
conda activate trtllm
export PATH="/home/jake/miniconda3/envs/trtllm/bin:$PATH"

# 量化
python scripts/quantize_methods.py --method fp8 --force
python scripts/quantize_methods.py --method int8_smoothquant --force --calib-samples 128

# 全部 5 种一次运行
python scripts/quantize_methods.py --method all --force --calib-samples 64

# 推理
python -c "
from vllm import LLM, SamplingParams
llm = LLM(model='models/DeepSeek-R1-Distill-Qwen-1.5B-FP8',
          quantization='fp8', max_model_len=2048,
          gpu_memory_utilization=0.85, enforce_eager=True)
for o in llm.generate(['你好'], SamplingParams(max_tokens=256)):
    print(o.outputs[0].text)
"
```

---

## 9. 技能展示

### 技术广度

| 领域 | 具体技能 |
|------|---------|
| **模型结构** | Qwen2 架构 (GQA, SwiGLU, RMSNorm, RoPE), 参数量计算, 蒸馏模型 |
| **量化理论** | FP8 E4M3, INT8/INT4, per-tensor/per-group/per-channel, SmoothQuant |
| **推理框架** | vLLM V1 引擎, CUTLASS kernel 调度, FlashAttention, FlashInfer, NCCL |
| **GPU 硬件** | SM89 Ada Lovelace, WGMMA, Tensor Core, KV Cache 管理 |
| **数值表示** | IEEE 754 变体, E4M3 格式, 量化误差分析 (scale/2) |
| **工具链** | PyTorch, safetensors, compressed-tensors, transformers, ninja/nvcc |

### 工程能力

| 能力 | 体现 |
|------|------|
| **调试** | 追踪 llm-compressor 安装失败 → compressed-tensors observer 不触发 → 绕开整个工具链 |
| **性能分析** | 定位 JIT 编译 432s 瓶颈, WSL2 特有的 spawn/CUDA Graph/PATH 问题 |
| **系统理解** | vLLM 多进程架构, EngineCore 生命周期, kernel dispatch 机制 |
| **实验设计** | 5 种方案按校准需求分组, 控制变量法对比 |
| **文档** | 结构化项目文档, 代码注释, 使用说明 |

### 踩坑经验

1. **WSL2 的特殊性**: CUDA Graph 不支持, fork → spawn, ninja PATH, GPU 驱动转发延迟
2. **工具链成熟度**: llm-compressor 安装有问题 → 不要盲目依赖第三方工具, 理解底层原理后手动实现
3. **vLLM 架构边界**: ParallelLMHead 不走 LinearMethod → 理解模型特定的 dispatch 机制
4. **JIT 编译**: 首次启动慢不代表性能差 → 区分冷启动和稳态
5. **量化不是银弹**: FP8 是最佳选择 (精度+速度+简单), INT8 需要大量技巧才能接近
