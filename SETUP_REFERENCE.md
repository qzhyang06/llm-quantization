# WSL2 + RTX 4070 Ti 12GB 部署 LLM 环境记录

## 环境
- GPU: RTX 4070 Ti 12GB (Ada Lovelace SM89)
- OS: Ubuntu 26.04 WSL2
- RAM: 15GB | Swap: 4GB
- CUDA: 13.1 | Driver: 591.86
- Python: conda 环境 `trtllm` (Python 3.12)

## vLLM 安装与修复

### 安装
```bash
conda activate trtllm
pip install vllm -i https://pypi.tuna.tsinghua.edu.cn/simple --trusted-host pypi.tuna.tsinghua.edu.cn
pip uninstall torchcodec -y  # 修复 libavutil.so.56 缺失
```

### 必需修复（WSL2 特有）

**1. UVA 检测 patch** — WSL2 实际支持 pinned memory，但 vLLM 检测返回 False
```bash
# 修改文件: $CONDA_PREFIX/lib/python3.12/site-packages/vllm/utils/platform_utils.py
# 第 56 行，将:
#   return is_pin_memory_available() or current_platform.is_cpu()
# 改为:
#   return True  # Patched for WSL2
```

**2. FlashInfer 缓存权限** — Docker 用 root 创建的目录，普通用户无法写入
```bash
rm -rf ~/.cache/flashinfer
```

### 可用模型

| 模型 | 大小 | 量化 | 状态 |
|------|------|------|------|
| `casperhansen/deepseek-r1-distill-qwen-7b-awq` | 5.2GB | AWQ INT4 | ✅ 已验证 |
| `boboliu/Qwen2-7B-Instruct-FP8-CN` | 8.2GB | FP8 (compressed-tensors) | 模型已下载，待验证 |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 15GB | FP16 | ❌ 太大会 OOM |

### 运行命令

```bash
conda activate trtllm

python3 -c "
from vllm import LLM, SamplingParams
llm = LLM(
    model='/home/jake/LLM-Deploy/TensorRT-LLM/models/DeepSeek-R1-Distill-Qwen-7B-AWQ',
    max_model_len=2048,
    gpu_memory_utilization=0.85,
    enforce_eager=True,       # WSL2 必须关闭 CUDA Graph
)
for o in llm.generate(['你的 prompt'], SamplingParams(max_tokens=256, temperature=0.6)):
    print(o.outputs[0].text)
"
```

## TRT-LLM (Docker) 经验

### Docker 镜像
```bash
docker run -d --rm --ipc=host --gpus all \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -p 8000:8000 \
    -v ~/.cache:/root/.cache:rw \
    -v $(pwd):/workspace:rw \
    --name trtllm-dev \
    nvcr.io/nvidia/tensorrt-llm/release:1.2.1 \
    sleep infinity
```

### TRT-LLM 权重加载 OOM 修复
TRT-LLM 并行加载权重时 16+ 线程同时 `.to("cuda")` 导致碎片化 OOM。
修复：加载前 monkey-patch `run_concurrently` 强制 `num_workers=1`。

```python
from tensorrt_llm._torch.models import modeling_utils
_original = modeling_utils.run_concurrently
def _patched(func, args_list, reduce_func=None, pbar=None, num_workers=None):
    return _original(func, args_list, reduce_func, pbar, num_workers=1)
modeling_utils.run_concurrently = _patched
```

### TRT-LLM 量化兼容性
- ✅ 支持: `hf_quant_config.json` (ModelOpt 格式), `quant_method: fp8` (config.json)
- ❌ 不支持: `compressed-tensors`, `awq`, `gptq` (PyTorch 后端)

## HF 镜像
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

## 关键教训
1. **12GB GPU + 15GB RAM 不能做 7B PTQ**：FP16 模型 14.3GB 装不进内存
2. **本地量化的上限是 1.5B 模型**：FP16 ~3GB，可以装入 15GB RAM
3. **WSL2 有 UVA/CUDA Graph 兼容性问题**：必须 `enforce_eager=True`
4. **社区 FP8 模型全是 compressed-tensors 格式**：TRT-LLM 不支持，vLLM 支持
5. **Docker 用 root 创建的文件普通用户删不掉**：需要 `sudo rm -rf`
