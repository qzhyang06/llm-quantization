"""最小化 profiling 脚本 — 避免 JIT 编译进入 profile 范围"""
import torch
from vllm import LLM, SamplingParams

# 预初始化（触发 JIT 编译，不在 profile 范围内）
print("预初始化模型（触发 JIT）...")
llm = LLM(
    model='models/DeepSeek-R1-Distill-Qwen-1.5B-FP8',
    quantization='fp8',
    max_model_len=256,
    gpu_memory_utilization=0.80,
    enforce_eager=True,
)
# 预热一次
_ = llm.generate(['热身'], SamplingParams(max_tokens=4))
print("预热完成，开始 profiling...")

# Profile 范围：只包含推理，不包含 JIT
torch.cuda.profiler.start()
result = llm.generate(
    ['用中文解释量子计算'],
    SamplingParams(max_tokens=16, temperature=0.6),
)
torch.cuda.profiler.stop()

print(f"输出: {result[0].outputs[0].text[:100]}")
print("Profiling 完成")
