#!/usr/bin/env python3
"""
vLLM 量化脚本 — 对 DeepSeek-R1-Distill-Qwen-1.5B 进行多种量化

直接使用 PyTorch + safetensors 进行权重量化，输出格式兼容 vLLM。

支持:
  - FP8     — 权重 FP8 (E4M3)，vLLM --quantization fp8
  - INT8    — 权重 INT8 (per-channel)，vLLM --quantization compressed-tensors
  - W4A16   — 权重 INT4 (per-group)，vLLM --quantization compressed-tensors

用法:
  python quantize_vllm.py --method fp8           # FP8 量化（推荐）
  python quantize_vllm.py --method int8          # INT8 权重量化
  python quantize_vllm.py --method w4a16         # INT4 权重量化
  python quantize_vllm.py --method all           # 全部量化
  python quantize_vllm.py --method fp8 --verify  # 量化后 vLLM 验证

环境:
  conda activate trtllm
"""

import argparse
import gc
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from transformers import AutoConfig, AutoTokenizer

# ── 路径配置 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_DIR / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_BASE = PROJECT_DIR / "models"


# ════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════

def check_model():
    """检查源模型是否完整"""
    safetensors = MODEL_DIR / "model.safetensors"
    config = MODEL_DIR / "config.json"
    if not safetensors.exists():
        print(f"❌ 模型文件不存在: {safetensors}")
        sys.exit(1)
    if not config.exists():
        print(f"❌ config.json 不存在: {config}")
        sys.exit(1)
    size_gb = safetensors.stat().st_size / (1024**3)
    print(f"✅ 源模型就绪: {MODEL_DIR} ({size_gb:.1f} GB)")
    return str(MODEL_DIR)


def get_linear_layers(state_dict):
    """找出 state_dict 中所有 Linear 层的权重名"""
    linear_weights = []
    for name, tensor in state_dict.items():
        if name.endswith(".weight") and tensor.ndim == 2:
            # 排除 embedding 和 lm_head（通常保留原精度）
            if "embed" not in name.lower() and "lm_head" not in name.lower():
                linear_weights.append(name)
    # 注意: lm_head 不量化（vLLM 的 ParallelLMHead 不支持外部 scale）
    return linear_weights


# ════════════════════════════════════════════════════════════
#  FP8 量化 (E4M3, per-tensor symmetric)
# ════════════════════════════════════════════════════════════

def quantize_to_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将 bf16/fp16 权重量化为 FP8 (E4M3) 格式。
    返回: (fp8_weight, fp8_scale)

    FP8 E4M3 格式:
    - 指数: 4 bits, 尾数: 3 bits
    - 可表示范围: ±448.0
    - max norm: ±240.0, min norm: ±0.001953125
    """
    # FP8 E4M3 max value
    FP8_MAX = 240.0  # 使用安全值而非理论最大 448.0

    # Per-tensor symmetric scale
    amax = weight.abs().max()
    scale = amax / FP8_MAX

    # 防止 scale 过小导致精度损失
    scale = torch.clamp(scale, min=1e-12)

    # 量化
    fp8_weight = weight.float() / scale
    fp8_weight = fp8_weight.clamp(-FP8_MAX, FP8_MAX)
    fp8_weight = fp8_weight.to(torch.float8_e4m3fn)

    return fp8_weight, scale.to(weight.dtype)


def quantize_fp8(model_path: str, output_dir: str):
    """
    FP8 权重量化 — 每个 Linear 层使用 per-tensor scale。
    输出兼容 vLLM --quantization fp8。
    """
    print("\n" + "=" * 60)
    print("  FP8 权重量化 (E4M3)")
    print("=" * 60)

    safetensors_path = os.path.join(model_path, "model.safetensors")
    config_path = os.path.join(model_path, "config.json")

    print("  加载权重...")
    state_dict = load_file(safetensors_path)
    linear_names = get_linear_layers(state_dict)
    print(f"  找到 {len(linear_names)} 个 Linear 层")

    # 量化每个 Linear 层
    new_state_dict = {}
    fp8_scales = {}
    quantized_count = 0
    skipped_count = 0

    for name, tensor in state_dict.items():
        if name in linear_names and tensor.ndim == 2:
            try:
                fp8_w, scale = quantize_to_fp8(tensor)
                new_state_dict[name] = fp8_w
                # vLLM 期望 scale 以 _scale_inv 命名（1/scale）或 _scale
                scale_name = name.replace(".weight", ".weight_scale")
                new_state_dict[scale_name] = scale
                fp8_scales[name] = scale
                quantized_count += 1
            except Exception as e:
                print(f"    ⚠️  量化失败 {name}: {e}, 保留原精度")
                new_state_dict[name] = tensor
                skipped_count += 1
        else:
            # 非 Linear 层保持原精度（layernorm, embedding, bias 等）
            new_state_dict[name] = tensor

    print(f"  量化: {quantized_count} 层, 跳过: {skipped_count} 层")

    # 保存量化权重
    os.makedirs(output_dir, exist_ok=True)
    output_safetensors = os.path.join(output_dir, "model.safetensors")
    print(f"  保存量化权重...")
    save_file(new_state_dict, output_safetensors)

    # 更新 config.json
    with open(config_path) as f:
        config = json.load(f)

    config["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
    }
    # FP8 模型用 bfloat16 做计算
    config["torch_dtype"] = "bfloat16"

    output_config = os.path.join(output_dir, "config.json")
    with open(output_config, "w") as f:
        json.dump(config, f, indent=2)

    # 拷贝 tokenizer 等文件
    for fname in ["tokenizer.json", "tokenizer_config.json", "generation_config.json",
                   "vocab.json", "merges.txt", "special_tokens_map.json"]:
        src = os.path.join(model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    # 统计大小
    total_mb = sum(os.path.getsize(os.path.join(output_dir, f))
                   for f in os.listdir(output_dir)) / (1024**2)
    print(f"✅ FP8 量化完成！模型大小: {total_mb:.1f} MB")
    return output_dir


# ════════════════════════════════════════════════════════════
#  INT8 量化 (per-channel symmetric, group_size=128)
# ════════════════════════════════════════════════════════════

def quantize_to_int8(weight: torch.Tensor, group_size: int = 128
                     ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将权重量化为 INT8 (per-group symmetric)。
    返回: (int8_weight, scale)
    """
    out_features, in_features = weight.shape

    if in_features % group_size != 0:
        # 填充到 group_size 的倍数
        pad_size = group_size - (in_features % group_size)
        weight = torch.cat([weight, torch.zeros(out_features, pad_size,
                                                 dtype=weight.dtype)], dim=1)
        out_features, in_features = weight.shape

    n_groups = in_features // group_size
    reshaped = weight.view(out_features, n_groups, group_size)

    # Per-group symmetric scale: scale = max(|w|) / 127
    amax = reshaped.abs().amax(dim=-1, keepdim=True)  # [out, n_groups, 1]
    amax = torch.clamp(amax, min=1e-12)
    scale = (amax / 127.0).to(weight.dtype)  # [out, n_groups, 1]

    # 量化
    q_weight = reshaped.float() / scale.float()
    q_weight = q_weight.round().clamp(-128, 127).to(torch.int8)
    q_weight = q_weight.view(out_features, in_features)

    # 如果填充过，去掉填充
    if in_features != weight.shape[0]:
        q_weight = q_weight[:weight.shape[0], :weight.shape[1]]
        scale = scale[:weight.shape[0], :n_groups]

    return q_weight, scale.squeeze(-1)  # [out, n_groups]


def quantize_int8(model_path: str, output_dir: str, group_size: int = 128):
    """
    INT8 权重量化 — per-group symmetric。
    输出兼容 vLLM --quantization compressed-tensors。
    """
    print("\n" + "=" * 60)
    print(f"  INT8 权重量化 (group_size={group_size})")
    print("=" * 60)

    safetensors_path = os.path.join(model_path, "model.safetensors")
    config_path = os.path.join(model_path, "config.json")

    print("  加载权重...")
    state_dict = load_file(safetensors_path)
    linear_names = get_linear_layers(state_dict)
    print(f"  找到 {len(linear_names)} 个 Linear 层")

    new_state_dict = {}
    quantized_count = 0

    for name, tensor in state_dict.items():
        if name in linear_names and tensor.ndim == 2:
            try:
                q_weight, scale = quantize_to_int8(tensor, group_size)
                new_state_dict[name] = q_weight

                scale_name = name.replace(".weight", ".weight_scale")
                new_state_dict[scale_name] = scale

                # INT8 zero point (symmetric → all zeros)
                zp_name = name.replace(".weight", ".weight_zero_point")
                new_state_dict[zp_name] = torch.zeros_like(scale, dtype=torch.int8)

                quantized_count += 1
            except Exception as e:
                print(f"    ⚠️  量化失败 {name}: {e}, 保留原精度")
                new_state_dict[name] = tensor
        else:
            new_state_dict[name] = tensor

    print(f"  量化: {quantized_count} 层")

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_safetensors = os.path.join(output_dir, "model.safetensors")
    print(f"  保存量化权重...")
    save_file(new_state_dict, output_safetensors)

    # 更新 config.json — compressed-tensors 格式
    with open(config_path) as f:
        config = json.load(f)

    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 8,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": group_size,
                },
                "targets": ["Linear"],
            }
        },
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
    }
    config["torch_dtype"] = "bfloat16"

    output_config = os.path.join(output_dir, "config.json")
    with open(output_config, "w") as f:
        json.dump(config, f, indent=2)

    # 拷贝 tokenizer 文件
    for fname in ["tokenizer.json", "tokenizer_config.json", "generation_config.json",
                   "vocab.json", "merges.txt", "special_tokens_map.json"]:
        src = os.path.join(model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    total_mb = sum(os.path.getsize(os.path.join(output_dir, f))
                   for f in os.listdir(output_dir)) / (1024**2)
    print(f"✅ INT8 量化完成！模型大小: {total_mb:.1f} MB")
    return output_dir


# ════════════════════════════════════════════════════════════
#  INT4 量化 (per-group symmetric, group_size=128)
# ════════════════════════════════════════════════════════════

def quantize_to_int4(weight: torch.Tensor, group_size: int = 128
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    INT4 权重量化 — per-group symmetric。
    返回: (int4_weight_packed, scale, zero_point)

    注意: INT4 值打包为 int8 (两个 INT4 值放在一个 INT8 字节中)。
    """
    out_features, in_features = weight.shape

    if in_features % group_size != 0:
        pad_size = group_size - (in_features % group_size)
        weight = torch.cat([weight, torch.zeros(out_features, pad_size,
                                                 dtype=weight.dtype)], dim=1)
        out_features, in_features = weight.shape

    n_groups = in_features // group_size
    reshaped = weight.view(out_features, n_groups, group_size)

    # Per-group symmetric scale
    amax = reshaped.abs().amax(dim=-1, keepdim=True)
    amax = torch.clamp(amax, min=1e-12)
    scale = (amax / 7.0).to(weight.dtype)  # INT4 max = 7 (symmetric)

    # 量化到 [-7, 7]
    q_weight = reshaped.float() / scale.float()
    q_weight = q_weight.round().clamp(-8, 7).to(torch.int8)
    q_weight = q_weight.view(out_features, in_features)

    if in_features % 2 != 0:
        q_weight = torch.cat([q_weight, torch.zeros(out_features, 1, dtype=torch.int8)], dim=1)

    # 打包两个 INT4 → 一个 INT8
    packed_len = q_weight.shape[1] // 2
    q_even = q_weight[:, 0::2].clone()
    q_odd = q_weight[:, 1::2].clone()
    packed = ((q_even & 0x0F) | ((q_odd & 0x0F) << 4)).to(torch.uint8)

    return packed, scale.squeeze(-1), torch.zeros_like(scale.squeeze(-1), dtype=torch.int8)


def quantize_w4a16(model_path: str, output_dir: str, group_size: int = 128):
    """
    INT4 权重量化 — per-group symmetric。
    输出兼容 vLLM --quantization compressed-tensors。
    """
    print("\n" + "=" * 60)
    print(f"  INT4 权重量化 (group_size={group_size})")
    print("=" * 60)

    safetensors_path = os.path.join(model_path, "model.safetensors")
    config_path = os.path.join(model_path, "config.json")

    print("  加载权重...")
    state_dict = load_file(safetensors_path)
    linear_names = get_linear_layers(state_dict)
    # 排除 lm_head (保留原精度)
    linear_names = [n for n in linear_names if "lm_head" not in n.lower()]
    print(f"  找到 {len(linear_names)} 个 Linear 层")

    new_state_dict = {}
    quantized_count = 0

    for name, tensor in state_dict.items():
        if name in linear_names and tensor.ndim == 2:
            try:
                packed, scale, zp = quantize_to_int4(tensor, group_size)
                new_state_dict[name] = packed

                scale_name = name.replace(".weight", ".weight_scale")
                new_state_dict[scale_name] = scale

                zp_name = name.replace(".weight", ".weight_zero_point")
                new_state_dict[zp_name] = zp

                # g_idx for group quantization
                g_idx_name = name.replace(".weight", ".weight_g_idx")
                in_f = tensor.shape[1]
                n_groups_val = (in_f + group_size - 1) // group_size
                g_idx = torch.arange(in_f, dtype=torch.int32)
                g_idx = g_idx // group_size
                new_state_dict[g_idx_name] = g_idx

                quantized_count += 1
            except Exception as e:
                print(f"    ⚠️  量化失败 {name}: {e}, 保留原精度")
                new_state_dict[name] = tensor
        else:
            new_state_dict[name] = tensor

    print(f"  量化: {quantized_count} 层")

    # 保存
    os.makedirs(output_dir, exist_ok=True)
    output_safetensors = os.path.join(output_dir, "model.safetensors")
    print(f"  保存量化权重...")
    save_file(new_state_dict, output_safetensors)

    # 更新 config.json
    with open(config_path) as f:
        config = json.load(f)

    config["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {
            "group_0": {
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": True,
                    "strategy": "group",
                    "group_size": group_size,
                },
                "targets": ["Linear"],
            }
        },
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
    }
    config["torch_dtype"] = "bfloat16"

    output_config = os.path.join(output_dir, "config.json")
    with open(output_config, "w") as f:
        json.dump(config, f, indent=2)

    for fname in ["tokenizer.json", "tokenizer_config.json", "generation_config.json",
                   "vocab.json", "merges.txt", "special_tokens_map.json"]:
        src = os.path.join(model_path, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    total_mb = sum(os.path.getsize(os.path.join(output_dir, f))
                   for f in os.listdir(output_dir)) / (1024**2)
    print(f"✅ INT4 量化完成！模型大小: {total_mb:.1f} MB")
    return output_dir


# ════════════════════════════════════════════════════════════
#  vLLM 推理验证
# ════════════════════════════════════════════════════════════

def verify_quantized_model(model_dir: str, quant_method: str):
    """验证量化模型在 vLLM 中能正常加载和推理"""
    print(f"\n--- vLLM 验证: {quant_method} @ {model_dir} ---")

    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        return False

    quant_arg_map = {
        "fp8": "fp8",
        "int8": "compressed-tensors",
        "w4a16": "compressed-tensors",
    }
    quant_arg = quant_arg_map.get(quant_method, "compressed-tensors")

    test_script = f"""
import sys
from vllm import LLM, SamplingParams

try:
    llm = LLM(
        model="{model_dir}",
        quantization="{quant_arg}",
        max_model_len=1024,
        gpu_memory_utilization=0.80,
        enforce_eager=True,
        trust_remote_code=True,
        max_num_seqs=1,
    )
    result = llm.generate(
        ["Hello, how are you?"],
        SamplingParams(max_tokens=32, temperature=0.0),
    )
    output = result[0].outputs[0].text
    print(f"推理成功! 输出: {{output[:100]}}")
    print("✅ 量化模型验证通过")
except Exception as e:
    print(f"❌ 验证失败: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
"""
    python_bin = "/home/jake/miniconda3/envs/trtllm/bin/python"
    result = subprocess.run(
        [python_bin, "-c", test_script],
        capture_output=True, text=True, timeout=600,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        return False
    return True


# ════════════════════════════════════════════════════════════
#  模型大小对比
# ════════════════════════════════════════════════════════════

def print_size_comparison(original_path: str, quantized_paths: dict):
    """打印量化前后大小对比"""
    print("\n" + "=" * 60)
    print("  模型大小对比")
    print("=" * 60)

    def get_dir_size_gb(path):
        if not path or not os.path.exists(path):
            return 0
        total = 0
        for root, dirs, files in os.walk(path):
            if ".cache" in root:
                continue
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / (1024**3)

    orig_size = get_dir_size_gb(original_path)
    print(f"\n  {'模型':<35} {'大小':>10} {'压缩比':>10}")
    print(f"  {'─' * 35} {'─' * 10} {'─' * 10}")
    print(f"  {'BF16 (原始)':<35} {orig_size:>8.2f} GB {'1.00x':>10}")

    for name, path in quantized_paths.items():
        if path:
            qsize = get_dir_size_gb(path)
            ratio = orig_size / qsize if qsize > 0 else 0
            mem_est = qsize * 1.1  # 估算推理时显存占用 (含 KV cache)
            print(f"  {name:<35} {qsize:>8.2f} GB {ratio:>8.2f}x")


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

METHOD_MAP = {
    "fp8": ("DeepSeek-R1-Distill-Qwen-1.5B-FP8", quantize_fp8, "fp8"),
    "int8": ("DeepSeek-R1-Distill-Qwen-1.5B-INT8", quantize_int8, "int8"),
    "w4a16": ("DeepSeek-R1-Distill-Qwen-1.5B-W4A16", quantize_w4a16, "w4a16"),
}


def main():
    parser = argparse.ArgumentParser(
        description="vLLM 模型量化脚本 — DeepSeek-R1-Distill-Qwen-1.5B",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python quantize_vllm.py --method fp8           # FP8 量化（推荐，~1.7 GB）
  python quantize_vllm.py --method int8          # INT8 权重量化（~1.7 GB）
  python quantize_vllm.py --method w4a16         # INT4 权重量化（~0.9 GB）
  python quantize_vllm.py --method all           # 全部量化
  python quantize_vllm.py --method fp8 --verify  # 量化后 vLLM 验证
        """,
    )
    parser.add_argument(
        "--method", "-m",
        choices=["fp8", "int8", "w4a16", "all"],
        default="fp8",
        help="量化方法 (default: fp8)",
    )
    parser.add_argument(
        "--model", "-i",
        default=None,
        help=f"源模型路径 (default: {MODEL_DIR})",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="自定义输出目录",
    )
    parser.add_argument(
        "--verify", "-v",
        action="store_true",
        help="量化完成后用 vLLM 验证推理",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="INT8/INT4 的 group size (default: 128)",
    )
    args = parser.parse_args()

    model_path = args.model or str(MODEL_DIR)
    check_model()

    methods = list(METHOD_MAP.keys()) if args.method == "all" else [args.method]
    quantized_paths = {}

    for method in methods:
        folder_name, quant_func, vllm_quant_name = METHOD_MAP[method]
        output_dir = args.output_dir or str(OUTPUT_BASE / folder_name)

        if os.path.exists(output_dir):
            resp = input(f"\n⚠️  输出目录已存在: {output_dir}\n   覆盖? [y/N]: ")
            if resp.lower() == "y":
                print(f"   删除旧目录: {output_dir}")
                shutil.rmtree(output_dir)
            else:
                print(f"   跳过 {method}")
                quantized_paths[method] = output_dir
                continue

        try:
            if method == "fp8":
                result = quant_func(model_path, output_dir)
            else:
                result = quant_func(model_path, output_dir, group_size=args.group_size)
            quantized_paths[method] = result

            if args.verify and result:
                verify_quantified_model(result, vllm_quant_name)
        except Exception as e:
            print(f"❌ {method} 量化失败: {e}")
            import traceback
            traceback.print_exc()
            quantized_paths[method] = None

    # 打印对比
    print_size_comparison(model_path, quantized_paths)

    # 输出使用说明
    successful = {k: v for k, v in quantized_paths.items() if v}
    if successful:
        print("\n" + "=" * 60)
        print("  vLLM 运行命令")
        print("=" * 60)
        for method, path in successful.items():
            vllm_quant = METHOD_MAP[method][2]
            quant_flag = "fp8" if vllm_quant == "fp8" else "compressed-tensors"
            print(f"\n  # {method.upper()} — {path}")
            print(
                f"  python -c \"\n"
                f"  from vllm import LLM, SamplingParams\n"
                f"  llm = LLM(\n"
                f"      model='{path}',\n"
                f"      quantization='{quant_flag}',\n"
                f"      max_model_len=2048,\n"
                f"      gpu_memory_utilization=0.85,\n"
                f"      enforce_eager=True,\n"
                f"  )\n"
                f"  for o in llm.generate(['你好'], SamplingParams(max_tokens=256)):\n"
                f"      print(o.outputs[0].text)\n"
                f"  \""
            )


if __name__ == "__main__":
    main()
