#!/usr/bin/env python3
"""
vLLM 量化脚本 (校准版本) — 5 种量化方案对比

方法 1: fp8              FP8 权重 (per-tensor) + 动态 FP8 激活 (on-the-fly)
方法 2: int8_w8a16        INT8 权重 (per-group) + BF16 激活 (不量化)
方法 3: int8_w8a8_tensor  INT8 权重 (per-group) + INT8 激活 (per-tensor 静态校准)
方法 4: int8_w8a8_channel INT8 权重 (per-group) + INT8 激活 (per-channel 静态校准)
方法 5: int8_smoothquant  SmoothQuant 平滑 + INT8 W8A8 (per-tensor)

方法 1-2 不需要校准数据（纯权重变换）
方法 3-5 需要校准数据（需要统计激活值范围）

用法:
  # 方法 1-2 (不需要 GPU 加载模型)
  python quantize_methods.py --method fp8
  python quantize_methods.py --method int8_w8a16

  # 方法 3-5 (需要 GPU + 校准数据)
  python quantize_methods.py --method int8_w8a8_tensor --calib-samples 128
  python quantize_methods.py --method int8_w8a8_channel --calib-samples 128
  python quantize_methods.py --method int8_smoothquant --calib-samples 128

  # 全部 5 种
  python quantize_methods.py --method all
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── 路径配置 ──────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MODEL_DIR = PROJECT_DIR / "models" / "DeepSeek-R1-Distill-Qwen-1.5B"
OUTPUT_BASE = PROJECT_DIR / "models"

MAX_SEQ_LEN = 2048
CALIB_SAMPLES = 128


# ════════════════════════════════════════════════════════════
#  校准数据
# ════════════════════════════════════════════════════════════

def get_calibration_data(tokenizer, num_samples=CALIB_SAMPLES):
    """获取校准数据"""
    try:
        from datasets import load_dataset
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True, trust_remote_code=True)
        samples = []
        for item in ds:
            text = item["text"].strip()
            if 200 < len(text) < MAX_SEQ_LEN * 4:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                if 512 <= len(tokens) <= MAX_SEQ_LEN:
                    samples.append(text)
            if len(samples) >= num_samples:
                break
        print(f"  ✅ 从 c4 加载了 {len(samples)} 条校准样本")
        return samples
    except Exception as e:
        print(f"  ⚠️  c4 不可用 ({e})，使用合成数据")
        templates = [
            "The quick brown fox jumps over the lazy dog. ",
            "Machine learning is a fascinating field of artificial intelligence. ",
            "Deep learning models have achieved remarkable results in NLP. ",
            "Quantization techniques can significantly reduce model size. ",
            "The transformer architecture uses self-attention mechanisms. ",
        ]
        samples = []
        for i in range(num_samples):
            text = templates[i % len(templates)] * 80
            samples.append(text[:MAX_SEQ_LEN * 4])
        return samples


# ════════════════════════════════════════════════════════════
#  激活值收集 (用于方法 3-5)
# ════════════════════════════════════════════════════════════

class ActivationCollector:
    """通过 forward hook 收集每层 Linear 的输入激活值"""

    def __init__(self, model):
        self.activation_stats = defaultdict(list)  # layer_name → [tensor_max, ...]
        self.hooks = []
        self._register_hooks(model)

    def _register_hooks(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                hook = module.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)

    def _make_hook(self, name):
        def hook(module, input, output):
            # input[0] shape: [batch*seq, in_features] or [batch, seq, in_features]
            x = input[0].detach().float()
            if x.ndim == 3:
                x = x.view(-1, x.shape[-1])
            # 只保留每列的 absmax，省内存
            if x.shape[0] > 0:
                self.activation_stats[name].append(
                    x.abs().amax(dim=0).cpu()  # [in_features]
                )
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def run_calibration(model, tokenizer, calib_data):
    """用校准数据跑前向，收集激活统计"""
    collector = ActivationCollector(model)
    model.eval()

    print(f"  校准中 ({len(calib_data)} 样本)...")
    t0 = time.time()

    with torch.no_grad():
        for i, text in enumerate(calib_data):
            inputs = tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=MAX_SEQ_LEN,
            ).to(model.device)
            _ = model(**inputs)

            if (i + 1) % 32 == 0:
                print(f"    [{i + 1}/{len(calib_data)}]")

    elapsed = time.time() - t0
    print(f"  校准完成: {elapsed:.1f}s ({len(calib_data) / elapsed:.1f} samples/s)")

    # 聚合统计: 每层取所有样本的最大值
    stats = {}
    for name, max_list in collector.activation_stats.items():
        stacked = torch.stack(max_list)  # [N_samples, in_features]
        stats[name] = {
            "per_tensor_max": stacked.max().item(),       # 标量 — 用于 per-tensor
            "per_channel_max": stacked.amax(dim=0),       # [in_features] — 用于 per-channel
        }
    collector.remove()
    return stats


# ════════════════════════════════════════════════════════════
#  量化函数 (纯权重，不需要校准)
# ════════════════════════════════════════════════════════════

def get_linear_layers(state_dict):
    linear_weights = []
    for name, tensor in state_dict.items():
        if name.endswith(".weight") and tensor.ndim == 2:
            if "embed" not in name.lower() and "lm_head" not in name.lower():
                linear_weights.append(name)
    return linear_weights


def quantize_to_fp8(weight: torch.Tensor):
    FP8_MAX = 240.0
    amax = weight.abs().max()
    scale = (amax / FP8_MAX).clamp(min=1e-12)
    fp8_weight = (weight.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return fp8_weight, scale.to(weight.dtype)


def quantize_to_int8(weight: torch.Tensor, group_size: int = 128):
    out_f, in_f = weight.shape
    if in_f % group_size != 0:
        pad = group_size - (in_f % group_size)
        weight = torch.cat([weight, torch.zeros(out_f, pad, dtype=weight.dtype)], dim=1)
        out_f, in_f = weight.shape

    n_groups = in_f // group_size
    reshaped = weight.view(out_f, n_groups, group_size)
    amax = reshaped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = (amax / 127.0).to(weight.dtype)
    q_weight = reshaped.float().div(scale.float()).round().clamp(-128, 127)
    q_weight = q_weight.view(out_f, in_f).to(torch.int8)
    return q_weight, scale.squeeze(-1)


def copy_non_quantized_files(src_dir, dst_dir):
    for fname in ["tokenizer.json", "tokenizer_config.json", "generation_config.json",
                   "vocab.json", "merges.txt", "special_tokens_map.json"]:
        src = os.path.join(src_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fname))


# ════════════════════════════════════════════════════════════
#  方法 1: FP8 权重 + 动态 FP8 激活
# ════════════════════════════════════════════════════════════

def method_fp8(model_path, output_dir):
    """FP8 权重量化 (per-tensor) + vLLM 动态 FP8 激活"""
    print("\n" + "=" * 60)
    print("  方法 1: FP8 W8A8 (权重 per-tensor + 激活 dynamic)")
    print("=" * 60)

    st = load_file(os.path.join(model_path, "model.safetensors"))
    linear_names = get_linear_layers(st)
    print(f"  Linear 层: {len(linear_names)}")

    new_st = {}
    for name, t in st.items():
        if name in linear_names and t.ndim == 2:
            w, s = quantize_to_fp8(t)
            new_st[name] = w
            new_st[name.replace(".weight", ".weight_scale")] = s
        else:
            new_st[name] = t

    os.makedirs(output_dir, exist_ok=True)
    save_file(new_st, os.path.join(output_dir, "model.safetensors"))

    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {"quant_method": "fp8", "activation_scheme": "dynamic"}
    cfg["torch_dtype"] = "bfloat16"
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    copy_non_quantized_files(model_path, output_dir)
    size_mb = sum(os.path.getsize(os.path.join(output_dir, f))
                  for f in os.listdir(output_dir)) / (1024**2)
    print(f"  ✅ 完成: {size_mb:.1f} MB")
    return output_dir


# ════════════════════════════════════════════════════════════
#  方法 2: INT8 权重 + BF16 激活 (W8A16)
# ════════════════════════════════════════════════════════════

def method_int8_w8a16(model_path, output_dir, group_size=128):
    """INT8 权重量化 (per-group) + BF16 激活 (不量化)"""
    print("\n" + "=" * 60)
    print("  方法 2: INT8 W8A16 (权重 per-group + 激活 BF16)")
    print("=" * 60)

    st = load_file(os.path.join(model_path, "model.safetensors"))
    linear_names = get_linear_layers(st)
    print(f"  Linear 层: {len(linear_names)}")

    new_st = {}
    for name, t in st.items():
        if name in linear_names and t.ndim == 2:
            try:
                w, s = quantize_to_int8(t, group_size)
                new_st[name] = w
                new_st[name.replace(".weight", ".weight_scale")] = s
                new_st[name.replace(".weight", ".weight_zero_point")] = (
                    torch.zeros_like(s, dtype=torch.int8))
            except Exception as e:
                print(f"    ⚠️ {name}: {e}")
                new_st[name] = t
        else:
            new_st[name] = t

    os.makedirs(output_dir, exist_ok=True)
    save_file(new_st, os.path.join(output_dir, "model.safetensors"))

    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    cfg["quantization_config"] = {
        "quant_method": "compressed-tensors",
        "format": "int-quantized",
        "config_groups": {"group_0": {
            "weights": {"num_bits": 8, "type": "int", "symmetric": True,
                         "strategy": "group", "group_size": group_size},
            "targets": ["Linear"],
        }},
        "quantization_status": "compressed",
        "ignore": ["lm_head"],
    }
    cfg["torch_dtype"] = "bfloat16"
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    copy_non_quantized_files(model_path, output_dir)
    size_mb = sum(os.path.getsize(os.path.join(output_dir, f))
                  for f in os.listdir(output_dir)) / (1024**2)
    print(f"  ✅ 完成: {size_mb:.1f} MB")
    return output_dir


# ════════════════════════════════════════════════════════════
#  方法 3-5 共用: 加载模型 + 校准 + 量化为 W8A8
# ════════════════════════════════════════════════════════════

class W8A8Quantizer:
    """
    INT8 W8A8 量化器。
    先加载模型跑校准收集激活统计，然后根据策略计算激活 scale。
    """

    def __init__(self, model_path, calib_data, group_size=128, device="cuda"):
        self.model_path = model_path
        self.group_size = group_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        print("  加载模型到 GPU...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device, trust_remote_code=True,
        )
        print("  收集激活统计...")
        self.act_stats = run_calibration(self.model, self.tokenizer, calib_data)

    def get_layer_stats(self, layer_name):
        """获取某层的激活统计，如果没收集到就返回 None"""
        for prefix in ["model.layers.", ""]:
            key = prefix + layer_name if prefix else layer_name
            if key in self.act_stats:
                return self.act_stats[key]
        # Try to find by suffix
        for k, v in self.act_stats.items():
            if k.endswith(layer_name) or layer_name in k:
                return v
        return None

    def _quantize_weight(self, weight):
        return quantize_to_int8(weight, self.group_size)

    def _save_model(self, state_dict, config, output_dir):
        """保存量化模型"""
        os.makedirs(output_dir, exist_ok=True)
        save_file(state_dict, os.path.join(output_dir, "model.safetensors"))
        config["torch_dtype"] = "bfloat16"
        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        copy_non_quantized_files(self.model_path, output_dir)

    # ── 方法 3: per-tensor 激活量化 ──

    def method_w8a8_tensor(self, output_dir):
        """INT8 权重 (per-group) + INT8 激活 (per-tensor static)"""
        print("\n" + "=" * 60)
        print("  方法 3: INT8 W8A8 tensor (权重 per-group + 激活 per-tensor static)")
        print("=" * 60)

        st = load_file(os.path.join(self.model_path, "model.safetensors"))
        linear_names = get_linear_layers(st)
        new_st = {}
        act_scales = {}  # 存储激活 scale

        for name, t in st.items():
            if name in linear_names and t.ndim == 2:
                # 量化权重
                w, w_scale = self._quantize_weight(t)
                new_st[name] = w
                new_st[name.replace(".weight", ".weight_scale")] = w_scale
                new_st[name.replace(".weight", ".weight_zero_point")] = (
                    torch.zeros_like(w_scale, dtype=torch.int8))

                # 计算激活 scale (per-tensor)
                stats = self.get_layer_stats(name.replace(".weight", ""))
                if stats is not None:
                    act_max = stats["per_tensor_max"]
                else:
                    act_max = 6.0  # 默认值 (经验)
                act_scale = max(act_max, 1e-8) / 127.0
                scale_name = name.replace(".weight", ".input_scale")
                new_st[scale_name] = torch.tensor(act_scale, dtype=torch.bfloat16)
                act_scales[name] = act_scale
            else:
                new_st[name] = t

        with open(os.path.join(self.model_path, "config.json")) as f:
            cfg = json.load(f)
        cfg["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {"group_0": {
                "weights": {"num_bits": 8, "type": "int", "symmetric": True,
                             "strategy": "group", "group_size": self.group_size},
                "input_activations": {"num_bits": 8, "type": "int", "symmetric": True,
                                       "strategy": "tensor"},
                "targets": ["Linear"],
            }},
            "quantization_status": "compressed",
            "ignore": ["lm_head"],
        }
        self._save_model(new_st, cfg, output_dir)

        print(f"  激活 scale 范围: "
              f"[{min(act_scales.values()):.4f}, {max(act_scales.values()):.4f}]")
        size_mb = sum(os.path.getsize(os.path.join(output_dir, f))
                      for f in os.listdir(output_dir)) / (1024**2)
        print(f"  ✅ 完成: {size_mb:.1f} MB")
        return output_dir

    # ── 方法 4: per-channel 激活量化 ──

    def method_w8a8_channel(self, output_dir):
        """INT8 权重 (per-group) + INT8 激活 (per-channel static)"""
        print("\n" + "=" * 60)
        print("  方法 4: INT8 W8A8 channel (权重 per-group + 激活 per-channel static)")
        print("=" * 60)

        st = load_file(os.path.join(self.model_path, "model.safetensors"))
        linear_names = get_linear_layers(st)
        new_st = {}
        act_scales_list = []

        for name, t in st.items():
            if name in linear_names and t.ndim == 2:
                # 量化权重
                w, w_scale = self._quantize_weight(t)
                new_st[name] = w
                new_st[name.replace(".weight", ".weight_scale")] = w_scale
                new_st[name.replace(".weight", ".weight_zero_point")] = (
                    torch.zeros_like(w_scale, dtype=torch.int8))

                # 计算激活 scale (per-input-channel)
                #   每列输入有自己的 scale: scale_j = max(|X_j|) / 127
                #   存储 shape: (1, in_features) — 每个输入 channel 独立
                in_f = t.shape[1]
                stats = self.get_layer_stats(name.replace(".weight", ""))
                if stats is not None:
                    act_max_per_channel = stats["per_channel_max"].clamp(min=1e-8)  # [in_f]
                    act_scale_tensor = (act_max_per_channel / 127.0).unsqueeze(0).to(torch.bfloat16)
                else:
                    act_scale_tensor = torch.full((1, in_f), 6.0 / 127.0, dtype=torch.bfloat16)

                scale_name = name.replace(".weight", ".input_scale")
                new_st[scale_name] = act_scale_tensor
                act_scales_list.append(act_scale_tensor.mean().item())
            else:
                new_st[name] = t

        with open(os.path.join(self.model_path, "config.json")) as f:
            cfg = json.load(f)
        cfg["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {"group_0": {
                "weights": {"num_bits": 8, "type": "int", "symmetric": True,
                             "strategy": "group", "group_size": self.group_size},
                "input_activations": {"num_bits": 8, "type": "int", "symmetric": True,
                                       "strategy": "channel"},
                "targets": ["Linear"],
            }},
            "quantization_status": "compressed",
            "ignore": ["lm_head"],
        }
        self._save_model(new_st, cfg, output_dir)

        if act_scales_list:
            print(f"  激活 scale 范围: [{min(act_scales_list):.4f}, {max(act_scales_list):.4f}]")
        size_mb = sum(os.path.getsize(os.path.join(output_dir, f))
                      for f in os.listdir(output_dir)) / (1024**2)
        print(f"  ✅ 完成: {size_mb:.1f} MB")
        return output_dir

    # ── 方法 5: SmoothQuant ──

    def method_smoothquant(self, output_dir, alpha=0.5):
        """
        SmoothQuant: 把激活量化难度通过数学变换迁移到权重上。

        对每个 Linear Y = X @ W^T:
          s_j = max(|X_j|)^α / max(|W_j|)^(1-α)
          X'_j = X_j / s_j    (通过修改前一层的 LayerNorm 实现)
          W'_j = W_j * s_j    (直接修改权重)

        然后对变换后的权重做 INT8 per-group 量化，对变换后的激活做 per-tensor 量化。
        """
        print("\n" + "=" * 60)
        print(f"  方法 5: SmoothQuant W8A8 (α={alpha})")
        print("=" * 60)

        st = load_file(os.path.join(self.model_path, "model.safetensors"))
        linear_names = get_linear_layers(st)
        new_st = {}

        # SmoothQuant: 对每组共享输入的 Linear，计算并应用平滑因子
        # Qwen2 的共享输入组:
        #   q_proj, k_proj, v_proj 共享 input_layernorm 的输出
        #   gate_proj, up_proj 共享 post_attention_layernorm 的输出
        #   o_proj, down_proj 有独立输入

        layer_count = 0
        for layer_idx in range(100):  # 最多 100 层
            prefix = f"model.layers.{layer_idx}."
            if f"{prefix}input_layernorm.weight" not in st:
                break
            layer_count += 1

        print(f"  模型层数: {layer_count}")

        for layer_idx in range(layer_count):
            prefix = f"model.layers.{layer_idx}."

            # ── Attention 组 (q,k,v 共享 input_layernorm) ──
            attn_weights = ["self_attn.q_proj.weight", "self_attn.k_proj.weight",
                            "self_attn.v_proj.weight"]
            attn_full = [prefix + w for w in attn_weights if prefix + w in st]

            if attn_full:
                # 计算联合平滑因子
                sq_max = None
                for wn in attn_full:
                    w_tensor = st[wn].float()
                    stats = self.get_layer_stats(wn.replace(".weight", ""))
                    if stats is not None:
                        act_max = stats["per_channel_max"].float()  # [in_f]
                    else:
                        act_max = torch.ones(w_tensor.shape[1]) * 6.0
                    w_max = w_tensor.abs().amax(dim=0).float()  # [in_f]
                    s = (act_max ** alpha) / (w_max ** (1 - alpha) + 1e-8)
                    if sq_max is None:
                        sq_max = s
                    else:
                        sq_max = torch.max(sq_max, s)

                # 吸收到 input_layernorm
                ln_name = prefix + "input_layernorm.weight"
                if ln_name in st:
                    st[ln_name] = (st[ln_name].float() / sq_max).to(st[ln_name].dtype)

                # 缩放 Q/K/V 权重
                for wn in attn_full:
                    if wn in st:
                        w_tensor = st[wn]
                        in_f = w_tensor.shape[1]
                        s_clamped = sq_max[:in_f].to(w_tensor.dtype)
                        st[wn] = (w_tensor.float() * s_clamped).to(w_tensor.dtype)

            # ── MLP 组 (gate, up 共享 post_attention_layernorm) ──
            mlp_weights = ["mlp.gate_proj.weight", "mlp.up_proj.weight"]
            mlp_full = [prefix + w for w in mlp_weights if prefix + w in st]

            if mlp_full:
                sq_max = None
                for wn in mlp_full:
                    w_tensor = st[wn].float()
                    stats = self.get_layer_stats(wn.replace(".weight", ""))
                    if stats is not None:
                        act_max = stats["per_channel_max"].float()
                    else:
                        act_max = torch.ones(w_tensor.shape[1]) * 6.0
                    w_max = w_tensor.abs().amax(dim=0).float()
                    s = (act_max ** alpha) / (w_max ** (1 - alpha) + 1e-8)
                    if sq_max is None:
                        sq_max = s
                    else:
                        sq_max = torch.max(sq_max, s)

                ln_name = prefix + "post_attention_layernorm.weight"
                if ln_name in st:
                    st[ln_name] = (st[ln_name].float() / sq_max).to(st[ln_name].dtype)

                for wn in mlp_full:
                    if wn in st:
                        w_tensor = st[wn]
                        in_f = w_tensor.shape[1]
                        s_clamped = sq_max[:in_f].to(w_tensor.dtype)
                        st[wn] = (w_tensor.float() * s_clamped).to(w_tensor.dtype)

            # ── o_proj, down_proj: 独立输入，单独处理 ──
            for proj_name in ["self_attn.o_proj.weight", "mlp.down_proj.weight"]:
                fn = prefix + proj_name
                if fn not in st:
                    continue
                w_tensor = st[fn].float()
                stats = self.get_layer_stats(proj_name.replace(".weight", ""))
                if stats is not None:
                    act_max = stats["per_channel_max"].float()
                else:
                    act_max = torch.ones(w_tensor.shape[1]) * 6.0
                w_max = w_tensor.abs().amax(dim=0).float()
                s = (act_max ** alpha) / (w_max ** (1 - alpha) + 1e-8)
                st[fn] = (w_tensor * s.to(w_tensor.dtype)).to(st[fn].dtype)

        # SmoothQuant 变换完成，需要重新加载模型并重校准
        print("  保存变换后的权重到临时文件...")
        import tempfile
        tmpdir = tempfile.mkdtemp()
        tmp_safetensors = os.path.join(tmpdir, "model.safetensors")
        save_file(dict(st), tmp_safetensors)

        # 重新加载变换后的权重
        print("  重新加载变换后的模型...")
        self.model.load_state_dict(
            load_file(tmp_safetensors), strict=False, assign=True)

        # 重新校准
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        calib_data = get_calibration_data(tokenizer, num_samples=CALIB_SAMPLES // 2)
        print("  变换后重校准...")
        self.act_stats = run_calibration(self.model, tokenizer, calib_data)

        import shutil as _shutil
        _shutil.rmtree(tmpdir, ignore_errors=True)

        # 用新的激活统计量化
        print("  SmoothQuant 变换完成，量化权重...")
        quantized = 0
        for name, t in st.items():
            if name in linear_names and t.ndim == 2:
                try:
                    w, w_scale = self._quantize_weight(t)
                    new_st[name] = w
                    new_st[name.replace(".weight", ".weight_scale")] = w_scale
                    new_st[name.replace(".weight", ".weight_zero_point")] = (
                        torch.zeros_like(w_scale, dtype=torch.int8))

                    # 激活 per-tensor scale (SmoothQuant 后更均匀)
                    stats = self.get_layer_stats(name.replace(".weight", ""))
                    act_max = stats["per_tensor_max"] if stats else 6.0
                    act_scale = max(act_max, 1e-8) / 127.0
                    new_st[name.replace(".weight", ".input_scale")] = (
                        torch.tensor(act_scale, dtype=torch.bfloat16))
                    quantized += 1
                except Exception as e:
                    print(f"    ⚠️ {name}: {e}")
                    new_st[name] = t
            elif name not in new_st:
                new_st[name] = t

        print(f"  量化: {quantized} 层")
        with open(os.path.join(self.model_path, "config.json")) as f:
            cfg = json.load(f)
        cfg["quantization_config"] = {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "config_groups": {"group_0": {
                "weights": {"num_bits": 8, "type": "int", "symmetric": True,
                             "strategy": "group", "group_size": self.group_size},
                "input_activations": {"num_bits": 8, "type": "int", "symmetric": True,
                                       "strategy": "tensor"},
                "targets": ["Linear"],
            }},
            "quantization_status": "compressed",
            "ignore": ["lm_head"],
        }
        self._save_model(new_st, cfg, output_dir)

        size_mb = sum(os.path.getsize(os.path.join(output_dir, f))
                      for f in os.listdir(output_dir)) / (1024**2)
        print(f"  ✅ 完成: {size_mb:.1f} MB")
        return output_dir

    def cleanup(self):
        del self.model
        gc.collect()
        torch.cuda.empty_cache()


# ════════════════════════════════════════════════════════════
#  模型大小对比
# ════════════════════════════════════════════════════════════

def print_comparison(original_path, quantized_paths):
    print("\n" + "=" * 60)
    print("  5 种量化方案对比")
    print("=" * 60)

    def get_size_gb(path):
        if not path or not os.path.exists(path):
            return 0
        total = 0
        for root, dirs, files in os.walk(path):
            if ".cache" in root:
                continue
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total / (1024**3)

    orig = get_size_gb(original_path)
    print(f"\n  {'方法':<35} {'大小':>10} {'压缩比':>10} {'校准':>8}")
    print(f"  {'─' * 35} {'─' * 10} {'─' * 10} {'─' * 8}")
    print(f"  {'0. BF16 (原始)':<35} {orig:>8.2f} GB {'1.00x':>10} {'─':>8}")

    labels = {
        "fp8": "1. FP8 W8A8 dynamic",
        "int8_w8a16": "2. INT8 W8A16",
        "int8_w8a8_tensor": "3. INT8 W8A8 tensor",
        "int8_w8a8_channel": "4. INT8 W8A8 channel",
        "int8_smoothquant": "5. INT8 SmoothQuant",
    }
    needs_calib = {"fp8", "int8_w8a16"}  # 不需要校准的

    for method, path in quantized_paths.items():
        if path:
            size = get_size_gb(path)
            ratio = orig / size if size > 0 else 0
            calib = "否" if method in needs_calib else "是"
            print(f"  {labels.get(method, method):<35} {size:>8.2f} GB {ratio:>8.2f}x {calib:>8}")


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

METHOD_MAP = {
    "fp8": ("DeepSeek-R1-Distill-Qwen-1.5B-FP8", "fp8"),           # 方法 1
    "int8_w8a16": ("DeepSeek-R1-Distill-Qwen-1.5B-INT8-W8A16", "int8_w8a16"),  # 方法 2
    "int8_w8a8_tensor": ("DeepSeek-R1-Distill-Qwen-1.5B-INT8-W8A8-Tensor", "int8_w8a8_tensor"),
    "int8_w8a8_channel": ("DeepSeek-R1-Distill-Qwen-1.5B-INT8-W8A8-Channel", "int8_w8a8_channel"),
    "int8_smoothquant": ("DeepSeek-R1-Distill-Qwen-1.5B-SmoothQuant", "int8_smoothquant"),
}


def main():
    parser = argparse.ArgumentParser(
        description="5 种 vLLM 量化方案对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
方法:
  fp8                FP8 权重 (per-tensor) + 动态 FP8 激活 (推荐)
  int8_w8a16         INT8 权重 (per-group) + BF16 激活 (不量化)
  int8_w8a8_tensor   INT8 权重 (per-group) + INT8 激活 (per-tensor 校准)
  int8_w8a8_channel  INT8 权重 (per-group) + INT8 激活 (per-channel 校准)
  int8_smoothquant   SmoothQuant + INT8 W8A8
  all                全部 5 种

示例:
  python quantize_methods.py --method fp8
  python quantize_methods.py --method int8_smoothquant --calib-samples 128
  python quantize_methods.py --method all --calib-samples 256
        """,
    )
    parser.add_argument("--method", "-m", default="fp8",
                        choices=["fp8", "int8_w8a16", "int8_w8a8_tensor",
                                  "int8_w8a8_channel", "int8_smoothquant", "all"])
    parser.add_argument("--model", "-i", default=str(MODEL_DIR))
    parser.add_argument("--output-dir", "-o", default=None)
    parser.add_argument("--calib-samples", type=int, default=CALIB_SAMPLES)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--smoothquant-alpha", type=float, default=0.5)
    parser.add_argument("--force", "-f", action="store_true", help="覆盖已存在的输出目录")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(os.path.join(model_path, "model.safetensors")):
        print(f"❌ 模型不存在: {model_path}")
        sys.exit(1)

    orig_size = os.path.getsize(os.path.join(model_path, "model.safetensors")) / (1024**3)
    print(f"✅ 源模型: {model_path} ({orig_size:.1f} GB)")

    methods = (list(METHOD_MAP.keys()) if args.method == "all"
               else [args.method])

    # 哪些方法需要校准
    calibration_methods = {"int8_w8a8_tensor", "int8_w8a8_channel", "int8_smoothquant"}
    needs_calib = bool(set(methods) & calibration_methods)

    # 准备校准数据和量化器（如果至少一个方法需要）
    quantizer = None
    if needs_calib:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        calib_data = get_calibration_data(tokenizer, args.calib_samples)
        quantizer = W8A8Quantizer(model_path, calib_data, group_size=args.group_size)

    quantized_paths = {}
    for method in methods:
        folder_name, method_key = METHOD_MAP[method]
        output_dir = args.output_dir or str(OUTPUT_BASE / folder_name)

        if os.path.exists(output_dir):
            if not args.force:
                try:
                    resp = input(f"\n⚠️  已存在: {output_dir}\n   覆盖? [y/N]: ")
                    if resp.lower() != "y":
                        print(f"   跳过 {method}")
                        quantized_paths[method] = output_dir
                        continue
                except EOFError:
                    print(f"   跳过 {method} (无法交互)")
                    quantized_paths[method] = output_dir
                    continue
            print(f"   覆盖: {output_dir}")
            shutil.rmtree(output_dir)

        try:
            if method == "fp8":
                result = method_fp8(model_path, output_dir)
            elif method == "int8_w8a16":
                result = method_int8_w8a16(model_path, output_dir, args.group_size)
            elif method == "int8_w8a8_tensor":
                result = quantizer.method_w8a8_tensor(output_dir)
            elif method == "int8_w8a8_channel":
                result = quantizer.method_w8a8_channel(output_dir)
            elif method == "int8_smoothquant":
                result = quantizer.method_smoothquant(output_dir, args.smoothquant_alpha)
            quantized_paths[method] = result
        except Exception as e:
            print(f"❌ {method} 失败: {e}")
            import traceback
            traceback.print_exc()
            quantized_paths[method] = None

    if quantizer:
        quantizer.cleanup()

    print_comparison(model_path, quantized_paths)


if __name__ == "__main__":
    main()
