# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DeepMachine contributors.
"""Strict adapters from the validated AnyPaint runtime. Apache-2.0."""
import hashlib,json,random
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load_file
LOCK=json.loads((Path(__file__).parent/"runtime_lock.json").read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()

def validate_seed(seed):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0xffffffffffffffff:
        raise ValueError("Seed must be an integer between 0 and 2**64 - 1")

def seed_all(seed):
    validate_seed(seed)
    random.seed(seed)
    # NumPy's legacy RNG accepts uint32; retain the full uint64 for Torch noise.
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def quantize_frozen(model):
    from optimum.quanto import quantize, freeze, qfloat8
    model.requires_grad_(False).eval().to(device="cuda", dtype=torch.bfloat16)
    quantize(model, weights=qfloat8)
    freeze(model)
    return model

def add_lora(dit):
    from diffsynth.diffusion import DiffusionTrainingModule
    # Match attention + MLP in every joint block; no adapters on VAE or Qwen.
    targets = r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)"
    return DiffusionTrainingModule().add_lora_to_model(dit, targets, 32, lora_alpha=32, upcast_dtype=torch.bfloat16)

def add_distillation(dit, path):
    from peft import LoraConfig, inject_adapter_in_model
    from peft.tuners.lora import LoraLayer
    spec = LOCK["distillation"]
    path = Path(path)
    if sha(path) != spec["files"]["lora_dmd2.safetensors"]["sha256"]:
        raise RuntimeError("Distillation adapter checksum mismatch")
    state = load_file(str(path))
    targets = r"layers\.\d+\.proj\.(q|k|v|o|ffn_up_gate|ffn_down)"
    inject_adapter_in_model(LoraConfig(r=128, lora_alpha=128, target_modules=targets), dit, adapter_name="distillation")
    parameters = dict(dit.named_parameters())
    copied, used = set(), set()
    with torch.no_grad():
        for name, module in dit.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            prefix = "diffusion_model." + name
            alpha_key = prefix + ".alpha"
            if float(state[alpha_key]) != 128:
                raise RuntimeError(f"Unexpected distillation alpha: {name}")
            used.add(alpha_key)
            # Keep the small distilled delta in FP32 and unmerged, as required
            # by its model card; never round it into the qfloat8 base weights.
            for source, target in [("lora_down", "lora_A"), ("lora_up", "lora_B")]:
                key = prefix + "." + source + ".weight"
                parameter_name = name + "." + target + ".distillation.weight"
                value = state[key]
                parameter = parameters[parameter_name]
                if value.dtype != torch.float32 or value.shape != parameter.shape or not torch.isfinite(value).all():
                    raise RuntimeError(f"Distillation shape/dtype mismatch: {key}")
                parameter.data = value.to(device="cuda", dtype=torch.float32)
                copied.add(parameter_name)
                used.add(key)
            active = (["default"] if "default" in module.lora_A else []) + ["distillation"]
            module.set_adapter(active)
    if used != set(state) or len(copied) != 504:
        raise RuntimeError("Partial distillation adapter load")
    dit.eval().requires_grad_(False)
    print("DISTILLATION", spec["repo"], spec["revision"], "504 FP32 tensors, unmerged", flush=True)
    return dit

def load_lora(dit, path):
    state = load_file(str(path))
    expected = {name for name, p in dit.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise RuntimeError("LoRA keys differ from the configured adapter targets")
    # Quanto's QModule deserializer expects quantized base weights even with
    # strict=False. An adapter-only file must only touch the named LoRA params.
    parameters = dict(dit.named_parameters())
    with torch.no_grad():
        for name, value in state.items():
            if value.shape != parameters[name].shape or value.dtype != torch.bfloat16:
                raise RuntimeError(f"Adapter shape/precision mismatch: {name}")
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite adapter: {name}")
            parameters[name].copy_(value)
