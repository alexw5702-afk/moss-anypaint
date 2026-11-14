# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DeepMachine contributors.
"""MOSS AnyPaint inference shared by the HF example and ComfyUI nodes.

Apache-2.0. Uses the AnyPaint adapter and unmerged FP32 Base DMD2.
"""
from __future__ import annotations

import gc
import json
import os
from pathlib import Path
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

os.environ.setdefault("DIFFSYNTH_ATTENTION_IMPLEMENTATION", "torch")

try:
    from .anypaint import prepare_anypaint
    from ._adapters import LOCK, sha, validate_seed
except ImportError:
    from anypaint import prepare_anypaint
    from _adapters import LOCK, sha, validate_seed


class MOSSAnyPaintPipeline:
    """CUDA qfloat8/BF16 runtime. An ordinary text-to-image LoRA loader is insufficient."""

    def __init__(self, adapter_path=None, *, model_dir=None, distillation_path=None):
        from huggingface_hub import hf_hub_download, snapshot_download

        if not torch.cuda.is_available():
            raise RuntimeError("This release runtime requires a CUDA GPU with BF16 support")
        self.adapter_path = Path(adapter_path or Path(__file__).parent / LOCK["task"]["file"])
        if not self.adapter_path.is_file():
            self.adapter_path = Path(hf_hub_download(LOCK["task"]["repo"], LOCK["task"]["file"]))
        if sha(self.adapter_path) != LOCK["task"]["sha256"]:
            raise ValueError("AnyPaint adapter does not match this release")
        self.model_dir = Path(model_dir or snapshot_download(
            LOCK["base_model"]["repo"], revision=LOCK["base_model"]["revision"],
            allow_patterns=["transformer/*", "text_encoder/*", "tokenizer/*", "vae/*"]))
        self.distillation_path = Path(distillation_path or hf_hub_download(
            LOCK["distillation"]["repo"], "lora_dmd2.safetensors", revision=LOCK["distillation"]["revision"]))
        self.pipe = None
        self._text = None
        self._prompt_cache = {}
        self._lock = threading.RLock()

    def load(self):
        if self.pipe is not None:
            return
        from diffsynth.pipelines.moss_image import MOSSImagePipeline, ModelConfig
        try:
            from ._adapters import quantize_frozen, add_lora, add_distillation, load_lora
            from .moss_model import install
        except ImportError:
            from _adapters import quantize_frozen, add_lora, add_distillation, load_lora
            from moss_model import install
        root = self.model_dir
        configs = [
            ModelConfig(path=sorted(map(str, (root / "text_encoder").glob("model*.safetensors"))), computation_dtype=torch.bfloat16),
            ModelConfig(path=sorted(map(str, (root / "transformer").glob("diffusion_pytorch_model*.safetensors"))), computation_dtype=torch.bfloat16),
            ModelConfig(path=str(root / "vae/diffusion_pytorch_model.safetensors"), computation_dtype=torch.float32),
        ]
        pipe = MOSSImagePipeline.from_pretrained(torch_dtype=torch.bfloat16, device="cpu", model_configs=configs,
                                               tokenizer_config=ModelConfig(path=str(root / "tokenizer")))
        pipe.device = torch.device("cuda")
        self._text = quantize_frozen(pipe.text_encoder)
        self._text.to("cpu")
        pipe.text_encoder = None
        torch.cuda.empty_cache()
        dit = add_lora(quantize_frozen(pipe.dit))
        load_lora(dit, self.adapter_path)
        pipe.dit = install(add_distillation(dit, self.distillation_path)).eval().requires_grad_(False)
        pipe.vae.eval().requires_grad_(False).to("cuda")
        self.pipe = pipe

    def _encode_prompt(self, prompt):
        from diffsynth.pipelines.moss_image import MOSSImageUnit_PromptEmbedder
        if prompt not in self._prompt_cache:
            self._text.to("cuda")
            self.pipe.text_encoder = self._text
            try:
                text = MOSSImageUnit_PromptEmbedder().encode_prompt(self.pipe, prompt, 4096)
            finally:
                self._text.to("cpu")
                self.pipe.text_encoder = None
                torch.cuda.empty_cache()
            if len(self._prompt_cache) >= 8:
                self._prompt_cache.pop(next(iter(self._prompt_cache)))
            self._prompt_cache[prompt] = {k: v.cpu() for k, v in text.items()}
        return {k: v.to("cuda") for k, v in self._prompt_cache[prompt].items()}

    @torch.no_grad()
    def generate(self, prepared, prompt, *, seed=2026, steps=8):
        validate_seed(seed)
        if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 50:
            raise ValueError("Steps must be an integer between 1 and 50")
        with self._lock:
            self.load()
            try:
                from .moss_model import precompute_reference
                from ._adapters import seed_all
            except ImportError:
                from moss_model import precompute_reference
                from _adapters import seed_all
            start = time.monotonic()
            pipe = self.pipe
            text = self._encode_prompt(prompt)
            reference = pipe.vae.encode(pipe.preprocess_image(prepared.condition).to("cuda", torch.float32)).bfloat16()
            known = pipe.vae.encode(pipe.preprocess_image(prepared.known_image).to("cuda", torch.float32))
            length = int(text["prompt_embeds_mask"].sum())
            kv = precompute_reference(pipe.dit, reference, known.shape[-2:], length)
            ht, wt = known.shape[-2:]
            keep = torch.from_numpy(np.asarray(prepared.keep_mask).copy()).float()[None, None].to("cuda") / 255
            keep = F.pixel_unshuffle(F.interpolate(keep, size=(ht * 2, wt * 2), mode="nearest"), 2).amin(1, keepdim=True) > .5
            seed_all(seed)
            initial = torch.randn(known.shape, generator=torch.Generator("cpu").manual_seed(seed), dtype=torch.float32).to("cuda")
            latent = initial.clone()
            pipe.scheduler.set_timesteps(steps, shift=5.)
            for i, timestep in enumerate(pipe.scheduler.timesteps):
                prediction = pipe.dit(latent.bfloat16(), timestep.to("cuda").reshape(1), text["prompt_embeds"],
                                      text["prompt_embeds_mask"], reference_kv=kv).float()
                latent = pipe.scheduler.step(prediction, timestep, latent)
                sigma = float(pipe.scheduler.sigmas[i + 1]) if i + 1 < len(pipe.scheduler.sigmas) else 0.
                latent = torch.where(keep, (1 - sigma) * known + sigma * initial, latent)
            if not torch.isfinite(latent).all():
                raise RuntimeError("Non-finite model output")
            image = pipe.vae_output_to_image(pipe.vae.decode(latent))
            self.last_metadata = {"checkpoint_step": 4303, "seed": seed, "steps": steps,
                                  "adapter_sha256": LOCK["task"]["sha256"], "seconds": time.monotonic() - start,
                                  "output_contract": "raw_model_output_no_final_rgb_composite"}
            return image

    def __call__(self, source, mask, prompt, *, canvas_size=None, bbox=None, seed=2026, steps=8):
        prepared = prepare_anypaint(source, mask, canvas_size or source.size, bbox)
        return self.generate(prepared, prompt, seed=seed, steps=steps)

    def close(self):
        with self._lock:
            self.pipe = None
            self._text = None
            self._prompt_cache.clear()
            gc.collect()
            torch.cuda.empty_cache()
