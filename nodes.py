# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DeepMachine contributors.
"""MOSS AnyPaint nodes sharing the reviewed portable runtime. Apache-2.0."""
import json
import math
import numpy as np
from PIL import Image
import torch

import folder_paths
import comfy.model_management

from .moss_runtime import MOSSAnyPaintPipeline, prepare_anypaint


def _image(value):
    if value.ndim != 4 or value.shape[0] != 1 or value.shape[-1] < 3:
        raise ValueError("Use one IMAGE per request; queue multiple seeds for batches")
    array = value[0, ..., :3].detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
    return Image.fromarray(array)


def _tensor(image):
    return torch.from_numpy(np.asarray(image.convert("RGB")).copy()).float().div(255)[None]


class MOSSAnyPaintLoader:
    @classmethod
    def INPUT_TYPES(cls):
        files = folder_paths.get_filename_list("loras")
        return {"required": {
            "anypaint_lora": (files or ["moss_anypaint_rank32.safetensors"], {"default": "moss_anypaint_rank32.safetensors"}),
            "dmd2_lora": (files or ["lora_dmd2.safetensors"], {"default": "lora_dmd2.safetensors"}),
            "base_model_dir": ("STRING", {"default": "", "tooltip": "Optional local MOSS-Image-Base HF folder. Blank downloads the pinned official revision."}),
        }}

    RETURN_TYPES = ("MOSS_ANYPAINT_PIPELINE",)
    FUNCTION = "load"
    CATEGORY = "MOSS/AnyPaint"

    def load(self, anypaint_lora, dmd2_lora, base_model_dir=""):
        runtime = MOSSAnyPaintPipeline(
            folder_paths.get_full_path_or_raise("loras", anypaint_lora),
            model_dir=base_model_dir.strip() or None,
            distillation_path=folder_paths.get_full_path_or_raise("loras", dmd2_lora))
        return (runtime,)


class MOSSAnyPaintPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "source": ("IMAGE",),
            **{name: ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}) for name in ["left", "top", "right", "bottom"]},
        }, "optional": {"generated_mask": ("MASK",)}}

    RETURN_TYPES = ("MOSS_ANYPAINT_INPUT", "IMAGE", "MASK")
    RETURN_NAMES = ("prepared", "input_with_mask", "generated_mask")
    FUNCTION = "prepare"
    CATEGORY = "MOSS/AnyPaint"

    def prepare(self, source, left=0, top=0, right=0, bottom=0, generated_mask=None):
        image = _image(source)
        if min(left, top, right, bottom) < 0:
            raise ValueError("Padding cannot be negative")
        width = math.ceil((image.width + left + right) / 16) * 16
        height = math.ceil((image.height + top + bottom) / 16) * 16
        mask = Image.new("L", (width, height), 0)
        if generated_mask is not None:
            values = generated_mask.detach().float().cpu()
            while values.ndim > 2:
                if values.shape[0] != 1:
                    raise ValueError("Use one MASK per request")
                values = values[0]
            brush = Image.fromarray((values.clamp(0, 1).numpy() > .5).astype(np.uint8) * 255)
            if brush.size == (width, height):
                mask = brush
            else:
                if brush.size != image.size:
                    brush = brush.resize(image.size, Image.Resampling.NEAREST)
                mask.paste(brush, (left, top))
        prepared = prepare_anypaint(image, mask, (width, height), (left, top, left + image.width, top + image.height))
        pixels = np.asarray(prepared.known_image).copy()
        generated = np.asarray(prepared.generated_mask) > 0
        pixels[generated] = (.35 * pixels[generated] + .65 * np.array([80, 230, 140])).astype(np.uint8)
        return (prepared, _tensor(Image.fromarray(pixels)), torch.from_numpy(generated.astype(np.float32))[None])


class MOSSAnyPaintGenerate:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "pipeline": ("MOSS_ANYPAINT_PIPELINE",),
            "prepared": ("MOSS_ANYPAINT_INPUT",),
            "prompt": ("STRING", {"multiline": True}),
            "seed": ("INT", {"default": 2026, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "steps": ("INT", {"default": 8, "min": 1, "max": 50}),
            "keep_model_loaded": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "metadata")
    FUNCTION = "generate"
    CATEGORY = "MOSS/AnyPaint"
    DESCRIPTION = "Use the shared MOSS runtime with FP32 unmerged DMD2; standard IMAGE/MASK I/O, not a KSampler patch."

    def generate(self, pipeline, prepared, prompt, seed=2026, steps=8, keep_model_loaded=True):
        comfy.model_management.unload_all_models()
        comfy.model_management.throw_exception_if_processing_interrupted()
        try:
            image = pipeline.generate(prepared, prompt, seed=int(seed), steps=int(steps))
            return (_tensor(image), json.dumps(pipeline.last_metadata))
        finally:
            if not keep_model_loaded:
                pipeline.close()


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in [MOSSAnyPaintLoader, MOSSAnyPaintPrepare, MOSSAnyPaintGenerate]}
NODE_DISPLAY_NAME_MAPPINGS = {name: name.replace("MOSSAnyPaint", "MOSS AnyPaint ") for name in NODE_CLASS_MAPPINGS}
