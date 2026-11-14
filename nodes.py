"""ComfyUI nodes for the yijunwang2 MOSS AnyPaint adapter."""

from __future__ import annotations

import math

import torch

import comfy.conds
import node_helpers

from . import anypaint_helpers as ah
from . import moss_registered_core as core


class MOSSAnyPaintPrepare:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source": ("IMAGE",),
                "left": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "top": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "right": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "bottom": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 16}),
                "reference_max_edge": ("INT", {"default": 384, "min": 128, "max": 768, "step": 16}),
                "boundary_redraw_px": ("INT", {"default": 32, "min": 0, "max": 256, "step": 1}),
            },
            "optional": {
                "generated_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MASK", "MASK", "INT", "INT")
    RETURN_NAMES = (
        "semantic_reference",
        "known_image",
        "generated_mask",
        "keep_mask",
        "canvas_width",
        "canvas_height",
    )
    FUNCTION = "prepare"
    CATEGORY = "MOSS/AnyPaint"
    DESCRIPTION = (
        "Connect source IMAGE and a MASK from SAM, Mask Editor or another mask node. "
        "White/values above 0.5 generate; black/other values preserve. "
        "For a separate black/white image use ImageToMask (red). "
        "Padding defaults to zero; add padding only for outpainting."
    )

    def prepare(
        self,
        source,
        left,
        top,
        right,
        bottom,
        reference_max_edge,
        boundary_redraw_px,
        generated_mask=None,
    ):
        if generated_mask is None:
            generated_mask = torch.zeros(
                (source.shape[0], source.shape[1], source.shape[2]),
                dtype=torch.float32,
                device=source.device,
            )
        return ah.prepare_canvas(
            source,
            generated_mask,
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            reference_max_edge=reference_max_edge,
            seam_px=boundary_redraw_px,
        )


def _encode_reference_latent(vae, image):
    samples = image.movedim(-1, 1)
    height, width = samples.shape[-2:]
    target_h = ah.align_up(height)
    target_w = ah.align_up(width)
    if (target_h, target_w) != (height, width):
        samples = torch.nn.functional.interpolate(
            samples,
            (target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )
    return vae.encode(samples.movedim(1, -1)[..., :3])


class MOSSAnyPaintEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
                "vae": ("VAE",),
                "semantic_reference": ("IMAGE",),
                "known_image": ("IMAGE",),
                "keep_mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "latent")
    FUNCTION = "encode"
    CATEGORY = "MOSS/AnyPaint"
    DESCRIPTION = (
        "Encode the full-canvas semantic reference and create a standard Comfy "
        "latent/noise_mask pair. KSampler performs timestep-matched known-region "
        "replacement; no final source composite is used."
    )

    def encode(
        self,
        clip,
        prompt,
        vae,
        semantic_reference,
        known_image,
        keep_mask,
    ):
        tokens = clip.tokenize(prompt)

        conditioning = clip.encode_from_tokens_scheduled(tokens)
        reference_latent = _encode_reference_latent(vae, semantic_reference)
        conditioning = node_helpers.conditioning_set_values(
            conditioning,
            {"reference_latents": [reference_latent]},
            append=True,
        )

        known_latent = vae.encode(known_image[..., :3])
        if known_latent.ndim not in (4, 5):
            raise ValueError(f"AnyPaint expects a 4D or 5D image latent, got {tuple(known_latent.shape)}")
        if known_latent.ndim == 5 and known_latent.shape[2] != 1:
            raise ValueError(
                "AnyPaint handles still images only; the latent time dimension must be 1, "
                f"got {tuple(known_latent.shape)}"
            )
        noise_mask = ah.token_aligned_noise_mask(
            keep_mask,
            known_latent.shape[-2],
            known_latent.shape[-1],
            patch_size=1,
        )
        return conditioning, {"samples": known_latent, "noise_mask": noise_mask}


class MOSSAnyPaintModelPatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "kv_cache": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "MOSS/AnyPaint"
    DESCRIPTION = (
        "Register the semantic reference over the complete target canvas and "
        "cache its isolated K/V once per sampling run. Place after Load LoRA."
    )

    def patch(self, model, kv_cache=True):
        bbox_norm = [0.0, 0.0, 1.0, 1.0]
        patched = model.clone()
        base_model = patched.model
        dit = patched.get_model_object("diffusion_model")

        original_extra_conds = base_model.extra_conds
        original_extra_conds_shapes = base_model.extra_conds_shapes
        original_forward = dit.forward

        def extra_conds(**kwargs):
            # Keep native text/control conditions but defer registered reference KV
            # until the target grid and actual text length are available in forward.
            ordinary = {k: v for k, v in kwargs.items() if k not in ("reference_latents", "reference_latents_method")}
            output = original_extra_conds(**ordinary)
            reference_latents = kwargs.get("reference_latents")
            if reference_latents is not None:
                output["ref_latents"] = comfy.conds.CONDList(
                    [base_model.process_latent_in(latent) for latent in reference_latents]
                )
            return output

        def extra_conds_shapes(**kwargs):
            output = original_extra_conds_shapes(**kwargs)
            reference_latents = kwargs.get("reference_latents")
            if reference_latents is not None:
                output["ref_latents"] = [
                    1,
                    128,
                    sum(math.prod(latent.size()) for latent in reference_latents) // 128,
                ]
            return output

        state = {"last_sigma": None, "caches": {}}

        def forward(
            x,
            timestep,
            context,
            transformer_options={},
            ref_latents=None,
            **kwargs,
        ):
            if not ref_latents:
                return original_forward(x, timestep, context,
                                        transformer_options=transformer_options, **kwargs)
            sigma = float(timestep.max())
            sample_sigmas = transformer_options.get("sample_sigmas")
            new_run = state["last_sigma"] is None or sigma > state["last_sigma"]
            if sample_sigmas is not None and sigma == float(sample_sigmas[0]) and sigma != state["last_sigma"]:
                new_run = True
            if new_run:
                state["caches"].clear()
            state["last_sigma"] = sigma
            key = core.ref_fingerprint(ref_latents, x, context.shape[1], bbox_norm)
            reference_kv = state["caches"].get(key) if kv_cache else None
            if reference_kv is None:
                reference_kv = core.precompute_registered_refs(
                    dit, ref_latents, x.shape[-2:], context.shape[1],
                    transformer_options, bbox_norm=bbox_norm)
                if kv_cache:
                    state["caches"][key] = reference_kv
            # Native MOSS already implements isolated ref-KV attention. Keep the
            # training text/image coordinates; only reference placement changes.
            return original_forward(
                x, timestep, context, transformer_options=transformer_options,
                ref_kv=reference_kv,
                ref_latents_method="index_timestep_zero_text_start", **kwargs)

        patched.add_object_patch("extra_conds", extra_conds)
        patched.add_object_patch("extra_conds_shapes", extra_conds_shapes)
        patched.add_object_patch("diffusion_model.forward", forward)
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "MOSSAnyPaintPrepare": MOSSAnyPaintPrepare,
    "MOSSAnyPaintEncode": MOSSAnyPaintEncode,
    "MOSSAnyPaintModelPatch": MOSSAnyPaintModelPatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MOSSAnyPaintPrepare": "MOSS AnyPaint Prepare",
    "MOSSAnyPaintEncode": "MOSS AnyPaint Encode",
    "MOSSAnyPaintModelPatch": "MOSS AnyPaint Model Patch",
}
