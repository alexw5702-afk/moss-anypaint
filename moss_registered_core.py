"""Registered-reference adapter for native ComfyUI MOSS.

Reuses the Krea AnyPaint reference-cache contract. Projection, attention,
quantization, prefetch and memory management remain in ComfyUI's MOSS model.
"""
import torch
import comfy.ldm.flux.math


def reference_rope(dit, refs, target_hw, text_length, bbox_norm):
    ht, wt = target_hw
    positions = []
    for index, ref in enumerate(refs):
        hs, ws = ref.shape[-2:]
        y, x = torch.meshgrid(torch.arange(hs, device=ref.device),
                              torch.arange(ws, device=ref.device), indexing="ij")
        ids = torch.zeros((1, hs * ws, 4), device=ref.device, dtype=torch.float32)
        ids[..., 0] = index + 1
        ids[..., 1] = text_length
        x0, y0, x1, y1 = bbox_norm
        ids[..., 2] = (y.flatten().float() + .5) * (y1 - y0) * ht / hs + y0 * ht - .5 - (ht - 1) / 2
        ids[..., 3] = (x.flatten().float() + .5) * (x1 - x0) * wt / ws + x0 * wt - .5 - (wt - 1) / 2
        positions.append(ids)
    ids = torch.cat(positions, dim=1)
    pe = torch.cat([comfy.ldm.flux.math.rope(ids[..., axis], width, 10000)
                    for axis, width in enumerate(dit.rope_axes_dim)], dim=-3)
    return pe.unsqueeze(1).transpose(1, 2).contiguous()


def precompute_registered_refs(dit, refs, target_hw, text_length, transformer_options, bbox_norm):
    # Reuse the native implementation rather than duplicate its block/prefetch
    # loop. Override its reference-only positions for this call, then restore.
    # ModelPatcher executes one forward at a time for a given model instance.
    original_rope = dit.rope
    pe = reference_rope(dit, refs, target_hw, text_length, bbox_norm)
    def registered_rope(h, w, text_len, device, transformer_options={}, ref_shapes=(), f_offset=0):
        if h == 0 and w == 0 and text_len == 0:
            return pe
        return original_rope(h, w, text_len, device, transformer_options, ref_shapes, f_offset)
    dit.rope = registered_rope
    try:
        return dit.preprocess_refs(refs, transformer_options)
    finally:
        dit.rope = original_rope


def ref_fingerprint(refs, x, text_length, bbox_norm):
    # The registered cache additionally depends on destination geometry and text length.
    key = [tuple(x.shape), text_length, str(x.device), x.dtype, tuple(bbox_norm)]
    for ref in refs:
        value = ref.float()
        key.append((tuple(ref.shape), str(ref.device), ref.dtype,
                    float(value.sum()), float(value.square().sum())))
    return tuple(key)
