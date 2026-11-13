# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DeepMachine contributors.
# Modified from DiffSynth-Studio MOSS DiT (55fdfb23987672528d2cf71f4a9d44aba44c1863).
# Changes: registered clean reference, isolated attention/KV cache, reference RoPE.
# Original upstream rights retained; see THIRD_PARTY_NOTICES.md.
"""Registered clean-reference conditioning for the pinned MOSS DiT.

Retains all pretrained parameter names. The no-reference forward is untouched.
Reference tokens use t=0, an independent reference RoPE axis, and self-only
attention; target/text can attend reference K/V. Batch size is explicitly one.
"""
from __future__ import annotations

import types
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from diffsynth.models.moss_image_dit import rotate, situ, validate_skip_layers


def reference_rope(dit, target_hw, source_hw, text_length, bbox, dtype, device):
    ht, wt = target_hw
    hs, ws = source_hw
    x0, y0, x1, y1 = bbox
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("Invalid registered reference bbox")
    y, x = torch.meshgrid(torch.arange(hs, device=device), torch.arange(ws, device=device), indexing="ij")
    coords = torch.stack((torch.ones_like(y), torch.full_like(y, text_length), y, x), -1).flatten(0, 1).float()
    coords[:, 2] = y0 * ht + (coords[:, 2] + 0.5) * (y1 - y0) * ht / hs - 0.5 - (ht - 1) / 2
    coords[:, 3] = x0 * wt + (coords[:, 3] + 0.5) * (x1 - x0) * wt / ws - 0.5 - (wt - 1) / 2
    phase = coords.unsqueeze(-1) * dit.rope.bands.to(device)
    return torch.cat((phase.sin(), phase.cos()), -2).flatten(-2).unsqueeze(0).to(dtype)


def time_conditions(dit, timestep, dtype, device):
    times = torch.as_tensor(timestep, device=device, dtype=torch.float32).reshape(-1)
    if times.numel() != 1:
        raise ValueError("AnyPaint currently uses batch size 1")
    half = dit.freq_dim // 2
    freq = 10000 ** (-torch.arange(half, device=device, dtype=torch.float64) / half)
    phase = times.double()[:, None] * freq
    emb = torch.cat((phase.cos(), phase.sin()), -1).float()
    time = dit.time_embedding(emb.to(dtype)).float()
    mod = dit.shared_block_modulation(time.to(dtype)).view(1, 6, -1)
    head = dit.video_head_modulation(time.to(dtype)).view(1, 2, -1)
    return mod, head


def block_forward(layer, x, rope, mod, ref_mod, split, cached=None, capture=False):
    if split and x.shape[1] > split:
        # Run the independent reference branch with exactly the same GEMM
        # shapes as precompute_reference. Joint BF16 GEMMs can select different
        # kernels and their tiny differences accumulate across 42 layers.
        reference, ref_k, ref_v = block_forward(layer, x[:, split:], rope[:, split:], ref_mod, ref_mod, 0, capture=True)
        target = block_forward(layer, x[:, :split], rope[:, :split], mod, ref_mod, split, (ref_k, ref_v))
        combined = torch.cat((target, reference), 1)
        return (combined, ref_k, ref_v) if capture else combined
    m = mod.unbind(1)
    r = ref_mod.unbind(1)

    def affine(h, shift, scale):
        return torch.cat((h[:, :split] * (1 + m[scale][:, None]) + m[shift][:, None],
                          h[:, split:] * (1 + r[scale][:, None]) + r[shift][:, None]), 1).to(x.dtype)

    def gated(h, gate):
        return torch.cat((h[:, :split] * m[gate][:, None], h[:, split:] * r[gate][:, None]), 1)

    y = affine(layer.norm1(x), 0, 1)
    heads = layer.heads
    q = layer.q_norm(layer.proj["q"](y).unflatten(-1, (heads, -1)))
    k = layer.k_norm(layer.proj["k"](y).unflatten(-1, (heads, -1)))
    v = layer.proj["v"](y).unflatten(-1, (heads, -1)).transpose(1, 2)
    q = rotate(q, rope).to(v.dtype).transpose(1, 2)
    k = rotate(k, rope).to(v.dtype).transpose(1, 2)
    ref_k, ref_v = (k[:, :, split:], v[:, :, split:]) if cached is None else cached
    key = torch.cat((k, ref_k), 2) if cached is not None else k
    val = torch.cat((v, ref_v), 2) if cached is not None else v
    # Two flash-SDPA calls implement the asymmetric mask without an O(N^2) mask.
    outputs = []
    if split:
        outputs.append(F.scaled_dot_product_attention(q[:, :, :split], key, val))
    if x.shape[1] > split:
        outputs.append(F.scaled_dot_product_attention(q[:, :, split:], ref_k, ref_v))
    attended = torch.cat(outputs, 2).transpose(1, 2).flatten(2)
    x = x + gated(layer.proj["o"](attended), 2)
    y = affine(layer.norm2(x), 3, 4)
    hidden, gate = layer.proj["ffn_up_gate"](y).chunk(2, -1)
    x = x + gated(layer.proj["ffn_down"](situ(hidden, gate)), 5)
    return (x, ref_k, ref_v) if capture else x


def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask=None,
            skip_layers=None, use_gradient_checkpointing=False,
            use_gradient_checkpointing_offload=False, reference_latents=None,
            reference_bbox=(0., 0., 1., 1.), reference_kv=None):
    if reference_latents is None and reference_kv is None:
        return self._anypaint_original_forward(
            hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
            skip_layers, use_gradient_checkpointing, use_gradient_checkpointing_offload)
    if hidden_states.shape[0] != 1 or encoder_hidden_states.shape[0] != 1:
        raise ValueError("AnyPaint currently uses batch size 1")
    if use_gradient_checkpointing_offload:
        raise ValueError("Use the tested resident gradient-checkpointing path")
    skip = validate_skip_layers(skip_layers, len(self.layers))
    _, channels, ht, wt = hidden_states.shape
    if channels != self.latent_channels:
        raise ValueError("Incorrect MOSS latent channels")
    context = encoder_hidden_states
    if encoder_attention_mask is not None:
        context = context[:, encoder_attention_mask[0].bool()]
    length = context.shape[1]
    dtype, device = hidden_states.dtype, hidden_states.device
    rope = self.rope(ht, wt, length, device).to(dtype)
    text = self.text_embedder(context, rope[:, ht * wt:])
    video = self.video_embedder(hidden_states.flatten(2).transpose(1, 2).contiguous())
    x = torch.cat((video, text), 1)
    split = x.shape[1]
    mod, head = time_conditions(self, timestep, dtype, device)
    ref_mod, _ = time_conditions(self, torch.zeros(1, device=device), dtype, device)
    if reference_kv is None:
        hs, ws = reference_latents.shape[-2:]
        ref = self.video_embedder(reference_latents.flatten(2).transpose(1, 2).contiguous())
        x = torch.cat((x, ref), 1)
        rope = torch.cat((rope, reference_rope(self, (ht, wt), (hs, ws), length, reference_bbox, dtype, device)), 1)
    for i, layer in enumerate(self.layers):
        if i in skip:
            continue
        cached = reference_kv[i] if reference_kv is not None else None
        if use_gradient_checkpointing:
            x = checkpoint(block_forward, layer, x, rope, mod, ref_mod, split, cached, use_reentrant=False)
        else:
            x = block_forward(layer, x, rope, mod, ref_mod, split, cached)
    x = self.final_norm_video(x[:, :ht * wt])
    x = x * (1 + head[:, 1, None]) + head[:, 0, None]
    return self.final_linear_video(x).transpose(1, 2).reshape(1, channels, ht, wt)


@torch.no_grad()
def precompute_reference(dit, reference, target_hw, text_length, bbox=(0., 0., 1., 1.)):
    hs, ws = reference.shape[-2:]
    x = dit.video_embedder(reference.flatten(2).transpose(1, 2).contiguous())
    rope = reference_rope(dit, target_hw, (hs, ws), text_length, bbox, reference.dtype, reference.device)
    mod, _ = time_conditions(dit, torch.zeros(1, device=reference.device), reference.dtype, reference.device)
    cache = []
    for layer in dit.layers:
        x, key, value = block_forward(layer, x, rope, mod, mod, 0, capture=True)
        cache.append((key, value))
    return cache


def install(dit):
    if not hasattr(dit, "_anypaint_original_forward"):
        dit._anypaint_original_forward = dit.forward
        dit.forward = types.MethodType(forward, dit)
    return dit
