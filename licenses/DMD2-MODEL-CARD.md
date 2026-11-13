---
license: apache-2.0
base_model: OpenMOSS-Team/MOSS-Image-Base
---
# MOSS-Image-Base, 8-step data-space DMD

Work-in-progress distillation of [OpenMOSS-Team/MOSS-Image-Base](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)
(50 steps, CFG 3) to 8 network evaluations, no CFG. Data-space DMD (arXiv 2610.12102): the
generator iterates on its own clean estimate from fixed noise. Trained on compact-JSON `scene_graph` captions.

Checkpoint: update 12000 (EMA), float32 weights.

# Usage (recommended): the LoRA alone, standard sampler

Load `lora.safetensors` on the stock MOSS-Image-Base transformer (e.g. a ComfyUI LoRA loader, strength 1) and
sample with the model's usual flow-matching Euler sampler, shift 5, **CFG 1**, 4-8 steps. Nothing else from
this repo is needed.

# Data-space sampler (the training-time form)

`transformer/` is a stock `MOSSImageTransformer` (load it with the release `pipeline_moss_image.py`); the
text encoder, tokenizer and VAE are the base repo's.

Precision: the distillation delta is smaller than bf16's rounding step, so `transformer/` must be loaded AND run
in fp32 (`torch_dtype=torch.float32`, no autocast); casting it to bf16 discards most of the distillation. For fast
bf16 inference keep the base repo's bf16 transformer and apply the adapter files unmerged:

- `lora.safetensors`: fp32 LoRA, rank 128 / alpha 128 on every block's `proj.{q,k,v,o,ffn_up_gate,ffn_down}`,
  kohya keys `diffusion_model.<module>.lora_down.weight` / `lora_up.weight` / `alpha`; apply as
  `y = W x + (alpha / rank) * up(down(x))`.
- `time_embedding.safetensors`: fp32 replacements for `time_embedding.*` (the step embedder) AND
  `shared_block_modulation.bias` / `video_head_modulation.bias`; the t=1 time vector is folded into those biases,
  so the embedder only works together with them. Keep these three modules in fp32.
- `generator.safetensors`: the same tensors in one file (`<module>.lora_A/B` keys) plus Q and the step codes.

The sampler differs from the release scheduler:

```python
from safetensors.torch import load_file
ds = load_file("dataspace.safetensors")            # latent_mix.Q [128, 128], step_codes [8]
Q, codes = ds["latent_mix.Q"].cuda(), ds["step_codes"].cuda()
z = torch.randn(1, 128, H // 16, W // 16, device="cuda")   # fp32
x0 = torch.zeros_like(z)
for r in range(8):                            # fewer steps also work (each prefix is trained)
    x = z + torch.einsum("oc,bchw->bohw", Q, x0)
    v = transformer(x, codes[r], prompt_embeds, prompt_mask).sample.float()
    x0 = z - v
# decode x0 with the release VAE (BN denormalize + 2x2 unpatchify)
```
