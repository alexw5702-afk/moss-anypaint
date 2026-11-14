# ComfyUI MOSS AnyPaint

Arbitrary-mask editing and outpainting with the selected4303-step
[MOSS AnyPaint adapter](https://huggingface.co/yijunwang2/moss-anypaint).

These nodes **share the portable HF runtime**, preserving the tested qfloat8/
BF16 path and unmerged FP32 DMD2. They are not a native KSampler MODEL patch.
Inputs/outputs use standard ComfyUI IMAGE/MASK, so ordinary image nodes work
before and after the workflow. No Space or external API service is used.

## Install

Clone/copy this repository to `ComfyUI/custom_nodes/moss-anypaint`, then install
`requirements.txt` with the Python interpreter used by your ComfyUI instance.
Use Python 3.11+, a recent ComfyUI build and a CUDA GPU with BF16 support.
The validated GPU is RTX 5090 32GB. Dependencies include Transformers5.5.3,
PEFT0.18.1, Quanto0.2.4 and the pinned DiffSynth revision.

Download into `ComfyUI/models/loras/`:

- [moss_anypaint_rank32.safetensors](https://huggingface.co/yijunwang2/moss-anypaint/resolve/main/moss_anypaint_rank32.safetensors)
- [lora_dmd2.safetensors](https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd/resolve/da05cb3b52899179e2bb5f7efe1058fefcbae21b/lora_dmd2.safetensors)

The Loader downloads the pinned original **MOSS-Image-Base HF directory** on
first use. Or set `base_model_dir` to a local folder containing `transformer/`,
`text_encoder/`, `tokenizer/` and `vae/`. Native Comfy-Org single-file model
checkpoints and SFT weights are not inputs to this loader.

## Workflow

Import `moss_anypaint_workflow.json`. Copy
`examples/moss_anypaint_source.png` and `examples/moss_anypaint_mask.png`
into ComfyUI's input folder and select them in the two Load Image nodes.
The second image is converted to a generated mask through ImageToMask/red.
Alternatively use the source Load Image Mask Editor output directly.

The supplied PNG is losslessly converted from the reviewed source's Pillow RGB.
Recent ComfyUI Load Image uses a different WebP decode path, which can slightly
change RGB values before sampling. Use this PNG for strict fixed-seed matching.

```text
Load Image + generated MASK -> MOSS AnyPaint Prepare
MOSS AnyPaint Loader + prepared input -> MOSS AnyPaint Generate -> Save Image
Prepare.input_with_mask -> Preview Image
```

White mask pixels and padded pixels are generated. For inpainting use zero
padding; for outpainting add left/top/right/bottom space, in pixels. The image
is not resized by padding; right/bottom alignment pixels are added if needed.
Only exterior seams gain32px latent redraw; internal brushes remain exact.

The default example is seed 2026,8 steps. The sampler returns raw decoded RGB,
with no final source composite. The loader checks the selected task and DMD2
hashes, and the two adapters stay separate.

Seeds accept the full uint64 range (0 through2**64−1), including ComfyUI's
randomize mode. Source-sized masks follow the padding offset; canvas-sized masks
already use canvas coordinates. For the portable HF helper the same rule applies
to bbox placement/scaling; when source/canvas sizes are equal, canvas coordinates
take priority.

`keep_model_loaded=true` keeps the shared runtime for subsequent seeds. Set it
false to release its GPU/CPU model allocations after the current image when
switching to another model or needing VRAM. Prompt text is cached independently
of the sample seed. The JSON metadata output includes the actual seed/step/hash.

`moss_anypaint_api.json` is the same graph in ComfyUI API form.

## Showcase

All 12 cases use seed 2026 and the selected4303-step weights. Inputs show the
actual mask overlay, alongside each result. Original sources and binary masks
remain available for reproduction.

![MOSS AnyPaint inputs with mask and results](showcase/contact_sheet.webp)

## Credits and license

Apache-2.0. See [LICENSE](LICENSE), [PIPELINE_LICENSE](PIPELINE_LICENSE),
[NOTICE](NOTICE) and [component licenses/provenance](THIRD_PARTY_NOTICES.md). Mask/padding/workflow interactions follow the
previous Krea AnyPaint nodes; MOSS model code comes from the shared released
runtime, not the Krea DiT patch. Base and DMD2 assets are downloaded separately.

## Validation

Validated with the pinned runtime on RTX 5090: internal editing, outpainting and
mixed editing match the selected fixed-seed outputs pixel for pixel. ComfyUI
0.45.0 API workflow validation uses the canonical PNG example input to preserve
the same decoded RGB values. See the runtime lock for exact model revisions.

The distribution includes SHA256SUMS for all shipped files. Release assets also
include a ComfyUI ZIP, its checksum, and both UI/API workflow JSON files.

## v0.1.1 update

Fixes the large-seed failure in v0.1.0 and the portable source-mask placement
behavior. The builders now reject unlisted source files and symlinks before
copying. The selected4303 adapter, seed2026 examples, sampler and precision are
unchanged; update the node code/runtime together, with no new weights required.
The v0.1.0 tag and assets remain available as historical artifacts.
