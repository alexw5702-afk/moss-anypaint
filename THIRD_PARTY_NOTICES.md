# Licenses and provenance

The selected AnyPaint adapter and DeepMachine's runtime/node modifications are
Apache-2.0, Copyright 2026 DeepMachine contributors. `LICENSE` applies to the
adapter and package; `PIPELINE_LICENSE` explicitly supplies the code license.
Existing upstream rights are retained. No upstream model weights are bundled.

| Component | Role in this release | License and pinned source |
| --- | --- | --- |
| MOSS-Image-Base | Separately downloaded original Base components | Apache-2.0; [Hub revision](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/tree/85975793d8b3d432f1b6e27f352abaa4c2a4c0d3); full text: [MOSS-Image-Base-LICENSE](licenses/MOSS-Image-Base-LICENSE) |
| Official MOSS-Image code | Original model/code provenance; not vendored as a complete repository | Apache-2.0; [source revision](https://github.com/OpenMOSS/MOSS-Image/tree/6be4453024c5acae9dcd0a191e26ee524e743310); full text: [MOSS-Image-code-LICENSE](licenses/MOSS-Image-code-LICENSE) |
| DiffSynth-Studio | Installed runtime dependency; modified model forward in moss_model.py derives from its MOSS implementation | Apache-2.0; [source revision](https://github.com/modelscope/DiffSynth-Studio/tree/55fdfb23987672528d2cf71f4a9d44aba44c1863); full text: [DiffSynth-Studio-LICENSE](licenses/DiffSynth-Studio-LICENSE) |
| lodestones Base 8-step DMD2 | Separately downloaded, FP32 unmerged distillation adapter | Apache-2.0 declaration in the [pinned model card](https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd/blob/da05cb3b52899179e2bb5f7efe1058fefcbae21b/README.md); preserved as [DMD2-MODEL-CARD.md](licenses/DMD2-MODEL-CARD.md) |

The DMD2 source has no separate LICENSE file at the pinned revision; its model
card declaration is recorded rather than inventing an upstream license file.
The complete standard Apache-2.0 terms are supplied in LICENSE. No upstream
NOTICE file was present in the pinned MOSS/DiffSynth roots; this package's
NOTICE records the sources and the changes.

The principal model-forward modifications are registered clean t=0 reference
tokens, isolated reference attention/KV caching and full-canvas reference RoPE.
Canvas preparation and workflow interactions follow DeepMachine's evaluated
AnyPaint implementation. The older Krea model weights and model-specific DiT
patch are not bundled; their model license is not applied to MOSS weights.
The ComfyUI wrapper uses this same runtime through standard IMAGE/MASK I/O.

PyTorch, Torchvision, Transformers, PEFT, Quanto, DiffSynth, NumPy, Pillow,
OpenCV, Safetensors, Hugging Face Hub, Accelerate and ComfyUI are installed
separately and remain subject to their own licenses. They are not vendored here.
Synthetic showcase sources were already distributed in DeepMachine's preceding
Krea AnyPaint release; the selected MOSS outputs and masks are supplied here
for reproduction. No private training images or training manifests are included.
