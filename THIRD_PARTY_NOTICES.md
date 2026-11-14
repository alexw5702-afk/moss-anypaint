# Licenses and provenance

Node code is MIT licensed; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
It reuses the Krea AnyPaint node interfaces and canvas helpers. The MOSS
registered-reference adapter calls the installed ComfyUI implementation.

| Component | Role in this release | License and source |
| --- | --- | --- |
| Krea AnyPaint nodes | Node interfaces, workflow and canvas helpers | MIT; [source](https://github.com/alexw5702-afk/krea2-anypaint) |
| ComfyUI | Installed native model loaders, encoding, attention and sampler | GPL-3.0; [source](https://github.com/Comfy-Org/ComfyUI); not bundled |
| MOSS-Image-Base | Separately downloaded base model | Apache-2.0; [source](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base); [full license](licenses/MOSS-Image-Base-LICENSE) |
| Official MOSS-Image code | Model provenance | Apache-2.0; [source](https://github.com/OpenMOSS/MOSS-Image); [full license](licenses/MOSS-Image-code-LICENSE) |
| Base 8-step DMD2 | Separately downloaded distillation LoRA | Apache-2.0; [source model card](https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd); [preserved model card](licenses/DMD2-MODEL-CARD.md) |

The MOSS AnyPaint weights retain Apache-2.0. Model weights and private training
data are not bundled. PyTorch, NumPy, Pillow, einops and ComfyUI are installed
separately and retain their licenses. Showcase sources are synthetic examples
from the earlier Krea AnyPaint release; results here use MOSS AnyPaint.
