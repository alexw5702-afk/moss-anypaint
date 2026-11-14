# ComfyUI MOSS AnyPaint

ComfyUI nodes for the
[`yijunwang2/moss-anypaint`](https://huggingface.co/yijunwang2/moss-anypaint)
functional adapter. AnyPaint adds arbitrary-mask **inpainting**, **outpainting**,
and mixed **image editing** to MOSS-Image-Base.

[Model and portable inference example](https://huggingface.co/yijunwang2/moss-anypaint)
| [ComfyUI releases](https://github.com/alexw5702-afk/moss-anypaint/releases)

![AnyPaint examples](showcase/contact_sheet.webp)

<!-- SHOWCASE_EXAMPLES_START -->
<details>
<summary><strong>Open the complete source / mask / result set and prompts</strong></summary>

### 01_photo_internal - Photographic internal edit

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/01_photo_internal_source.webp) | ![](showcase/01_photo_internal_mask.png) | ![](showcase/01_photo_internal_result.webp) |

**Prompt:** a sunlit modern kitchen interior with a handmade ceramic bowl of fresh lemons on the center island, coherent morning light, complete architectural photograph

### 02_watercolor_outpaint - Watercolor downward outpainting

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/02_watercolor_outpaint_source.webp) | ![](showcase/02_watercolor_outpaint_mask.png) | ![](showcase/02_watercolor_outpaint_result.webp) |

**Prompt:** an original watercolor greenhouse and coastal garden, the path and flowering plants continuing naturally downward, airy washes, complete vertical composition

### 03_3d_outpaint - Stylized 3D horizontal outpainting

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/03_3d_outpaint_source.webp) | ![](showcase/03_3d_outpaint_mask.png) | ![](showcase/03_3d_outpaint_result.webp) |

**Prompt:** an original retro-futuristic exploration rover crossing a colorful alien desert, distant mineral formations extending to the right, stylized 3D cinematic render

### 04_portrait_mixed - Original-character mixed edit

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/04_portrait_mixed_source.webp) | ![](showcase/04_portrait_mixed_mask.png) | ![](showcase/04_portrait_mixed_result.webp) |

**Prompt:** an original adult woman with short silver hair wearing a richly textured coral coat in a spacious contemporary gallery, complete full-body fashion photograph

### 05_botanical_disconnected - Disconnected honeybee insertion

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/05_botanical_disconnected_source.webp) | ![](showcase/05_botanical_disconnected_mask.png) | ![](showcase/05_botanical_disconnected_result.webp) |

**Prompt:** an original botanical field illustration of blue and white meadow flowers with three small honeybees hovering naturally between the blossoms, delicate wings, fine paper texture, naturalistic painted details

### 07_product_narrow - Narrow cable recolor

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/07_product_narrow_source.webp) | ![](showcase/07_product_narrow_mask.png) | ![](showcase/07_product_narrow_result.webp) |

**Prompt:** an original sculptural brushed aluminum desk lamp with a vivid orange braided power cable extending from the lamp across the tabletop, premium neutral product photography

### 08_landscape_complex - Complex semantic landscape edit

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/08_landscape_complex_source.webp) | ![](showcase/08_landscape_complex_mask.png) | ![](showcase/08_landscape_complex_result.webp) |

**Prompt:** an original alpine valley after rain with clusters of luminous rainbow-colored fantasy mushrooms growing along the winding riverbanks and foreground rocks, subtle magical glow reflected in wet stone, coherent wide cinematic landscape photograph


### 10_female_accessories - Female disconnected accessories

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/10_female_accessories_source.webp) | ![](showcase/10_female_accessories_mask.png) | ![](showcase/10_female_accessories_result.webp) |

**Prompt:** a polished portrait of an adult woman with shoulder-length auburn hair, a dark green enamel hair clip above her left ear and matching small geometric earrings, simple cream blouse, coherent natural daylight and pale gray studio background

### 11_product_restyle - Shoes and bag local redesign

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/11_product_restyle_source.webp) | ![](showcase/11_product_restyle_mask.png) | ![](showcase/11_product_restyle_result.webp) |

**Prompt:** an overhead catalog photograph with a pair of deep forest-green suede sneakers on the left and a cobalt-blue structured leather shoulder bag on the right, refined minimal hardware, balanced composition, coherent soft studio shadows

### 12_desk_objects - Two desktop object insertions

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/12_desk_objects_source.webp) | ![](showcase/12_desk_objects_mask.png) | ![](showcase/12_desk_objects_result.webp) |

**Prompt:** a clean oak writing desk in a softly lit home office with a handmade dark green ceramic mug on the left side and a small potted succulent in a matte white planter on the right, coherent restrained contemporary interior photograph

### 13_living_room_objects - Living-room multi-region insertion

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/13_living_room_objects_source.webp) | ![](showcase/13_living_room_objects_mask.png) | ![](showcase/13_living_room_objects_result.webp) |

**Prompt:** a bright neutral living room with a tall brushed-brass floor lamp beside the left end of the cream sofa and a round woven storage basket beside the right end, simple coffee table, coherent warm daylight and understated architectural photography

### 14_window_view - Complete window-view replacement

| Source | Generated mask | AnyPaint result |
|:---:|:---:|:---:|
| ![](showcase/14_window_view_source.webp) | ![](showcase/14_window_view_mask.png) | ![](showcase/14_window_view_result.webp) |

**Prompt:** a quiet contemporary reading room whose large central picture window overlooks a snowy mountain lake beneath a clear pale-blue winter sky, armchair in the foreground, coherent daylight, reflections and realistic interior photography


</details>
<!-- SHOWCASE_EXAMPLES_END -->

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/alexw5702-afk/moss-anypaint
```

Install `requirements.txt` with the Python interpreter used by your ComfyUI
instance, then restart ComfyUI.

## Requirements

- Python 3.11+, a current ComfyUI build, and a CUDA GPU with BF16 support.
- [MOSS AnyPaint LoRA](https://huggingface.co/yijunwang2/moss-anypaint/blob/main/moss_anypaint_rank32.safetensors)
  in `ComfyUI/models/loras`.
- [Base 8-step DMD2 `lora_dmd2.safetensors`](https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd/resolve/da05cb3b52899179e2bb5f7efe1058fefcbae21b/lora_dmd2.safetensors)
  in `ComfyUI/models/loras`.

The Loader downloads the MOSS-Image-Base HF components on first use. Alternatively,
set `base_model_dir` to a local folder containing `transformer/`, `text_encoder/`,
`tokenizer/` and `vae/`. Native Comfy-Org single-file checkpoints and SFT weights
are not inputs to this loader.

## Nodes

### MOSS AnyPaint Prepare

Takes a source image, optional arbitrary mask, and four padding values. White
mask pixels and all padded canvas pixels are generated; black pixels are
preserved. The node outputs:

- the prepared canvas and mask for the shared runtime;
- the input with its mask overlay;
- the generated mask.

Use the `MASK` output from ComfyUI's `Load Image` Mask Editor directly. For
inpainting, keep all padding at zero. For outpainting, add space on any side.
Both can be combined in one request.

### MOSS AnyPaint Loader

Loads the MOSS AnyPaint adapter, Base 8-step DMD2 adapter and MOSS-Image-Base
components into the shared HF runtime. Select both adapters from the LoRA folder;
leave `base_model_dir` blank to download the Base components automatically.

### MOSS AnyPaint Generate

Takes the loaded pipeline, prepared input, prompt, seed and inference steps.
Returns a standard ComfyUI `IMAGE` and generation metadata. Set
`keep_model_loaded=true` to reuse the loaded runtime for subsequent requests;
set it false to release model allocations after generation.

## Workflow

Drag [`moss_anypaint_workflow.json`](moss_anypaint_workflow.json) into
ComfyUI. The important wiring is:

```text
Load Image (image + mask) -> MOSS AnyPaint Prepare
MOSS AnyPaint Loader + prepared input -> MOSS AnyPaint Generate -> Save Image
MOSS AnyPaint Prepare.input_with_mask -> Preview Image
```

Copy `examples/moss_anypaint_source.png` and `examples/moss_anypaint_mask.png`
into ComfyUI's input folder for the supplied example. Connect the mask through
`ImageToMask/red`, or use the source image's Mask Editor output directly.
`moss_anypaint_api.json` provides the same graph in ComfyUI API form.

Recommended settings: 8 steps, Euler, CFG 1, shift 5, and both adapter strengths
at 1. The runtime uses qfloat8 frozen DiT/text weights, BF16 compute/task adapter,
FP32 VAE and an unmerged FP32 DMD2 adapter. These nodes use the shared HF runtime
rather than a native KSampler MODEL patch.

Source-sized masks follow the padding offset; canvas-sized masks use canvas
coordinates. Only exterior source seams have a 32-pixel inward redraw band;
internal masks are exact. The output is the raw model decode, with no final
source composite.

Use v0.1.1 or later for full uint64 seed support and corrected source-sized mask
placement. Update the node code and shared runtime together; no new adapter
weight download is required.

## Credits

The nodes use the shared
[MOSS AnyPaint runtime](https://huggingface.co/yijunwang2/moss-anypaint), with model
helpers derived from [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio).
Base and DMD2 weights are downloaded separately. See `NOTICE` and
`THIRD_PARTY_NOTICES.md` for the complete attribution chain.

Node code and adapter weights are Apache-2.0 licensed. See `LICENSE` and
`PIPELINE_LICENSE` for the license terms.
