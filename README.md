# ComfyUI MOSS AnyPaint

Native ComfyUI nodes for the
[`yijunwang2/moss-anypaint`](https://huggingface.co/yijunwang2/moss-anypaint)
functional adapter. AnyPaint adds arbitrary-mask **inpainting**, **outpainting**,
and mixed **image editing** to MOSS-Image-Base.

[Model and portable inference example](https://huggingface.co/yijunwang2/moss-anypaint)
| [ComfyUI releases](https://github.com/alexw5702-afk/moss-anypaint/releases)

![AnyPaint examples](showcase.webp)

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

Restart ComfyUI, then place
`moss_anypaint_rank32.safetensors` in `ComfyUI/models/loras`.

## Requirements

- A current ComfyUI build with native MOSS support.
- [`moss_image_base_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/diffusion_models/moss_image_base_int8_convrot.safetensors)
  in `ComfyUI/models/diffusion_models`.
- [`qwen3.5_9b_int8_convrot.safetensors`](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/text_encoders/qwen3.5_9b_int8_convrot.safetensors)
  in `ComfyUI/models/text_encoders`.
- [`flux2-vae_small.safetensors`](https://huggingface.co/Comfy-Org/MOSS-Image/blob/main/vae/flux2-vae_small.safetensors)
  in `ComfyUI/models/vae`.
- [MOSS AnyPaint LoRA](https://huggingface.co/yijunwang2/moss-anypaint/blob/main/moss_anypaint_rank32.safetensors).
- [Base 8-step DMD2 LoRA](https://huggingface.co/lodestones/MOSS-Image-Base-8-steps-dmd/blob/main/lora_dmd2.safetensors)
  in `ComfyUI/models/loras`. Load it with a second ordinary Load LoRA node.

No custom sampler and no final source composite are required.

## Nodes

### MOSS AnyPaint Prepare

Takes a source image, optional arbitrary mask, and four padding values. White
mask pixels and all padded canvas pixels are generated; black pixels are
preserved. The node outputs:

- a 384px semantic reference for MOSS reference attention;
- the full-resolution known canvas;
- generated and preserved masks;
- aligned canvas dimensions.

Use the `MASK` output from ComfyUI's `Load Image` Mask Editor directly. For
inpainting, keep all padding at zero. For outpainting, add space on any side.
Both can be combined in one request.

### MOSS AnyPaint Encode

Encodes the prompt/reference and returns a standard ComfyUI `LATENT` containing
the known image plus a token-aligned `noise_mask`. ComfyUI's sampler applies the
same timestep-matched preservation used by the standalone reference pipeline:

```text
known_at_sigma = sigma * initial_noise + (1 - sigma) * known_latent
```

### MOSS AnyPaint Model Patch

Registers the semantic reference over the complete target grid and precomputes
its isolated reference K/V cache. Connect this node after `Load LoRA` and before
`KSampler`.

## Workflow

Drag [`moss_anypaint_workflow.json`](moss_anypaint_workflow.json) into
ComfyUI. The important wiring is:

```text
Load Image (image + mask) -> AnyPaint Prepare
AnyPaint Prepare + CLIP + VAE -> AnyPaint Encode
Load Diffusion Model -> Load LoRA (AnyPaint) -> Load LoRA (DMD2)
    -> AnyPaint Model Patch -> KSampler
AnyPaint Encode.latent -> KSampler.latent_image
KSampler -> VAE Decode -> Save Image
```

Recommended MOSS-Image-Base settings: 8 steps, Euler, simple scheduler, CFG 1.0,
LoRA strength 1.0, K/V cache enabled.

The output is the raw model decode. Do not add the old registered-outpaint
composite node: AnyPaint preserves known pixels during every denoising step.

## Usage

### Inpaint without changing image size

1. Import `moss_anypaint_workflow.json` and select your installed model, text
   encoder, VAE, AnyPaint LoRA and DMD2 LoRA in their loader nodes.
2. Load your source image and connect a `MASK` from **SAM or another mask node**
   to Prepare's `generated_mask`. The source `IMAGE` connects to Prepare's `source`.
   You can also use the already-wired Load Image `MASK` output with Mask Editor.
3. Leave Prepare's `left`, `top`, `right` and `bottom` at **0**. This is the default;
   it does not request outpainting. Dimensions are aligned upward to multiples
   of 16, so non-aligned inputs can still gain a few right/bottom pixels.
4. Enter the **complete desired image description** in **MOSS AnyPaint Encode →
   prompt**, not Prepare or the empty negative-prompt node. For example, to remove
   a mug from a table, describe the table and the background you want to remain.
   This guides generation; exact object removal is not guaranteed.
5. Queue the workflow. KSampler → VAE Decode → Save Image produces the result.

### Outpaint

Set only the side(s) you want to extend in Prepare: `left`, `top`, `right`,
`bottom`, in pixels. For example, `right=256` adds 256 pixels on the right;
leave the other sides at 0. Padding creates the generated area automatically,
so no hand-drawn mask is needed for a pure outpaint. You can also draw an internal
mask to combine inpainting and outpainting. The source image is not stretched.

### Connect a mask from SAM or other nodes

Connect the source `IMAGE` to Prepare's `source`, and a `MASK` from SAM or any
other mask-producing node to Prepare's `generated_mask`. There is no required
mask tool. **White / 1 means regenerate; black / 0 means preserve.** For object
removal, the selected object should be white. Preview the mask to check the
selection and invert it if the generating node uses the opposite convention.

```text
Load Image.IMAGE ──────────────────────> AnyPaint Prepare.source
       └──> SAM / segmentation node ──> AnyPaint Prepare.generated_mask
```

A segmentation prompt selects **what to mask**; the AnyPaint generation prompt
in Encode describes **the image you want afterward**.
If a mask misses an object's outline, hair or shadow, optionally use a standard
Grow Mask node before Prepare. Choose the expansion yourself; this workflow
does not add an automatic growth setting or a required segmentation dependency.

Other optional mask sources:

- **Mask Editor:** use the source Load Image's `MASK` output after saving the edit.
  The bundled single-Load-Image workflow is wired this way; replace that MASK
  connection with your SAM output if preferred.
- **Separate black/white image:** Load Image `IMAGE` → ImageToMask (`red`) →
  Prepare `generated_mask`. Its ordinary Load Image `MASK` output reads inverse
  transparency, not black/white colors.

Masks are binary: values above 0.5 generate, all other values preserve. Gray or
partially transparent masks do not provide proportional edit strength.

### Update existing workflows

Import the updated workflow: it uses ComfyUI's native model, text encoder, VAE,
LoRA loaders and KSampler. The earlier MOSS Loader/Generate workflow is replaced
by the same Prepare → Encode → Model Patch layout used by Krea 2 AnyPaint.
Keep padding at 0 for inpainting. No new AnyPaint weight is required.

## Credits

Node interfaces, workflow and canvas helpers reuse
[`alexw5702-afk/krea2-anypaint`](https://github.com/alexw5702-afk/krea2-anypaint).
MOSS model loading, text encoding, attention and sampling use native ComfyUI.
See `NOTICE` for the complete attribution chain.

Node code is MIT licensed. The separately downloaded MOSS AnyPaint and Base
8-step DMD2 weights are Apache-2.0 licensed and retain their upstream terms.
