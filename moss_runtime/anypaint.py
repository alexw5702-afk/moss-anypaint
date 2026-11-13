# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 DeepMachine contributors.
"""Portable AnyPaint canvas preparation; synced from DeepMachine's evaluated runtime.

Internal brushes are exact; only source-exterior seams redraw32px inward.
Apache-2.0. See NOTICE for source attribution.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
REFERENCE_ERASE_PX=0
OUTPAINT_SEAM_PX=32
BRUSH_SEAM_PX=0
SEAM_PX=32

def validate_app_resolution(width, height):
    if width <= 0 or height <= 0 or width % 16 or height % 16:
        raise ValueError("Canvas dimensions must be positive multiples of16")

def build_edge_aware_keep_mask(
    generated_mask: Image.Image,
    seam_px: int = SEAM_PX,
) -> Image.Image:
    """Release a boundary band around any generated region for coherent redraw."""
    generated = np.where(np.asarray(generated_mask.convert("L")) > 0, 255, 0).astype(
        np.uint8
    )
    if seam_px > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (seam_px * 2 + 1, seam_px * 2 + 1),
        )
        generated = cv2.dilate(generated, kernel)
    return ImageOps.invert(Image.fromarray(generated, mode="L"))

@dataclass(frozen=True)
class OutpaintBox:
    x: float
    y: float
    width: float
    height: float
    rotation: float = 0.0

    def validate(self, *, allow_outside: bool = False) -> None:
        values = (self.x, self.y, self.width, self.height)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Outpaint box values must be finite")
        if not math.isfinite(self.rotation):
            raise ValueError("Outpaint box rotation must be finite")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("Outpaint box must have positive width and height")
        if allow_outside:
            if self.width > 4.0 or self.height > 4.0:
                raise ValueError("Outpaint box dimensions are unreasonably large")
            if self.x < -4.0 or self.y < -4.0 or self.x > 4.0 or self.y > 4.0:
                raise ValueError("Outpaint box position is unreasonably far outside the canvas")
            return
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError("Outpaint box values must be normalized to [0, 1]")
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("Outpaint box must stay inside the canvas")

def resize_condition(image: Image.Image, max_edge: int = 384) -> Image.Image:
    scale = min(1.0, max_edge / max(image.size))
    width = max(16, int(round(image.width * scale)) // 16 * 16)
    height = max(16, int(round(image.height * scale)) // 16 * 16)
    return image.resize((width, height), Image.Resampling.LANCZOS)

@dataclass(frozen=True)
class InOutInput:
    condition: Image.Image
    known_image: Image.Image
    reference_mask: Image.Image
    keep_mask: Image.Image
    generated_mask: Image.Image
    canvas_width: int
    canvas_height: int
    bbox_pixels: tuple[int, int, int, int]
    bbox_normalized: list[float]

    @property
    def metadata(self) -> dict[str, object]:
        generated = np.asarray(self.generated_mask, dtype=np.uint8) > 0
        return {
            "canvas_size": [self.canvas_width, self.canvas_height],
            "condition_size": list(self.condition.size),
            "bbox_pixels": list(self.bbox_pixels),
            "bbox_normalized": self.bbox_normalized,
            "generated_fraction": round(float(generated.mean()), 6),
            "mask_convention": "0=known,255=generated",
            "condition_fill": "median-known-rgb",
            "reference_erase_px": REFERENCE_ERASE_PX,
            "seam_px": OUTPAINT_SEAM_PX,
            "brush_seam_px": BRUSH_SEAM_PX,
            "seam_scope": "canvas-exterior-only",
        }

def _median_fill(image: Image.Image, known: np.ndarray) -> np.ndarray:
    values = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    known_values = values[known]
    fill = (
        np.median(known_values, axis=0).round().astype(np.uint8)
        if len(known_values)
        else np.array([127, 127, 127], dtype=np.uint8)
    )
    values[~known] = fill
    return values

def _place_transformed_source(
    source: Image.Image,
    width: int,
    height: int,
    box: OutpaintBox | None,
) -> tuple[Image.Image, Image.Image, tuple[int, int, int, int]]:
    """Rasterize the source transform and return its canvas-space alpha footprint."""
    if box is None:
        source_ratio = source.width / source.height
        canvas_ratio = width / height
        placed_width = width
        placed_height = height
        if source_ratio >= canvas_ratio:
            placed_height = max(1, round(width / source_ratio))
        else:
            placed_width = max(1, round(height * source_ratio))
        x0 = (width - placed_width) // 2
        y0 = (height - placed_height) // 2
        box = OutpaintBox(
            x=x0 / width,
            y=y0 / height,
            width=placed_width / width,
            height=placed_height / height,
        )

    box.validate(allow_outside=True)
    placed_width = max(1, round(box.width * width))
    placed_height = max(1, round(box.height * height))
    center_x = (box.x + box.width / 2) * width
    center_y = (box.y + box.height / 2) * height

    patch = source.convert("RGBA").resize(
        (placed_width, placed_height), Image.Resampling.LANCZOS
    )
    if abs(box.rotation) > 1e-6:
        # CSS positive angles rotate clockwise in screen coordinates; Pillow uses
        # positive counter-clockwise angles, hence the sign inversion.
        patch = patch.rotate(
            -box.rotation,
            resample=Image.Resampling.BICUBIC,
            expand=True,
        )

    paste_x = round(center_x - patch.width / 2)
    paste_y = round(center_y - patch.height / 2)
    rgba_canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    crop_left = max(0, -paste_x)
    crop_top = max(0, -paste_y)
    crop_right = min(patch.width, width - paste_x)
    crop_bottom = min(patch.height, height - paste_y)
    if crop_left >= crop_right or crop_top >= crop_bottom:
        raise ValueError("Transformed source must overlap the canvas")
    visible_patch = patch.crop((crop_left, crop_top, crop_right, crop_bottom))
    rgba_canvas.alpha_composite(
        visible_patch,
        (max(0, paste_x), max(0, paste_y)),
    )
    alpha = rgba_canvas.getchannel("A")
    bbox = alpha.getbbox()
    if bbox is None:
        raise ValueError("Transformed source has no visible pixels")
    return rgba_canvas.convert("RGB"), alpha, bbox

def prepare_in_out(
    source: Image.Image,
    width: int,
    height: int,
    box: OutpaintBox | None,
    brush_mask: Image.Image | None,
) -> InOutInput:
    validate_app_resolution(width, height)
    placed_canvas, source_alpha, source_bbox = _place_transformed_source(
        source.convert("RGB"), width, height, box
    )
    outside_generated = Image.fromarray(
        np.where(np.asarray(source_alpha, dtype=np.uint8) > 127, 0, 255).astype(np.uint8),
        mode="L",
    )
    generated = outside_generated.copy()
    brush_values = np.zeros((height, width), dtype=np.uint8)
    if brush_mask is not None:
        brush = brush_mask.convert("L")
        if brush.size != (width, height):
            brush = brush.resize((width, height), Image.Resampling.NEAREST)
        brush_values = np.asarray(brush, dtype=np.uint8)
        generated = Image.fromarray(
            np.maximum(
                np.asarray(generated, dtype=np.uint8),
                brush_values,
            ),
            mode="L",
        )

    generated_values = np.asarray(generated, dtype=np.uint8) > 0
    if not generated_values.any():
        raise ValueError("In&Out mask has no area to generate")

    placed_values = np.asarray(placed_canvas, dtype=np.uint8)
    source_known = np.asarray(source_alpha, dtype=np.uint8) > 127
    source_values = placed_values[source_known]
    source_fill = np.median(source_values, axis=0).round().astype(np.uint8)
    canvas = np.empty((height, width, 3), dtype=np.uint8)
    canvas[:] = source_fill
    canvas[source_known] = placed_values[source_known]
    known_image = Image.fromarray(canvas, mode="RGB")
    reference_keep_mask = build_edge_aware_keep_mask(generated, REFERENCE_ERASE_PX)
    reference_keep_values = np.asarray(reference_keep_mask, dtype=np.uint8) > 0
    reference_mask = Image.fromarray(
        np.where(reference_keep_values, 0, 255).astype(np.uint8),
        mode="L",
    )
    condition = Image.fromarray(_median_fill(known_image, reference_keep_values), mode="RGB")
    outpaint_keep = np.asarray(
        build_edge_aware_keep_mask(outside_generated, OUTPAINT_SEAM_PX),
        dtype=np.uint8,
    )
    brush_keep = np.where(brush_values > 0, 0, 255).astype(np.uint8)
    keep_mask = Image.fromarray(np.minimum(outpaint_keep, brush_keep), mode="L")

    return InOutInput(
        condition=resize_condition(condition, max_edge=384),
        known_image=known_image,
        reference_mask=reference_mask,
        keep_mask=keep_mask,
        generated_mask=generated,
        canvas_width=width,
        canvas_height=height,
        bbox_pixels=source_bbox,
        bbox_normalized=[0.0, 0.0, 1.0, 1.0],
    )

AnyPaintInput = InOutInput

def prepare_anypaint(source, generated_mask, canvas_size, bbox=None):
    """Place a source/canvas mask; white pixels generate, bbox uses canvas pixels.

    A canvas-sized mask is already registered. A source-sized mask follows the
    source's bbox transform. Other sizes are rejected instead of stretched.
    """
    width,height=map(int,canvas_size)
    validate_app_resolution(width, height)
    x0,y0,x1,y1=bbox or (0,0,width,height)
    box=OutpaintBox(x0/width,y0/height,(x1-x0)/width,(y1-y0)/height)
    box.validate(allow_outside=True)
    if generated_mask is not None:
        generated_mask = generated_mask.convert("L")
        if generated_mask.size != (width, height):
            if generated_mask.size != source.size:
                raise ValueError("Mask must match either the source image or the output canvas")
            placed_width = max(1, round(box.width * width))
            placed_height = max(1, round(box.height * height))
            paste_x = round((box.x + box.width / 2) * width - placed_width / 2)
            paste_y = round((box.y + box.height / 2) * height - placed_height / 2)
            placed_mask = generated_mask.resize((placed_width, placed_height), Image.Resampling.NEAREST)
            generated_mask = Image.new("L", (width, height), 0)
            generated_mask.paste(placed_mask, (paste_x, paste_y))
    return prepare_in_out(source,width,height,box,generated_mask)
