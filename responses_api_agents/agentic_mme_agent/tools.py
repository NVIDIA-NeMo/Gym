# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent implementation of Agentic-MME's public atomic image interface."""

import base64
import math
from io import BytesIO
from typing import Annotated, Any, Literal

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pydantic import BaseModel, ConfigDict, Field


class ImageArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    image_index: int = Field(ge=0)
    label: str = Field(default="", max_length=1000)


class CropArgs(ImageArgs):
    bbox_2d: list[Annotated[float, Field(ge=0, le=1000)]] = Field(min_length=4, max_length=4)
    zoom_scale: float = Field(default=1.0, gt=0, le=16)


class RotateArgs(ImageArgs):
    angle: float = Field(ge=-360, le=360)
    expand: bool = True


class FlipArgs(ImageArgs):
    direction: Literal["horizontal", "vertical", "both"] = "horizontal"


class ResizeArgs(ImageArgs):
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    scale: float | None = Field(default=None, gt=0, le=16)


class EnhanceArgs(ImageArgs):
    brightness: float | None = Field(default=None, ge=0, le=100)
    contrast: float | None = Field(default=None, ge=0, le=100)
    sharpness: float | None = Field(default=None, ge=0, le=100)


class AutocontrastArgs(ImageArgs):
    cutoff: float = Field(default=0, ge=0, lt=50)


class BlurArgs(ImageArgs):
    radius: int = Field(default=2, ge=0, le=100)


class DenoiseArgs(ImageArgs):
    strength: int = Field(default=10, ge=0, le=100)


class EdgeArgs(ImageArgs):
    method: Literal["canny", "sobel", "simple"] = "canny"


class ThresholdArgs(ImageArgs):
    value: int = Field(default=128, ge=0, le=255)
    mode: Literal["binary", "binary_inv", "trunc", "tozero"] = "binary"


IMAGE_TOOLS: dict[str, tuple[type[ImageArgs], str]] = {
    "crop": (CropArgs, "Crop bbox_2d=[x1,y1,x2,y2] in 0–1000 coordinates; optionally zoom."),
    "rotate": (RotateArgs, "Rotate counterclockwise by angle degrees; optionally expand the canvas."),
    "flip": (FlipArgs, "Mirror horizontally, vertically, or both."),
    "resize": (ResizeArgs, "Resize using width and height, or a scale multiplier."),
    "enhance": (EnhanceArgs, "Adjust brightness, contrast, and sharpness; 1 means unchanged."),
    "grayscale": (ImageArgs, "Convert to grayscale."),
    "autocontrast": (AutocontrastArgs, "Stretch contrast, ignoring cutoff percent at each end."),
    "blur": (BlurArgs, "Apply Gaussian blur."),
    "sharpen": (ImageArgs, "Apply the sharpening filter."),
    "denoise": (DenoiseArgs, "Apply OpenCV non-local means color denoising."),
    "edge_detect": (EdgeArgs, "Find edges using canny, sobel, or the simple Pillow filter."),
    "invert": (ImageArgs, "Invert RGB colors."),
    "equalize": (ImageArgs, "Equalize the image histogram."),
    "threshold": (ThresholdArgs, "Threshold grayscale pixels using value and mode."),
}


def function_schema(name: str, model: type[BaseModel], description: str) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": model.model_json_schema(),
        "strict": False,
    }


def image_data_url(image: Image.Image) -> str:
    stream = BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


class ImageWorkspace:
    """Per-episode image indices; no model-controlled filesystem or URL access."""

    def __init__(self, *, max_pixels: int = 16_000_000, max_total_pixels: int = 64_000_000) -> None:
        self.images: list[Image.Image] = []
        self.max_pixels = max_pixels
        self.max_total_pixels = max_total_pixels
        self.total_pixels = 0

    def check_size(self, width: int, height: int) -> None:
        pixels = width * height
        if min(width, height) < 1 or pixels > self.max_pixels:
            raise ValueError("image dimensions exceed the configured per-image limit")
        if pixels + self.total_pixels > self.max_total_pixels:
            raise ValueError("image workspace pixel budget exhausted")

    def add(self, image: Image.Image) -> int:
        self.check_size(*image.size)
        self.images.append(image.convert("RGB"))
        self.total_pixels += image.width * image.height
        return len(self.images) - 1

    def load(self, data_url: str) -> int:
        prefix, separator, encoded = data_url.partition(",")
        if not separator or prefix not in {
            "data:image/png;base64",
            "data:image/jpeg;base64",
            "data:image/webp;base64",
        }:
            raise ValueError("input images must be PNG/JPEG/WebP base64 data URLs; convert the dataset first")
        if len(encoded) > 64_000_000:
            raise ValueError("encoded image exceeds 64 MB")
        with Image.open(BytesIO(base64.b64decode(encoded, validate=True))) as image:
            self.check_size(*image.size)
            return self.add(image)

    def get(self, index: int) -> Image.Image:
        if not 0 <= index < len(self.images):
            raise ValueError("image_index does not exist in this rollout")
        return self.images[index]

    def apply(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name not in IMAGE_TOOLS:
            raise ValueError("unknown image tool")
        args = IMAGE_TOOLS[name][0].model_validate(arguments)
        image = self.get(args.image_index)
        if name not in {"crop", "resize", "rotate"}:
            self.check_size(*image.size)
        if isinstance(args, CropArgs):
            x1, y1, x2, y2 = args.bbox_2d
            box = (
                int(min(x1, x2) * image.width / 1000),
                int(min(y1, y2) * image.height / 1000),
                int(max(x1, x2) * image.width / 1000),
                int(max(y1, y2) * image.height / 1000),
            )
            zoom = max(1.0, args.zoom_scale)
            size = (int((box[2] - box[0]) * zoom), int((box[3] - box[1]) * zoom))
            self.check_size(*size)
            out = image.crop(box)
            if zoom > 1:
                out = out.resize(size, Image.Resampling.LANCZOS)
        elif isinstance(args, RotateArgs):
            angle = math.radians(args.angle)
            width, height = image.size
            if args.expand:
                # Conservative bound before Pillow allocates the expanded canvas.
                width = math.ceil(abs(image.width * math.cos(angle)) + abs(image.height * math.sin(angle))) + 2
                height = math.ceil(abs(image.width * math.sin(angle)) + abs(image.height * math.cos(angle))) + 2
            self.check_size(width, height)
            out = image.rotate(args.angle, expand=args.expand)
        elif isinstance(args, FlipArgs):
            out = ImageOps.mirror(image) if args.direction in {"horizontal", "both"} else image
            if args.direction in {"vertical", "both"}:
                out = ImageOps.flip(out)
        elif isinstance(args, ResizeArgs):
            if args.width is not None and args.height is not None:
                size = (args.width, args.height)
            elif args.scale is not None:
                size = (int(image.width * args.scale), int(image.height * args.scale))
            else:
                raise ValueError("resize needs width and height, or scale")
            self.check_size(*size)
            out = image.resize(size, Image.Resampling.LANCZOS)
        elif isinstance(args, EnhanceArgs):
            out = image
            for enhancer, value in (
                (ImageEnhance.Brightness, args.brightness),
                (ImageEnhance.Contrast, args.contrast),
                (ImageEnhance.Sharpness, args.sharpness),
            ):
                if value is not None:
                    out = enhancer(out).enhance(value)
        elif isinstance(args, AutocontrastArgs):
            out = ImageOps.autocontrast(image, cutoff=args.cutoff)
        elif isinstance(args, BlurArgs):
            out = image.filter(ImageFilter.GaussianBlur(args.radius))
        elif isinstance(args, DenoiseArgs):
            out = Image.fromarray(
                cv2.fastNlMeansDenoisingColored(np.asarray(image), None, args.strength, args.strength, 7, 21)
            )
        elif isinstance(args, EdgeArgs):
            gray = np.asarray(image.convert("L"))
            if args.method == "canny":
                out = Image.fromarray(cv2.Canny(gray, 100, 200))
            elif args.method == "sobel":
                x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                out = Image.fromarray(np.clip(np.hypot(x, y), 0, 255).astype(np.uint8))
            else:
                out = image.filter(ImageFilter.FIND_EDGES)
        elif isinstance(args, ThresholdArgs):
            values = np.asarray(image.convert("L"))
            above = values > args.value
            if args.mode == "binary":
                values = np.where(above, 255, 0)
            elif args.mode == "binary_inv":
                values = np.where(above, 0, 255)
            elif args.mode == "trunc":
                values = np.minimum(values, args.value)
            else:
                values = np.where(above, values, 0)
            out = Image.fromarray(values.astype(np.uint8))
        elif name == "grayscale":
            out = ImageOps.grayscale(image)
        elif name == "sharpen":
            out = image.filter(ImageFilter.SHARPEN)
        elif name == "invert":
            out = ImageOps.invert(image)
        else:
            out = ImageOps.equalize(image)
        index = self.add(out)
        return {
            "ok": True,
            "op": name,
            "source_image_index": args.image_index,
            "new_image_index": index,
            "label": args.label,
            "size": list(out.size),
            "image_url": image_data_url(self.images[index]),
        }
