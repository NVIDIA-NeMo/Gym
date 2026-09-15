# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generic Gym environment for the parameter-only image tool suite."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import uuid
from io import BytesIO
from math import ceil, floor
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, TypedDict


if TYPE_CHECKING:
    from PIL import Image


IMAGE_TOOL_NAMES = frozenset(
    {
        "image_zoom_in_tool",
        "image_crop_tool",
        "image_rotate_tool",
        "image_flip_tool",
        "image_diff_tool",
        "image_side_by_side_tool",
        "image_overlay_tool",
        "count_objects_tool",
        "find_color_tool",
        "color_at_tool",
    }
)

_XML_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_JSON_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_XML_PARAMETER_RE = re.compile(
    r"<parameter=([^>\s]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
_NUMBER_RE = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


class ImageToolsGymToolConfig(TypedDict, total=False):
    crop_dir: str
    crop_format: str
    crop_jpeg_quality: int
    crop_min_pixels: int
    crop_max_pixels: int
    max_tool_calls: int
    max_tool_calls_per_turn: int
    tool_success_reward: float
    tool_success_reward_cap: float
    invalid_tool_call_penalty: float
    duplicate_tool_call_penalty: float
    force_final_after_duplicate: bool
    terminate_on_invalid_tool_call: bool
    forced_final_prompt: str
    duplicate_iou_threshold: float
    stop_strings: Optional[list[str]]


class ImageToolsGymToolMetadata(TypedDict, total=False):
    ground_truth: str
    image_paths: list[str]
    dataset: str
    tool_call_count: int
    tool_error_count: int
    crop_paths: list[str]
    seen_tool_sigs: list[str]
    seen_bboxes: dict[str, list[list[float]]]
    force_final_next: bool


def _parse_bbox(value: str) -> Any:
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass
    numbers = re.findall(_NUMBER_RE, value)
    if len(numbers) >= 4:
        return [float(number) for number in numbers[:4]]
    return value


def _parse_parameter_value(value: str) -> Any:
    """Parse the JSON-compatible parameter values used by the image tools."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_image_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse XML or JSON image tool calls from model text."""
    parsed_calls: list[dict[str, Any]] = []
    for idx, match in enumerate(_XML_TOOL_CALL_RE.finditer(text or "")):
        tool_name = match.group(1).strip()
        body = match.group(2)
        arguments: dict[str, Any] = {}
        for param_match in _XML_PARAMETER_RE.finditer(body):
            param_name = param_match.group(1).strip()
            value = param_match.group(2).strip()
            arguments[param_name] = _parse_parameter_value(value)
        parsed_calls.append(
            {
                "id": f"xml_call_{idx}",
                "name": tool_name,
                "arguments": arguments,
                "source": "xml",
            }
        )

    for idx, match in enumerate(_JSON_TOOL_CALL_RE.finditer(text or "")):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        name = payload.get("name") or payload.get("tool_name")
        arguments = payload.get("arguments") or {}
        if name:
            parsed_calls.append(
                {
                    "id": f"json_call_{idx}",
                    "name": name,
                    "arguments": arguments,
                    "source": "json",
                }
            )
    return parsed_calls


def has_malformed_image_tool_markup(text: str) -> bool:
    """Return True when tool-call markup is present but not a clean parsed call.

    The vLLM reasoning parser should remove the reasoning boundary before the
    image-zoom agent sees message content, so clean tool content is just
    ``<tool_call>...</tool_call>``. Any leftover reasoning tag or nested
    tool-call tag is a parser artifact that should not be executed or trained
    as a valid tool call.
    """
    text = text or ""
    has_tool_marker = any(marker in text for marker in ("<tool_call>", "</tool_call>", "<function=", "</function>"))
    if not has_tool_marker:
        return False

    parsed_calls = parse_image_tool_calls(text)
    if not parsed_calls:
        return True

    xml_call_count = sum(1 for call in parsed_calls if call.get("source") == "xml")
    if text.count("<tool_call>") != len(parsed_calls):
        return True
    if text.count("</tool_call>") != len(parsed_calls):
        return True
    if text.count("<function=") != xml_call_count:
        return True
    if text.count("</function>") != xml_call_count:
        return True
    if "</think>" in text or "<think>" in text:
        return True

    return False


def has_malformed_image_tool_raw_generation(text: str) -> bool:
    """Validate raw model continuations that may legitimately close thinking.

    The chat template opens the assistant turn with ``<think>`` before
    generation, so raw generated tokens for a clean tool call usually contain a
    bare ``</think>`` followed by ``<tool_call>``. That closing tag is expected
    in raw generation but would be malformed in already parsed Responses
    message content.

    A generation that never closes thinking is a reasoning-only parser artifact,
    not visible assistant content. Likewise, a thinking boundary inside the XML
    tool call means the parser and trainer will not see a stable assistant turn.
    """
    text = text or ""
    if "<think>" in text:
        return True

    think_close_positions = [match.start() for match in re.finditer(re.escape("</think>"), text)]
    first_tool_call = text.find("<tool_call>")

    if first_tool_call >= 0:
        closes_before_tool = [pos for pos in think_close_positions if pos < first_tool_call]
        if len(closes_before_tool) != 1:
            return True
        if any(pos > first_tool_call for pos in think_close_positions):
            return True
        text = text[: closes_before_tool[0]] + text[closes_before_tool[0] + len("</think>") :]
    else:
        if len(think_close_positions) == 0:
            return True
        if len(think_close_positions) > 1:
            return True
        text = text[: think_close_positions[0]] + text[think_close_positions[0] + len("</think>") :]

    return has_malformed_image_tool_markup(text)


def extract_last_assistant_text(message_log: list[dict[str, Any]]) -> str:
    for message in reversed(message_log):
        if message.get("role") == "assistant":
            return str(message.get("content") or "")
    return ""


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 56 * 56,
    max_pixels: int = 12845056,
) -> tuple[int, int]:
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _open_image(image_ref: str) -> Image.Image:
    from PIL import Image

    if image_ref.startswith("data:"):
        _, encoded = image_ref.split(",", 1)
        return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
    image_arg = image_ref[len("file://") :] if image_ref.startswith("file://") else image_ref
    if image_arg.startswith(("http://", "https://")):
        import requests

        response = requests.get(image_arg, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_arg).convert("RGB")


def coerce_bbox(value: Any) -> list[float] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = re.findall(_NUMBER_RE, value)[:4]
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in value)
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _bbox_to_pixels(bbox: list[float], img_w: int, img_h: int) -> tuple[float, float, float, float, str]:
    x1, y1, x2, y2 = bbox
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) > 1000.0:
        return x1, y1, x2, y2, "pixel"
    return (
        x1 / 1000.0 * img_w,
        y1 / 1000.0 * img_h,
        x2 / 1000.0 * img_w,
        y2 / 1000.0 * img_h,
        "normalized",
    )


def _maybe_resize_bbox(
    left: float,
    top: float,
    right: float,
    bottom: float,
    img_width: int,
    img_height: int,
) -> list[int]:
    left = max(0, floor(left))
    top = max(0, floor(top))
    right = min(img_width, ceil(right))
    bottom = min(img_height, ceil(bottom))

    height = bottom - top
    width = right - left
    if height < 32 or width < 32:
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        ratio = 32 / max(1, min(height, width))
        new_half_height = ceil(height * ratio * 0.5)
        new_half_width = ceil(width * ratio * 0.5)
        left = max(0, floor(center_x - new_half_width))
        right = min(img_width, ceil(center_x + new_half_width))
        top = max(0, floor(center_y - new_half_height))
        bottom = min(img_height, ceil(center_y + new_half_height))

    return [left, top, right, bottom]


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _save_image_output(
    image: Image.Image,
    work_dir: str | os.PathLike[str],
    *,
    image_format: str,
    jpeg_quality: int,
    max_side: int = 4096,
) -> tuple[str, list[int]]:
    from PIL import Image

    image = image.convert("RGB")
    if max(image.size) > max_side:
        image = image.copy()
        image.thumbnail((max_side, max_side), resample=Image.Resampling.LANCZOS)

    output_dir = Path(work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    normalized_format = image_format.lower()
    if normalized_format in {"jpg", "jpeg"}:
        output_path = output_dir / f"{uuid.uuid4()}.jpg"
        image.save(output_path, format="JPEG", quality=jpeg_quality)
    elif normalized_format == "png":
        output_path = output_dir / f"{uuid.uuid4()}.png"
        image.save(output_path, format="PNG")
    else:
        raise ValueError(f"Unsupported crop_format: {image_format}")
    return str(output_path.resolve()), [image.width, image.height]


def _image_at(image_paths: list[str], value: Any, parameter: str) -> tuple[int, Image.Image]:
    try:
        img_idx = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{parameter} must be an integer") from exc
    if img_idx < 0 or img_idx >= len(image_paths):
        raise ValueError(f"{parameter} {img_idx} is out of range for {len(image_paths)} image(s)")
    return img_idx, _open_image(image_paths[img_idx])


def _normalized_bbox(bbox: tuple[int, int, int, int] | list[int], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [
        max(0, min(1000, round(x1 / width * 1000))),
        max(0, min(1000, round(y1 / height * 1000))),
        max(0, min(1000, round(x2 / width * 1000))),
        max(0, min(1000, round(y2 / height * 1000))),
    ]


def _normalized_point(point: tuple[float, float], width: int, height: int) -> list[int]:
    x, y = point
    return [
        max(0, min(1000, round(x / width * 1000))),
        max(0, min(1000, round(y / height * 1000))),
    ]


def _crop_box(arguments: dict[str, Any], image: Image.Image) -> tuple[list[float], list[int], str]:
    bbox = coerce_bbox(arguments.get("bbox_2d"))
    if bbox is None:
        raise ValueError("bbox_2d must be a list of four numbers")
    x1, y1, x2, y2, coord_mode = _bbox_to_pixels(bbox, image.width, image.height)
    box = [
        max(0, floor(x1)),
        max(0, floor(y1)),
        min(image.width, ceil(x2)),
        min(image.height, ceil(y2)),
    ]
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"Invalid crop after clipping: {box}")
    return bbox, box, coord_mode


def _rgb(value: Any, *, required: bool) -> list[int] | None:
    if value is None and not required:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError("color must be an [R, G, B] list")
    try:
        color = [int(channel) for channel in value]
    except (TypeError, ValueError) as exc:
        raise ValueError("color must contain three integers") from exc
    if any(channel < 0 or channel > 255 for channel in color):
        raise ValueError("color channels must be in the 0-255 range")
    return color


def _connected_components(mask: Any, min_size: int, max_blobs: int = 40) -> list[dict[str, Any]]:
    import numpy as np
    from scipy import ndimage

    labels, _ = ndimage.label(mask)
    objects = ndimage.find_objects(labels)
    blobs: list[dict[str, Any]] = []
    for label_id, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        component = labels[slices] == label_id
        pixels = int(component.sum())
        if pixels < min_size:
            continue
        local_y, local_x = np.nonzero(component)
        y_offset = int(slices[0].start or 0)
        x_offset = int(slices[1].start or 0)
        xs = local_x + x_offset
        ys = local_y + y_offset
        blobs.append(
            {
                "bbox_pixels": [
                    int(xs.min()),
                    int(ys.min()),
                    int(xs.max()) + 1,
                    int(ys.max()) + 1,
                ],
                "center_pixels": [float(xs.mean()), float(ys.mean())],
                "pixels": pixels,
            }
        )
        if len(blobs) >= max_blobs:
            break
    return blobs


def execute_image_tool(
    tool_name: str,
    arguments: dict[str, Any],
    image_paths: list[str],
    work_dir: str | os.PathLike[str],
    *,
    crop_format: str = "png",
    crop_jpeg_quality: int = 95,
    crop_min_pixels: int = 256 * 32 * 32,
    crop_max_pixels: int = 12845056,
) -> dict[str, Any]:
    """Execute one of the dataset's parameter-only image tools.

    The returned ``result`` is JSON-serializable and uses normalized 0-1000
    coordinates. ``path`` points to the image that must be appended to the
    rollout's image store and exposed as the next ``img_idx``.
    """
    import numpy as np
    from PIL import Image, ImageDraw

    if tool_name not in IMAGE_TOOL_NAMES:
        raise ValueError(f"Unknown tool name: {tool_name}")
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be an object")

    label = str(arguments.get("label", ""))
    result: dict[str, Any]
    output: Image.Image

    if tool_name == "image_zoom_in_tool":
        # Zoom is crop plus a resample: `factor` scales the crop, then
        # _smart_resize snaps it to the 32-pixel grid the vision tower expects
        # and clamps it into [crop_min_pixels, crop_max_pixels].
        bbox = coerce_bbox(arguments.get("bbox_2d"))
        if bbox is None:
            raise ValueError("bbox_2d must be a list of four numbers")
        img_idx, image = _image_at(image_paths, arguments.get("img_idx", 0), "img_idx")
        img_width, img_height = image.size
        abs_x1, abs_y1, abs_x2, abs_y2, _ = _bbox_to_pixels(bbox, img_width, img_height)
        left, top, right, bottom = _maybe_resize_bbox(abs_x1, abs_y1, abs_x2, abs_y2, img_width, img_height)
        if right <= left or bottom <= top:
            raise ValueError(f"Invalid crop after clipping: {[left, top, right, bottom]}")
        factor = float(arguments.get("factor", 3.0))
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("factor must be a positive finite number")
        new_h, new_w = _smart_resize(
            max(1, int(round((bottom - top) * factor))),
            max(1, int(round((right - left) * factor))),
            factor=32,
            min_pixels=crop_min_pixels,
            max_pixels=crop_max_pixels,
        )
        output = image.crop((left, top, right, bottom)).resize((new_w, new_h), resample=Image.Resampling.BICUBIC)
        result = {
            "box": bbox,
            "factor": factor,
            "label": label,
            "img_idx": img_idx,
        }

    elif tool_name == "image_crop_tool":
        img_idx, image = _image_at(image_paths, arguments.get("img_idx", 0), "img_idx")
        bbox, box, _ = _crop_box(arguments, image)
        output = image.crop(tuple(box))
        result = {
            "box": bbox,
            "size": list(output.size),
            "label": label,
            "img_idx": img_idx,
        }

    elif tool_name == "image_rotate_tool":
        img_idx, image = _image_at(image_paths, arguments.get("img_idx", 0), "img_idx")
        try:
            degrees = float(arguments["degrees"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("degrees must be a number") from exc
        if not math.isfinite(degrees):
            raise ValueError("degrees must be finite")
        output = image.rotate(degrees, expand=True, fillcolor=(255, 255, 255))
        result = {
            "degrees": degrees,
            "size": list(output.size),
            "label": label,
            "img_idx": img_idx,
        }

    elif tool_name == "image_flip_tool":
        img_idx, image = _image_at(image_paths, arguments.get("img_idx", 0), "img_idx")
        axis = str(arguments.get("axis", ""))
        if axis == "horizontal":
            output = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif axis == "vertical":
            output = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        else:
            raise ValueError('axis must be "horizontal" or "vertical"')
        result = {
            "axis": axis,
            "size": list(output.size),
            "label": label,
            "img_idx": img_idx,
        }

    elif tool_name == "image_diff_tool":
        idx_a, image_a = _image_at(image_paths, arguments.get("img_idx_a"), "img_idx_a")
        idx_b, image_b = _image_at(image_paths, arguments.get("img_idx_b"), "img_idx_b")
        image_a = image_a.convert("RGB")
        image_b = image_b.convert("RGB").resize(image_a.size, Image.Resampling.BICUBIC)
        difference = np.abs(np.asarray(image_a, dtype=np.int16) - np.asarray(image_b, dtype=np.int16)).astype(np.uint8)
        gray = difference.max(axis=2)
        changed = gray > 30
        result = {
            "mean_abs_diff": round(float(difference.mean()), 2),
            "changed_pixel_fraction": round(float(changed.mean()), 4),
            "label": label,
            "img_idx_a": idx_a,
            "img_idx_b": idx_b,
        }
        if changed.any():
            ys, xs = np.nonzero(changed)
            result["changed_bbox"] = _normalized_bbox(
                [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
                image_a.width,
                image_a.height,
            )
        heat = np.zeros_like(difference)
        heat[..., 0] = np.clip(gray.astype(np.uint16) * 3, 0, 255).astype(np.uint8)
        output = Image.blend(image_a, Image.fromarray(heat), 0.6)

    elif tool_name == "image_side_by_side_tool":
        raw_indices = arguments.get("img_indices")
        if not isinstance(raw_indices, (list, tuple)) or not 2 <= len(raw_indices) <= 6:
            raise ValueError("img_indices must contain between two and six image indices")
        loaded = [_image_at(image_paths, value, "img_indices")[1].convert("RGB") for value in raw_indices]
        indices = [int(value) for value in raw_indices]
        height = max(image.height for image in loaded)
        loaded = [
            image.resize(
                (max(1, round(image.width * height / image.height)), height),
                Image.Resampling.BICUBIC,
            )
            for image in loaded
        ]
        labels = arguments.get("labels") or []
        if not isinstance(labels, (list, tuple)):
            raise ValueError("labels must be an array when provided")
        padding = 8
        output = Image.new(
            "RGB",
            (
                sum(image.width for image in loaded) + padding * (len(loaded) - 1),
                height,
            ),
            (255, 255, 255),
        )
        x = 0
        for idx, image in enumerate(loaded):
            if idx < len(labels):
                image = image.copy()
                ImageDraw.Draw(image).text((4, 4), str(labels[idx]), fill=(255, 0, 0))
            output.paste(image, (x, 0))
            x += image.width + padding
        result = {"img_indices": indices, "size": list(output.size), "label": label}

    elif tool_name == "image_overlay_tool":
        idx_a, image_a = _image_at(image_paths, arguments.get("img_idx_a"), "img_idx_a")
        idx_b, image_b = _image_at(image_paths, arguments.get("img_idx_b"), "img_idx_b")
        try:
            alpha = float(arguments.get("alpha", 0.5))
        except (TypeError, ValueError) as exc:
            raise ValueError("alpha must be a number") from exc
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in the 0-1 range")
        image_a = image_a.convert("RGB")
        image_b = image_b.convert("RGB").resize(image_a.size, Image.Resampling.BICUBIC)
        output = Image.blend(image_a, image_b, alpha)
        result = {
            "alpha": alpha,
            "label": label,
            "img_idx_a": idx_a,
            "img_idx_b": idx_b,
        }

    elif tool_name in {"count_objects_tool", "find_color_tool"}:
        img_idx, image = _image_at(image_paths, arguments.get("img_idx", 0), "img_idx")
        image = image.convert("RGB")
        array = np.asarray(image)
        try:
            tolerance = int(arguments.get("tolerance", 40))
        except (TypeError, ValueError) as exc:
            raise ValueError("tolerance must be an integer") from exc
        if not 0 <= tolerance <= 255:
            raise ValueError("tolerance must be in the 0-255 range")
        color = _rgb(arguments.get("color"), required=tool_name == "find_color_tool")
        if color is not None:
            target = np.asarray(color, dtype=np.int16)
            mask = (np.abs(array.astype(np.int16) - target) <= tolerance).all(axis=2)
        else:
            border = np.concatenate([array[0], array[-1], array[:, 0], array[:, -1]], axis=0).reshape(-1, 3)
            background = np.median(border, axis=0).astype(np.int16)
            mask = (np.abs(array.astype(np.int16) - background) > tolerance).any(axis=2)
        try:
            min_size = int(arguments.get("min_size", 30))
        except (TypeError, ValueError) as exc:
            raise ValueError("min_size must be an integer") from exc
        if min_size < 1:
            raise ValueError("min_size must be positive")
        blobs = _connected_components(mask, min_size if tool_name == "count_objects_tool" else 20)
        normalized_blobs = []
        for blob in blobs:
            bbox = _normalized_bbox(blob["bbox_pixels"], image.width, image.height)
            center = _normalized_point(blob["center_pixels"], image.width, image.height)
            normalized_blobs.append(
                {
                    "bbox": bbox,
                    "center": center,
                    "centroid": center,
                    "pixels": blob["pixels"],
                }
            )
        if tool_name == "count_objects_tool":
            output = image.copy()
            draw = ImageDraw.Draw(output)
            for number, blob in enumerate(blobs, start=1):
                x1, y1, x2, y2 = blob["bbox_pixels"]
                draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
                draw.text((x1 + 2, y1 + 2), str(number), fill=(255, 0, 0))
            result = {
                "count": len(normalized_blobs),
                "blobs": normalized_blobs,
                "label": label,
                "img_idx": img_idx,
            }
        else:
            overlay = array.copy()
            overlay[mask] = [255, 0, 255]
            output = Image.fromarray(overlay)
            result = {
                "match_fraction": round(float(mask.mean()), 4),
                "count": len(normalized_blobs),
                "blobs": normalized_blobs,
                "label": label,
                "img_idx": img_idx,
            }

    elif tool_name == "color_at_tool":
        img_idx, image = _image_at(image_paths, arguments.get("img_idx", 0), "img_idx")
        point = arguments.get("point_2d")
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("point_2d must be an [x, y] list")
        try:
            normalized_x, normalized_y = (float(value) for value in point)
        except (TypeError, ValueError) as exc:
            raise ValueError("point_2d must contain two numbers") from exc
        if not (0 <= normalized_x <= 1000 and 0 <= normalized_y <= 1000):
            raise ValueError("point_2d coordinates must be in the 0-1000 range")
        x = min(image.width - 1, max(0, round(normalized_x / 1000 * image.width)))
        y = min(image.height - 1, max(0, round(normalized_y / 1000 * image.height)))
        image = image.convert("RGB")
        rgb = list(image.getpixel((x, y)))
        left, top = max(0, x - 12), max(0, y - 12)
        patch = image.crop((left, top, min(image.width, x + 13), min(image.height, y + 13)))
        output = patch.resize((150, 150), Image.Resampling.NEAREST)
        result = {
            "point_2d": [round(normalized_x), round(normalized_y)],
            "rgb": rgb,
            "label": label,
            "img_idx": img_idx,
        }

    else:  # pragma: no cover - IMAGE_TOOL_NAMES and branches are kept in lockstep.
        raise AssertionError(f"Unhandled image tool: {tool_name}")

    path, saved_size = _save_image_output(
        output,
        work_dir,
        image_format=crop_format,
        jpeg_quality=crop_jpeg_quality,
    )
    result["size"] = saved_size
    return {"path": path, "result": result}


class ImageToolsGymToolLogic:
    def __init__(self, cfg: ImageToolsGymToolConfig):
        self.cfg = cfg
        self.max_tool_calls = int(cfg.get("max_tool_calls", 8))
        self.max_tool_calls_per_turn = int(cfg.get("max_tool_calls_per_turn", 1))
        self.tool_success_reward = float(cfg.get("tool_success_reward", 0.0))
        self.invalid_tool_call_penalty = float(cfg.get("invalid_tool_call_penalty", -0.05))
        self.duplicate_tool_call_penalty = float(cfg.get("duplicate_tool_call_penalty", -0.02))
        self.force_final_after_duplicate = bool(cfg.get("force_final_after_duplicate", True))
        self.terminate_on_invalid_tool_call = bool(cfg.get("terminate_on_invalid_tool_call", True))
        self.duplicate_iou_threshold = float(cfg.get("duplicate_iou_threshold", 0.5))
        self.crop_dir = cfg.get("crop_dir") or os.path.join(os.getcwd(), "image_tool_outputs")
        self.crop_format = str(cfg.get("crop_format", "png"))
        self.crop_jpeg_quality = int(cfg.get("crop_jpeg_quality", 95))
        self.crop_min_pixels = int(cfg.get("crop_min_pixels", 256 * 32 * 32))
        self.crop_max_pixels = int(cfg.get("crop_max_pixels", 12845056))
        self.stop_strings = cfg.get("stop_strings", None)
        self.forced_final_prompt = cfg.get(
            "forced_final_prompt",
            (
                "This is your FINAL turn. Based on the original images and any image "
                "tool results already provided, answer the original question now. Do not "
                "call any more tools."
            ),
        )

    @staticmethod
    def _copy_metadata(
        metadata: ImageToolsGymToolMetadata,
    ) -> ImageToolsGymToolMetadata:
        copied: ImageToolsGymToolMetadata = dict(metadata)
        copied["image_paths"] = list(metadata.get("image_paths", []))
        copied["crop_paths"] = list(metadata.get("crop_paths", []))
        copied["seen_tool_sigs"] = list(metadata.get("seen_tool_sigs", []))
        copied["seen_bboxes"] = {
            str(k): [list(bbox) for bbox in v] for k, v in metadata.get("seen_bboxes", {}).items()
        }
        return copied

    def _force_final_observation(
        self, metadata: ImageToolsGymToolMetadata
    ) -> tuple[dict[str, Any], float, bool, list[str] | None, ImageToolsGymToolMetadata, None]:
        next_metadata = self._copy_metadata(metadata)
        next_metadata["force_final_next"] = True
        return (
            {"role": "user", "content": self.forced_final_prompt},
            0.0,
            False,
            None,
            next_metadata,
            None,
        )

    def _invalid_observation(
        self,
        message: str,
        metadata: ImageToolsGymToolMetadata,
    ) -> tuple[
        dict[str, Any],
        float,
        bool,
        list[str] | None,
        ImageToolsGymToolMetadata | None,
        None,
    ]:
        next_metadata = self._copy_metadata(metadata)
        next_metadata["tool_error_count"] = int(next_metadata.get("tool_error_count", 0)) + 1
        if self.terminate_on_invalid_tool_call:
            return (
                {
                    "role": "user",
                    "content": f"<tool_response>\n{message}\n</tool_response>",
                },
                self.invalid_tool_call_penalty,
                True,
                None,
                next_metadata,
                None,
            )
        return (
            {
                "role": "user",
                "content": f"<tool_response>\n{message}\n</tool_response>",
            },
            self.invalid_tool_call_penalty,
            False,
            self.stop_strings,
            next_metadata,
            None,
        )

    def process_nonterminal_turn(
        self,
        message_log: list[dict[str, Any]],
        metadata: ImageToolsGymToolMetadata,
    ) -> tuple[
        dict[str, Any],
        float,
        bool,
        list[str] | None,
        ImageToolsGymToolMetadata | None,
        str | None,
        str | None,
    ]:
        assistant_text = extract_last_assistant_text(message_log)
        tool_calls = parse_image_tool_calls(assistant_text)
        malformed_tool_attempt = has_malformed_image_tool_markup(assistant_text)

        if malformed_tool_attempt:
            obs, rew, done, stops, meta, ans = self._invalid_observation("Invalid image tool call format.", metadata)
            return obs, rew, done, stops, meta, ans, None

        if not tool_calls:
            return (
                {"role": "environment", "content": ""},
                0.0,
                True,
                None,
                None,
                assistant_text,
                metadata.get("ground_truth", ""),
            )

        if metadata.get("force_final_next", False):
            obs, rew, done, stops, meta, ans = self._invalid_observation(
                "No more tool calls are allowed. A final answer was required.",
                metadata,
            )
            return obs, rew, done, stops, meta, ans, None

        tool_call_count = int(metadata.get("tool_call_count", 0))
        if tool_call_count >= self.max_tool_calls:
            obs, rew, done, stops, meta, ans = self._force_final_observation(metadata)
            return obs, rew, done, stops, meta, ans, None

        selected_calls = tool_calls[: min(self.max_tool_calls_per_turn, self.max_tool_calls - tool_call_count)]
        next_metadata = self._copy_metadata(metadata)
        image_paths = list(next_metadata.get("image_paths", []))
        if not image_paths:
            obs, rew, done, stops, meta, ans = self._invalid_observation(
                "Image tools are unavailable because no source images were provided.",
                metadata,
            )
            return obs, rew, done, stops, meta, ans, None

        response_content: list[dict[str, Any]] = [{"type": "text", "text": "<tool_response>\n"}]
        reward = 0.0
        force_final_next = False

        for tool_call in selected_calls:
            tool_name = str(tool_call.get("name") or "")
            if tool_name not in IMAGE_TOOL_NAMES:
                obs, rew, done, stops, meta, ans = self._invalid_observation(
                    f"Unknown tool name: {tool_name}", metadata
                )
                return obs, rew, done, stops, meta, ans, None

            arguments = tool_call.get("arguments") or {}
            if not isinstance(arguments, dict):
                obs, rew, done, stops, meta, ans = self._invalid_observation(
                    "Tool arguments must be an object.", metadata
                )
                return obs, rew, done, stops, meta, ans, None

            try:
                call_sig = json.dumps(
                    {"name": tool_name, "arguments": arguments},
                    sort_keys=True,
                    ensure_ascii=False,
                    default=str,
                )
                is_exact_duplicate = call_sig in next_metadata["seen_tool_sigs"]
                next_metadata["seen_tool_sigs"].append(call_sig)

                near_duplicate_iou = None
                if tool_name in {"image_zoom_in_tool", "image_crop_tool"}:
                    bbox = coerce_bbox(arguments.get("bbox_2d"))
                    img_idx = int(arguments.get("img_idx", 0))
                    if bbox is None:
                        raise ValueError("bbox_2d must be a list of four numbers")
                    img_key = f"{tool_name}:{img_idx}"
                    seen_for_image = next_metadata["seen_bboxes"].setdefault(img_key, [])
                    for previous_bbox in seen_for_image:
                        iou = bbox_iou(previous_bbox, bbox)
                        if iou >= self.duplicate_iou_threshold:
                            near_duplicate_iou = iou
                            break
                    seen_for_image.append(bbox)

                sample_work_dir = os.path.join(
                    self.crop_dir,
                    str(next_metadata.get("dataset", "unknown")),
                    str(uuid.uuid4()),
                )
                result = execute_image_tool(
                    tool_name,
                    arguments,
                    image_paths,
                    sample_work_dir,
                    crop_format=self.crop_format,
                    crop_jpeg_quality=self.crop_jpeg_quality,
                    crop_min_pixels=self.crop_min_pixels,
                    crop_max_pixels=self.crop_max_pixels,
                )
            except Exception as exc:
                obs, rew, done, stops, meta, ans = self._invalid_observation(f"{tool_name} failed: {exc}", metadata)
                return obs, rew, done, stops, meta, ans, None

            warning = None
            if is_exact_duplicate:
                warning = (
                    "You already requested this exact tool call. Use the evidence you "
                    "have and provide the final answer."
                )
                reward += self.duplicate_tool_call_penalty
                force_final_next = self.force_final_after_duplicate
            elif near_duplicate_iou is not None:
                warning = (
                    f"This region heavily overlaps a previous {tool_name} result "
                    f"(IoU={near_duplicate_iou:.2f}). Use the evidence you have and "
                    "provide the final answer."
                )
                reward += self.duplicate_tool_call_penalty
                force_final_next = self.force_final_after_duplicate
            else:
                reward += self.tool_success_reward

            new_img_idx = len(image_paths)
            tool_result = {
                "ok": True,
                **result["result"],
                "new_img_indices": [new_img_idx],
            }
            if warning is not None:
                tool_result["warning"] = warning
            response_content.append(
                {
                    "type": "text",
                    "text": json.dumps(tool_result, ensure_ascii=False) + "\n",
                }
            )
            response_content.append({"type": "image", "image": result["path"]})
            response_content.append({"type": "text", "text": "\n"})
            image_paths.append(result["path"])
            next_metadata["image_paths"] = image_paths
            next_metadata["crop_paths"].append(result["path"])
            next_metadata["tool_call_count"] = int(next_metadata.get("tool_call_count", 0)) + 1

        if len(tool_calls) > len(selected_calls):
            response_content.append(
                {
                    "type": "text",
                    "text": (
                        "Only the first "
                        f"{len(selected_calls)} tool call(s) were executed this turn. "
                        "Call tools one turn at a time.\n"
                    ),
                }
            )
        response_content.append({"type": "text", "text": "</tool_response>"})
        next_metadata["force_final_next"] = force_final_next
        return (
            {"role": "user", "content": response_content},
            reward,
            False,
            self.stop_strings,
            next_metadata,
            None,
            None,
        )
