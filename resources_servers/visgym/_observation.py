# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import io
import logging
from typing import Any

from PIL import Image

from resources_servers.visgym._metadata import sanitize_metadata
from resources_servers.visgym.schemas import VisGymEnvStateEasyInputMessage


def image_to_data_url(image: Image.Image, fmt: str = "PNG", jpeg_quality: int = 90) -> str:
    """Encode a PIL image as an OpenAI-compatible base64 data URL."""

    normalized_fmt = fmt.upper()
    buf = io.BytesIO()
    if normalized_fmt == "JPEG":
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        mime = "jpeg"
    else:
        image.save(buf, format="PNG")
        mime = "png"

    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{mime};base64,{encoded}"


logger = logging.getLogger(__name__)


def _is_numpy_array(value: Any) -> bool:
    return value.__class__.__module__.startswith("numpy") and hasattr(value, "dtype")


def _array_to_image(value: Any) -> Image.Image:
    import numpy as np

    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Unsupported image array shape: {arr.shape}")
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating):
            max_value = float(np.nanmax(arr)) if arr.size else 1.0
            if max_value <= 1.0:
                arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def coerce_images(value: Any) -> list[Image.Image]:
    """Convert common VisGym observation/render image values to PIL images."""

    if value is None:
        return []
    if isinstance(value, Image.Image):
        return [value]
    if _is_numpy_array(value):
        try:
            return [_array_to_image(value)]
        except ValueError:
            # A state vector, a (H,W,2) mask, anything not image-shaped. This
            # is called from /step and /seed_session outside their try/except,
            # so raising here surfaces as an unhandled 500 that kills the whole
            # rollout batch -- and in seed_session the environment has already
            # been registered, so it leaks too. An environment that returns a
            # non-image observation is a normal thing to handle, not a crash:
            # the caller falls back to env.render() when configured.
            logger.debug("Observation array is not image-like; ignoring for image extraction")
            return []
    if isinstance(value, (list, tuple)):
        images: list[Image.Image] = []
        for item in value:
            images.extend(coerce_images(item))
        return images
    if isinstance(value, dict):
        for key in ("image", "rgb", "render", "frame", "obs", "observation"):
            if key in value:
                images = coerce_images(value[key])
                if images:
                    return images
    return []


def observation_to_user_message(
    *,
    image_value: Any,
    env_id: str,
    prefix_text: str | None = None,
    feedback_text: str | None = None,
    image_format: str = "PNG",
    image_jpeg_quality: int = 90,
    skip_images: bool = False,
) -> VisGymEnvStateEasyInputMessage:
    """Build the multimodal user message emitted by the VisGym server."""

    parts: list[dict[str, Any]] = []
    text_parts = [text for text in (prefix_text, feedback_text) if text]
    if text_parts:
        parts.append({"type": "input_text", "text": "\n\n".join(text_parts)})

    if not skip_images:
        for image in coerce_images(image_value):
            parts.append(
                {
                    "type": "input_image",
                    "image_url": image_to_data_url(
                        image,
                        fmt=image_format,
                        jpeg_quality=image_jpeg_quality,
                    ),
                    "detail": "auto",
                }
            )

    return VisGymEnvStateEasyInputMessage(role="user", content=parts, env_info=None)


def attach_env_info(
    obs_msg: VisGymEnvStateEasyInputMessage,
    info_dict: dict[str, Any],
) -> VisGymEnvStateEasyInputMessage:
    return obs_msg.model_copy(update={"env_info": sanitize_metadata(info_dict)})
