# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Copied from https://github.com/katerynaCh/Gym at
# 168e95ee80780737273dda011d31d1d5c58f35ce, Apache-2.0:
# responses_api_agents/opencode_multimodal_agent/media.py (stage_inputs).
# Only the public Gym type import is updated; preprocessing is unchanged.
import base64
import io
from pathlib import Path

from PIL import Image

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming


def stage_inputs(
    body: NeMoGymResponseCreateParamsNonStreaming,
    directory: Path,
    *,
    max_source_pixels: int,
    max_images: int | None,
    preserve_format: bool = False,
) -> tuple[str, str | None, list[Path]]:
    """Stage initial images without exposing verifier metadata to the runtime.

    This entrypoint starts fresh sessions. Explicitly reject replay/history rather
    than silently treating a later user message as a new independent task.
    """
    if isinstance(body.input, str):
        raise ValueError("A multimodal task requires input_image parts")
    texts, systems, paths = [], [], []
    for item in body.model_dump()["input"]:
        if item.get("role") not in {"user", "system", "developer"}:
            raise ValueError("OpenCodeMultimodalAgent accepts initial prompts, not replayed trajectories")
        content = item["content"]
        parts = [{"type": "input_text", "text": content}] if isinstance(content, str) else content
        for part in parts:
            if part["type"] == "input_text":
                (texts if item["role"] == "user" else systems).append(part["text"])
                continue
            if part["type"] != "input_image" or item["role"] != "user":
                raise ValueError("Only user images and text prompts are supported")
            if max_images is not None and len(paths) >= max_images:
                raise ValueError("Initial image count exceeds max_images")
            source = part.get("image_url")
            if not isinstance(source, str):
                raise ValueError("Images require image_url, not file_id")
            if source.startswith("data:"):
                header, payload = source.split(",", 1)
                if not header.startswith("data:image/") or not header.endswith(";base64"):
                    raise ValueError("Expected a base64 image data URL")
                image_source = io.BytesIO(base64.b64decode(payload, validate=True))
            else:
                if "://" in source and not source.startswith("file://"):
                    raise ValueError("Stage remote image URLs locally before rollout")
                image_source = Path(source.removeprefix("file://"))
            with Image.open(image_source) as image:
                if image.width * image.height > max_source_pixels:
                    raise ValueError("Visual input exceeds max_source_pixels")
                suffix = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp", "GIF": ".gif"}.get(image.format)
                if preserve_format and suffix:
                    path = directory / f"input-{len(paths)}{suffix}"
                    raw = (
                        image_source.getvalue() if isinstance(image_source, io.BytesIO) else image_source.read_bytes()
                    )
                    path.write_bytes(raw)
                else:
                    path = directory / f"input-{len(paths)}.png"
                    image.convert("RGB").save(path)
            paths.append(path)
    if not paths:
        raise ValueError("A multimodal task requires an input image")
    return "\n\n".join(texts), "\n\n".join(systems) or None, paths
