# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode the inline images that OpenCode can attach to a user turn."""

import base64
import binascii
from collections.abc import Mapping
from dataclasses import dataclass


_IMAGE_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/gif": "gif", "image/webp": "webp"}
# OpenCode 1.17.11 forwards images within these attachment limits without resizing or re-encoding.
IMAGE_LIMITS = {"max_width": 2**31 - 1, "max_height": 2**31 - 1, "max_base64_bytes": 2**31 - 1}


@dataclass(frozen=True)
class ImageAttachment:
    """Decoded bytes and a validated extension; the session chooses the destination."""

    content: bytes
    extension: str


def decode_image_part(part: Mapping[str, object], *, index: int) -> ImageAttachment:
    """Decode a base64 PNG, JPEG, GIF or WebP data URI, without fetching remote URLs."""
    image_url = part.get("image_url")
    if not isinstance(image_url, str) or not image_url:
        raise ValueError(f"input_image part {index} requires image_url; file_id inputs are not supported")
    if not image_url.startswith("data:"):
        raise ValueError(f"input_image part {index} requires an inline data URI; remote URLs are not supported")
    header, separator, payload = image_url[5:].partition(",")
    media_type, *parameters = header.split(";")
    if not separator or "base64" not in parameters:
        raise ValueError(f"input_image part {index} requires a base64 data URI")
    media_type = media_type.strip().lower()
    extension = _IMAGE_EXTENSIONS.get(media_type)
    if extension is None:
        raise ValueError(f"input_image part {index} requires a PNG, JPEG, GIF or WebP image")
    try:
        content = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"input_image part {index} carries invalid base64") from error
    signatures = {
        "image/png": content.startswith(b"\x89PNG\r\n\x1a\n"),
        "image/jpeg": content.startswith(b"\xff\xd8\xff"),
        "image/gif": content[:6] in (b"GIF87a", b"GIF89a"),
        "image/webp": content[:4] == b"RIFF" and content[8:12] == b"WEBP",
    }
    if not signatures[media_type]:
        raise ValueError(f"input_image part {index} bytes do not match its declared image media type")
    return ImageAttachment(content=content, extension=extension)
