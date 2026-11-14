# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read verified Agent-X image bytes from source, embedded rows, or portable bundles."""

import base64
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image


def read_image(value: str, *, media_root: Path | None = None) -> tuple[bytes, str]:
    """Validate the image and return its bytes and MIME type without requiring host paths at inference."""
    declared_mime = None
    if value.startswith("data:"):
        header, separator, encoded = value.partition(",")
        if not separator or not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("Agent-X embedded image must be a base64 image data URL")
        declared_mime = header[5:-7]
        data = base64.b64decode(encoded, validate=True)
    else:
        relative = Path(value)
        path = relative
        if media_root is not None:
            path = (media_root / relative).resolve(strict=True)
            if relative.is_absolute() or not path.is_relative_to(media_root):
                raise ValueError("Agent-X bundle image must be relative to media_root")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != path.stem:
            raise ValueError(f"Agent-X image checksum mismatch: {relative}")
    with Image.open(BytesIO(data)) as image:
        mime = Image.MIME[image.format]
        image.verify()
    if declared_mime is not None and declared_mime != mime:
        raise ValueError("Agent-X image MIME type differs from its decoded format")
    return data, mime


def image_url(data: bytes, mime: str) -> str:
    """Embed verified bytes for policy endpoints that cannot read the preparation host."""
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
