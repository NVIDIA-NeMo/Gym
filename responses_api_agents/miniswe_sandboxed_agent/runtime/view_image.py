#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Return a sandbox image to the experimental mini-SWE observation formatter."""

import argparse
import base64
import hashlib
import json
from pathlib import Path


MAX_IMAGE_BYTES = 20 * 1024 * 1024


def image_result(path: Path) -> dict:
    """Read one supported image without placing its bytes in a text observation."""
    with path.open("rb") as image:
        data = image.read(MAX_IMAGE_BYTES + 1)
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds the 20 MiB limit; create a smaller image or crop first")
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        mime = "image/gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        raise ValueError("Unsupported image format; use PNG, JPEG, GIF, or WebP")
    return {
        "_nemo_gym_view_image": 1,
        "path": str(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "content": [{"type": "image", "mimeType": mime, "data": base64.b64encode(data).decode("ascii")}],
    }


def main() -> None:
    """Emit the image envelope for a path supplied through native bash."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        result = image_result(args.path)
    except (OSError, ValueError) as error:
        parser.exit(1, f"view_image: {error}\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
