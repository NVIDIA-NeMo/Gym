# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Identify an already-read attachment without decoding pixels or rereading files."""

from __future__ import annotations

import base64
import binascii
import io
import sys

from PIL import Image, UnidentifiedImageError


def image_identifiable(encoded: str) -> bool:
    """Match the model proxy's definite-invalid check, including decoder tolerance.

    The caller supplies the original attachment, regardless of its declared MIME.
    Pixel/decompression limits belong to the engine. Unexpected decoder failures
    propagate instead of being reported as an invalid user image.
    """
    try:
        payload = base64.b64decode(encoded)
    except (binascii.Error, ValueError):
        return False
    try:
        with Image.open(io.BytesIO(payload)):
            return True
    except UnidentifiedImageError:
        return False
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        return True


def main() -> int:
    """Read base64 on stdin; exit 0=identifiable, 2=invalid, 1=validator failure.

    No payload, filename, diagnostic text, or credential is written to stdout or
    stderr. Image size and runtime remain governed by the existing read tool and
    agent/container lifetime; this helper adds no file, pixel, or time limit.
    """
    try:
        return 0 if image_identifiable(sys.stdin.read()) else 2
    except Exception:
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
