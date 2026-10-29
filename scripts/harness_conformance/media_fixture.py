# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small deterministic PNGs for image delivery, with no model/decoder dependency."""

import base64
import struct
import zlib


def png(red: int, green: int, blue: int) -> bytes:
    """Build an 8x8 RGB PNG; no resizing is needed by image-reading harness tools."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack("!I", len(data)) + kind + data + struct.pack("!I", zlib.crc32(kind + data))

    pixels = (b"\x00" + bytes((red, green, blue)) * 8) * 8
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack("!IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


PROMPT_BYTES = (png(255, 0, 0), png(0, 255, 0))
TOOL_BYTES = png(0, 0, 255)


def image_part(data: bytes) -> dict:
    """Encode a fixture in the native Responses task format."""
    return {
        "type": "input_image",
        "image_url": "data:image/png;base64," + base64.b64encode(data).decode(),
        "detail": "auto",
    }


def prompt_content() -> list[dict]:
    """Keep distinguishable text and image parts in the task submitted to Gym."""
    return [
        {"type": "input_text", "text": "First attachment:"},
        image_part(PROMPT_BYTES[0]),
        {"type": "input_text", "text": "Second attachment:"},
        image_part(PROMPT_BYTES[1]),
        {"type": "input_text", "text": "Read the image file requested by the model, then report CONFORMANCE_DONE."},
    ]
