# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared image-capacity policy for visual training and evaluation requests."""

from copy import deepcopy
from typing import Any


IMAGE_LIMIT_NOTICE = "Image limit reached. This additional image was omitted. You cannot receive additional images."


def limit_request_images(body: dict[str, Any], *, max_images: int | None) -> dict[str, Any]:
    """Omit excess chat images without mutating the caller's stored history.

    Training invokes this before lineage resolution and capture so the notice,
    rather than the omitted image, becomes part of the actual captured input.
    """
    if max_images is None:
        return body
    if max_images <= 0:
        raise ValueError("max_images must be positive or None")
    result = deepcopy(body)
    count = 0
    for message in result.get("messages", []):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for index, part in enumerate(content):
            if part.get("type") == "image_url":
                count += 1
                if count > max_images:
                    content[index] = {"type": "text", "text": IMAGE_LIMIT_NOTICE}
    return result
