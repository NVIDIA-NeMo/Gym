# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image validation before shared session staging."""

import io

from PIL import Image

from nemo_gym.core import MediaFile


def prepare_image(image: MediaFile, *, max_source_pixels: int) -> MediaFile:
    """Check benchmark pixel limits without changing decoded media bytes."""
    try:
        with Image.open(io.BytesIO(image.data)) as source:
            if source.width * source.height > max_source_pixels:
                raise ValueError("Visual input exceeds max_source_pixels")
            return image
    except OSError as error:
        raise ValueError("Invalid or truncated input image") from error
