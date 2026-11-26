# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64

import numpy as np
from PIL import Image

from resources_servers.visgym._observation import (
    coerce_images,
    image_to_data_url,
    observation_to_user_message,
)


def test_image_to_data_url_round_trips_png() -> None:
    image = Image.new("RGB", (2, 2), (12, 34, 56))

    url = image_to_data_url(image)

    assert url.startswith("data:image/png;base64,")
    base64.b64decode(url.split(",", 1)[1])


def test_coerce_images_accepts_numpy_rgb_array() -> None:
    arr = np.zeros((3, 4, 3), dtype=np.uint8)
    arr[:, :, 0] = 255

    images = coerce_images(arr)

    assert len(images) == 1
    assert images[0].size == (4, 3)


def test_observation_to_user_message_contains_text_and_image() -> None:
    msg = observation_to_user_message(
        image_value=np.zeros((3, 4, 3), dtype=np.uint8),
        env_id="maze_2d/easy",
        prefix_text="Navigate the maze.",
        feedback_text="Action executed successfully.",
    )

    assert msg.role == "user"
    assert msg.content[0]["type"] == "input_text"
    assert "Navigate the maze." in msg.content[0]["text"]
    assert msg.content[1]["type"] == "input_image"
    assert msg.content[1]["image_url"].startswith("data:image/png;base64,")


def test_non_image_observation_does_not_raise() -> None:
    """A state-vector observation must not escape as an unhandled 500.

    coerce_images is called from /step and /seed_session outside their
    try/except blocks, so a ValueError here fails the whole rollout batch --
    and in seed_session the environment is already registered, so it leaks.
    """
    import numpy as np

    assert coerce_images(np.zeros((25,), dtype=np.float32)) == []
    assert coerce_images(np.zeros((8, 8, 2), dtype=np.uint8)) == []
    assert coerce_images({"state": np.zeros((4,), dtype=np.float32)}) == []
    # A genuine frame still converts.
    assert len(coerce_images(np.zeros((4, 4, 3), dtype=np.uint8))) == 1
