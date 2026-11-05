# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tool reply's images leave the reply and keep their order and labels."""

import pytest
from pydantic import ValidationError

from nemo_gym.core.tool_output_images import TOOL_OUTPUT_IMAGES_KEY, ToolOutputImage, split_tool_output_images


PNG_URL = "data:image/png;base64,iVBORw0KGgo="
JPEG_URL = "data:image/jpeg;base64,/9j/4AAQ"


def test_split_moves_the_images_out_and_keeps_the_rest() -> None:
    reply = {
        "ok": True,
        "new_image_index": 2,
        TOOL_OUTPUT_IMAGES_KEY: [{"image_url": PNG_URL, "label": "Image 2"}, {"image_url": JPEG_URL}],
    }

    rest, images = split_tool_output_images(reply)

    assert rest == {"ok": True, "new_image_index": 2}
    assert TOOL_OUTPUT_IMAGES_KEY in reply, "the caller's reply must not change"
    assert [(image.label, image.media_type, image.data) for image in images] == [
        ("Image 2", "image/png", "iVBORw0KGgo="),
        (None, "image/jpeg", "/9j/4AAQ"),
    ]


@pytest.mark.parametrize("reply", [{"ok": True}, "plain text", ["a", "list"], None])
def test_a_reply_without_images_comes_back_unchanged(reply: object) -> None:
    assert split_tool_output_images(reply) == (reply, [])


@pytest.mark.parametrize(
    "images",
    [
        "not a list",
        [{"image_url": "https://example.com/a.png"}],
        [{"image_url": "data:text/plain;base64,aGk="}],
        [{"image_url": PNG_URL, "path": "/etc/passwd"}],
    ],
)
def test_a_malformed_image_list_is_a_fault_of_the_tool(images: object) -> None:
    with pytest.raises(ValidationError):
        split_tool_output_images({"ok": True, TOOL_OUTPUT_IMAGES_KEY: images})


def test_an_image_is_a_base64_data_url_only() -> None:
    assert ToolOutputImage(image_url=PNG_URL).media_type == "image/png"
    with pytest.raises(ValidationError):
        ToolOutputImage(image_url="data:image/png,raw-bytes")
