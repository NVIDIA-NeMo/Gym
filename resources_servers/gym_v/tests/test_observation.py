# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import base64
import io
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from resources_servers.gym_v._observation import (
    _attach_env_info,
    image_to_data_url,
    observation_to_user_message,
)


@dataclass
class _StubObservation:
    """Lightweight stand-in for `gym_v.Observation` used in helper-level tests."""

    text: str | None = None
    image: Image.Image | list[Image.Image] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _decode_data_url(data_url: str) -> Image.Image:
    prefix, encoded = data_url.split(",", 1)
    assert prefix == "data:image/png;base64"
    return Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_image_to_data_url_round_trips_png() -> None:
    image = Image.new("RGB", (4, 4), (12, 34, 56))
    data_url = image_to_data_url(image)

    decoded = _decode_data_url(data_url)

    assert decoded.size == (4, 4)
    assert decoded.getpixel((0, 0)) == (12, 34, 56)


def test_image_to_data_url_resizes_when_over_cap() -> None:
    image = Image.new("RGB", (1024, 512), (0, 0, 0))
    data_url = image_to_data_url(image, max_image_wh=256)

    decoded = _decode_data_url(data_url)

    assert max(decoded.size) == 256
    assert decoded.size == (256, 128)


def test_observation_to_user_message_includes_text_image_and_visible_metadata() -> None:
    obs = _StubObservation(
        text="board caption",
        image=Image.new("RGB", (4, 4), (0, 0, 0)),
        metadata={
            "state_text": "[[0,1],[1,0]]",
            "text_prompt": "Predict the next board.",
            "oracle_answer": "do not leak",
        },
    )

    message = observation_to_user_message(
        obs,
        env_id="Puzzles/Maze-QA-v0",
        prefix_text="Rules text",
    )

    content = message.content
    assert isinstance(content, list)
    assert content[0]["type"] == "input_text"
    assert "Rules text" in content[0]["text"]
    assert "board caption" in content[0]["text"]
    assert "state_text: [[0,1],[1,0]]" in content[0]["text"]
    assert "text_prompt" not in content[0]["text"]
    assert "oracle_answer" not in content[0]["text"]
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_observation_to_user_message_suppresses_metadata_for_unlisted_env() -> None:
    obs = _StubObservation(
        text=None,
        image=Image.new("RGB", (2, 2), (0, 0, 0)),
        metadata={"state_text": "should_not_leak"},
    )

    message = observation_to_user_message(obs, env_id="Games/FrozenLake-v0")

    text_parts = [c for c in message.content if c["type"] == "input_text"]
    assert not any("should_not_leak" in c["text"] for c in text_parts)


def test_observation_to_user_message_skip_images_omits_image_parts() -> None:
    obs = _StubObservation(
        text="caption",
        image=Image.new("RGB", (2, 2), (0, 0, 0)),
        metadata={},
    )

    message = observation_to_user_message(
        obs,
        env_id="Games/FrozenLake-v0",
        skip_images=True,
    )

    assert all(c["type"] != "input_image" for c in message.content)


def test_attach_env_info_populates_side_channel() -> None:
    obs = _StubObservation(text="hi", image=None, metadata={})
    message = observation_to_user_message(obs, env_id="Games/FrozenLake-v0")

    updated = _attach_env_info(message, {"step": 3, "score": 0.5})

    assert updated.env_info == {"step": 3, "score": 0.5}
