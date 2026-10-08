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
from unittest.mock import MagicMock

import pytest
from app import (
    GuiCoordinateResourcesServer,
    GuiCoordinateResourcesServerConfig,
    _compute_reward,
    _parse_gt,
    _parse_prediction,
)

from nemo_gym.server_utils import ServerClient


class TestApp:
    def test_sanity(self) -> None:
        config = GuiCoordinateResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
        GuiCoordinateResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    @pytest.mark.parametrize(
        ("expected_answer", "expected"),
        [
            ("0.5,0.25", (0.5, 0.25)),
            ("not,coords", None),
            ("0.1,0.2,0.3", None),
        ],
    )
    def test_parse_gt(self, expected_answer: str, expected) -> None:
        assert _parse_gt(expected_answer) == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # Coordinates are emitted in thousandths and normalized to [0, 1].
            ("<point>(500, 250)</point>", (0.5, 0.25)),
            ("<point>[(500,250)]</point>", (0.5, 0.25)),
            ("I will click <point>( 500 , 250 )</point> now.", (0.5, 0.25)),
            # Missing the parentheses the pattern requires.
            ("<point>500 250</point>", None),
            ("no point here", None),
        ],
    )
    def test_parse_prediction(self, text: str, expected) -> None:
        assert _parse_prediction(text) == expected

    @pytest.mark.parametrize(
        ("pred", "expected_reward"),
        [
            ((0.5, 0.25), 1.0),  # exact hit
            ((0.55, 0.25), 0.25),  # half of max_dist away -> (1 - 0.5)^2
            ((0.9, 0.9), 0.0),  # beyond max_dist
        ],
    )
    def test_compute_reward(self, pred, expected_reward: float) -> None:
        reward, dist = _compute_reward((0.5, 0.25), pred, 0.1)
        assert reward == pytest.approx(expected_reward)
        assert dist >= 0.0
