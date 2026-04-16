# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.circle_count.app import (
    CircleCountConfig,
    CircleCountResourcesServer,
    CircleCountVerifyRequest,
)


CIRCLES = [
    {"x": 100, "y": 100, "radius": 45, "color": "red"},
    {"x": 250, "y": 200, "radius": 45, "color": "red"},
    {"x": 400, "y": 300, "radius": 45, "color": "blue"},
]

MINIMAL_RESPONSES_CREATE_PARAMS = {
    "input": [{"role": "user", "content": "test"}],
}


def _make_server() -> CircleCountResourcesServer:
    config = CircleCountConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return CircleCountResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_verify_request(text: str, target_color: str = "red") -> CircleCountVerifyRequest:
    return CircleCountVerifyRequest(
        responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
        response=_make_response(text),
        circles=CIRCLES,
        target_color=target_color,
    )


class TestCircleCountServer:
    def test_sanity(self) -> None:
        assert _make_server() is not None

    async def test_correct_count(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request(r"There are 2 red circles. \boxed{2}"))
        assert result.reward == 1.0
        assert result.correct is True
        assert result.predicted_count == 2
        assert result.expected_count == 2

    async def test_wrong_count_too_low(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request(r"\boxed{1}"))
        assert result.reward == 0.0
        assert result.correct is False
        assert result.predicted_count == 1
        assert result.expected_count == 2

    async def test_wrong_count_too_high(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request(r"\boxed{5}"))
        assert result.reward == 0.0
        assert result.correct is False

    async def test_correct_count_blue(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request(r"\boxed{1}", target_color="blue"))
        assert result.reward == 1.0
        assert result.expected_count == 1

    async def test_zero_count_for_absent_color(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request(r"\boxed{0}", target_color="green"))
        assert result.reward == 1.0
        assert result.expected_count == 0

    async def test_no_boxed_returns_zero(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request("I think there are two red circles."))
        assert result.reward == 0.0
        assert result.predicted_count is None

    async def test_first_boxed_used(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request(r"First I thought \boxed{2} but actually \boxed{1}"))
        assert result.predicted_count == 2
        assert result.reward == 1.0

    async def test_boxed_zero(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request(r"\boxed{0}", target_color="green"))
        assert result.predicted_count == 0
        assert result.reward == 1.0

    async def test_no_match_gives_none(self) -> None:
        server = _make_server()
        result = await server.verify(_make_verify_request("seven", target_color="red"))
        assert result.predicted_count is None
        assert result.reward == 0.0
