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
import json
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI

from nemo_gym.mcp_test_utils import assert_transport_parity, mcp_call, mcp_handshake, mcp_list_tools, seed_token
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient
from resources_servers.circle_click.app import (
    CircleClickConfig,
    CircleClickResourcesServer,
    CircleClickVerifyRequest,
    ClickRequest,
    ClickResponse,
)


CIRCLES = [
    {"x": 100, "y": 100, "radius": 45, "color": "red"},
    {"x": 250, "y": 200, "radius": 45, "color": "blue"},
    {"x": 350, "y": 300, "radius": 45, "color": "green"},
]

MINIMAL_RESPONSES_CREATE_PARAMS = {
    "input": [{"role": "user", "content": "test"}],
    "parallel_tool_calls": True,
}


def _make_server() -> CircleClickResourcesServer:
    config = CircleClickConfig(host="0.0.0.0", port=8080, entrypoint="", name="")
    return CircleClickResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def _make_response(click_x: int, click_y: int, tool_name: str = "click") -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=[
            {
                "call_id": "call_1",
                "name": tool_name,
                "arguments": json.dumps({"x": click_x, "y": click_y}),
                "type": "function_call",
            }
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_verify_request(
    click_x: int,
    click_y: int,
    target_color: str = "red",
    tool_name: str = "click",
) -> CircleClickVerifyRequest:
    return CircleClickVerifyRequest(
        responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
        response=_make_response(click_x, click_y, tool_name),
        circles=CIRCLES,
        target_color=target_color,
    )


class TestCircleClickServer:
    def test_sanity(self) -> None:
        server = _make_server()
        assert server is not None

    async def test_hit_center_of_target_circle(self) -> None:
        server = _make_server()
        body = _make_verify_request(click_x=100, click_y=100, target_color="red")
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.hit is True
        assert result.clicked_x == 100
        assert result.clicked_y == 100

    async def test_click_wrong_circle(self) -> None:
        server = _make_server()
        body = _make_verify_request(click_x=250, click_y=200, target_color="red")
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.hit is False

    async def test_click_at_edge_of_target_circle(self) -> None:
        server = _make_server()
        body = _make_verify_request(click_x=100 + 45, click_y=100, target_color="red")
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.hit is True

    async def test_click_just_outside_target_circle(self) -> None:
        server = _make_server()
        body = _make_verify_request(click_x=100 + 46, click_y=100, target_color="red")
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.hit is False

    async def test_no_tool_call_returns_zero(self) -> None:
        server = _make_server()
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        body = CircleClickVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=response,
            circles=CIRCLES,
            target_color="red",
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.clicked_x is None
        assert result.clicked_y is None

    async def test_wrong_tool_name_ignored(self) -> None:
        server = _make_server()
        body = _make_verify_request(click_x=100, click_y=100, target_color="red", tool_name="tap")
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.clicked_x is None

    async def test_malformed_arguments_returns_zero(self) -> None:
        server = _make_server()
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "click",
                    "arguments": "not valid json",
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )
        body = CircleClickVerifyRequest(
            responses_create_params=MINIMAL_RESPONSES_CREATE_PARAMS,
            response=response,
            circles=CIRCLES,
            target_color="red",
        )
        result = await server.verify(body)
        assert result.reward == 0.0
        assert result.clicked_x is None

    async def test_target_color_blue(self) -> None:
        server = _make_server()
        body = _make_verify_request(click_x=250, click_y=200, target_color="blue")
        result = await server.verify(body)
        assert result.reward == 1.0
        assert result.hit is True

    async def test_point_in_circle_static(self) -> None:
        circle = {"x": 200, "y": 200, "radius": 50}
        assert CircleClickResourcesServer._point_in_circle(200, 200, circle) is True
        assert CircleClickResourcesServer._point_in_circle(250, 200, circle) is True
        assert CircleClickResourcesServer._point_in_circle(251, 200, circle) is False
        assert CircleClickResourcesServer._point_in_circle(0, 0, circle) is False


def _reference_app() -> FastAPI:
    """The pre-migration hand-written /click route: the byte-parity oracle."""
    app = FastAPI()

    async def click(body: ClickRequest) -> ClickResponse:
        return ClickResponse(x=body.x, y=body.y)

    app.post("/click")(click)
    return app


class TestClickHTTPReplay:
    """Wire-format preservation: /click responses must match the hand-written route byte-for-byte."""

    REPLAY_PAYLOADS = [
        {"x": 100, "y": 200},  # nominal integer click
        {"x": [1, 2], "y": "abc"},  # x/y are Any: echoed verbatim, no coercion
        {"x": 100},  # error path: missing required field -> 422
        {"x": None, "y": None},  # Any accepts null
    ]

    def test_click_bytes_match_handwritten_route(self) -> None:
        pytest.importorskip("mcp")
        from fastapi.testclient import TestClient

        server = _make_server()
        with (
            TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as migrated,
            TestClient(_reference_app(), base_url="http://127.0.0.1:8000") as reference,
        ):
            for payload in self.REPLAY_PAYLOADS:
                got = migrated.post("/click", json=payload)
                want = reference.post("/click", json=payload)
                assert got.status_code == want.status_code, payload
                assert got.content == want.content, payload
                assert got.headers["content-type"] == want.headers["content-type"], payload

    def test_click_non_json_body_bytes_match(self) -> None:
        pytest.importorskip("mcp")
        from fastapi.testclient import TestClient

        server = _make_server()
        with (
            TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as migrated,
            TestClient(_reference_app(), base_url="http://127.0.0.1:8000") as reference,
        ):
            got = migrated.post("/click", content=b"not json", headers={"Content-Type": "application/json"})
            want = reference.post("/click", content=b"not json", headers={"Content-Type": "application/json"})
            assert got.status_code == want.status_code == 422
            assert got.content == want.content


class TestClickMCP:
    """MCP round-trip: tools/list advertises click, tools/call echoes the coordinates."""

    def test_tools_list_and_call_round_trip(self) -> None:
        pytest.importorskip("mcp")
        from fastapi.testclient import TestClient

        server = _make_server()
        with TestClient(server.setup_webserver(), base_url="http://127.0.0.1:8000") as client:
            token = seed_token(client)

            mcp_handshake(client, token=token)

            tools = mcp_list_tools(client, token=token)
            assert set(tools) == {"click"}
            assert set(tools["click"]["inputSchema"]["properties"]) == {"x", "y"}
            assert set(tools["click"]["inputSchema"]["required"]) == {"x", "y"}

            result = mcp_call(client, "click", {"x": 10, "y": 20}, token=token)
            assert result.get("isError") is not True
            assert result["structuredContent"] == {"x": 10, "y": 20}


class TestTransportParity:
    def test_mcp_names_match_http_tool_routes(self) -> None:
        pytest.importorskip("mcp")
        from fastapi.testclient import TestClient

        server = _make_server()
        app = server.setup_webserver()
        with TestClient(app, base_url="http://127.0.0.1:8000") as client:
            assert_transport_parity(app, client, {"click"})
