# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from unittest.mock import MagicMock

from app import (
    XlamFcResourcesServer,
    XlamFcResourcesServerConfig,
    XlamFcVerifyRequest,
)
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import ResourcesSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponse
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.testing.session_conformance import check_resources_session_contract


def _server() -> XlamFcResourcesServer:
    return XlamFcResourcesServer(
        config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name="xlam_fc"),
        server_client=MagicMock(spec=ServerClient),
    )


def _response(output: list[dict]) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="resp_test",
        created_at=0.0,
        model="dummy",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _call(call_id: str, name: str, arguments: dict) -> dict:
    return {"call_id": call_id, "name": name, "arguments": json.dumps(arguments), "type": "function_call"}


def _tool_output(call_id: str, output: dict) -> dict:
    return {"call_id": call_id, "output": json.dumps(output), "type": "function_call_output"}


def _message(text: str) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


LONDON = {"name": "timezone_for_location", "arguments": {"location": "London", "area": "Europe"}}


class TestApp:
    def test_sanity(self) -> None:
        config = XlamFcResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        XlamFcResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    async def test_verify_exact_match_single_call(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "San Francisco"}),
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                ],
            },
            response=response,
            expected_answers=[{"name": "get_weather", "arguments": {"city": "San Francisco"}}],
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.num_expected == 1
        assert result.num_predicted == 1
        assert result.num_correct == 1

    async def test_verify_exact_match_multiple_calls(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "live_giveaways_by_type",
                    "arguments": json.dumps({"type": "beta"}),
                    "type": "function_call",
                },
                {
                    "call_id": "call_2",
                    "name": "live_giveaways_by_type",
                    "arguments": json.dumps({"type": "game"}),
                    "type": "function_call",
                },
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "Where can I find live giveaways?"},
                ],
            },
            response=response,
            expected_answers=[
                {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
                {"name": "live_giveaways_by_type", "arguments": {"type": "game"}},
            ],
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.num_expected == 2
        assert result.num_predicted == 2
        assert result.num_correct == 2

    async def test_verify_partial_match(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "live_giveaways_by_type",
                    "arguments": json.dumps({"type": "beta"}),
                    "type": "function_call",
                },
                {
                    "call_id": "call_2",
                    "name": "live_giveaways_by_type",
                    "arguments": json.dumps({"type": "loot"}),
                    "type": "function_call",
                },
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "Where can I find live giveaways?"},
                ],
            },
            response=response,
            expected_answers=[
                {"name": "live_giveaways_by_type", "arguments": {"type": "beta"}},
                {"name": "live_giveaways_by_type", "arguments": {"type": "game"}},
            ],
        )

        result = await server.verify(verify_request)
        assert result.reward == 0.0
        assert result.num_expected == 2
        assert result.num_predicted == 2
        assert result.num_correct == 1

    async def test_verify_no_match(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "wrong_function",
                    "arguments": json.dumps({"city": "New York"}),
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                ],
            },
            response=response,
            expected_answers=[{"name": "get_weather", "arguments": {"city": "San Francisco"}}],
        )

        result = await server.verify(verify_request)
        assert result.reward == 0.0
        assert result.num_expected == 1
        assert result.num_predicted == 1
        assert result.num_correct == 0

    async def test_verify_wrong_arguments(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "New York"}),
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                ],
            },
            response=response,
            expected_answers=[{"name": "get_weather", "arguments": {"city": "San Francisco"}}],
        )

        result = await server.verify(verify_request)
        assert result.reward == 0.0
        assert result.num_expected == 1
        assert result.num_predicted == 1
        assert result.num_correct == 0

    async def test_verify_no_predictions(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_1",
                    "content": [
                        {
                            "annotations": [],
                            "text": "I don't know how to help with that.",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                ],
            },
            response=response,
            expected_answers=[{"name": "get_weather", "arguments": {"city": "San Francisco"}}],
        )

        result = await server.verify(verify_request)
        assert result.reward == 0.0
        assert result.num_expected == 1
        assert result.num_predicted == 0
        assert result.num_correct == 0

    async def test_verify_extra_arguments(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "San Francisco", "units": "celsius"}),
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "What's the weather in San Francisco?"},
                ],
            },
            response=response,
            expected_answers=[{"name": "get_weather", "arguments": {"city": "San Francisco"}}],
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.num_expected == 1
        assert result.num_predicted == 1
        assert result.num_correct == 1

    async def test_verify_no_expected_no_predicted(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "id": "msg_1",
                    "content": [
                        {
                            "annotations": [],
                            "text": "Here's some information.",
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "Tell me something."},
                ],
            },
            response=response,
            expected_answers=[],
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.num_expected == 0
        assert result.num_predicted == 0
        assert result.num_correct == 0

    async def test_verify_multiple_arguments(self) -> None:
        response = NeMoGymResponse(
            id="resp_test",
            created_at=0.0,
            model="dummy",
            object="response",
            output=[
                {
                    "call_id": "call_1",
                    "name": "t3ma",
                    "arguments": json.dumps({"symbol": "ETH/BTC", "interval": "1h", "time_period": 14}),
                    "type": "function_call",
                }
            ],
            parallel_tool_calls=True,
            tool_choice="auto",
            tools=[],
        )

        server = XlamFcResourcesServer(
            config=XlamFcResourcesServerConfig(host="0.0.0.0", port=8080, entrypoint="", name=""),
            server_client=MagicMock(spec=ServerClient),
        )

        verify_request = XlamFcVerifyRequest(
            responses_create_params={
                "input": [
                    {"role": "user", "content": "Get T3MA for ETH/BTC"},
                ],
            },
            response=response,
            expected_answers=[
                {"name": "t3ma", "arguments": {"symbol": "ETH/BTC", "interval": "1h", "time_period": 14}}
            ],
        )

        result = await server.verify(verify_request)
        assert result.reward == 1.0
        assert result.num_expected == 1
        assert result.num_predicted == 1
        assert result.num_correct == 1


class TestToolRoutes:
    """The model must never see a 404 for a tool the dataset asked it to call (NVBug 6900453)."""

    def test_any_tool_call_is_acknowledged(self) -> None:
        client = TestClient(_server().setup_webserver())
        response = client.post("/timezone_for_location", json={"location": "London", "area": "Europe"})
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["status"] == "ok"
        assert "detail" not in body

    def test_catch_all_tool_route_keeps_the_session_contract(self) -> None:
        check_resources_session_contract(
            _server().setup_webserver(),
            ResourcesSeedSessionRequest(
                resources_session_id="contract-session",
                episode_id=EpisodeId(rollout_id="rollout", attempt=0),
                task_id=TaskId(taskset="example", task_id="0"),
                task_data={},
            ),
            keeps_state=False,
        )

    def test_verify_grades_a_call_the_model_answered_after_the_acknowledgement(self) -> None:
        # The whole flow through the app: the tool route answers, then /verify (registered before the catch-all)
        # still reaches its handler and grades the call.
        client = TestClient(_server().setup_webserver())
        acknowledgement = client.post(f"/{LONDON['name']}", json=LONDON["arguments"]).json()
        request = XlamFcVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "What time zone is London in?"}]},
            response=_response(
                [
                    _call("call_1", LONDON["name"], LONDON["arguments"]),
                    _tool_output("call_1", acknowledgement),
                    _message("London is in the Europe/London time zone."),
                ]
            ),
            expected_answers=[LONDON],
        )
        result = client.post("/verify", json=request.model_dump(mode="json")).json()
        assert (result["reward"], result["num_expected"], result["num_predicted"], result["num_correct"]) == (
            1.0,
            1,
            1,
            1,
        )

    def test_verify_counts_a_call_repeated_after_the_acknowledgement(self) -> None:
        # README rule: every function_call in the response is graded, so a call repeated after the
        # acknowledgement costs the reward.
        client = TestClient(_server().setup_webserver())
        acknowledgement = client.post(f"/{LONDON['name']}", json=LONDON["arguments"]).json()
        request = XlamFcVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "What time zone is London in?"}]},
            response=_response(
                [
                    _call("call_1", LONDON["name"], LONDON["arguments"]),
                    _tool_output("call_1", acknowledgement),
                    _call("call_2", LONDON["name"], LONDON["arguments"]),
                ]
            ),
            expected_answers=[LONDON],
        )
        result = client.post("/verify", json=request.model_dump(mode="json")).json()
        assert result["reward"] == 0.0
        assert (result["num_expected"], result["num_predicted"]) == (1, 2)
