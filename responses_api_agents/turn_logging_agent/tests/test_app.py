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
import json
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent.app import ModelServerRef, ResourcesServerRef
from responses_api_agents.turn_logging_agent.app import (
    TurnLoggingAgent,
    TurnLoggingAgentConfig,
)


def make_config() -> TurnLoggingAgentConfig:
    return TurnLoggingAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="turn_logging_agent",
        model_server=ModelServerRef(type="responses_api_models", name="model server"),
        resources_server=ResourcesServerRef(type="resources_servers", name="resources server"),
    )


def usage(input_tokens: int, output_tokens: int, cached: int = 0, reasoning: int = 0) -> dict:
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "input_tokens_details": {"cached_tokens": cached},
        "output_tokens_details": {"reasoning_tokens": reasoning},
    }


def model_response(output: list, usage_dict: dict, response_id: str = "resp_1") -> dict:
    return {
        "id": response_id,
        "created_at": 0.0,
        "model": "dummy_model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": usage_dict,
    }


FN_CALL = {
    "type": "function_call",
    "id": "fc_1",
    "call_id": "call_1",
    "name": "get_weather",
    "arguments": json.dumps({"city": "SF"}),
    "status": "completed",
}
FINAL_MESSAGE = {
    "type": "message",
    "id": "msg_1",
    "role": "assistant",
    "status": "completed",
    "content": [{"type": "output_text", "text": "It is cold.", "annotations": []}],
}


def http_mock(payload) -> MagicMock:
    mock = MagicMock()
    mock.read = AsyncMock(return_value=json.dumps(payload))
    mock.content.read = AsyncMock(return_value=json.dumps(payload).encode())
    mock.cookies = {}
    return mock


class TestTurnLogging:
    def test_two_turn_loop_captures_per_turn_telemetry(self) -> None:
        server = TurnLoggingAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        client = TestClient(server.setup_webserver())

        turn1 = model_response([FN_CALL], usage(1000, 20, cached=0), "resp_turn1")
        tool_result = {"content": [{"type": "text", "text": "cold"}], "isError": False}
        turn2 = model_response([FINAL_MESSAGE], usage(1200, 30, cached=950, reasoning=5), "resp_turn2")
        server.server_client.post = AsyncMock(side_effect=[http_mock(turn1), http_mock(tool_result), http_mock(turn2)])

        result = client.post(
            "/v1/responses",
            json={
                "input": [{"role": "user", "content": "weather in SF?"}],
                "metadata": {"_turn_log_id": "trial-42"},
            },
        )
        assert result.status_code == 200, result.text
        body = result.json()

        # Aggregate usage sums across turns WITHOUT zeroing cached/reasoning details.
        assert body["usage"]["input_tokens"] == 2200
        assert body["usage"]["output_tokens"] == 50
        assert body["usage"]["input_tokens_details"]["cached_tokens"] == 950
        assert body["usage"]["output_tokens_details"]["reasoning_tokens"] == 5

        # Metadata was stripped before every model call.
        for call in server.server_client.post.call_args_list:
            if call.kwargs.get("server_name") == "model server":
                assert call.kwargs["json"].metadata is None

        # Per-turn telemetry captured under the correlation id.
        turns = server.turn_logs["trial-42"]
        assert len(turns) == 2
        first, second = turns
        assert first["turn"] == 0
        assert first["tool_call_names"] == ["get_weather"]
        assert first["input_tokens"] == 1000
        assert first["cached_input_tokens"] == 0
        assert first["response_id"] == "resp_turn1"
        assert first["timestamp"].endswith("+00:00")
        # Turn 1 slice covers the function_call plus its tool output.
        assert (first["output_start_index"], first["output_end_index"]) == (0, 2)
        assert second["turn"] == 1
        assert second["input_tokens"] == 1200
        assert second["cached_input_tokens"] == 950
        assert second["reasoning_tokens"] == 5
        assert second["assistant_text"] == "It is cold."
        assert second["tool_call_names"] == []
        assert (second["output_start_index"], second["output_end_index"]) == (2, 3)

        # The final trajectory matches simple_agent semantics: fn_call, tool output, message.
        assert [o["type"] for o in body["output"]] == ["function_call", "function_call_output", "message"]

    def test_run_attaches_turns_to_verify_response(self, monkeypatch: MonkeyPatch) -> None:
        server = TurnLoggingAgent(config=make_config(), server_client=MagicMock(spec=ServerClient))
        client = TestClient(server.setup_webserver())

        fixed_id = MagicMock()
        fixed_id.hex = "fixed-trial-id"
        monkeypatch.setattr("responses_api_agents.turn_logging_agent.app.uuid4", lambda: fixed_id)

        # Pre-fill the turn log the (mocked) inner /v1/responses call would have produced.
        server.turn_logs["fixed-trial-id"] = [
            {"turn": 0, "timestamp": "2026-07-07T00:00:00+00:00"},
            {"turn": 1, "timestamp": "2026-07-07T00:00:09+00:00"},
        ]

        responses_create_params = {"input": [{"role": "user", "content": "hi"}]}
        final_response = model_response([FINAL_MESSAGE], usage(10, 5))
        verify_payload = {
            "responses_create_params": responses_create_params,
            "response": final_response,
            "reward": 1.0,
        }
        server.server_client.post = AsyncMock(
            side_effect=[http_mock({}), http_mock(final_response), http_mock(verify_payload)]
        )

        result = client.post("/run", json={"responses_create_params": responses_create_params})
        assert result.status_code == 200, result.text
        body = result.json()

        assert body["reward"] == 1.0
        assert body["num_turns"] == 2
        assert [t["turn"] for t in body["turns"]] == [0, 1]
        assert body["trial_started_at"] == "2026-07-07T00:00:00+00:00"
        assert body["trial_finished_at"] == "2026-07-07T00:00:09+00:00"

        # The inner /v1/responses call carried the correlation metadata...
        inner_call = server.server_client.post.call_args_list[1]
        assert inner_call.kwargs["json"].metadata == {"_turn_log_id": "fixed-trial-id"}
        # ...and the log entry was popped after use.
        assert server.turn_logs == {}
