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
"""End-to-end #1483 contract check through a real Gym model server.

Drives the actual SimpleModelServer.setup_webserver() install path (only the upstream OpenAI
client mocked), so the captured response is a real NeMoGymResponse that went through
model_validate + serialization -- proving the contract fields survive the canonical type.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from nemo_gym.trajectory_capture import (
    CaptureStore,
    aggregate_rollout_metrics,
    assemble_rollout,
    assemble_step_records,
)
from responses_api_models.openai_model.app import (
    NeMoGymAsyncOpenAI,
    SimpleModelServer,
    SimpleModelServerConfig,
)


def _server(tmp_path, *, enabled: bool = True) -> SimpleModelServer:
    config = SimpleModelServerConfig(
        host="0.0.0.0",
        port=8081,
        openai_base_url="https://api.openai.com/v1",
        openai_api_key="dummy_key",  # pragma: allowlist secret
        openai_model="dummy_model",
        entrypoint="",
        name="srv-e2e",
        observability_enabled=enabled,
        trajectory_capture_dir=str(tmp_path),
    )
    return SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient))


def _response(text, *, tool=None, reasoning=None, cached=0, reasoning_tokens=0, in_tok=12, out_tok=7) -> dict:
    """A valid OpenAI Responses payload (the shape the upstream model returns)."""
    output = []
    if reasoning is not None:
        output.append({"type": "reasoning", "id": "rs", "summary": [{"type": "summary_text", "text": reasoning}]})
    output.append(
        {
            "type": "message",
            "id": "m",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    )
    if tool is not None:
        output.append(
            {
                "type": "function_call",
                "id": "fc",
                "call_id": tool["call_id"],
                "name": tool["name"],
                "arguments": tool["arguments"],
                "status": "completed",
            }
        )
    return {
        "id": "resp_x",
        "created_at": 1753983920.0,
        "model": "dummy_model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": in_tok + out_tok,
            "input_tokens_details": {"cached_tokens": cached},
            "output_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


def test_e2e_full_contract_through_real_model_server(tmp_path):
    server = _server(tmp_path)
    app = server.setup_webserver()  # real install path -> install_trajectory_capture(app, config)
    client = TestClient(app)

    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    server._client.create_response = AsyncMock(
        side_effect=[
            _response(
                "let me check",
                reasoning="weigh the options",
                tool={"call_id": "c1", "name": "calc", "arguments": '{"x": 1}'},
                cached=4,
                reasoning_tokens=2,
                in_tok=12,
                out_tok=7,
            ),
            _response("the answer is 42", in_tok=20, out_tok=5),
        ]
    )

    h = {"x-nemo-gym-rollout-id": "r1", "x-nemo-gym-trial-index": "0"}
    r1 = client.post("/v1/responses", json={"input": "solve"}, headers={**h, "x-nemo-gym-turn-index": "0"})
    r2 = client.post(
        "/v1/responses",
        json={"input": [{"type": "function_call_output", "call_id": "c1", "output": "42"}]},
        headers={**h, "x-nemo-gym-turn-index": "1"},
    )
    assert r1.status_code == 200 and r2.status_code == 200

    store = CaptureStore(tmp_path)
    steps = assemble_step_records(store, "r1", run_id="run-e2e")

    # indices + attribution survive the round trip
    assert [s.step_index for s in steps] == [0, 1]
    assert [s.turn_index for s in steps] == [0, 1]
    assert all(s.trial_index == 0 and s.model_server == "srv-e2e" and s.dialect == "responses" for s in steps)

    # token stats survive NeMoGymResponse validation + serialization (the real point of this test)
    assert (steps[0].tokens_in, steps[0].tokens_out, steps[0].tokens_total) == (12, 7, 19)
    assert steps[0].tokens_reasoning == 2
    assert steps[0].cache_hit is True and steps[0].cached_tokens == 4
    assert steps[0].reasoning_content == "weigh the options"
    assert steps[0].tool_calls == [{"call_id": "c1", "name": "calc", "arguments": {"x": 1}}]
    assert steps[0].latency_total_ms is not None
    assert steps[1].tokens_in == 20

    # per-rollout aggregates
    agg = aggregate_rollout_metrics(store, "r1")
    assert agg["tokens_in"] == 32 and agg["tokens_out"] == 12
    assert agg["num_turns"] == 2 and agg["num_steps"] == 2

    # eval-only ordered trajectory (no token-ids surfaced)
    items = assemble_rollout(store, "r1")
    assert [type(i).__name__ for i in items] == [
        "NeMoGymResponseOutputMessage",
        "NeMoGymResponseFunctionToolCall",
        "NeMoGymFunctionCallOutput",
        "NeMoGymResponseOutputMessage",
    ]
    assert not hasattr(items[0], "generation_token_ids")


def test_e2e_error_category_through_real_model_server(tmp_path):
    server = _server(tmp_path)
    app = server.setup_webserver()
    client = TestClient(app, raise_server_exceptions=False)

    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    server._client.create_response = AsyncMock(side_effect=asyncio.TimeoutError())

    r = client.post("/v1/responses", json={"input": "x"}, headers={"x-nemo-gym-rollout-id": "r-timeout"})
    assert r.status_code == 500  # response unchanged for the caller

    steps = assemble_step_records(CaptureStore(tmp_path), "r-timeout")
    assert len(steps) == 1 and steps[0].error_category == "timeout"


def test_e2e_off_by_default_writes_nothing(tmp_path):
    server = _server(tmp_path, enabled=False)
    app = server.setup_webserver()
    client = TestClient(app)

    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    server._client.create_response = AsyncMock(return_value=_response("hi"))

    r = client.post("/v1/responses", json={"input": "x"}, headers={"x-nemo-gym-rollout-id": "r-off"})
    assert r.status_code == 200
    assert CaptureStore(tmp_path).read("r-off") == []  # disabled => no capture


def test_e2e_per_rollout_url_prefix(tmp_path):
    """The per-rollout URL prefix correlates capture through the real server install path; the
    standard /v1/... endpoint is reached unchanged (prefix stripped before routing)."""
    server = _server(tmp_path)
    app = server.setup_webserver()
    client = TestClient(app)

    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    server._client.create_chat_completion = AsyncMock(
        return_value={
            "id": "chatcmpl-x",
            "choices": [{"finish_reason": "stop", "index": 0, "message": {"content": "hi", "role": "assistant"}}],
            "created": 1753983922,
            "model": "dummy_model",
            "object": "chat.completion",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
    )

    r = client.post(
        "/ng-rollout/task9-roll3/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 200

    steps = assemble_step_records(CaptureStore(tmp_path), "task9-roll3")
    assert len(steps) == 1
    assert steps[0].dialect == "chat" and steps[0].tokens_total == 7
