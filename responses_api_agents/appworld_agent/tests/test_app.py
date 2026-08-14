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
"""Unit tests for the AppWorld agent harness.

The harness drives a code-as-action loop against the resources server
(``seed_session`` -> ``step`` -> ``close`` -> ``verify``) while sampling turns
from the policy model. ``ServerClient.post`` is mocked with a small router that
returns canned payloads per ``url_path``, so the whole episode loop runs without
a live server or a real AppWorld.
"""

import json
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock

from pytest import fixture, mark, raises
from tenacity import RetryError, wait_none

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.appworld_agent.app import (
    AppWorldAgent,
    AppWorldAgentConfig,
    AppWorldAgentRunRequest,
    _extract_code,
)


class FakeContent:
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def read(self) -> bytes:
        return self._body


class FakeResponse:
    """Minimal stand-in for the aiohttp response the harness consumes."""

    def __init__(self, payload: Dict[str, Any], status: int = 200) -> None:
        self._payload = payload
        self.status = status
        self.ok = status < 400
        self.cookies: Dict[str, str] = {}
        self.content = FakeContent(json.dumps(payload).encode())

    def raise_for_status(self) -> None:
        return None

    async def read(self) -> bytes:
        return json.dumps(self._payload).encode()

    async def json(self) -> Dict[str, Any]:
        return self._payload


def post_router(routes: Dict[str, List[Dict[str, Any]]], log: Optional[List[tuple]] = None) -> Callable[..., Any]:
    """Async ``post`` side effect dispatching on ``url_path``.

    Each path maps to a FIFO queue; the last entry repeats forever so a
    steady-state reply can serve an unbounded loop.
    """
    queues = {path: list(payloads) for path, payloads in routes.items()}

    async def _post(*, server_name: str, url_path: str, json: Any = None, cookies: Any = None) -> FakeResponse:
        if log is not None:
            log.append((url_path, json))
        queue = queues[url_path]
        payload = queue.pop(0) if len(queue) > 1 else queue[0]
        return FakeResponse(payload)

    return _post


def seed_payload(obs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    return {
        "env_id": "env-1",
        "task_id": "82e2fac_1",
        "obs": [
            {"role": "system", "content": "You are a super intelligent AI Assistant."},
            {"role": "user", "content": "Task: what is my most-liked song?"},
        ]
        if obs is None
        else obs,
        "tools": [
            {
                "type": "function",
                "name": "execute_ipython_code",
                "description": "Run python.",
                "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
                "strict": True,
            }
        ],
    }


def function_call_response(
    arguments: str = '{"code": "print(1)"}',
    name: str = "execute_ipython_code",
    call_id: str = "call-1",
    extra_calls: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    output = [{"arguments": arguments, "call_id": call_id, "name": name, "type": "function_call"}]
    output.extend(extra_calls or [])
    return {
        "id": "resp-fn",
        "created_at": 1,
        "model": "policy-model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def text_response(text: str = "I give up.") -> Dict[str, Any]:
    return {
        "id": "resp-text",
        "created_at": 1,
        "model": "policy-model",
        "object": "response",
        "output": [
            {
                "id": "msg-1",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


def step_payload(done: bool, output: str = "Execution successful.", num_interactions: int = 1) -> Dict[str, Any]:
    return {"output": output, "done": done, "reward": 0.0, "num_interactions": num_interactions}


def verify_payload(reward: float = 1.0) -> Dict[str, Any]:
    return {
        "responses_create_params": {"input": []},
        "response": {
            "id": "resp-fn",
            "created_at": 1,
            "model": "policy-model",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
            "env_id": "env-1",
            "task_id": "82e2fac_1",
        },
        "reward": reward,
        "success": reward == 1.0,
        "num_tests": 2,
        "num_passed": 2 if reward == 1.0 else 0,
    }


@fixture
def agent() -> AppWorldAgent:
    config = AppWorldAgentConfig(
        name="appworld_agent",
        host="0.0.0.0",
        port=8081,
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="appworld"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    return AppWorldAgent(config=config, server_client=MagicMock(spec=ServerClient))


def run_request(task_id: str = "82e2fac_1") -> AppWorldAgentRunRequest:
    return AppWorldAgentRunRequest(
        task_id=task_id,
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
    )


# ---------------------------------------------------------------------------
# episode loop
# ---------------------------------------------------------------------------


@mark.asyncio
async def test_seed_observations_and_tools_start_the_rollout(agent):
    log: List[tuple] = []
    agent.server_client.post = AsyncMock(
        side_effect=post_router(
            {
                "/seed_session": [seed_payload()],
                "/v1/responses": [function_call_response()],
                "/step": [step_payload(done=True)],
                "/close": [{"message": "Success", "success": True}],
            },
            log,
        )
    )

    response = await agent.responses(run_request())

    # The first model call carries the seeded prompt and the single tool.
    _, first_model_body = next((path, body) for path, body in log if path == "/v1/responses")
    assert [message.role for message in first_model_body.input] == ["system", "user"]
    assert [tool["name"] for tool in first_model_body.tools] == ["execute_ipython_code"]
    assert response.env_id == "env-1"
    assert response.task_id == "82e2fac_1"
    # Trajectory holds the model turn plus the environment's observation.
    assert [item.type for item in response.output] == ["function_call", "function_call_output"]
    assert response.output[1].output == "Execution successful."


@mark.asyncio
async def test_rollout_stops_when_the_environment_reports_done(agent):
    log: List[tuple] = []
    agent.server_client.post = AsyncMock(
        side_effect=post_router(
            {
                "/seed_session": [seed_payload()],
                "/v1/responses": [function_call_response(), function_call_response()],
                "/step": [step_payload(done=False), step_payload(done=True)],
                "/close": [{"message": "Success", "success": True}],
            },
            log,
        )
    )

    await agent.responses(run_request())

    assert [path for path, _ in log].count("/step") == 2
    assert [path for path, _ in log].count("/v1/responses") == 2


@mark.asyncio
async def test_a_turn_without_a_tool_call_ends_the_rollout(agent):
    log: List[tuple] = []
    agent.server_client.post = AsyncMock(
        side_effect=post_router(
            {
                "/seed_session": [seed_payload()],
                "/v1/responses": [text_response()],
                "/close": [{"message": "Success", "success": True}],
            },
            log,
        )
    )

    response = await agent.responses(run_request())

    assert [path for path, _ in log] == ["/seed_session", "/v1/responses", "/close"]
    assert [item.type for item in response.output] == ["message"]


@mark.asyncio
async def test_max_steps_caps_a_model_that_never_finishes(agent):
    agent.config.max_steps = 3
    log: List[tuple] = []
    agent.server_client.post = AsyncMock(
        side_effect=post_router(
            {
                "/seed_session": [seed_payload()],
                "/v1/responses": [function_call_response()],
                "/step": [step_payload(done=False)],
                "/close": [{"message": "Success", "success": True}],
            },
            log,
        )
    )

    await agent.responses(run_request())

    assert [path for path, _ in log].count("/step") == 3


@mark.asyncio
async def test_close_runs_even_when_the_loop_raises(agent):
    log: List[tuple] = []
    router = post_router(
        {
            "/seed_session": [seed_payload()],
            "/v1/responses": [function_call_response()],
            "/close": [{"message": "Success", "success": True}],
        },
        log,
    )

    async def _post(*, server_name, url_path, json=None, cookies=None):
        if url_path == "/step":
            raise RuntimeError("resources server exploded")
        return await router(server_name=server_name, url_path=url_path, json=json, cookies=cookies)

    agent.server_client.post = AsyncMock(side_effect=_post)

    with raises(RuntimeError):
        await agent.responses(run_request())

    # The worker lease must always be handed back.
    assert ("/close", {"env_id": "env-1"}) in log


@mark.asyncio
async def test_seed_session_without_observations_is_retried_then_fails(agent):
    agent._seed_session.retry.wait = wait_none()
    agent.server_client.post = AsyncMock(side_effect=post_router({"/seed_session": [seed_payload(obs=[])]}))

    with raises(RetryError):
        await agent._seed_session("82e2fac_1")

    assert agent.server_client.post.await_count == 3


@mark.asyncio
async def test_tool_outputs_survive_exclude_unset_serialisation(agent):
    """Regression: `ServerClient.post` dumps with `exclude_unset=True`.

    A defaulted `type` on the tool result is therefore dropped from the wire
    payload, and the provider sees a tool call with no matching result — Bedrock
    rejects the whole turn, others silently lose the observation. Assert on the
    serialised form, not the object, since the object looks fine either way.
    """
    log: List[tuple] = []
    agent.server_client.post = AsyncMock(
        side_effect=post_router(
            {
                "/seed_session": [seed_payload()],
                "/v1/responses": [function_call_response(), text_response()],
                "/step": [step_payload(done=False)],
                "/close": [{"message": "Success", "success": True}],
            },
            log,
        )
    )

    await agent.responses(run_request())

    second_turn = [body for path, body in log if path == "/v1/responses"][1]
    wire_items = second_turn.model_dump(exclude_unset=True)["input"]
    assert wire_items[-1]["type"] == "function_call_output"
    assert wire_items[-1]["call_id"] == "call-1"


@mark.asyncio
async def test_a_string_prompt_is_normalised_to_a_message(agent):
    log: List[tuple] = []
    agent.server_client.post = AsyncMock(
        side_effect=post_router(
            {
                "/seed_session": [seed_payload()],
                "/v1/responses": [text_response()],
                "/close": [{"message": "Success", "success": True}],
            },
            log,
        )
    )
    request = AppWorldAgentRunRequest(
        task_id="82e2fac_1",
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="do the thing"),
    )

    await agent.responses(request)

    _, model_body = next((path, body) for path, body in log if path == "/v1/responses")
    assert [message.role for message in model_body.input] == ["user", "system", "user"]


@mark.asyncio
async def test_a_model_server_error_ends_the_rollout_and_is_logged_with_its_body(agent, caplog):
    log: List[tuple] = []
    router = post_router(
        {
            "/seed_session": [seed_payload()],
            "/v1/responses": [function_call_response()],
            "/step": [step_payload(done=False)],
            "/close": [{"message": "Success", "success": True}],
        },
        log,
    )
    calls = {"count": 0}

    async def _post(*, server_name, url_path, json=None, cookies=None):
        if url_path == "/v1/responses":
            calls["count"] += 1
            if calls["count"] == 2:
                return FakeResponse({"error": "tool_use ids without tool_result"}, status=500)
        return await router(server_name=server_name, url_path=url_path, json=json, cookies=cookies)

    agent.server_client.post = AsyncMock(side_effect=_post)

    response = await agent.responses(run_request())

    # The first turn is kept, and the lease is still returned.
    assert [item.type for item in response.output] == ["function_call", "function_call_output"]
    assert ("/close", {"env_id": "env-1"}) in log
    # The provider's message must reach the log, else a rejected multi-turn
    # payload is indistinguishable from the model giving up.
    assert "tool_use ids without tool_result" in caplog.text


@mark.asyncio
async def test_an_invalid_model_response_ends_the_rollout(agent):
    log: List[tuple] = []
    agent.server_client.post = AsyncMock(
        side_effect=post_router(
            {
                "/seed_session": [seed_payload()],
                "/v1/responses": [function_call_response(), {"not": "a response"}],
                "/step": [step_payload(done=False)],
                "/close": [{"message": "Success", "success": True}],
            },
            log,
        )
    )

    await agent.responses(run_request())

    assert [path for path, _ in log][-1] == "/close"


# ---------------------------------------------------------------------------
# tool-call handling
# ---------------------------------------------------------------------------


@mark.asyncio
async def test_malformed_tool_arguments_are_reported_to_the_model(agent):
    log: List[tuple] = []
    agent.server_client.post = AsyncMock(
        side_effect=post_router(
            {
                "/seed_session": [seed_payload()],
                "/v1/responses": [function_call_response(arguments="{not json"), text_response()],
                "/close": [{"message": "Success", "success": True}],
            },
            log,
        )
    )

    response = await agent.responses(run_request())

    assert "/step" not in [path for path, _ in log]
    assert "Invalid tool call arguments" in response.output[1].output


@mark.asyncio
async def test_an_unknown_tool_name_is_reported_to_the_model(agent):
    agent.server_client.post = AsyncMock(
        side_effect=post_router(
            {
                "/seed_session": [seed_payload()],
                "/v1/responses": [function_call_response(name="bash"), text_response()],
                "/close": [{"message": "Success", "success": True}],
            }
        )
    )

    response = await agent.responses(run_request())

    assert "Unknown tool 'bash'" in response.output[1].output


@mark.asyncio
async def test_parallel_tool_calls_run_in_order_and_stop_once_done(agent):
    log: List[tuple] = []
    second_call = {
        "arguments": '{"code": "print(2)"}',
        "call_id": "call-2",
        "name": "execute_ipython_code",
        "type": "function_call",
    }
    agent.server_client.post = AsyncMock(
        side_effect=post_router(
            {
                "/seed_session": [seed_payload()],
                "/v1/responses": [function_call_response(extra_calls=[second_call])],
                "/step": [step_payload(done=True)],
                "/close": [{"message": "Success", "success": True}],
            },
            log,
        )
    )

    response = await agent.responses(run_request())

    # AppWorld's shell is sequential: the second call is not executed once the
    # first one ended the episode.
    assert [path for path, _ in log].count("/step") == 1
    outputs = [item.output for item in response.output if item.type == "function_call_output"]
    assert outputs == ["Execution successful.", "The episode has already ended."]


def test_extract_code_accepts_a_well_formed_call():
    call = MagicMock(arguments=json.dumps({"code": "print(1)"}))
    assert _extract_code(call) == ("print(1)", None)


def test_extract_code_rejects_a_non_object_payload():
    call = MagicMock(arguments="[1, 2]")
    code, error = _extract_code(call)
    assert code == ""
    assert "expected a JSON object" in error


def test_extract_code_rejects_a_missing_code_argument():
    call = MagicMock(arguments=json.dumps({"script": "print(1)"}))
    code, error = _extract_code(call)
    assert code == ""
    assert "Missing required argument 'code'" in error


# ---------------------------------------------------------------------------
# run()
# ---------------------------------------------------------------------------


@mark.asyncio
async def test_run_returns_the_verify_response(agent):
    log: List[tuple] = []
    agent.server_client.post = AsyncMock(
        side_effect=post_router(
            {
                "/seed_session": [seed_payload()],
                "/v1/responses": [function_call_response()],
                "/step": [step_payload(done=True)],
                "/close": [{"message": "Success", "success": True}],
                "/verify": [verify_payload(reward=1.0)],
            },
            log,
        )
    )

    result = await agent.run(run_request())

    assert result.reward == 1.0
    assert result.success is True
    # /verify comes after /close, so the reward reflects the scored episode.
    paths = [path for path, _ in log]
    assert paths.index("/close") < paths.index("/verify")


@mark.asyncio
async def test_aggregate_metrics_comes_from_the_resources_server(agent):
    """Scenario Goal Completion needs every variant at once, so it lives there."""
    log: List[tuple] = []
    agent.server_client.post = AsyncMock(
        side_effect=post_router(
            {"/aggregate_metrics": [{"agent_metrics": {"mean/scenario_goal_completion": 0.5}}]},
            log,
        )
    )

    metrics = await agent.aggregate_metrics({"verify_responses": []})

    assert [path for path, _ in log] == ["/aggregate_metrics"]
    assert metrics.agent_metrics["mean/scenario_goal_completion"] == 0.5


@mark.asyncio
async def test_run_propagates_failures(agent):
    agent._seed_session.retry.wait = wait_none()
    agent.server_client.post = AsyncMock(side_effect=RuntimeError("resources server down"))

    with raises(RetryError):
        await agent.run(run_request())
