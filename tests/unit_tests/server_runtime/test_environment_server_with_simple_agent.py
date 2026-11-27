# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An Environment Server episode that drives a real Simple Agent continuation session through its tools."""

import asyncio
import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from nemo_gym.agents.base_responses_api_agent import AGENT_ACTIVATION_HEADER, AgentSeedSessionRequest
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import ModelServerRef
from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.server_runtime.base_environment_server import ServedTools
from nemo_gym.trajectory.trajectory_record import TrajectoryRecord
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig
from tests.unit_tests.server_runtime.test_base_environment_server_with_tools import (
    _app,
    _Counter,
    _request,
    _Response,
    _Server,
)


def _model_reply(response_id: str, output: list[dict[str, Any]]) -> MagicMock:
    reply = MagicMock(status=200, ok=True, cookies={}, headers={})
    reply.read = AsyncMock(
        return_value=json.dumps(
            {
                "id": response_id,
                "created_at": 1.0,
                "model": "model",
                "object": "response",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "output": output,
            }
        )
    )
    return reply


_SUBMIT = {
    "id": "fc-submit",
    "call_id": "call-submit",
    "name": "submit",
    "arguments": json.dumps({"amount": 2}),
    "type": "function_call",
    "status": "completed",
}
_DONE = {
    "id": "msg-done",
    "content": [{"annotations": [], "text": "Done.", "type": "output_text"}],
    "role": "assistant",
    "status": "completed",
    "type": "message",
}


class _ToolReply:
    """The environment's reply, in the shape the agent reads from its HTTP client."""

    def __init__(self, reply: httpx.Response) -> None:
        self.status = reply.status_code
        self.headers = reply.headers
        self.cookies: dict[str, Any] = {}
        self.content = MagicMock(read=AsyncMock(return_value=reply.content))


def _run_episode(
    monkeypatch: pytest.MonkeyPatch,
    second_model_reply: MagicMock | Exception,
    second_activation: dict[str, Any] | None = None,
) -> tuple[_Response, AsyncMock, dict[str, Any]]:
    """Run one episode whose agent session has two activations: the first submits, the second sends
    ``second_activation`` and gets ``second_model_reply`` from the model. Return the episode's response, the model
    calls and what the episode saw."""
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    server_client.post = AsyncMock(side_effect=[_model_reply("resp-1", [_SUBMIT]), second_model_reply])
    agent = SimpleAgent(
        config=SimpleAgentConfig(
            host="127.0.0.1",
            port=8080,
            entrypoint="",
            name="simple",
            model_server=ModelServerRef(type="responses_api_models", name="model"),
            max_steps=10,
        ),
        server_client=server_client,
    )
    agent_http = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=agent.setup_webserver(), raise_app_exceptions=False),
        base_url="http://agent",
    )
    seen: dict[str, Any] = {"activations": []}

    async def episode(server: _Server, counter: _Counter, served: ServedTools) -> None:
        seed = await agent_http.post(
            "/v1/agent_sessions",
            json=AgentSeedSessionRequest(
                agent_session_id="agent-session",
                episode_id=EpisodeId(rollout_id="3-0"),
                task_id=TaskId(taskset="test:default", task_id="task"),
                tool_accesses=served.tool_accesses,
                continuation={"observations": ["trajectory", "timing"]},
            ).model_dump(mode="json"),
        )
        assert seed.status_code == 200, seed.text
        activations = [{"input": "Add two and submit."}, second_activation or {"input": "What is the total?"}]
        for index, body in enumerate(activations, start=1):
            reply = await agent_http.post("/v1/responses", json=body, headers={AGENT_ACTIVATION_HEADER: str(index)})
            if reply.status_code == 200:
                seen["activations"].append((200, [item["type"] for item in reply.json()["output"]]))
            else:
                seen["activations"].append((reply.status_code, reply.text))
        close = await agent_http.post(
            "/v1/agent_sessions/close",
            json={"agent_session_id": "agent-session", "episode_id": {"rollout_id": "3-0", "attempt": 0}},
        )
        seen["trajectory"] = TrajectoryRecord.model_validate(close.json()["trajectory"])
        seen["total"] = counter.total

    server, app = _app(episode)
    environment_http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://environment")

    async def tool_call(*, method: str, url: str, json: Any, headers: dict[str, str], **_: Any) -> _ToolReply:
        return _ToolReply(await environment_http.request(method, url, json=json, headers=headers))

    monkeypatch.setattr("responses_api_agents.simple_agent.app.http_request", tool_call)
    return asyncio.run(server.run_request(_request())), server_client.post, seen


def test_a_tool_that_ends_the_activation_ends_the_turn_and_the_next_activation_continues_the_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first activation stops at ``submit`` without another model call, and the second sends the whole
    conversation so far. Close returns both activations and the tool call."""
    response, model_calls, seen = _run_episode(monkeypatch, _model_reply("resp-2", [_DONE]))

    assert response.result == "total=2 calls=1" and seen["total"] == 2
    # The first activation ends at the submission's reply: one model call for it, one for the second activation.
    assert model_calls.await_count == 2
    assert seen["activations"] == [(200, ["function_call", "function_call_output"]), (200, ["message"])]
    second_input = model_calls.await_args_list[1].kwargs["json"].model_dump(mode="json")["input"]
    assert [(item.get("type"), item.get("role")) for item in second_input] == [
        ("message", "user"),
        ("function_call", None),
        ("function_call_output", None),
        ("message", "user"),
    ]
    assert json.loads(second_input[2]["output"]) == {"total": 2}
    trajectory = seen["trajectory"]
    assert [(invocation.invocation_id, invocation.activation_id) for invocation in trajectory.invocations] == [
        ("root", 1),
        ("activation-2", 2),
    ]
    assert [(call.tool_name, call.status) for call in trajectory.tool_calls] == [("submit", "completed")]


def test_a_failed_activation_stays_in_the_trajectory_the_episode_closes_with(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Environment Server gets the failed turn at close, with a gap, after the turns that completed."""
    _, _, seen = _run_episode(monkeypatch, RuntimeError("model down"))

    assert [status for status, _ in seen["activations"]] == [200, 500]
    trajectory = seen["trajectory"]
    assert [(i.invocation_id, i.activation_id, i.status, i.error_type) for i in trajectory.invocations] == [
        ("root", 1, "completed", None),
        ("activation-2", 2, "failed", "RuntimeError"),
    ]
    assert ("agent_activation_interrupted", "activation-2") in {
        (gap.code, gap.invocation_id) for gap in trajectory.gaps
    }


@pytest.mark.parametrize(
    "second_activation, detail",
    [
        pytest.param({"input": []}, "A continuation activation requires new input", id="no-input"),
        pytest.param(
            {"input": "What is the total?", "tool_choice": "none"},
            "Simple Agent continuation cannot change session tool_choice",
            id="changed-setting",
        ),
    ],
)
def test_an_activation_the_session_cannot_run_is_refused_before_a_model_call(
    monkeypatch: pytest.MonkeyPatch, second_activation: dict[str, Any], detail: str
) -> None:
    """Every activation runs with the first one's settings on the conversation so far, so the Environment Server gets a
    422 for one with no new input or changed settings, not a turn that silently drops them."""
    _, model_calls, seen = _run_episode(monkeypatch, AssertionError("not called"), second_activation)

    status, text = seen["activations"][1]
    assert status == 422 and detail in text
    assert model_calls.await_count == 1
