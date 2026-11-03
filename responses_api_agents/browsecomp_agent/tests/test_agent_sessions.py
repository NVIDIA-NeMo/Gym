# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agent sessions an Environment Server (single_agent_turn) seeds, activates once, and closes."""

import json as jsonlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi.testclient import TestClient

from nemo_gym.agents import (
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    OutcomeAction,
    agent_outcome_policy,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    EpisodeId,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    TaskId,
)
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_runtime import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from nemo_gym.testing.agent_session_conformance import check_agent_seed_rejected
from responses_api_agents.browsecomp_agent.app import BrowsecompAgent, BrowsecompAgentConfig


EPISODE = EpisodeId(rollout_id="7-0", attempt=0)
GRANT = DirectHTTPToolAccess(
    name="harness.direct_http",
    required=True,
    base_url="http://harness:9000",
    cookies={"harness": "seed-cookie"},
    headers={"x-grant": "1"},
)
QUESTION = {"input": [{"role": "user", "content": "What is the capital of France?"}]}


def _seed(**overrides) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="agent-session",
        episode_id=EPISODE,
        task_id=TaskId(taskset="browsecomp", task_id="7"),
        **({"tool_accesses": [GRANT]} | overrides),
    )


def _message(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _tool_call(keystrokes: str) -> NeMoGymResponseFunctionToolCall:
    return NeMoGymResponseFunctionToolCall(
        id="fc", call_id="call", name="bash_command", arguments=jsonlib.dumps({"keystrokes": keystrokes})
    )


def _model_response(*output) -> dict:
    return NeMoGymResponse(
        id="resp",
        created_at=0.0,
        model="policy",
        object="response",
        output=list(output),
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
    ).model_dump(mode="json")


class _Harness:
    """Stands in for the resources server behind the grant; each call sets the next cookie value."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, **kwargs) -> MagicMock:
        self.calls.append(kwargs)
        reply = MagicMock(status=200, cookies={"harness": SimpleNamespace(value=f"after-call-{len(self.calls)}")})
        reply.content.read = AsyncMock(return_value=b'{"results_string": "manifest.tsv\\npages"}')
        return reply


def _agent(model_replies: list, monkeypatch: pytest.MonkeyPatch, **config) -> tuple[TestClient, MagicMock, _Harness]:
    """An agent app whose model returns ``model_replies`` in order (a dict, or an int HTTP error status)."""

    async def post(server_name=None, url_path=None, json=None, cookies=None):
        reply = model_replies.pop(0)
        if isinstance(reply, int):
            response = MagicMock(ok=False, status=reply, cookies={})
            request_info = SimpleNamespace(url="http://model", method="POST", headers={}, real_url="http://model")
            response.raise_for_status = MagicMock(
                side_effect=ClientResponseError(request_info, (), status=reply, message="refused")
            )
            response.content.read = AsyncMock(return_value=b"model refused the request")
            return response
        response = MagicMock(ok=True, status=200, cookies={})
        response.read = AsyncMock(return_value=jsonlib.dumps(reply).encode())
        return response

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    server_client.post = AsyncMock(side_effect=post)
    harness = _Harness()
    monkeypatch.setattr("responses_api_agents.browsecomp_agent.app.http_request", harness)
    agent = BrowsecompAgent(
        config=BrowsecompAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="browsecomp_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="configured_harness"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            nudge_steps=False,
            **config,
        ),
        server_client=server_client,
    )
    return TestClient(agent.setup_webserver()), server_client, harness


@pytest.mark.parametrize(
    ("unsupported", "reason"),
    [
        ({"tool_accesses": []}, "needs exactly one direct HTTP tool access to its resources server, got 0"),
        (
            {"tool_accesses": [GRANT, GRANT.model_copy(update={"name": "other.direct_http"})]},
            "needs exactly one direct HTTP tool access to its resources server, got 2",
        ),
        (
            {
                "tool_accesses": [
                    GRANT,
                    MCPToolAccess(
                        name="resources",
                        required=True,
                        connection=MCPStreamableHTTPConnection(url="http://harness:9000/mcp"),
                    ),
                ]
            },
            "does not support required MCP tool access: resources",
        ),
        (
            {
                "sandbox_access": SandboxAccess(
                    connection=DirectSandboxConnection(provider_config_ref="runtime", descriptor={}),
                    workdir="/work",
                )
            },
            "does not support sandbox access",
        ),
    ],
)
def test_a_seed_it_cannot_serve_is_refused_as_final(
    unsupported: dict, reason: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _ = _agent([], monkeypatch)
    assert reason in check_agent_seed_rejected(client.app, _seed(**unsupported))


def test_tool_calls_use_the_grant_and_close_returns_the_resources_cookies(monkeypatch: pytest.MonkeyPatch) -> None:
    client, server_client, harness = _agent(
        [_model_response(_tool_call("ls")), _model_response(_message("Exact Answer: Paris"))], monkeypatch
    )
    assert client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json")).status_code == 200

    reply = client.post("/v1/responses", json=QUESTION)

    assert reply.status_code == 200, reply.text
    assert [call.kwargs["server_name"] for call in server_client.post.await_args_list] == ["policy_model"] * 2
    [call] = harness.calls
    assert call["url"] == "http://harness:9000/bash_command"
    assert call["json"] == {"keystrokes": "ls"}
    assert call["cookies"] == {"harness": "seed-cookie"}
    assert call["headers"] == {"x-grant": "1"}
    # The tool output reaches the model unwrapped, as on the /run path.
    assert reply.json()["output"][1]["output"] == "manifest.tsv\npages"
    # The resources cookie returns through the close, not on the reply, where it would join the agent's cookies.
    assert "harness" not in reply.cookies
    close = client.post(
        "/v1/agent_sessions/close",
        json=AgentCloseSessionRequest(agent_session_id="agent-session", episode_id=EPISODE).model_dump(mode="json"),
    )
    assert close.status_code == 200
    assert close.json()["resources_cookies"] == {"harness": "after-call-1"}


def test_an_empty_answer_is_retried_inside_the_activation(monkeypatch: pytest.MonkeyPatch) -> None:
    client, server_client, harness = _agent(
        [
            _model_response(_tool_call("ls")),
            _model_response(_message("<think>still thinking</think>")),
            _model_response(_tool_call("cat manifest.tsv")),
            _model_response(_message("Exact Answer: Paris")),
        ],
        monkeypatch,
        max_run_retries=2,
    )
    assert client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json")).status_code == 200

    reply = client.post("/v1/responses", json=QUESTION)

    assert reply.status_code == 200, reply.text
    assert reply.json()["output"][-1]["content"][0]["text"] == "Exact Answer: Paris"
    assert server_client.post.await_count == 4
    # Like run(), the retry keeps the session's resources state: its tool call carries the first attempt's cookie.
    assert [call["cookies"] for call in harness.calls] == [{"harness": "seed-cookie"}, {"harness": "after-call-1"}]


def test_the_last_attempt_returns_its_empty_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    client, server_client, _ = _agent([_model_response(_message("<think>no answer</think>"))], monkeypatch)
    assert client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json")).status_code == 200

    reply = client.post("/v1/responses", json=QUESTION)

    assert reply.status_code == 200, reply.text
    assert reply.json()["output"][-1]["content"][0]["text"] == "<think>no answer</think>"
    assert server_client.post.await_count == 1


@pytest.mark.parametrize(
    ("status", "action"),
    [(400, OutcomeAction.VERIFY), (503, OutcomeAction.RETRY), (429, OutcomeAction.RETRY), (404, OutcomeAction.FAIL)],
)
def test_a_model_http_error_is_a_failed_response_for_the_outcome_policy(
    status: int, action: OutcomeAction, monkeypatch: pytest.MonkeyPatch
) -> None:
    client, _, _ = _agent([_model_response(_tool_call("ls")), status], monkeypatch)
    assert client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json")).status_code == 200

    reply = client.post("/v1/responses", json=QUESTION)

    assert reply.status_code == 200, reply.text
    response = NeMoGymResponse.model_validate(reply.json())
    assert response.status == "failed"
    assert response.output == []
    assert f"Model HTTP {status}: model refused the request" in response.error.message
    assert agent_outcome_policy(response) == action
