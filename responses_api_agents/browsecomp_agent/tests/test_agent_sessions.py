# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The agent sessions an Environment Server (single_agent_turn) seeds, activates once, and closes."""

import asyncio
import json as jsonlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi import Request, Response
from fastapi.testclient import TestClient

from nemo_gym.agents import (
    AGENT_SESSION_COOKIE_KEY,
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    OutcomeAction,
    SandboxSession,
    agent_outcome_policy,
    check_agent_seed_rejected,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import (
    EpisodeId,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    TaskId,
)
from nemo_gym.server_runtime import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from responses_api_agents.browsecomp_agent.app import BrowsecompAgent, BrowsecompAgentConfig
from responses_api_agents.browsecomp_agent.sandbox import BrowsecompSandboxSession


EPISODE = EpisodeId(rollout_id="7-0", attempt=0)
GRANT = DirectHTTPToolAccess(
    name="harness.direct_http",
    required=True,
    base_url="http://harness:9000",
    cookies={"harness": "seed-cookie"},
    headers={"x-grant": "1"},
)
QUESTION = {"input": [{"role": "user", "content": "What is the capital of France?"}]}
ROUTE = f"/ng-rollout/{EPISODE.capture_key}/v1/responses"


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
    monkeypatch.setattr("responses_api_agents.browsecomp_agent.app.get_global_config_dict", lambda: {})
    monkeypatch.setattr(
        "responses_api_agents.browsecomp_agent.app.resolve_provider_config", lambda *args: {"docker": {}}
    )
    monkeypatch.setattr("responses_api_agents.browsecomp_agent.app.resolve_provider_metadata", lambda *args: {})
    monkeypatch.setattr(
        "responses_api_agents.browsecomp_agent.app.create_provider", lambda *args: MagicMock(aclose=AsyncMock())
    )
    sandbox = MagicMock(start=AsyncMock(), stop=AsyncMock())
    monkeypatch.setattr(
        "responses_api_agents.browsecomp_agent.app.AsyncSandbox",
        MagicMock(return_value=sandbox, connect=AsyncMock(return_value=sandbox)),
    )
    monkeypatch.setattr(BrowsecompSandboxSession, "install_runtime", AsyncMock())
    monkeypatch.setattr(SandboxSession, "close", AsyncMock())
    monkeypatch.setattr(BrowsecompAgent, "resolve_model_base_url", lambda *args: "http://model:9001/v1")

    # Exercise the original loop and HTTP grants independently from provider transport; the runner
    # subprocess and supervisor lifecycle have separate tests and real cluster smoke coverage.
    async def execute_loop(self, request, body, state):
        from fastapi import Response

        return await self._run_session_loop(request, Response(), body, state)

    monkeypatch.setattr(BrowsecompAgent, "_sandbox_responses", execute_loop)
    agent = BrowsecompAgent(
        config=BrowsecompAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="browsecomp_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="configured_harness"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            nudge_steps=False,
            **({"sandbox_provider": "sandbox"} | config),
        ),
        server_client=server_client,
    )
    return TestClient(agent.setup_webserver()), server_client, harness


@pytest.mark.parametrize(
    ("unsupported", "reason"),
    [
        (
            {"tool_accesses": [GRANT, GRANT.model_copy(update={"name": "other.direct_http"})]},
            "supports at most one direct HTTP tool access",
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

    reply = client.post(ROUTE, json=QUESTION)

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

    reply = client.post(ROUTE, json=QUESTION)

    assert reply.status_code == 200, reply.text
    assert reply.json()["output"][-1]["content"][0]["text"] == "Exact Answer: Paris"
    assert server_client.post.await_count == 4
    # Like run(), the retry keeps the session's resources state: its tool call carries the first attempt's cookie.
    assert [call["cookies"] for call in harness.calls] == [{"harness": "seed-cookie"}, {"harness": "after-call-1"}]


def test_the_last_attempt_returns_its_empty_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    client, server_client, _ = _agent([_model_response(_message("<think>no answer</think>"))], monkeypatch)
    assert client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json")).status_code == 200

    reply = client.post(ROUTE, json=QUESTION)

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

    reply = client.post(ROUTE, json=QUESTION)

    assert reply.status_code == 200, reply.text
    response = NeMoGymResponse.model_validate(reply.json())
    assert response.status == "failed"
    assert response.output == []
    assert f"Model HTTP {status}: model refused the request" in response.error.message
    assert agent_outcome_policy(response) == action


def _agent_of(client: TestClient) -> BrowsecompAgent:
    return next(route.endpoint.__self__ for route in client.app.routes if route.path == "/v1/browsecomp/token_count")


def test_a_model_http_error_keeps_only_the_start_of_a_long_upstream_body(monkeypatch: pytest.MonkeyPatch) -> None:
    client, _, _ = _agent([], monkeypatch)
    request_info = SimpleNamespace(url="http://model", method="POST", headers={}, real_url="http://model")
    error = ClientResponseError(request_info, (), status=502, message="Bad Gateway")
    error.response_content = b"<html>" + b"x" * 10_000
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(QUESTION)
    response = _agent_of(client)._model_request_failure(body, error)
    # The rollout row carries this message, so an upstream error page cannot bloat it.
    assert response.error.message == "Model HTTP 502: <html>" + "x" * 3994


async def test_a_cancelled_waiter_leaves_the_activation_running_for_its_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """A disconnected HTTP request must not cancel the shared activation; the identical retry gets its result."""
    client, _, _ = _agent([], monkeypatch)
    agent = _agent_of(client)
    seed_scope = {"type": "http", "headers": [], "session": {}}
    await agent.seed_agent_session(Request(seed_scope), _seed(), Response())
    started, finish = asyncio.Event(), asyncio.Event()
    runs = []

    async def activation(self, request, body, state):
        runs.append(body)
        started.set()
        await finish.wait()
        return NeMoGymResponse.model_validate(_model_response(_message("Paris")))

    monkeypatch.setattr(BrowsecompAgent, "_sandbox_responses", activation)
    scope = {
        "type": "http",
        "headers": [],
        "session": {AGENT_SESSION_COOKIE_KEY: "agent-session"},
        "path_params": {"rollout_id": EPISODE.capture_key},
    }
    body = NeMoGymResponseCreateParamsNonStreaming.model_validate(QUESTION)
    waiter = asyncio.create_task(agent.responses(Request(scope), Response(), body))
    await asyncio.wait_for(started.wait(), 1)
    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    state = agent._require_agent_session("agent-session")
    assert state.task is not None and not state.task.done()
    finish.set()
    retried = await asyncio.wait_for(agent.responses(Request(scope), Response(), body), 1)
    assert retried.output[0].content[0].text == "Paris"
    assert len(runs) == 1


@pytest.mark.parametrize("enabled", [False, True])
def test_the_token_count_endpoint_serves_only_a_seeded_session(monkeypatch: pytest.MonkeyPatch, enabled: bool) -> None:
    """The sandbox's only path to token counting: refused without the session cookie, and when disabled."""
    monkeypatch.setattr(
        "responses_api_agents.browsecomp_agent.app.get_first_server_config_dict",
        lambda *args: {"base_url": "http://10.0.0.5:9001/v1", "api_key": "unused"},
    )
    client, _, _ = _agent([], monkeypatch, save_model_call_using_vllm_tokenize_endpoint=enabled)
    counted = AsyncMock(return_value=1234)
    monkeypatch.setattr(BrowsecompAgent, "_count_prompt_tokens", counted)
    assert client.post("/v1/browsecomp/token_count", json=QUESTION).status_code == 409
    assert client.post("/v1/agent_sessions", json=_seed().model_dump(mode="json")).status_code == 200
    reply = client.post("/v1/browsecomp/token_count", json=QUESTION)
    if enabled:
        assert reply.status_code == 200 and reply.json() == {"count": 1234}
        counted.assert_awaited_once()
    else:
        assert reply.status_code == 422
        counted.assert_not_awaited()
