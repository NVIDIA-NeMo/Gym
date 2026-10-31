# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
import orjson
import pytest
from aiohttp import ClientResponseError, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict
from yarl import URL

from nemo_gym.base_environment_server import CleanupContext
from nemo_gym.base_responses_api_agent import AgentSeedSessionRequest
from nemo_gym.base_user_simulation_environment_server import (
    BaseUserSimulationEnvironmentServer,
    Stop,
    Transcript,
    TurnRecord,
    UserMessage,
    UserSimulationEnvironmentServerConfig,
    UserSimulationFailure,
    UserSimulationResult,
)
from nemo_gym.config_types import AgentServerRef, AggregateMetrics, AggregateMetricsRequest
from nemo_gym.episode_tools import TypedToolset, episode_tool
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeId, MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.server_utils import BaseServerConfig, ServerClient


class _TaskInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str


class _Request(BaseEpisodeRequest[_TaskInput]):
    pass


class _Response(BaseEpisodeResponse[UserSimulationResult]):
    failure: UserSimulationFailure | None = None


class _Cookie:
    def __init__(self, value: str) -> None:
        self.value = value


class _Content:
    def __init__(self, body: bytes) -> None:
        self.body = body

    async def read(self) -> bytes:
        return self.body


class _HTTPResponse:
    def __init__(self, body: dict, *, status: int = 200, cookies: dict[str, str] | None = None) -> None:
        self.status = status
        self.ok = status < 400
        self.body = orjson.dumps(body)
        self.content = _Content(self.body)
        self.cookies = {name: _Cookie(value) for name, value in (cookies or {}).items()}
        self.headers: dict[str, str] = {}

    async def read(self) -> bytes:
        return self.body

    def raise_for_status(self) -> None:
        if not self.ok:
            url = URL("http://agent:1")
            request_info = RequestInfo(url, "POST", CIMultiDictProxy(CIMultiDict()), url)
            raise ClientResponseError(request_info=request_info, history=(), status=self.status, message="error")


Handler = Callable[[dict], Awaitable[_HTTPResponse]]


class _Client(ServerClient):
    """Answers each path with a handler and records every call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    handlers: dict[str, Handler]
    calls: list[tuple[str, dict]] = []

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> _HTTPResponse:
        body = kwargs.get("json")
        if isinstance(body, BaseModel):
            # The real client sends a model as model_dump(exclude_unset=True).
            body = orjson.loads(orjson.dumps(body.model_dump(mode="json", exclude_unset=True)))
        kwargs["json"] = body
        self.calls.append((url_path, kwargs))
        key = "/v1/responses" if url_path.endswith("/v1/responses") else url_path
        return await self.handlers[key](body)

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:2"


def _message(text: str) -> dict:
    return NeMoGymResponse(
        id=f"response-{text}",
        created_at=0,
        model="model",
        object="response",
        output=[
            {
                "type": "message",
                "role": "assistant",
                "id": "message",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    ).model_dump(mode="json")


async def _seeded(body: dict) -> _HTTPResponse:
    capabilities = {"mode": "resume_harness_session", "observations": ["trajectory", "timing"]}
    reply = {"agent_session_id": body["agent_session_id"]}
    if "continuation" in body:
        reply["capabilities"] = capabilities
    return _HTTPResponse(reply, cookies={"agent": "agent-cookie"})


async def _closed(body: dict) -> _HTTPResponse:
    observations = AgentObservationBundle(source="test", records=[]).model_dump(mode="json")
    return _HTTPResponse({"agent_session_id": body["agent_session_id"], "agent_observations": observations})


def _replies(*texts: str) -> Handler:
    remaining = list(texts)

    async def reply(_: dict) -> _HTTPResponse:
        return _HTTPResponse(_message(remaining.pop(0)))

    return reply


class _Amount(BaseModel):
    amount: int


class _Counter:
    def __init__(self) -> None:
        self.total = 0

    @episode_tool
    async def add(self, arguments: _Amount) -> dict[str, int]:
        """Add an amount to the episode's counter."""
        self.total += arguments.amount
        return {"total": self.total}


@dataclass
class _Episode:
    request: _Request
    user_turns: list[UserMessage | Stop]
    seen: list[TurnRecord] = field(default_factory=list)
    counter: _Counter = field(default_factory=_Counter)


class _Server(BaseUserSimulationEnvironmentServer[_Request, _Response, _Episode]):
    request_model = _Request
    response_model = _Response
    script: list[UserMessage | Stop] = []

    async def start_episode(self, request: _Request, cleanup: CleanupContext) -> _Episode:
        return _Episode(request=request, user_turns=list(self.script))

    async def responses_create_params(self, episode: _Episode) -> NeMoGymResponseCreateParamsNonStreaming:
        return NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {"role": "system", "content": "Answer briefly."},
                {"role": "user", "content": episode.request.task.task_input.question},
            ]
        )

    async def next_user_turn(self, episode: _Episode, turn: TurnRecord) -> UserMessage | Stop:
        episode.seen.append(turn)
        return episode.user_turns.pop(0)

    async def finish_episode(self, episode: _Episode, transcript: Transcript) -> _Response:
        return _Response(
            episode_id=episode.request.episode_id,
            task_id=episode.request.task.task_id,
            result=UserSimulationResult(reward=float(len(episode.seen)), **transcript.result_fields()),
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        return AggregateMetrics()


class _ToolServer(_Server):
    async def episode_tools(self, episode: _Episode) -> TypedToolset:
        return TypedToolset(episode.counter)


def _server(
    handlers: dict[str, Handler],
    script: list[UserMessage | Stop],
    *,
    max_turns: int = 4,
    turn_timeout_seconds: float = 5,
    interaction_timeout_seconds: float | None = None,
    server_class: type[_Server] = _Server,
) -> tuple[_Server, _Client]:
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(
            {"agent": {"responses_api_agents": {"test": {"host": "agent", "port": 1, "entrypoint": "app.py"}}}}
        ),
        handlers={"/v1/agent_sessions": _seeded, "/v1/agent_sessions/close": _closed} | handlers,
        calls=[],
    )
    config = UserSimulationEnvironmentServerConfig(
        name="environment",
        host="environment",
        port=2,
        entrypoint="app.py",
        agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
        max_turns=max_turns,
        turn_timeout_seconds=turn_timeout_seconds,
        interaction_timeout_seconds=interaction_timeout_seconds,
        cleanup_timeout_seconds=5,
    )
    server = server_class(config=config, server_client=client)
    server.script = script
    return server, client


def _request() -> _Request:
    return _Request(
        episode_id=EpisodeId(rollout_id="rollout", attempt=1),
        task=MaterializedTask(task_id=TaskId(taskset="test", task_id="task"), task_input=_TaskInput(question="Hi?")),
    )


def _paths(client: _Client) -> list[str]:
    return [path for path, _ in client.calls]


async def test_each_turn_sends_only_the_new_user_message_until_the_user_side_stops() -> None:
    server, client = _server(
        {"/v1/responses": _replies("Which year?", "Here it is.")},
        [UserMessage(content="2024."), Stop(reason="answered")],
    )

    response = await server.run_request(_request())

    assert response.failure is None
    assert response.result.stop_reason == "answered"
    assert response.result.reward == 2.0
    assert [turn.response.output_text for turn in response.result.turns] == ["Which year?", "Here it is."]
    assert _paths(client) == [
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a1/v1/responses",
        "/ng-rollout/rollout-a1/v1/responses",
        "/v1/agent_sessions/close",
    ]
    seed = AgentSeedSessionRequest.model_validate(client.calls[0][1]["json"])
    assert seed.continuation.mode == "resume_harness_session"
    assert seed.tool_accesses == [] and response.result.tool_calls == []
    first, second = (kwargs for path, kwargs in client.calls if path.endswith("/v1/responses"))
    assert [message["content"] for message in first["json"]["input"]] == ["Answer briefly.", "Hi?"]
    assert second["json"]["input"] == [{"role": "user", "content": "2024."}]
    assert [first["headers"], second["headers"]] == [{"x-ng-agent-activation": "1"}, {"x-ng-agent-activation": "2"}]
    assert second["cookies"] == {"agent": "agent-cookie"}
    assert float(second["json"]["metadata"]["agent_timeout_sec"]) <= 5
    assert response.result.ng_agent_observations.source == "test"


async def test_the_agent_calls_the_episode_tools_that_its_seed_grants_until_the_episode_ends() -> None:
    seeds: list[dict] = []
    apps = {}

    async def seeded(body: dict) -> _HTTPResponse:
        seeds.append(body)
        return await _seeded(body)

    async def call_add(amount: int) -> httpx.Response:
        (grant,) = seeds[0]["tool_accesses"]
        transport = httpx.ASGITransport(app=apps["app"])
        async with httpx.AsyncClient(transport=transport, base_url=grant["base_url"]) as http:
            return await http.post("/add", json={"amount": amount}, headers=grant["headers"])

    async def respond(_: dict) -> _HTTPResponse:
        reply = await call_add(2)
        assert (reply.status_code, reply.json()) == (200, {"total": 2})
        return _HTTPResponse(_message("Added."))

    server, _ = _server(
        {"/v1/agent_sessions": seeded, "/v1/responses": respond}, [Stop(reason="done")], server_class=_ToolServer
    )
    apps["app"] = server.setup_webserver()

    response = await server.run_request(_request())

    assert response.failure is None and response.result.stop_reason == "done"
    (grant,) = seeds[0]["tool_accesses"]
    assert grant["kind"] == "direct_http" and grant["base_url"].rstrip("/") == "http://environment:2"
    assert [(call.name, call.arguments, call.output) for call in response.result.tool_calls] == [
        ("add", {"amount": 2}, {"total": 2})
    ]
    # The grant ends with the episode.
    assert (await call_add(1)).status_code == 410


async def test_the_user_side_reads_the_last_turn_before_max_turns_ends_the_conversation() -> None:
    server, _ = _server(
        {"/v1/responses": _replies("one", "two")},
        [UserMessage(content="again"), UserMessage(content="unused")],
        max_turns=2,
    )

    response = await server.run_request(_request())

    assert response.result.stop_reason == "max_turns"
    # next_user_turn graded both turns, the last one included.
    assert response.result.reward == 2.0


async def test_one_turn_episodes_do_not_require_session_continuation() -> None:
    server, client = _server({"/v1/responses": _replies("done")}, [UserMessage(content="unused")], max_turns=1)

    response = await server.run_request(_request())

    assert response.result.stop_reason == "max_turns"
    assert "continuation" not in client.calls[0][1]["json"]


async def test_an_agent_without_continuation_fails_the_episode_at_seed() -> None:
    async def refuse(_: dict) -> _HTTPResponse:
        return _HTTPResponse(
            {"detail": "Agent does not support the required multi-turn continuation mode"}, status=422
        )

    server, client = _server({"/v1/agent_sessions": refuse}, [])

    response = await server.run_request(_request())

    assert response.failure.stage == "seed"
    assert response.failure.terminal is True
    # The session close was registered before the seed, so cleanup still closes it.
    assert _paths(client) == ["/v1/agent_sessions", "/v1/agent_sessions/close"]


async def test_a_failed_turn_keeps_the_completed_turns_and_the_agent_evidence() -> None:
    replies = _replies("first")

    async def respond(body: dict) -> _HTTPResponse:
        if body["input"][0]["content"] == "again":
            return _HTTPResponse({"detail": "model unavailable"}, status=503)
        return await replies(body)

    server, client = _server({"/v1/responses": respond}, [UserMessage(content="again")])

    response = await server.run_request(_request())

    assert response.failure.stage == "agent"
    assert response.failure.terminal is False
    assert [turn.turn for turn in response.failure.turns] == [1]
    assert response.failure.agent_observations.source == "test"
    assert _paths(client)[-1] == "/v1/agent_sessions/close"


async def test_a_turn_that_exceeds_its_timeout_fails_the_episode_for_a_retry() -> None:
    async def hang(_: dict) -> _HTTPResponse:
        await asyncio.sleep(10)
        raise AssertionError("unreachable")

    server, _ = _server({"/v1/responses": hang}, [], turn_timeout_seconds=0.05)

    response = await server.run_request(_request())

    assert response.failure.stage == "agent"
    assert response.failure.terminal is False
    assert "timed out" in response.failure.failure_reason


async def test_the_interaction_deadline_closes_the_agent_and_grades_the_episode() -> None:
    replies = _replies("first")

    async def respond(body: dict) -> _HTTPResponse:
        if body["input"][0]["content"] == "again":
            await asyncio.sleep(10)
        return await replies(body)

    server, client = _server(
        {"/v1/responses": respond},
        [UserMessage(content="again")],
        turn_timeout_seconds=5,
        interaction_timeout_seconds=0.2,
    )

    response = await server.run_request(_request())

    assert response.failure is None
    assert response.result.stop_reason == "interaction_timeout"
    assert [turn.turn for turn in response.result.turns] == [1]
    assert _paths(client)[-1] == "/v1/agent_sessions/close"


async def test_a_failed_agent_close_fails_the_episode_at_cleanup() -> None:
    async def fail_close(_: dict) -> _HTTPResponse:
        return _HTTPResponse({"detail": "harness still running"}, status=500)

    server, _ = _server(
        {"/v1/responses": _replies("done"), "/v1/agent_sessions/close": fail_close}, [Stop(reason="answered")]
    )

    response = await server.run_request(_request())

    assert response.failure.stage == "cleanup"
    assert [turn.turn for turn in response.failure.turns] == [1]


async def test_a_failed_verification_keeps_the_turns_after_the_agent_is_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, client = _server({"/v1/responses": _replies("done")}, [Stop(reason="answered")])

    async def fail_verification(self: _Server, episode: _Episode, transcript: Transcript) -> _Response:
        raise self.failure(TimeoutError("verifier lost"), stage="verification")

    monkeypatch.setattr(_Server, "finish_episode", fail_verification)

    response = await server.run_request(_request())

    assert (response.failure.stage, response.failure.terminal) == ("verification", False)
    assert [turn.turn for turn in response.failure.turns] == [1]
    assert response.failure.agent_observations.source == "test"
    assert _paths(client).count("/v1/agent_sessions/close") == 1


async def test_token_capture_moves_the_activation_onto_the_training_capture_path() -> None:
    server, client = _server({"/v1/responses": _replies("done")}, [Stop(reason="answered")])
    client.global_config_dict = OmegaConf.create(
        {
            "agent": {
                "responses_api_agents": {
                    "test": {"host": "agent", "port": 1, "entrypoint": "app.py", "token_id_capture": True}
                }
            },
            "token_id_capture": {"enabled": True},
        }
    )

    await server.run_request(_request())

    assert _paths(client)[1] == "/ng-rollout/rollout-a1/training-token-capture/v1/responses"


@pytest.mark.parametrize("value", [0, -1])
def test_max_turns_must_be_positive(value: int) -> None:
    with pytest.raises(ValueError, match="max_turns"):
        UserSimulationEnvironmentServerConfig(
            name="environment",
            host="environment",
            port=2,
            entrypoint="app.py",
            agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
            max_turns=value,
            turn_timeout_seconds=5,
            cleanup_timeout_seconds=5,
        )
