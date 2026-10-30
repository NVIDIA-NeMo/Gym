# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, ClassVar
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import Body
from pydantic import BaseModel

from nemo_gym.base_environment_server import BaseEnvironmentServerConfig, CleanupContext
from nemo_gym.base_environment_server_with_tools import BaseEnvironmentServerWithTools, ServedTools
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.episode_tokens import EPISODE_TOKEN_HEADER, EpisodeTokens
from nemo_gym.episode_tools import ToolResult, ToolSpec, TypedToolset, episode_tool
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeId, MaterializedTask, TaskId
from nemo_gym.server_utils import ServerClient, refuse_unsupported_num_workers
from nemo_gym.tool_access import DirectHTTPToolAccess


BASE_URL = "http://127.0.0.1:1234"


class _TaskInput(BaseModel):
    value: str


class _Request(BaseEpisodeRequest[_TaskInput]):
    pass


class _Response(BaseEpisodeResponse[str]):
    pass


class _Amount(BaseModel):
    amount: int


class _Nothing(BaseModel):
    pass


class _Counter:
    """An episode's state, with the tools that change it."""

    def __init__(self) -> None:
        self.total = 0
        self.running = 0
        self.most_running = 0
        self.release = asyncio.Event()

    @episode_tool
    async def add(self, arguments: _Amount) -> dict[str, int]:
        """Add an amount to the episode's total."""
        self.total += arguments.amount
        return {"total": self.total}

    @episode_tool
    async def slow_add(self, arguments: _Amount) -> dict[str, int]:
        """Add an amount after the test releases the call."""
        self.running += 1
        self.most_running = max(self.most_running, self.running)
        await self.release.wait()
        self.running -= 1
        return await self.add(arguments)

    @episode_tool
    async def fail(self, arguments: _Nothing) -> None:
        """Raise, as a broken tool does."""
        raise RuntimeError("broken")


class _Specs:
    """A toolset that lists fixed specs, as a toolset built from the task does."""

    def __init__(self, *names: str) -> None:
        self._specs = [ToolSpec(name=name, description="d", parameters={"type": "object"}) for name in names]

    def specs(self) -> list[ToolSpec]:
        return self._specs

    async def call(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return ToolResult(output=name)


Agent = Callable[["_Server", _Counter, ServedTools], Awaitable[None]]


class _Server(BaseEnvironmentServerWithTools[_Request, _Response]):
    request_model = _Request
    response_model = _Response
    agent: ClassVar[Agent | None] = None
    toolset: ClassVar[Callable[[_Counter], Any]] = TypedToolset

    async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
        counter = _Counter()
        served = await self.serve_tools(request, type(self).toolset(counter), cleanup)
        if type(self).agent is not None:
            await type(self).agent(self, counter, served)
        return _Response(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=f"total={counter.total} calls={len(served.calls)}",
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return AggregateMetrics()


def _server(agent: Agent | None = None, **overrides: Any) -> tuple[_Server, httpx.AsyncClient]:
    config = BaseEnvironmentServerConfig(
        name="counter_environment",
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        cleanup_timeout_seconds=1,
        **overrides,
    )
    client = MagicMock(spec=ServerClient)
    client._resolve_base_url.return_value = BASE_URL
    server_type = type("_Agented", (_Server,), {"agent": staticmethod(agent)} if agent else {})
    server = server_type(config=config, server_client=client)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=server.setup_webserver()), base_url=BASE_URL)
    return server, http


def _request(rollout_id: str = "3-0") -> _Request:
    return _Request(
        episode_id=EpisodeId(rollout_id=rollout_id),
        task=MaterializedTask(task_id=TaskId(taskset="test:default", task_id="task"), task_input=_TaskInput(value="")),
    )


def _headers(served: ServedTools) -> dict[str, str]:
    (grant,) = served.tool_accesses
    return dict(grant.headers)


def test_the_agent_calls_the_episodes_tools_while_it_runs() -> None:
    seen: dict[str, Any] = {}

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        seen["grant"] = served.tool_accesses
        for amount in (2, 3):
            reply = await http.post("/add", json={"amount": amount}, headers=_headers(served))
            seen.setdefault("replies", []).append((reply.status_code, reply.json()))

    server, http = _server(agent)
    response = asyncio.run(server.run_request(_request()))

    assert response.result == "total=5 calls=2"
    assert seen["replies"] == [(200, {"total": 2}), (200, {"total": 5})]
    (grant,) = seen["grant"]
    assert isinstance(grant, DirectHTTPToolAccess) and str(grant.base_url).rstrip("/") == BASE_URL
    assert grant.name == "counter_environment.direct_http" and EPISODE_TOKEN_HEADER in grant.headers


def test_each_call_is_recorded_in_order() -> None:
    recorded: list[Any] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        await http.post("/add", json={"amount": 2}, headers=_headers(served))
        await http.post("/add", json={"amount": "two"}, headers=_headers(served))
        recorded.extend(served.calls)

    server, http = _server(agent)
    asyncio.run(server.run_request(_request()))

    assert [(call.name, call.arguments, call.is_error) for call in recorded] == [
        ("add", {"amount": 2}, False),
        ("add", {"amount": "two"}, True),
    ]
    assert recorded[0].output == {"total": 2}


def test_a_call_after_the_episode_ends_is_gone() -> None:
    kept: dict[str, Any] = {}

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        kept["headers"] = _headers(served)

    server, http = _server(agent)

    async def scenario() -> httpx.Response:
        await server.run_request(_request())
        return await http.post("/add", json={"amount": 1}, headers=kept["headers"])

    assert asyncio.run(scenario()).status_code == 410


@pytest.mark.parametrize(
    "token, status",
    [
        pytest.param(None, 401, id="missing"),
        pytest.param("garbage", 401, id="garbage"),
        pytest.param(EpisodeTokens("another-server").mint(EpisodeId(rollout_id="3-0"))[0], 401, id="other-server"),
        pytest.param("forged", 403, id="guessed-nonce"),
        pytest.param("other-episode", 410, id="other-episode"),
    ],
)
def test_a_call_without_the_episodes_token_is_refused(token: str | None, status: int) -> None:
    replies: list[httpx.Response] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        key = server.get_session_middleware_key()
        sent = {
            "forged": EpisodeTokens(key).mint(EpisodeId(rollout_id="3-0"))[0],
            "other-episode": EpisodeTokens(key).mint(EpisodeId(rollout_id="4-0"))[0],
        }.get(token, token)
        replies.append(
            await http.post("/add", json={"amount": 1}, headers={EPISODE_TOKEN_HEADER: sent} if sent else {})
        )

    server, http = _server(agent)
    response = asyncio.run(server.run_request(_request()))

    assert replies[0].status_code == status
    assert response.result == "total=0 calls=0"


@pytest.mark.parametrize(
    "path, body, status, detail",
    [
        pytest.param("/subtract", {"amount": 1}, 404, "no tool 'subtract'", id="unknown-tool"),
        pytest.param("/add", {"amount": "two"}, 422, "amount", id="invalid-arguments"),
        pytest.param("/add", [1], 422, "JSON object", id="not-an-object"),
        pytest.param("/fail", {}, 500, "RuntimeError", id="broken-tool"),
    ],
)
def test_a_call_the_tool_cannot_run_answers_with_an_error_the_agent_sees(
    path: str, body: Any, status: int, detail: str
) -> None:
    replies: list[httpx.Response] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        replies.append(await http.post(path, json=body, headers=_headers(served)))

    server, http = _server(agent)
    asyncio.run(server.run_request(_request()))

    assert replies[0].status_code == status
    assert detail in replies[0].json()["detail"]


def test_a_call_under_a_rollout_prefix_reaches_the_tool() -> None:
    replies: list[httpx.Response] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        replies.append(await http.post("/ng-rollout/3-0/add", json={"amount": 4}, headers=_headers(served)))

    server, http = _server(agent)
    response = asyncio.run(server.run_request(_request()))

    assert replies[0].json() == {"total": 4}
    assert response.result == "total=4 calls=1"


def test_calls_to_one_episode_run_one_at_a_time() -> None:
    counters: list[_Counter] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        counters.append(counter)
        calls = [
            asyncio.create_task(http.post("/slow_add", json={"amount": 1}, headers=_headers(served))) for _ in range(3)
        ]
        await asyncio.sleep(0.05)
        counter.release.set()
        await asyncio.gather(*calls)

    server, http = _server(agent)
    response = asyncio.run(server.run_request(_request()))

    assert response.result == "total=3 calls=3"
    assert counters[0].most_running == 1


def test_two_episodes_have_separate_tools_and_tokens() -> None:
    tokens: dict[str, dict[str, str]] = {}
    both_open = asyncio.Event()

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        tokens[str(len(tokens))] = _headers(served)
        if len(tokens) == 2:
            both_open.set()
        await both_open.wait()
        await http.post(
            "/add", json={"amount": 10 if _headers(served) == tokens["0"] else 1}, headers=_headers(served)
        )

    server, http = _server(agent)

    async def scenario() -> list[_Response]:
        return await asyncio.gather(server.run_request(_request("3-0")), server.run_request(_request("4-0")))

    first, second = asyncio.run(scenario())

    assert sorted([first.result, second.result]) == ["total=1 calls=1", "total=10 calls=1"]


@pytest.mark.parametrize(
    "names, error",
    [
        pytest.param(("run",), "names of the server's routes: run", id="shadows-run"),
        pytest.param(("aggregate_metrics", "add"), "aggregate_metrics", id="shadows-aggregate-metrics"),
        pytest.param(("add", "add"), "unique", id="duplicate"),
    ],
)
def test_a_toolset_that_conflicts_fails_the_episode(names: tuple[str, ...], error: str) -> None:
    server, _ = _server()
    type(server).toolset = staticmethod(lambda counter: _Specs(*names))

    response = asyncio.run(server.run_request(_request()))

    assert response.failure is not None and error in response.failure.failure_reason


def test_one_episode_serves_its_tools_once() -> None:
    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        await server.serve_tools(_request(), TypedToolset(counter), CleanupContext(_request().episode_id, 1))

    server, _ = _server(agent)
    response = asyncio.run(server.run_request(_request()))

    assert response.failure is not None and "already serves its tools" in response.failure.failure_reason


def test_several_workers_are_refused_while_calls_reach_any_worker() -> None:
    server, _ = _server(num_workers=2)

    with pytest.raises(ValueError, match="needs num_workers=1"):
        refuse_unsupported_num_workers(server)
