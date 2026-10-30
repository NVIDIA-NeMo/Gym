# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any, ClassVar
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import Body, FastAPI
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, ValidationError

from nemo_gym.base_environment_server import CleanupContext
from nemo_gym.base_environment_server_with_tools import (
    BaseEnvironmentServerWithTools,
    BaseEnvironmentServerWithToolsConfig,
    ServedTools,
)
from nemo_gym.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.episode_tokens import EPISODE_TOKEN_HEADER, EpisodeTokens
from nemo_gym.episode_tools import ToolResult, ToolSpec, TypedToolset, episode_tool
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeId, MaterializedTask, TaskId
from nemo_gym.server_utils import ServerClient, refuse_unsupported_num_workers
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPToolAccess


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

    async def run_episode(self, request: _Request, cleanup: CleanupContext) -> _Response:
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


def _app(
    agent: Agent | None = None, server_class: type[_Server] | None = None, **overrides: Any
) -> tuple[_Server, FastAPI]:
    config = BaseEnvironmentServerWithToolsConfig(
        name="counter_environment",
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        cleanup_timeout_seconds=1,
        **overrides,
    )
    client = MagicMock(spec=ServerClient)
    client._resolve_base_url.return_value = BASE_URL
    server_type = type("_Agented", (server_class or _Server,), {"agent": staticmethod(agent)} if agent else {})
    server = server_type(config=config, server_client=client)
    return server, server.setup_webserver()


def _server(agent: Agent | None = None, **overrides: Any) -> tuple[_Server, httpx.AsyncClient]:
    server, app = _app(agent, **overrides)
    return server, httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE_URL)


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


def test_several_workers_route_each_episode_to_the_worker_that_runs_it() -> None:
    """Routing is tested in test_episode_tools_session_routing.py; here, the server takes several workers."""
    server, _ = _server(num_workers=4)

    refuse_unsupported_num_workers(server)

    assert server.routes_sessions_to_owner
    assert server.claimed_session_routing()["session_header"] == EPISODE_TOKEN_HEADER


class _Sandbox:
    """A resource whose release takes a moment, as stopping a remote sandbox does."""

    def __init__(self) -> None:
        self.stops_started = 0
        self.stopped = False

    async def stop(self) -> None:
        self.stops_started += 1
        await asyncio.sleep(0.05)
        self.stopped = True


def _sandbox_server(*, hold: bool, **overrides: Any) -> tuple[_Server, _Sandbox, asyncio.Event]:
    """A server whose episode registers a sandbox, then returns or, with ``hold``, runs until cancelled."""
    sandbox, started = _Sandbox(), asyncio.Event()

    async def run_episode(self: _Server, request: _Request, cleanup: CleanupContext) -> _Response:
        cleanup.register_cleanup("sandbox", sandbox.stop)
        started.set()
        if hold:
            await asyncio.Event().wait()
        return _Response(episode_id=request.episode_id, task_id=request.task.task_id, result="done")

    server, _ = _server(**overrides)
    type(server).run_episode = run_episode
    return server, sandbox, started


@pytest.mark.parametrize("transports", [["direct_http"], ["mcp"]], ids=["direct-http", "mcp-mounted"])
def test_shutdown_finishes_the_cleanups_of_a_running_episode(transports: list[str]) -> None:
    """With /mcp mounted, the MCP session manager's lifespan and the shutdown sweep both run."""

    async def scenario() -> _Sandbox:
        server, sandbox, started = _sandbox_server(hold=True, tool_transports=transports)
        app = server.setup_webserver()
        episode = asyncio.create_task(server.run_request(_request()))
        await started.wait()

        async with app.router.lifespan_context(app):
            pass
        assert sandbox.stopped

        episode.cancel()
        with pytest.raises(asyncio.CancelledError):
            await episode
        return sandbox

    sandbox = asyncio.run(scenario())

    assert sandbox.stops_started == 1


def test_shutdown_finishes_a_cleanup_that_a_second_cancellation_interrupted() -> None:
    """Shutdown cancels a running /run and does not wait for it; the event loop's teardown cancels it again."""

    async def scenario() -> _Sandbox:
        server, sandbox, started = _sandbox_server(hold=True)
        episode = asyncio.create_task(server.run_request(_request()))
        await started.wait()

        episode.cancel()
        await asyncio.sleep(0.01)
        assert sandbox.stops_started == 1 and not sandbox.stopped
        episode.cancel()
        with pytest.raises(asyncio.CancelledError):
            await episode
        assert not sandbox.stopped

        await server.finish_running_cleanups()
        return sandbox

    sandbox = asyncio.run(scenario())

    assert sandbox.stopped and sandbox.stops_started == 2


def test_an_episode_that_finished_is_not_cleaned_up_again_at_shutdown() -> None:
    async def scenario() -> _Sandbox:
        server, sandbox, _ = _sandbox_server(hold=False)
        await server.run_request(_request())
        await server.finish_running_cleanups()
        return sandbox

    sandbox = asyncio.run(scenario())

    assert sandbox.stopped and sandbox.stops_started == 1


def test_a_subclass_implements_run_episode_not_run() -> None:
    class _OnlyRun(BaseEnvironmentServerWithTools[_Request, _Response]):
        request_model = _Request
        response_model = _Response

        async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
            raise AssertionError("not reached")

        async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
            return AggregateMetrics()

    config = _server()[0].config
    with pytest.raises(TypeError, match="run_episode"):
        _OnlyRun(config=config, server_client=MagicMock(spec=ServerClient))


# MCP: the same episode tools for agents that take only MCP grants.


@asynccontextmanager
async def _mcp_session(app: FastAPI, headers: dict[str, str]) -> AsyncIterator[ClientSession]:
    """An MCP client session against ``app``'s /mcp, in-process."""

    def client_factory(headers=None, timeout=None, auth=None) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=app)
        return httpx.AsyncClient(transport=transport, base_url=BASE_URL, headers=headers, timeout=timeout, auth=auth)

    async with streamablehttp_client(f"{BASE_URL}/mcp", headers=headers, httpx_client_factory=client_factory) as (
        read,
        write,
        _,
    ):
        async with ClientSession(read, write) as session:
            await session.initialize()
            yield session


def _mcp_headers(served: ServedTools) -> dict[str, str]:
    (grant,) = [access for access in served.tool_accesses if isinstance(access, MCPToolAccess)]
    return dict(grant.connection.headers)


def _text(result: Any) -> str:
    return result.content[0].text


async def _run_with_lifespan(app: FastAPI, *runs: Awaitable[_Response]) -> list[_Response]:
    """Run episodes while the app's lifespan, which runs the MCP endpoint, is up."""
    async with app.router.lifespan_context(app):
        return await asyncio.gather(*runs)


@pytest.mark.parametrize(
    "transports, granted",
    [
        pytest.param(["direct_http"], [DirectHTTPToolAccess], id="direct-http"),
        pytest.param(["mcp"], [MCPToolAccess], id="mcp"),
        pytest.param(["mcp", "direct_http"], [MCPToolAccess, DirectHTTPToolAccess], id="both"),
    ],
)
def test_the_episode_is_granted_over_each_configured_transport(transports: list[str], granted: list[type]) -> None:
    seen: list[ServedTools] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        seen.append(served)

    server, app = _app(agent, tool_transports=transports)
    asyncio.run(_run_with_lifespan(app, server.run_request(_request())))

    accesses = seen[0].tool_accesses
    assert [type(access) for access in accesses] == granted
    tokens = {
        (access.connection.headers if isinstance(access, MCPToolAccess) else access.headers)[EPISODE_TOKEN_HEADER]
        for access in accesses
    }
    assert len(tokens) == 1
    for access in accesses:
        if isinstance(access, MCPToolAccess):
            assert access.name == "counter_environment" and str(access.connection.url) == f"{BASE_URL}/mcp"


def test_a_server_without_mcp_serves_no_mcp_endpoint() -> None:
    _, app = _app()

    assert "/mcp" not in {getattr(route, "path", None) for route in app.routes}


def test_mcp_lists_the_episodes_tools_and_shares_its_calls_with_direct_http() -> None:
    seen: dict[str, Any] = {}

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        async with _mcp_session(app, _mcp_headers(served)) as session:
            seen["tools"] = (await session.list_tools()).tools
            seen["mcp"] = await session.call_tool("add", {"amount": 2})
        direct = [access for access in served.tool_accesses if isinstance(access, DirectHTTPToolAccess)][0]
        http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE_URL)
        seen["http"] = (await http.post("/add", json={"amount": 3}, headers=dict(direct.headers))).json()
        seen["calls"] = list(served.calls)

    server, app = _app(agent, tool_transports=["mcp", "direct_http"])
    (response,) = asyncio.run(_run_with_lifespan(app, server.run_request(_request())))

    assert {tool.name: tool.description for tool in seen["tools"]} == {
        spec.name: spec.description for spec in TypedToolset(_Counter()).specs()
    }
    assert seen["mcp"].isError is False and seen["mcp"].structuredContent == {"total": 2}
    assert seen["http"] == {"total": 5}
    assert [(call.name, call.arguments) for call in seen["calls"]] == [("add", {"amount": 2}), ("add", {"amount": 3})]
    assert response.result == "total=5 calls=2"


@pytest.mark.parametrize(
    "headers",
    [pytest.param({}, id="missing"), pytest.param({EPISODE_TOKEN_HEADER: "garbage"}, id="garbage")],
)
def test_mcp_without_the_episodes_token_lists_no_tools_and_refuses_calls(headers: dict[str, str]) -> None:
    seen: dict[str, Any] = {}

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        async with _mcp_session(app, headers) as session:
            seen["tools"] = (await session.list_tools()).tools
            seen["call"] = await session.call_tool("add", {"amount": 2})

    server, app = _app(agent, tool_transports=["mcp"])
    (response,) = asyncio.run(_run_with_lifespan(app, server.run_request(_request())))

    assert seen["tools"] == []
    assert seen["call"].isError is True and "needs a valid" in _text(seen["call"])
    assert response.result == "total=0 calls=0"


def test_an_mcp_call_after_the_episode_ends_is_refused() -> None:
    kept: dict[str, Any] = {}

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        kept["headers"] = _mcp_headers(served)

    server, app = _app(agent, tool_transports=["mcp"])

    async def scenario() -> Any:
        async with app.router.lifespan_context(app):
            await server.run_request(_request())
            async with _mcp_session(app, kept["headers"]) as session:
                return await session.call_tool("add", {"amount": 1})

    late = asyncio.run(scenario())

    assert late.isError is True and "is not running" in _text(late)


class _TwoToolsets(_Server):
    """Episode 4-0 gets a toolset of its own, as a toolset built from the task does."""

    async def run_episode(self, request: _Request, cleanup: CleanupContext) -> _Response:
        counter = _Counter()
        toolset = _Specs("lookup") if request.episode_id.rollout_id == "4-0" else TypedToolset(counter)
        served = await self.serve_tools(request, toolset, cleanup)
        await type(self).agent(self, counter, served)
        return _Response(episode_id=request.episode_id, task_id=request.task.task_id, result="")


def test_each_episode_lists_only_its_own_tools_over_mcp() -> None:
    granted: list[ServedTools] = []
    listed: list[list[str]] = []
    both_open = asyncio.Event()

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        granted.append(served)
        if len(granted) == 2:
            both_open.set()
        await both_open.wait()
        async with _mcp_session(app, _mcp_headers(served)) as session:
            listed.append(sorted(tool.name for tool in (await session.list_tools()).tools))

    server, app = _app(agent, server_class=_TwoToolsets, tool_transports=["mcp"])
    asyncio.run(_run_with_lifespan(app, server.run_request(_request("3-0")), server.run_request(_request("4-0"))))

    assert sorted(listed) == [["add", "fail", "slow_add"], ["lookup"]]


def test_an_mcp_grant_refuses_tool_names_mcp_clients_reject() -> None:
    server, _ = _app(tool_transports=["mcp"])
    type(server).toolset = staticmethod(lambda counter: _Specs("lookup.v2"))

    response = asyncio.run(server.run_request(_request()))

    assert response.failure is not None and "lookup.v2" in response.failure.failure_reason


def test_tool_transports_cannot_repeat() -> None:
    with pytest.raises(ValidationError, match="must not repeat"):
        _app(tool_transports=["mcp", "mcp"])
