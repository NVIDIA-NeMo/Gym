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

from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.core.episode_context import current_rollout_id
from nemo_gym.core.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeId, MaterializedTask, TaskId
from nemo_gym.server_runtime.base_environment_server import (
    BaseEnvironmentServerWithTools,
    BaseEnvironmentServerWithToolsConfig,
    CleanupContext,
    ServedTools,
    handled_episode_failure,
)
from nemo_gym.server_runtime.base_server import refuse_unsupported_num_workers
from nemo_gym.server_runtime.episode_tokens import EPISODE_TOKEN_HEADER, EpisodeTokens, episode_token_secret
from nemo_gym.server_runtime.episode_tools import ToolResult, ToolSpec, TypedToolset, episode_tool
from nemo_gym.server_runtime.request_handling.request_context import install_request_context_isolation
from nemo_gym.server_runtime.tool_access import (
    TOOL_CALL_ID_HEADER,
    TOOL_ERROR_HEADER,
    DirectHTTPToolAccess,
    MCPToolAccess,
)


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

    @episode_tool
    async def time_out(self, arguments: _Nothing) -> None:
        """Raise TimeoutError, as a tool whose own deadline passed does."""
        raise TimeoutError("the database did not answer")

    @episode_tool
    async def overdraw(self, arguments: _Amount) -> ToolResult:
        """Refuse to take more than the total, with an error result the agent handles."""
        return ToolResult(output={"detail": f"the total is {self.total}"}, is_error=True)

    @episode_tool
    async def rollout(self, arguments: _Nothing) -> str | None:
        """Return the rollout the call runs under, as a tool that calls a model would send it."""
        return current_rollout_id()

    @episode_tool
    async def not_a_number(self, arguments: _Nothing) -> float:
        """Return NaN, which JSON cannot carry."""
        return float("nan")


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
    app = server.setup_webserver()
    # As in production: a request does not inherit the context of the task that sent it, here the episode's own.
    install_request_context_isolation(app)
    return server, app


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
        pytest.param("derivable-key", 401, id="session-middleware-key"),
    ],
)
def test_a_call_without_the_episodes_token_is_refused(token: str | None, status: int) -> None:
    replies: list[httpx.Response] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        key = episode_token_secret()
        sent = {
            "forged": EpisodeTokens(key).mint(EpisodeId(rollout_id="3-0"))[0],
            "other-episode": EpisodeTokens(key).mint(EpisodeId(rollout_id="4-0"))[0],
            # The session key is derived from the server's names; tokens are signed with a random secret instead.
            "derivable-key": EpisodeTokens(server.get_session_middleware_key()).mint(EpisodeId(rollout_id="3-0"))[0],
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
        pytest.param("/not_a_number", {}, 500, "not JSON", id="non-finite-output"),
    ],
)
def test_a_call_the_tool_cannot_run_answers_with_an_error_the_agent_sees(
    path: str, body: Any, status: int, detail: str
) -> None:
    """Each such call is also recorded as an error, so the calls the grader reads include it."""
    replies: list[httpx.Response] = []
    recorded: list[Any] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        replies.append(await http.post(path, json=body, headers=_headers(served)))
        recorded.extend(served.calls)

    server, http = _server(agent)
    asyncio.run(server.run_request(_request()))

    assert replies[0].status_code == status
    assert detail in replies[0].json()["detail"]
    assert [(call.name, call.arguments, call.is_error) for call in recorded] == [(path.strip("/"), body, True)]
    assert detail in recorded[0].output["detail"]


@pytest.mark.parametrize(
    "path, body, status, flagged",
    [
        pytest.param("/add", {"amount": 2}, 200, False, id="result"),
        pytest.param("/overdraw", {"amount": 2}, 200, True, id="error-result"),
        pytest.param("/add", {"amount": "two"}, 422, True, id="failed-call"),
    ],
)
def test_a_reply_says_whether_the_call_was_an_error(path: str, body: Any, status: int, flagged: bool) -> None:
    """Direct HTTP has no field for an error result, which MCP calls ``isError``, so the reply has a header."""
    replies: list[httpx.Response] = []
    recorded: list[Any] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        replies.append(await http.post(path, json=body, headers=_headers(served)))
        recorded.extend(served.calls)

    server, http = _server(agent)
    asyncio.run(server.run_request(_request()))

    assert replies[0].status_code == status
    assert (replies[0].headers.get(TOOL_ERROR_HEADER) == "true") is flagged
    assert [call.is_error for call in recorded] == [flagged]


@pytest.mark.parametrize("path", ["/run", "/aggregate_metrics", "/ng-rollout/3-0/run", "/add/../run", "/a/b"])
def test_an_episode_token_opens_only_the_episodes_tools(path: str) -> None:
    """A model that names a tool ``run`` cannot start another episode on the server that serves its tools."""
    replies: list[httpx.Response] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        replies.append(await http.post(path, json=_request("4-0").model_dump(mode="json"), headers=_headers(served)))

    server, http = _server(agent)
    response = asyncio.run(server.run_request(_request()))

    assert replies[0].status_code == 403
    assert "only the episode's tools" in replies[0].json()["detail"]
    assert response.result == "total=0 calls=0"


@pytest.mark.parametrize(
    "path, rollout_id",
    [
        pytest.param("/rollout", "3-0", id="bare"),
        pytest.param("/ng-rollout/3-0/rollout", "3-0", id="own-prefix"),
        pytest.param("/ng-rollout/4-0/rollout", "3-0", id="another-rollouts-prefix"),
    ],
)
def test_a_tool_runs_under_the_rollout_its_token_names(path: str, rollout_id: str) -> None:
    """A tool that calls a model or a Resources Server sends the call under the episode's rollout."""
    replies: list[httpx.Response] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        replies.append(await http.post(path, json={}, headers=_headers(served)))

    server, http = _server(agent)
    asyncio.run(server.run_request(_request(rollout_id)))

    assert replies[0].json() == rollout_id


def test_a_tool_keeps_the_attempt_of_its_episode() -> None:
    replies: list[httpx.Response] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        replies.append(await http.post("/rollout", json={}, headers=_headers(served)))

    server, http = _server(agent)
    request = _request().model_copy(update={"episode_id": EpisodeId(rollout_id="3-0", attempt=2)})
    asyncio.run(server.run_request(request))

    assert replies[0].json() == request.episode_id.capture_key


def test_a_call_waiting_when_the_episode_ends_does_not_run() -> None:
    """The episode's grant ends while a call waits for the one before it; the waiting call must not run."""
    seen: dict[str, Any] = {}

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        seen["counter"], seen["calls"] = counter, served.calls
        slow = asyncio.create_task(http.post("/slow_add", json={"amount": 1}, headers=_headers(served)))
        while counter.running == 0:
            await asyncio.sleep(0.01)
        seen["queued"] = asyncio.create_task(http.post("/add", json={"amount": 10}, headers=_headers(served)))
        seen["slow"] = slow
        await asyncio.sleep(0.05)

    server, http = _server(agent)

    async def scenario() -> None:
        await server.run_request(_request())
        seen["slow_reply"] = await seen["slow"]
        seen["queued_reply"] = await seen["queued"]

    asyncio.run(scenario())

    assert seen["slow_reply"].status_code == 410 and "ended while the call ran" in seen["slow_reply"].json()["detail"]
    assert seen["queued_reply"].status_code == 410
    assert seen["counter"].total == 0
    assert [(call.name, call.output) for call in seen["calls"]] == [
        ("slow_add", {"detail": "Tool 'slow_add' was cancelled"})
    ]


def test_a_call_whose_request_is_cancelled_is_recorded_and_frees_the_episode() -> None:
    """A tool can change state before its request is cancelled, so the record must not lose the call.

    uvicorn does not cancel a request whose client disconnects; ``tool_call_timeout_seconds`` bounds such a call.
    """
    seen: dict[str, Any] = {}

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        slow = asyncio.create_task(http.post("/slow_add", json={"amount": 1}, headers=_headers(served)))
        while counter.running == 0:
            await asyncio.sleep(0.01)
        slow.cancel()
        await asyncio.gather(slow, return_exceptions=True)
        seen["next"] = await http.post("/add", json={"amount": 2}, headers=_headers(served))
        seen["calls"] = list(served.calls)

    server, http = _server(agent)
    asyncio.run(server.run_request(_request()))

    assert seen["next"].json() == {"total": 2}
    assert [(call.name, call.is_error) for call in seen["calls"]] == [("slow_add", True), ("add", False)]


def test_a_call_that_runs_too_long_is_cancelled_and_frees_the_episode() -> None:
    """Calls to one episode run one at a time, so a hung call would make the episode's next calls wait."""
    seen: dict[str, Any] = {}

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        seen["slow"] = await http.post("/slow_add", json={"amount": 1}, headers=_headers(served))
        seen["next"] = await http.post("/add", json={"amount": 2}, headers=_headers(served))
        seen["calls"] = [(call.name, call.is_error, call.output) for call in served.calls]

    server, http = _server(agent, tool_call_timeout_seconds=0.1)
    # Without the timeout, slow_add waits forever; fail instead of hanging.
    response = asyncio.run(asyncio.wait_for(server.run_request(_request()), timeout=10))

    assert seen["slow"].status_code == 504 and seen["slow"].headers.get(TOOL_ERROR_HEADER) == "true"
    assert seen["next"].json() == {"total": 2}
    assert seen["calls"] == [
        ("slow_add", True, {"detail": "Tool 'slow_add' did not finish within 0.1 seconds"}),
        ("add", False, {"total": 2}),
    ]
    assert response.result == "total=2 calls=2"


def test_a_timeout_the_tool_raises_itself_is_a_failure_of_the_tool() -> None:
    replies: list[httpx.Response] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        replies.append(await http.post("/time_out", json={}, headers=_headers(served)))

    server, http = _server(agent, tool_call_timeout_seconds=60)
    asyncio.run(server.run_request(_request()))

    assert replies[0].status_code == 500 and "TimeoutError" in replies[0].json()["detail"]


def test_a_call_records_the_id_the_agent_sent() -> None:
    recorded: list[Any] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        await http.post("/add", json={"amount": 2}, headers=_headers(served) | {TOOL_CALL_ID_HEADER: "call-1"})
        await http.post("/add", json={"amount": 3}, headers=_headers(served))
        recorded.extend(served.calls)

    server, http = _server(agent)
    asyncio.run(server.run_request(_request()))

    assert [call.call_id for call in recorded] == ["call-1", None]


def test_a_call_that_repeats_the_last_one_runs_again() -> None:
    """A model can give two calls the same ID, and only the agent knows whether it resent one, so every call runs."""
    seen: dict[str, Any] = {}

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        headers = _headers(served) | {TOOL_CALL_ID_HEADER: "call-1"}
        replies = [await http.post("/add", json={"amount": 2}, headers=headers) for _ in range(2)]
        seen["replies"] = [reply.json() for reply in replies]
        seen["calls"] = [call.call_id for call in served.calls]

    server, http = _server(agent)
    response = asyncio.run(server.run_request(_request()))

    assert seen["replies"] == [{"total": 2}, {"total": 4}]
    assert seen["calls"] == ["call-1", "call-1"]
    assert response.result == "total=4 calls=2"


def test_a_server_that_exports_checkpoints_is_refused() -> None:
    with pytest.raises(TypeError, match="checkpoint_mode='exported'"):
        type("_Exported", (_Server,), {"checkpoint_mode": "exported"})


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
    assert server.owned_session_routing()["session_header"] == EPISODE_TOKEN_HEADER


def test_with_mcp_mounted_shutdown_still_finishes_the_cleanups_of_a_running_episode() -> None:
    """The app's lifespan runs both the MCP session manager and the base's shutdown sweep."""
    released: list[str] = []
    started = asyncio.Event()

    async def run(self: _Server, request: _Request, cleanup: CleanupContext) -> _Response:
        async def release() -> None:
            released.append(request.episode_id.capture_key)

        cleanup.register_cleanup("sandbox", release)
        started.set()
        await asyncio.Event().wait()

    server, _ = _server(tool_transports=["mcp"])
    type(server).run = run
    app = server.setup_webserver()

    async def scenario() -> None:
        episode = asyncio.create_task(server.run_request(_request()))
        await started.wait()
        async with app.router.lifespan_context(app):
            pass
        assert released == ["3-0"]
        episode.cancel()
        await asyncio.gather(episode, return_exceptions=True)

    asyncio.run(scenario())

    assert released == ["3-0"]


def test_a_handled_failure_in_a_tool_fails_the_episode() -> None:
    """A sandbox that crashes during a tool call is an environment failure, not the agent's."""
    replies: list[httpx.Response] = []

    class _Crashing(_Counter):
        @episode_tool
        async def execute(self, arguments: _Nothing) -> None:
            """Run a command in a sandbox that has crashed."""
            raise handled_episode_failure(terminal=False, failure_reason="The episode's sandbox crashed")

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        replies.append(await http.post("/execute", json={}, headers=_headers(served)))
        replies.append(await http.post("/execute", json={}, headers=_headers(served)))

    server, http = _server(agent)
    type(server).toolset = staticmethod(lambda counter: TypedToolset(_Crashing()))
    response = asyncio.run(server.run_request(_request()))

    assert [reply.status_code for reply in replies] == [500, 500]
    assert response.failure is not None and response.failure.terminal is False
    assert "sandbox crashed" in response.failure.failure_reason


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
    """With one transport its grant is required; with both, each is optional, so each agent takes the one it can use."""
    seen: list[ServedTools] = []

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        seen.append(served)

    server, app = _app(agent, tool_transports=transports)
    asyncio.run(_run_with_lifespan(app, server.run_request(_request())))

    accesses = seen[0].tool_accesses
    assert [type(access) for access in accesses] == granted
    assert [access.required for access in accesses] == [len(granted) == 1] * len(granted)
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
def test_mcp_without_the_episodes_token_refuses_the_listing_and_calls(headers: dict[str, str]) -> None:
    """A refused listing is an error, not an empty list: an MCP-only harness would run the episode with no tools."""
    seen: dict[str, Any] = {}

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        async with _mcp_session(app, headers) as session:
            try:
                await session.list_tools()
            # McpError in MCP SDK 1, MCPError in 2.
            except Exception as error:
                seen["listing"] = error
            seen["call"] = await session.call_tool("add", {"amount": 2})

    server, app = _app(agent, tool_transports=["mcp"])
    (response,) = asyncio.run(_run_with_lifespan(app, server.run_request(_request())))

    assert "needs a valid" in str(seen["listing"])
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

    async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
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

    assert sorted(listed) == [sorted(spec.name for spec in TypedToolset(_Counter()).specs()), ["lookup"]]


@pytest.mark.parametrize(
    "name, arguments, detail",
    [
        pytest.param("fail", {}, "RuntimeError", id="broken-tool"),
        pytest.param("add", {"amount": "two"}, "amount", id="invalid-arguments"),
        pytest.param("lookup", {}, "no tool 'lookup'", id="another-episodes-tool"),
    ],
)
def test_an_mcp_call_the_tool_cannot_run_is_an_error_result_and_recorded(
    name: str, arguments: dict[str, Any], detail: str
) -> None:
    """Over MCP, a failed call is a result with isError, as over direct HTTP it is an error status."""
    seen: dict[str, Any] = {}

    async def agent(server: _Server, counter: _Counter, served: ServedTools) -> None:
        async with _mcp_session(app, _mcp_headers(served)) as session:
            seen["result"] = await session.call_tool(name, arguments)
        seen["calls"] = list(served.calls)

    server, app = _app(agent, tool_transports=["mcp"])
    asyncio.run(_run_with_lifespan(app, server.run_request(_request())))

    assert seen["result"].isError is True and detail in _text(seen["result"])
    assert [(call.name, call.arguments, call.is_error) for call in seen["calls"]] == [(name, arguments, True)]


def test_an_mcp_grant_refuses_tool_names_mcp_clients_reject() -> None:
    server, _ = _app(tool_transports=["mcp"])
    type(server).toolset = staticmethod(lambda counter: _Specs("lookup.v2"))

    response = asyncio.run(server.run_request(_request()))

    assert response.failure is not None and "lookup.v2" in response.failure.failure_reason


def test_tool_transports_cannot_repeat() -> None:
    with pytest.raises(ValidationError, match="must not repeat"):
        _app(tool_transports=["mcp", "mcp"])
