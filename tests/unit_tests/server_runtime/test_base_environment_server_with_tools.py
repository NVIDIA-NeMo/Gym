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
from nemo_gym.server_runtime.tool_access import TOOL_CALL_ID_HEADER, TOOL_ERROR_HEADER, DirectHTTPToolAccess


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


def _server(agent: Agent | None = None, **overrides: Any) -> tuple[_Server, httpx.AsyncClient]:
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
    server_type = type("_Agented", (_Server,), {"agent": staticmethod(agent)} if agent else {})
    server = server_type(config=config, server_client=client)
    app = server.setup_webserver()
    # As in production: a request does not inherit the context of the task that sent it, here the episode's own.
    install_request_context_isolation(app)
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url=BASE_URL)
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


def test_several_workers_are_refused_while_calls_reach_any_worker() -> None:
    server, _ = _server(num_workers=2)

    with pytest.raises(ValueError, match="needs num_workers=1"):
        refuse_unsupported_num_workers(server)


class _Sandbox:
    """A resource whose release takes a moment, as stopping a remote sandbox does."""

    def __init__(self) -> None:
        self.stops_started = 0
        self.stopped = False

    async def stop(self) -> None:
        self.stops_started += 1
        await asyncio.sleep(0.05)
        self.stopped = True


def _sandbox_server(*, hold: bool) -> tuple[_Server, _Sandbox, asyncio.Event]:
    """A server whose episode registers a sandbox, then returns or, with ``hold``, runs until cancelled."""
    sandbox, started = _Sandbox(), asyncio.Event()

    async def run_episode(self: _Server, request: _Request, cleanup: CleanupContext) -> _Response:
        cleanup.register_cleanup("sandbox", sandbox.stop)
        started.set()
        if hold:
            await asyncio.Event().wait()
        return _Response(episode_id=request.episode_id, task_id=request.task.task_id, result="done")

    server, _ = _server()
    type(server).run_episode = run_episode
    return server, sandbox, started


def test_shutdown_finishes_the_cleanups_of_a_running_episode() -> None:
    async def scenario() -> _Sandbox:
        server, sandbox, started = _sandbox_server(hold=True)
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


@pytest.mark.parametrize("also_run_episode", [False, True], ids=["run-only", "run-and-run-episode"])
def test_a_subclass_implements_run_episode_not_run(also_run_episode: bool) -> None:
    """Overriding run would skip the tracking the shutdown sweep needs, even when run calls run_episode."""

    async def run(self: Any, request: _Request, cleanup: CleanupContext) -> _Response:
        return await self.run_episode(request, cleanup)

    members = {"__module__": __name__, "run": run, "request_model": _Request, "response_model": _Response}
    if also_run_episode:
        members["run_episode"] = _Server.run_episode

    with pytest.raises(TypeError, match="implements run_episode"):
        type("_OverridesRun", (_Server,), members)


def test_shutdown_runs_each_cleanup_under_its_episodes_rollout() -> None:
    """A cleanup that calls a Resources Server sends the call under the episode's attempt, also at shutdown."""
    seen: list[tuple[str, str | None]] = []
    started = asyncio.Event()

    async def run_episode(self: _Server, request: _Request, cleanup: CleanupContext) -> _Response:
        async def release() -> None:
            seen.append((request.episode_id.capture_key, current_rollout_id()))

        cleanup.register_cleanup("release", release)
        if len(self._running_cleanups) == 2:
            started.set()
        await asyncio.Event().wait()

    server, _ = _server()
    type(server).run_episode = run_episode

    async def scenario() -> None:
        requests = [_request("3-0").model_copy(update={"episode_id": EpisodeId(rollout_id="3-0", attempt=2)})]
        requests.append(_request("4-0"))
        episodes = [asyncio.create_task(server.run_request(request)) for request in requests]
        await started.wait()
        await server.finish_running_cleanups()
        for episode in episodes:
            episode.cancel()
        await asyncio.gather(*episodes, return_exceptions=True)

    asyncio.run(scenario())

    # The sweep sees each episode's own key; the episodes' cancelled requests find the cleanups already done.
    assert sorted(seen) == [("3-0-a2", "3-0-a2"), ("4-0", "4-0")]
    assert current_rollout_id() is None


def test_shutdown_stops_a_sandbox_even_when_another_cleanup_hangs(caplog: pytest.LogCaptureFixture) -> None:
    """In BIRD-Interact the agent-session close runs first and can retry an agent that is shutting down too."""
    sandbox = _Sandbox()
    started = asyncio.Event()

    async def run_episode(self: _Server, request: _Request, cleanup: CleanupContext) -> _Response:
        cleanup.register_cleanup("sandbox", sandbox.stop)
        cleanup.register_cleanup("agent session", asyncio.Event().wait)
        started.set()
        await asyncio.Event().wait()

    server, _ = _server()
    type(server).run_episode = run_episode
    server.config.cleanup_timeout_seconds = 0.3

    async def scenario() -> float:
        episode = asyncio.create_task(server.run_request(_request()))
        await started.wait()
        began = asyncio.get_running_loop().time()
        await server.finish_running_cleanups()
        took = asyncio.get_running_loop().time() - began
        episode.cancel()
        await asyncio.gather(episode, return_exceptions=True)
        return took

    took = asyncio.run(scenario())

    assert sandbox.stopped and sandbox.stops_started == 1
    assert took < 1
    assert "Episode cleanup unconfirmed at shutdown: episode_id=3-0" in caplog.text


def test_shutdown_retries_a_cleanup_that_failed_when_its_episode_ended() -> None:
    """A sandbox stop that got a provider error stays tracked, so shutdown stops the sandbox before its TTL."""
    sandbox = _Sandbox()
    failures = [RuntimeError("provider returned 503")]

    async def stop() -> None:
        if failures:
            raise failures.pop()
        await sandbox.stop()

    async def run_episode(self: _Server, request: _Request, cleanup: CleanupContext) -> _Response:
        cleanup.register_cleanup("sandbox", stop)
        return _Response(episode_id=request.episode_id, task_id=request.task.task_id, result="done")

    server, _ = _server()
    type(server).run_episode = run_episode

    async def scenario() -> None:
        await server.run_request(_request())
        assert not sandbox.stopped and len(server._running_cleanups) == 1
        await server.finish_running_cleanups()

    asyncio.run(scenario())

    assert sandbox.stopped and sandbox.stops_started == 1


def test_shutdown_during_an_interrupted_stop_waits_for_it_in_uvicorns_order() -> None:
    """Uvicorn cancels /run once, then exits the lifespan while the request's own cleanup still stops the sandbox."""

    async def scenario() -> tuple[_Sandbox, bool]:
        server, sandbox, started = _sandbox_server(hold=True)
        app = server.setup_webserver()
        episode = asyncio.create_task(server.run_request(_request()))
        await started.wait()

        episode.cancel()
        while sandbox.stops_started == 0:
            await asyncio.sleep(0)
        async with app.router.lifespan_context(app):
            pass
        stopped_at_lifespan_exit = sandbox.stopped
        await asyncio.gather(episode, return_exceptions=True)
        return sandbox, stopped_at_lifespan_exit

    sandbox, stopped_at_lifespan_exit = asyncio.run(scenario())

    assert stopped_at_lifespan_exit and sandbox.stops_started == 1


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
