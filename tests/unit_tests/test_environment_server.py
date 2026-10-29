# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import re
from typing import Any, ClassVar
from unittest.mock import MagicMock

import anyio
import pytest
from aiohttp import ClientConnectionError, ClientPayloadError, ClientResponseError, InvalidURL
from fastapi import Body
from fastapi.testclient import TestClient
from pydantic import BaseModel

from nemo_gym.base_environment_server import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    CleanupContext,
    dependency_failure,
    is_retryable_dependency_error,
    is_retryable_dependency_status,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeFailure, MaterializedTask


class _TaskInput(BaseModel):
    value: str


class _Request(BaseEpisodeRequest[_TaskInput]):
    pass


class _Response(BaseEpisodeResponse[str]):
    pass


class _ProtocolFailure(EpisodeFailure):
    stage: str | None = None


class _ProtocolResponse(BaseEpisodeResponse[str]):
    failure: _ProtocolFailure | None = None


class _EnvironmentServer(BaseEnvironmentServer[_Request, _Response]):
    request_model = _Request
    response_model = _Response

    async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
        return _Response(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=request.task.task_input.value,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        return AggregateMetrics()


def _request() -> _Request:
    return _Request(
        episode_id=EpisodeId(rollout_id="rollout", attempt=1),
        task=MaterializedTask(
            task_id=TaskId(taskset="test:default", task_id="task"),
            task_input=_TaskInput(value="result"),
        ),
    )


def _environment_server(**config_overrides: Any) -> _EnvironmentServer:
    config_values = {
        "name": "environment",
        "host": "127.0.0.1",
        "port": 1234,
        "entrypoint": "app.py",
        "default_episode_timeout_seconds": 1,
        "cleanup_timeout_seconds": 0.01,
    }
    config_values.update(config_overrides)
    config = BaseEnvironmentServerConfig(**config_values)
    return _EnvironmentServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.mark.parametrize(
    "declarations, error",
    [
        ({"checkpoint_mode": "stateless"}, "checkpoint_mode='stateless'"),
        ({"checkpoint_mode": "exportd"}, "checkpoint_mode='exportd'"),
    ],
)
def test_an_environment_server_with_an_unknown_checkpoint_declaration_fails_at_startup(
    declarations: dict[str, Any], error: str
) -> None:
    annotations = {name: ClassVar[Any] for name in declarations}
    server_type = type("Declared", (_EnvironmentServer,), {**declarations, "__annotations__": annotations})
    config = _environment_server().config

    with pytest.raises(ValueError, match=re.escape(error)):
        server_type(config=config, server_client=MagicMock(spec=ServerClient)).setup_webserver()


@pytest.mark.parametrize("mode", ["restart_only", "exported"])
def test_an_environment_servers_checkpoint_status_reports_its_declared_mode(mode: str) -> None:
    """A controller can refuse, from discovery alone, a server whose episodes start over after a crash."""
    server_type = type("Declared", (_EnvironmentServer,), {"checkpoint_mode": mode})
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    server = server_type(config=_environment_server().config, server_client=client)

    status = TestClient(server.setup_webserver()).get(
        "/ng-control/v1/checkpoint/status", headers={"Authorization": "Bearer t"}
    )

    assert (status.json()["kind"], status.json()["mode"]) == ("environment", mode)


def test_environment_server_binds_concrete_schema_and_rejects_invalid_input() -> None:
    app = _environment_server().setup_webserver()
    schema = app.openapi()["paths"]["/run"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert schema["$ref"].endswith("/_Request")
    assert (
        TestClient(app).post("/run", json={"task": {"task_input": {"value": "missing identity"}}}).status_code == 422
    )


def test_environment_server_runs_typed_request() -> None:
    response = asyncio.run(_environment_server().run_request(_request()))
    assert response.result == "result"


def test_episode_deadline_is_opt_in() -> None:
    config = BaseEnvironmentServerConfig(
        name="environment",
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        cleanup_timeout_seconds=1,
    )
    server = _EnvironmentServer(config=config, server_client=MagicMock(spec=ServerClient))

    response = asyncio.run(server.run_request(_request()))

    assert config.default_episode_timeout_seconds is None
    assert response.result == "result"


def test_failure_response_validates_base_failure_as_protocol_subclass() -> None:
    class _ProtocolEnvironmentServer(_EnvironmentServer):
        response_model = _ProtocolResponse

    server = _ProtocolEnvironmentServer(
        config=_environment_server().config,
        server_client=MagicMock(spec=ServerClient),
    )
    response = server.failure_response(
        _request(),
        EpisodeFailure(failure_reason="failed", terminal=True),
    )

    assert isinstance(response.failure, _ProtocolFailure)
    assert response.failure.stage is None


def test_cleanup_is_lifo() -> None:
    calls: list[str] = []
    context = CleanupContext(
        episode_id=_request().episode_id,
        cleanup_timeout_seconds=1,
    )

    async def first() -> None:
        calls.append("first")

    async def second() -> None:
        calls.append("second")

    context.register_cleanup("first", first)
    context.register_cleanup("second", second)
    asyncio.run(context.aclose())
    assert calls == ["second", "first"]


def test_hung_cleanup_does_not_skip_later_resource_owner() -> None:
    calls: list[str] = []
    context = CleanupContext(
        episode_id=_request().episode_id,
        cleanup_timeout_seconds=0.01,
    )

    async def first() -> None:
        calls.append("first")

    async def hung() -> None:
        calls.append("hung")
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            calls.append("cancelled")
            raise

    context.register_cleanup("first", first)
    context.register_cleanup("hung", hung)

    async def run() -> None:
        # A missing per-callback deadline must fail promptly instead of passing after 60 seconds.
        await asyncio.wait_for(context.aclose(), timeout=1)

    asyncio.run(run())
    assert calls == ["hung", "cancelled", "first"]


def test_each_cleanup_has_its_own_budget() -> None:
    calls: list[str] = []
    context = CleanupContext(
        episode_id=_request().episode_id,
        cleanup_timeout_seconds=0.05,
    )

    async def sandbox() -> None:
        calls.append("sandbox")

    async def slow_but_finishes() -> None:
        await asyncio.sleep(0.03)
        calls.append("slow")

    async def hung() -> None:
        calls.append("hung")
        await asyncio.sleep(60)

    context.register_cleanup("sandbox", sandbox)
    context.register_cleanup("slow", slow_but_finishes)
    context.register_cleanup("hung", hung)
    loop = asyncio.new_event_loop()
    try:
        started = loop.time()
        loop.run_until_complete(context.aclose())
        elapsed = loop.time() - started
    finally:
        loop.close()
    # Under one shared budget the slow close would have eaten the time the hung one left; here each
    # callback is bounded on its own and the sandbox stop at the bottom of the stack still happens.
    assert calls == ["hung", "slow", "sandbox"]
    assert elapsed < 0.05 * 3


def test_explicit_cleanup_is_bounded_and_final_unwind_retries() -> None:
    attempts = 0
    context = CleanupContext(
        episode_id=_request().episode_id,
        cleanup_timeout_seconds=0.01,
    )

    async def cleanup() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(60)

    handle = context.register_cleanup("cleanup", cleanup)

    async def run() -> None:
        with pytest.raises(TimeoutError):
            await handle.close()
        await context.aclose()

    asyncio.run(run())
    assert attempts == 2


def test_explicit_cleanup_and_final_unwind_do_not_run_concurrently() -> None:
    attempts = 0
    started = asyncio.Event()
    release = asyncio.Event()
    context = CleanupContext(
        episode_id=_request().episode_id,
        cleanup_timeout_seconds=1,
    )

    async def cleanup() -> None:
        nonlocal attempts
        attempts += 1
        started.set()
        await release.wait()

    handle = context.register_cleanup("cleanup", cleanup)

    async def run() -> None:
        explicit = asyncio.create_task(handle.close())
        await started.wait()
        unwind = asyncio.create_task(context.aclose())
        await asyncio.sleep(0)
        release.set()
        await asyncio.gather(explicit, unwind)

    asyncio.run(run())
    assert attempts == 1


def test_internal_timeout_error_is_not_reported_as_episode_timeout() -> None:
    class _InternalTimeoutEnvironmentServer(_EnvironmentServer):
        async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
            raise TimeoutError("dependency timed out")

    server = _InternalTimeoutEnvironmentServer(
        config=_environment_server().config,
        server_client=MagicMock(spec=ServerClient),
    )
    response = asyncio.run(server.run_request(_request()))
    assert response.failure is not None
    assert response.failure.terminal is True
    assert "TimeoutError: dependency timed out" in response.failure.failure_reason
    assert response.failure.failure_reason != "Episode timed out"


def test_episode_deadline_returns_retryable_typed_failure() -> None:
    class _TimedOutEnvironmentServer(_EnvironmentServer):
        async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
            await asyncio.sleep(60)
            raise AssertionError

    config = _environment_server(default_episode_timeout_seconds=0.01).config
    server = _TimedOutEnvironmentServer(config=config, server_client=MagicMock(spec=ServerClient))
    response = asyncio.run(server.run_request(_request()))
    assert response.failure is not None
    assert response.failure.failure_reason == "Episode timed out"
    assert response.failure.terminal is False


def test_unhandled_error_returns_terminal_typed_failure() -> None:
    class _FailedEnvironmentServer(_EnvironmentServer):
        async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
            raise ValueError("invalid protocol state")

    server = _FailedEnvironmentServer(
        config=_environment_server().config,
        server_client=MagicMock(spec=ServerClient),
    )
    response = asyncio.run(server.run_request(_request()))
    assert response.failure is not None
    assert response.failure.terminal is True
    assert "ValueError: invalid protocol state" in response.failure.failure_reason


def test_caller_cancellation_waits_for_cleanup() -> None:
    cleanup_finished = asyncio.Event()

    class _CancelledEnvironmentServer(_EnvironmentServer):
        async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
            async def release_resource() -> None:
                await asyncio.sleep(0.01)
                cleanup_finished.set()

            cleanup.register_cleanup("cleanup", release_resource)
            await asyncio.sleep(60)
            raise AssertionError

    async def run() -> None:
        config = _environment_server().config.model_copy(update={"cleanup_timeout_seconds": 1})
        environment_server = _CancelledEnvironmentServer(
            config=config,
            server_client=MagicMock(spec=ServerClient),
        )
        task = asyncio.create_task(environment_server.run_request(_request()))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert cleanup_finished.is_set()


def test_anyio_level_cancellation_waits_for_cleanup() -> None:
    cleanup_finished = anyio.Event()
    run_started = anyio.Event()

    class _CancelledEnvironmentServer(_EnvironmentServer):
        async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
            async def release_resource() -> None:
                await anyio.sleep(0.01)
                cleanup_finished.set()

            cleanup.register_cleanup("cleanup", release_resource)
            run_started.set()
            await anyio.sleep_forever()

    async def run() -> None:
        environment_server = _CancelledEnvironmentServer(
            config=_environment_server(cleanup_timeout_seconds=1).config,
            server_client=MagicMock(spec=ServerClient),
        )
        with anyio.CancelScope() as scope:
            async with anyio.create_task_group() as tasks:
                tasks.start_soon(environment_server.run_request, _request())
                await run_started.wait()
                scope.cancel()

    anyio.run(run)
    assert cleanup_finished.is_set()


@pytest.mark.parametrize("classified", [False, True])
def test_failure_metadata_survives_the_environment_http_boundary(classified: bool) -> None:
    class FailureEnvironmentServer(_EnvironmentServer):
        async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
            return self.failure_response(
                request,
                EpisodeFailure(
                    failure_reason="Judge unavailable",
                    terminal=False,
                    failure_kind="judge_failed" if classified else None,
                    stage="verification" if classified else None,
                ),
            )

    server = FailureEnvironmentServer(config=_environment_server().config, server_client=MagicMock(spec=ServerClient))
    response = TestClient(server.setup_webserver()).post("/run", json=_request().model_dump(mode="json"))
    assert response.status_code == 200
    expected = {"failure_reason": "Judge unavailable", "terminal": False}
    if classified:
        expected.update(failure_kind="judge_failed", stage="verification")
    assert response.json()["failure"] == expected
    assert response.json()["result"] is None


def test_admission_timeout_reports_its_stage_before_running_the_episode() -> None:
    server = _environment_server(max_concurrent_episodes=1, queue_timeout_seconds=0.01)

    async def run() -> _Response:
        async with server._admission:
            return await server.run_request(_request())

    response = asyncio.run(run())
    assert response.result is None
    assert response.failure.failure_reason == "Episode admission timed out"
    assert response.failure.stage == "admission"
    assert response.failure.terminal is False


@pytest.mark.parametrize("records_boundaries", [False, True])
async def test_the_episodes_of_a_protocol_without_boundaries_are_restarts(records_boundaries: bool) -> None:
    from nemo_gym._checkpoint.environment import EnvironmentParticipant

    started, release = asyncio.Event(), asyncio.Event()

    class _Blocking(_EnvironmentServer):
        checkpoint_mode = "exported" if records_boundaries else "restart_only"

        async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
            started.set()
            await release.wait()
            return await super().run(request, cleanup)

    config = _environment_server().config
    server = _Blocking(config=config, server_client=MagicMock(spec=ServerClient))
    participant = EnvironmentParticipant(mode="exported")
    server._checkpoint = participant
    running = asyncio.create_task(server.run_request(_request()))
    await started.wait()

    report = participant.readiness()
    release.set()
    await running

    # A protocol that records no boundaries cannot continue an episode, so it never holds up a checkpoint.
    assert report.restarts == ([] if records_boundaries else ["rollout-a1"])
    assert report.blocker_count == (1 if records_boundaries else 0)


async def test_a_refused_duplicate_run_leaves_the_live_episode_tracked() -> None:
    from nemo_gym._checkpoint.environment import EnvironmentParticipant
    from nemo_gym._checkpoint.errors import ControlError

    started, release = asyncio.Event(), asyncio.Event()

    class _Blocking(_EnvironmentServer):
        checkpoint_mode = "exported"

        async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
            started.set()
            await release.wait()
            return await super().run(request, cleanup)

    server = _Blocking(config=_environment_server().config, server_client=MagicMock(spec=ServerClient))
    participant = EnvironmentParticipant(mode="exported")
    server._checkpoint = participant
    running = asyncio.create_task(server.run_request(_request()))
    await started.wait()
    with pytest.raises(ControlError, match="already running"):
        await server.run_request(_request())
    # The duplicate owned nothing, so the live episode is still tracked: a checkpoint still waits for it.
    tracked = participant.readiness().blocker_count
    release.set()
    await running

    assert tracked == 1
    assert participant.readiness().blocker_count == 0


async def test_an_episode_is_untracked_even_if_its_cleanup_is_cancelled() -> None:
    from nemo_gym._checkpoint.environment import EnvironmentParticipant

    class _CancelledCleanup(_EnvironmentServer):
        checkpoint_mode = "exported"

        async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
            async def cancelled() -> None:
                # A retire's native task.cancel() landing while final cleanup runs.
                raise asyncio.CancelledError

            cleanup.register_cleanup("cancelled", cancelled)
            return await super().run(request, cleanup)

    server = _CancelledCleanup(config=_environment_server().config, server_client=MagicMock(spec=ServerClient))
    participant = EnvironmentParticipant(mode="exported")
    server._checkpoint = participant
    with pytest.raises(asyncio.CancelledError):
        await server.run_request(_request())

    assert participant.readiness().blocker_count == 0
    assert participant.export_records(None) == []


@pytest.mark.parametrize(
    ("status", "retryable"),
    [
        (400, False),
        (401, False),
        (404, False),
        (409, False),
        (413, False),
        (422, False),
        (408, True),
        (425, True),
        (429, True),
        (500, True),
        (503, True),
        (599, True),
    ],
)
def test_dependency_statuses_retry_only_timeouts_rate_limits_and_server_errors(status: int, retryable: bool) -> None:
    assert is_retryable_dependency_status(status) is retryable
    assert is_retryable_dependency_error(ClientResponseError(MagicMock(), (), status=status)) is retryable


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (ClientConnectionError("refused"), True),
        (ClientPayloadError("connection dropped mid-body"), True),
        (TimeoutError("timed out"), True),
        (InvalidURL("not a url"), False),
        (ValueError("reply failed validation"), False),
    ],
)
def test_only_transport_failures_retry_without_a_status(error: Exception, retryable: bool) -> None:
    assert is_retryable_dependency_error(error) is retryable


class _EvidenceFailure(EpisodeFailure):
    partial_response: str | None = None


def test_dependency_failure_keeps_the_dependencys_reason_and_classifies_it() -> None:
    refusal = ClientResponseError(MagicMock(), (), status=422, message="Unprocessable Content")
    refusal.response_content = b'{"detail":"agent does not support sandbox access"}'

    failure = dependency_failure(
        refusal, failure_model=_EvidenceFailure, stage="agent", partial_response="partial"
    ).failure

    assert type(failure) is _EvidenceFailure
    assert failure.stage == "agent"
    assert failure.terminal is True
    assert failure.failure_reason.startswith("422, message='Unprocessable Content'")
    assert "agent does not support sandbox access" in failure.failure_reason
    assert failure.partial_response == "partial"


def test_dependency_failure_names_the_error_type_bounds_the_reason_and_honors_terminal() -> None:
    failure = dependency_failure(
        TimeoutError("x" * 3000), failure_model=EpisodeFailure, stage="cleanup", name_error_type=True
    ).failure
    assert failure.terminal is False
    assert failure.failure_reason.startswith("TimeoutError: xxx")
    assert len(failure.failure_reason) == 2000

    forced = dependency_failure(
        TimeoutError("timed out"), failure_model=EpisodeFailure, stage="cleanup", terminal=True
    ).failure
    assert forced.terminal is True
    assert forced.failure_reason == "timed out"


def test_a_protocol_may_derive_the_episode_deadline_per_request() -> None:
    """`episode_timeout_seconds` is the one place the per-episode deadline comes from: the base returns the
    configured default, and a server that derives a longer one for a request (the Harbor server, for a multi-step
    task whose steps add up to more than the default) is held to that, not to the default."""

    class _SlowEnvironmentServer(_EnvironmentServer):
        async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
            await asyncio.sleep(0.05)
            return _Response(episode_id=request.episode_id, task_id=request.task.task_id, result="late")

    class _DerivingEnvironmentServer(_SlowEnvironmentServer):
        asked: list[str] = []

        async def episode_timeout_seconds(self, request: _Request) -> float | None:
            self.asked.append(request.task.task_input.value)
            return 5.0

    config = _environment_server(default_episode_timeout_seconds=0.01).config
    assert (
        asyncio.run(_environment_server(default_episode_timeout_seconds=0.01).episode_timeout_seconds(_request()))
        == 0.01
    )
    cut = asyncio.run(
        _SlowEnvironmentServer(config=config, server_client=MagicMock(spec=ServerClient)).run_request(_request())
    )
    assert cut.failure is not None and cut.failure.failure_reason == "Episode timed out"
    server = _DerivingEnvironmentServer(config=config, server_client=MagicMock(spec=ServerClient))
    response = asyncio.run(server.run_request(_request()))
    assert response.failure is None and response.result == "late"
    assert server.asked == [_request().task.task_input.value]


async def test_a_request_cancelled_while_its_bound_resolves_releases_its_admission_slot() -> None:
    """The bound is read after the admission slot is taken; the Harbor server awaits the task's snapshot there.
    A request cancelled (or failing) during that lookup must give the slot back, or a concurrency of one is spent
    for good and every later request times out at admission."""
    lookup = asyncio.Event()

    class _ResolvingEnvironmentServer(_EnvironmentServer):
        async def episode_timeout_seconds(self, request: _Request) -> float | None:
            await lookup.wait()
            return 1

    class _UnresolvedEnvironmentServer(_EnvironmentServer):
        async def episode_timeout_seconds(self, request: _Request) -> float | None:
            raise RuntimeError("snapshot lost")

    config = _environment_server(max_concurrent_episodes=1, queue_timeout_seconds=0.2).config
    server = _ResolvingEnvironmentServer(config=config, server_client=MagicMock(spec=ServerClient))
    first = asyncio.create_task(server.run_request(_request()))
    while not server._admission.locked():
        await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    lookup.set()
    response = await server.run_request(_request())
    assert response.failure is None and response.result == "result"

    server = _UnresolvedEnvironmentServer(config=config, server_client=MagicMock(spec=ServerClient))
    response = await server.run_request(_request())
    assert response.failure is not None and "RuntimeError: snapshot lost" in response.failure.failure_reason
    assert not server._admission.locked()
