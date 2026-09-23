# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any
from unittest.mock import MagicMock

import anyio
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from nemo_gym.base_environment_server import BaseEnvironmentServer, BaseEnvironmentServerConfig, CleanupContext
from nemo_gym.episode_types import (
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    EpisodeFailure,
    EpisodeId,
    MaterializedTask,
    TaskId,
)
from nemo_gym.server_utils import ServerClient


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
        EpisodeFailure(message="failed", terminal=True),
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


def test_cleanup_is_bounded() -> None:
    calls: list[str] = []
    context = CleanupContext(
        episode_id=_request().episode_id,
        cleanup_timeout_seconds=0.01,
    )

    async def first() -> None:
        calls.append("first")

    async def hung() -> None:
        calls.append("hung")
        await asyncio.sleep(60)

    context.register_cleanup("first", first)
    context.register_cleanup("hung", hung)
    asyncio.run(context.aclose())
    assert calls == ["hung"]


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
    assert "TimeoutError: dependency timed out" in response.failure.message
    assert response.failure.message != "Episode timed out"


def test_episode_deadline_returns_retryable_typed_failure() -> None:
    class _TimedOutEnvironmentServer(_EnvironmentServer):
        async def run(self, request: _Request, cleanup: CleanupContext) -> _Response:
            await asyncio.sleep(60)
            raise AssertionError

    config = _environment_server(default_episode_timeout_seconds=0.01).config
    server = _TimedOutEnvironmentServer(config=config, server_client=MagicMock(spec=ServerClient))
    response = asyncio.run(server.run_request(_request()))
    assert response.failure is not None
    assert response.failure.message == "Episode timed out"
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
    assert "ValueError: invalid protocol state" in response.failure.message


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
