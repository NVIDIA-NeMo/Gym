# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

from nemo_gym.base_environment_server import BaseEnvironmentServer, BaseEnvironmentServerConfig, EpisodeContext
from nemo_gym.episode_types import BaseEpisodeRequest, BaseEpisodeResponse, EpisodeId, MaterializedTask, TaskId
from nemo_gym.server_utils import ServerClient


class _TaskInput(BaseModel):
    value: str


class _Request(BaseEpisodeRequest[_TaskInput]):
    pass


class _Response(BaseEpisodeResponse[str]):
    pass


class _EnvironmentServer(BaseEnvironmentServer[_Request, _Response]):
    request_model = _Request
    response_model = _Response

    async def run(self, request: _Request, context: EpisodeContext) -> _Response:
        return _Response(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=request.task.task_input.value,
        )


def _request() -> _Request:
    return _Request(
        episode_id=EpisodeId(rollout_id="rollout", attempt=1),
        task=MaterializedTask(
            task_id=TaskId(taskset="test", task_id="task"),
            task_input=_TaskInput(value="result"),
        ),
    )


def _environment_server(**config_overrides: Any) -> _EnvironmentServer:
    config = BaseEnvironmentServerConfig(
        name="environment",
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        default_episode_timeout_seconds=1,
        cleanup_timeout_seconds=0.01,
        **config_overrides,
    )
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


def test_cleanup_is_lifo() -> None:
    calls: list[str] = []
    context = EpisodeContext(
        request=_request(),
        server_client=MagicMock(spec=ServerClient),
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
    context = EpisodeContext(
        request=_request(),
        server_client=MagicMock(spec=ServerClient),
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
    context = EpisodeContext(
        request=_request(),
        server_client=MagicMock(spec=ServerClient),
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


def test_caller_cancellation_waits_for_cleanup() -> None:
    cleanup_finished = asyncio.Event()

    class _CancelledEnvironmentServer(_EnvironmentServer):
        async def run(self, request: _Request, context: EpisodeContext) -> _Response:
            async def cleanup() -> None:
                await asyncio.sleep(0.01)
                cleanup_finished.set()

            context.register_cleanup("cleanup", cleanup)
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
