# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from pydantic import Field, PrivateAttr

from nemo_gym.client.server_client import SESSION_ID_KEY, ServerClient
from nemo_gym.core.episode import EpisodeId, TaskId
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
)
from nemo_gym.server_runtime.stateful_resources_server import StatefulResourcesServer


class TaskSeed(BaseSeedSessionRequest):
    value: int


class AsyncResourceServer(StatefulResourcesServer[int]):
    seed_request_model = TaskSeed
    state: dict[str, int] = Field(default_factory=dict)
    _initializing: asyncio.Event = PrivateAttr(default_factory=asyncio.Event)
    _initialization_gate: asyncio.Event | None = PrivateAttr(default=None)
    _fail_release: bool = PrivateAttr(default=False)

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self.state

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        assert isinstance(body, TaskSeed)
        self._initializing.set()
        if self._initialization_gate is not None:
            await self._initialization_gate.wait()
        self.state[session_id] = body.value

    async def release_session_state(self, session_id: str) -> None:
        await asyncio.sleep(0)
        if self._fail_release:
            raise RuntimeError("resource still running")
        self.state.pop(session_id, None)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=0)


def _server() -> AsyncResourceServer:
    return AsyncResourceServer(
        config=BaseResourcesServerConfig(host="", port=0, entrypoint="", name="async-resource"),
        server_client=MagicMock(spec=ServerClient),
    )


def _request() -> Request:
    return Request({"type": "http", "session": {SESSION_ID_KEY: "caller-cookie"}})


SEED = ResourcesSeedSessionRequest(
    resources_session_id="session",
    episode_id=EpisodeId(rollout_id="rollout"),
    task_id=TaskId(taskset="tasks", task_id="one"),
    task_data={"value": 42},
)
CLOSE = ResourcesCloseSessionRequest(resources_session_id="session", episode_id=SEED.episode_id)


async def _release(server: AsyncResourceServer, operation: str) -> None:
    if operation == "close":
        await server.close_resources_session(CLOSE)
    else:
        await server.retire_session_state("session")


@pytest.mark.parametrize("operation", ["close", "retire"])
async def test_release_waits_for_in_flight_initialization(operation: str) -> None:
    server = _server()
    request = _request()
    server._initialization_gate = asyncio.Event()
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(server.seed_session(request, SEED))
        await asyncio.wait_for(server._initializing.wait(), timeout=1)
        closing = tasks.create_task(_release(server, operation))
        await asyncio.sleep(0)
        assert not closing.done()
        assert request.session[SESSION_ID_KEY] == "caller-cookie"
        server._initialization_gate.set()
    assert request.session[SESSION_ID_KEY] == "session"
    assert server.state == {}
    assert server._session_episodes == {}
    assert server._fence.closed_episode("session") == SEED.episode_id


@pytest.mark.parametrize("operation", ["close", "retire"])
async def test_failed_release_retains_state_and_identity_for_retry(operation: str) -> None:
    server = _server()
    await server.seed_session(_request(), SEED)
    server._fail_release = True
    with pytest.raises(RuntimeError, match="resource still running"):
        await _release(server, operation)
    assert server.state == {"session": 42}
    assert server._session_episodes == {"session": (SEED.episode_id, SEED.task_id)}
    assert server._fence.closed_episode("session") is None
    server._fail_release = False
    await _release(server, operation)
    assert server.state == {}
    assert server._session_episodes == {}
    assert server._fence.closed_episode("session") == SEED.episode_id


async def test_invalid_task_never_initializes_or_binds_the_cookie() -> None:
    server = _server()
    request = _request()
    with pytest.raises(ResourcesSeedRejectedError) as error:
        await server.seed_session(request, SEED.model_copy(update={"task_data": {"value": "not an integer"}}))
    assert error.value.status_code == 422
    assert request.session[SESSION_ID_KEY] == "caller-cookie"
    assert not server._initializing.is_set()
    assert server.state == {}
    assert server._session_episodes == {}
