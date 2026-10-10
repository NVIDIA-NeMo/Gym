# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Typed resources sessions over a task sandbox, driven through a minimal server's HTTP routes."""

import asyncio
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import BaseModel, ConfigDict
from pytest import MonkeyPatch

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionRequest,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.swebench import sandbox_sessions
from resources_servers.swebench.sandbox_sessions import SandboxSessionResourcesServer


class _Config(BaseResourcesServerConfig):
    sandbox_provider: str = "test-provider-config"
    sandbox_config: dict[str, Any] = {}


class _Task(BaseModel):
    model_config = ConfigDict(extra="allow")

    instance_id: str


class _Server(SandboxSessionResourcesServer):
    """Starts a task sandbox per seed and captures it in verify, as the SWE servers do."""

    config: _Config

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox = {}
        self._session_id_to_prepared: dict[str, str] = {}
        self._sandboxes: list[MagicMock] = []
        self._fail_preparation = False

    async def _start_task_sandbox(self, session_id: str, task: _Task) -> str:
        sandbox = _sandbox(f"sb-{len(self._sandboxes) + 1}")
        self._sandboxes.append(sandbox)
        self._session_id_to_sandbox[session_id] = sandbox
        if self._fail_preparation:
            raise RuntimeError("preparation failed")
        self._session_id_to_prepared[session_id] = task.instance_id
        return "/testbed"

    def _forget_task_sandbox_state(self, session_id: str) -> None:
        self._session_id_to_prepared.pop(session_id, None)

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest):
        return await self.seed_task_sandbox_session(request, body, _Task)

    async def verify(self, request: Request, body: BaseVerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        self._claim_task_sandbox(session_id)
        sandbox = self._session_id_to_sandbox.pop(session_id)
        await self._release_task_sandbox(session_id, sandbox)
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


def _sandbox(sandbox_id: str) -> MagicMock:
    sandbox = MagicMock()
    sandbox.serialize = AsyncMock(return_value={"sandbox_id": sandbox_id})
    sandbox.stop = AsyncMock()
    return sandbox


def _server(**config: Any) -> _Server:
    return _Server(
        config=_Config(host="0.0.0.0", port=8080, entrypoint="", name="", **config),
        server_client=MagicMock(spec=ServerClient),
    )


def _seed(session_id: str = "resources-session", task_id: str = "repo__repo-1", rollout: str = "r") -> dict:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id=rollout),
        task_id=TaskId(taskset="swe", task_id=task_id),
        task_data={"instance_id": "repo__repo-1", "responses_create_params": {"input": "Fix it"}},
    ).model_dump(mode="json")


def _close(session_id: str = "resources-session", rollout: str = "r") -> dict:
    return {"resources_session_id": session_id, "episode_id": {"rollout_id": rollout}}


_VERIFY = {
    "responses_create_params": {"input": "Fix it"},
    "response": {
        "output": [],
        "id": "",
        "created_at": 0,
        "model": "",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    },
}


def test_typed_seed_hands_the_agent_the_task_sandbox_and_close_stops_it_once() -> None:
    server = _server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=_seed())
        assert seeded.status_code == 200, seeded.text
        assert seeded.json() == {
            "resources_session_id": "resources-session",
            "resources_tools": None,
            "sandbox_access": {
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "test-provider-config",
                    "descriptor": {"sandbox_id": "sb-1"},
                },
                "workdir": "/testbed",
            },
        }
        assert client.post("/seed_session", json=_seed()).json() == seeded.json()
        assert len(server._sandboxes) == 1
        # The agent gets an operate lease where the provider mints leases, so it cannot destroy the sandbox.
        server._sandboxes[0].serialize.assert_awaited_with(scope="operate")

        for _ in range(2):
            assert client.post("/close_session", json=_close()).json() == {"resources_session_id": "resources-session"}
        assert client.post("/seed_session", json=_seed()).status_code == 500
        assert len(server._sandboxes) == 1

    server._sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_prepared == {}


def test_verify_consumes_the_sandbox_and_a_repeated_verify_or_seed_is_rejected() -> None:
    server = _server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed()).status_code == 200
        assert client.post("/verify", json=_VERIFY).status_code == 200
        server._sandboxes[0].stop.assert_awaited_once()

        assert client.post("/verify", json=_VERIFY).status_code == 500
        assert client.post("/seed_session", json=_seed()).status_code == 500
        assert client.post("/close_session", json=_close()).status_code == 200

    server._sandboxes[0].stop.assert_awaited_once()


def test_close_retries_a_stop_that_failed_during_verify() -> None:
    server = _server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed()).status_code == 200
        server._sandboxes[0].stop.side_effect = [RuntimeError("provider unavailable"), None]
        assert client.post("/verify", json=_VERIFY).status_code == 200
        # The sandbox is kept for the close, but it is not handed out or graded again.
        assert client.post("/verify", json=_VERIFY).status_code == 500
        assert client.post("/seed_session", json=_seed()).status_code == 500

        assert client.post("/close_session", json=_close()).status_code == 200

    assert server._sandboxes[0].stop.await_count == 2
    assert server._session_id_to_sandbox == {}


def test_a_failed_close_is_not_fenced_and_a_retried_close_stops_the_sandbox() -> None:
    server = _server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed()).status_code == 200
        server._sandboxes[0].stop.side_effect = [RuntimeError("provider unavailable"), None]

        assert client.post("/close_session", json=_close()).status_code == 500
        assert server._session_id_to_sandbox, "a failed stop must keep the sandbox for the retry"
        assert client.post("/close_session", json=_close()).status_code == 200

    assert server._sandboxes[0].stop.await_count == 2
    assert server._session_id_to_sandbox == {}


def test_a_failed_seed_stops_its_sandbox_and_keeps_no_record() -> None:
    server = _server()
    server._fail_preparation = True
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed()).status_code == 500
        server._sandboxes[0].stop.assert_awaited_once()
        assert server._task_sessions == {} and server._session_id_to_sandbox == {}

        server._fail_preparation = False
        assert client.post("/seed_session", json=_seed()).status_code == 200


def test_a_failed_handoff_stops_the_sandbox() -> None:
    server = _server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        original_start = server._start_task_sandbox

        async def start_unserializable(session_id: str, task: _Task) -> str:
            workdir = await original_start(session_id, task)
            server._sandboxes[-1].serialize.side_effect = RuntimeError("not connectable")
            return workdir

        server._start_task_sandbox = start_unserializable
        assert client.post("/seed_session", json=_seed()).status_code == 500

    server._sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {} and server._session_id_to_prepared == {}


def test_a_seed_naming_another_task_or_episode_is_rejected() -> None:
    server = _server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        mismatched = client.post("/seed_session", json=_seed(task_id="repo__repo-2"))
        assert mismatched.status_code == 500
        assert server._sandboxes == [], "a mismatched task_id must be rejected before a sandbox starts"

        assert client.post("/seed_session", json=_seed()).status_code == 200
        assert client.post("/seed_session", json=_seed(rollout="other")).status_code == 500
        assert client.post("/close_session", json=_close(rollout="other")).status_code == 500
        assert len(server._sandboxes) == 1


def test_the_task_id_follows_task_materialization_precedence() -> None:
    server = _server()
    # Materialization takes a row's own task_id field before its instance_id.
    seed = _seed(task_id="package-7")
    seed["task_data"] = {"task_id": "package-7", "instance_id": "repo__repo-1"}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed).status_code == 200
        seed = _seed("other-session", task_id="repo__repo-1")
        seed["task_data"] = {"task_id": "package-7", "instance_id": "repo__repo-1"}
        assert client.post("/seed_session", json=seed).status_code == 500
    assert len(server._sandboxes) == 1


def test_a_legacy_session_verify_is_not_affected() -> None:
    server = _server()
    server._session_id_to_sandbox["cookie-session"] = sandbox = _sandbox("legacy")
    server._claim_task_sandbox("cookie-session")
    server._claim_task_sandbox("cookie-session")
    assert server._task_sessions == {}
    assert server._session_id_to_sandbox == {"cookie-session": sandbox}


def test_closed_session_fences_expire(monkeypatch: MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(sandbox_sessions, "monotonic", lambda: clock[0])
    server = _server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        for index in range(3):
            assert client.post("/seed_session", json=_seed(f"s-{index}")).status_code == 200
            assert client.post("/close_session", json=_close(f"s-{index}")).status_code == 200
        assert len(server._task_sessions) == 3

        clock[0] += sandbox_sessions.CLOSED_SESSION_RETENTION_S + 1
        assert client.post("/close_session", json=_close("never-seeded")).status_code == 200
        assert set(server._task_sessions) == {"never-seeded"}
        assert list(server._closed_task_sessions) == ["never-seeded"]


def test_typed_seeds_require_one_worker() -> None:
    server = _server(num_workers=2)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed()).status_code == 500
    assert server._sandboxes == []


def test_shutdown_stops_sandboxes_no_close_released() -> None:
    server = _server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed()).status_code == 200
    server._sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {} and server._task_sessions == {}


def test_session_conformance() -> None:
    check_resources_session_contract(
        _server().setup_webserver(),
        ResourcesSeedSessionRequest.model_validate(_seed("conformance-session")),
        keeps_state=True,
    )


def _request() -> SimpleNamespace:
    return SimpleNamespace(session={})


@pytest.mark.asyncio
async def test_a_retried_seed_stops_the_sandbox_a_failed_cleanup_left_behind() -> None:
    server = _server()
    server._fail_preparation = True
    original_start = server._start_task_sandbox

    async def start_with_failing_cleanup(session_id: str, task: _Task) -> str:
        try:
            return await original_start(session_id, task)
        finally:
            server._sandboxes[-1].stop.side_effect = [RuntimeError("provider unavailable"), None]

    server._start_task_sandbox = start_with_failing_cleanup
    seed = ResourcesSeedSessionRequest.model_validate(_seed())
    with pytest.raises(RuntimeError, match="preparation failed"):
        await server.seed_task_sandbox_session(_request(), seed, _Task)
    first = server._sandboxes[0]

    server._fail_preparation = False
    await server.seed_task_sandbox_session(_request(), seed, _Task)
    assert first.stop.await_count == 2, "the retry must stop the sandbox the failed cleanup left registered"
    assert server._session_id_to_sandbox == {"resources-session": server._sandboxes[1]}


@pytest.mark.asyncio
async def test_a_disconnect_during_seed_still_stops_the_sandbox() -> None:
    server = _server()
    preparing = asyncio.Event()
    original_start = server._start_task_sandbox

    async def slow_stop() -> None:
        await asyncio.sleep(0.01)

    async def start_slowly(session_id: str, task: _Task) -> str:
        workdir = await original_start(session_id, task)
        server._sandboxes[-1].stop = AsyncMock(side_effect=slow_stop)
        preparing.set()
        await asyncio.sleep(10)
        return workdir

    server._start_task_sandbox = start_slowly
    seed = ResourcesSeedSessionRequest.model_validate(_seed())
    async with anyio.create_task_group() as group:
        group.start_soon(server.seed_task_sandbox_session, _request(), seed, _Task)
        await preparing.wait()
        # ClientDisconnectCancellationMiddleware cancels the handler this way.
        group.cancel_scope.cancel()

    server._sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_sandbox == {} and server._task_sessions == {}


@pytest.mark.asyncio
async def test_a_stop_interrupted_by_a_disconnect_still_completes() -> None:
    server = _server()
    await server.seed_task_sandbox_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()), _Task)
    sandbox = server._session_id_to_sandbox.pop("resources-session")
    stopped = asyncio.Event()

    async def slow_stop() -> None:
        await asyncio.sleep(0.05)
        stopped.set()

    sandbox.stop = AsyncMock(side_effect=slow_stop)
    async with anyio.create_task_group() as group:
        group.start_soon(server._release_task_sandbox, "resources-session", sandbox)
        await asyncio.sleep(0.01)
        group.cancel_scope.cancel()
    assert stopped.is_set(), "the shielded stop must finish although its request was cancelled"


def test_a_session_that_is_never_closed_is_closed_after_its_sandbox_lifetime(monkeypatch: MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(sandbox_sessions, "monotonic", lambda: clock[0])
    server = _server(sandbox_config={"ttl_s": 600})
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed("abandoned")).status_code == 200
        clock[0] += 601
        assert client.post("/seed_session", json=_seed("next")).status_code == 200
        server._sandboxes[0].stop.assert_awaited_once()
        assert "abandoned" not in server._session_id_to_sandbox and "abandoned" not in server._open_task_sessions
        assert client.post("/seed_session", json=_seed("abandoned")).status_code == 500, "it is fenced like a close"

        clock[0] += sandbox_sessions.CLOSED_SESSION_RETENTION_S + 1
        assert client.post("/close_session", json=_close("next")).status_code == 200
        assert "abandoned" not in server._task_sessions
