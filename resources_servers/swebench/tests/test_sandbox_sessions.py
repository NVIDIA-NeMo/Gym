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

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import SESSION_EXPIRED, SESSION_LOST, EpisodeId, TaskId
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    check_resources_session_contract,
    lost_session_verification,
)
from resources_servers.swebench import sandbox_sessions
from resources_servers.swebench.sandbox_sessions import SandboxSessionResourcesServer


class _Config(BaseResourcesServerConfig):
    sandbox_provider: str = "test-provider-config"
    sandbox_config: dict[str, Any] = {}
    is_verifying_golden_patch: bool = False


class _Task(BaseModel):
    model_config = ConfigDict(extra="allow")

    instance_id: str


class _VerifyRequest(_Task, BaseVerifyRequest):
    pass


class _Server(SandboxSessionResourcesServer):
    """Starts a task sandbox per seed and captures it in verify, as the SWE servers do."""

    config: _Config

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._session_id_to_sandbox = {}
        self._session_id_to_prepared: dict[str, str] = {}
        self._sandboxes: list[MagicMock] = []
        self._fail_preparation = False
        self._capture = AsyncMock(return_value="original patch")
        self._grade = AsyncMock(return_value=1.0)

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

    async def seed_session(self, request: Request, body: ResourcesSeedSessionRequest) -> ResourcesSeedSessionResponse:
        return await self.seed_task_sandbox_session(request, body, _Task)

    async def verify(self, request: Request, body: _VerifyRequest) -> BaseVerifyResponse:
        return await self.verify_task_sandbox_session(
            request,
            body,
            verify=lambda: self._verify(request, body),
            lost_response=lambda reason, failure_kind: BaseVerifyResponse.model_validate(
                body.model_dump() | lost_session_verification(reason, failure_kind=failure_kind)
            ),
            allow_seedless=self.config.is_verifying_golden_patch,
        )

    async def _verify(self, request: Request, body: _VerifyRequest) -> BaseVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        patch = (
            "golden"
            if self.config.is_verifying_golden_patch
            else await self.capture_task_sandbox_patch(session_id, self._capture)
        )
        await self._stop_task_sandbox(session_id)
        reward = await self._grade(patch)
        return BaseVerifyResponse.model_validate(body.model_dump() | {"reward": reward})


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
    "instance_id": "repo__repo-1",
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
        server._sandboxes[0].serialize.assert_awaited_once_with(scope="operate")

        for _ in range(2):
            assert client.post("/close_session", json=_close()).json() == {"resources_session_id": "resources-session"}
        assert client.post("/seed_session", json=_seed()).status_code == 409
        assert len(server._sandboxes) == 1

    server._sandboxes[0].stop.assert_awaited_once()
    assert server._session_id_to_prepared == {}


def test_verify_replays_the_saved_verdict_without_a_task_sandbox() -> None:
    server = _server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed()).status_code == 200
        assert client.post("/verify", json=_VERIFY).status_code == 200
        server._sandboxes[0].stop.assert_awaited_once()

        assert not server._session_id_to_sandbox
        assert client.post("/verify", json=_VERIFY).json()["reward"] == 1.0
        server._capture.assert_awaited_once()
        server._grade.assert_awaited_once_with("original patch")
        assert client.post("/seed_session", json=_seed()).status_code == 409
        assert client.post("/close_session", json=_close()).status_code == 200

    server._sandboxes[0].stop.assert_awaited_once()


def test_close_retries_a_stop_that_failed_during_verify() -> None:
    server = _server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed()).status_code == 200
        server._sandboxes[0].stop.side_effect = [RuntimeError("provider unavailable"), None]
        assert client.post("/verify", json=_VERIFY).status_code == 500
        # The sandbox is kept for the close, but it is not handed out or graded again.
        server._grade.assert_not_awaited()
        assert client.post("/seed_session", json=_seed()).status_code == 409

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


@pytest.mark.parametrize("field", ["task_id", "episode_id", "task_data"])
def test_a_seed_naming_another_task_episode_or_data_is_rejected(field: str) -> None:
    server = _server()
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed()).status_code == 200
        changed = _seed()
        if field == "task_id":
            changed[field]["task_id"] = "other"
        elif field == "episode_id":
            changed[field]["rollout_id"] = "other"
        else:
            changed[field]["instance_id"] = "other"
        assert client.post("/seed_session", json=changed).status_code == 409
        assert client.post("/close_session", json=_close(rollout="other")).status_code == 409
        assert len(server._sandboxes) == 1


def test_caller_assigned_task_id_is_independent_of_nested_task_data() -> None:
    server = _server()
    # A canonical row may have an outer source index while task_data names its own instance.
    seed = _seed(task_id="0")
    seed["task_data"] = {"task_id": "package-7", "instance_id": "repo__repo-1"}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=seed).status_code == 200
        assert client.post("/seed_session", json=seed).status_code == 200
        assert client.post("/verify", json=_VERIFY).status_code == 200
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
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    server = _server(session_close_retry_window_seconds=10)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        for index in range(3):
            assert client.post("/seed_session", json=_seed(f"s-{index}")).status_code == 200
            assert client.post("/close_session", json=_close(f"s-{index}")).status_code == 200
        assert server._task_sessions == {}
        assert client.post("/seed_session", json=_seed("s-0")).status_code == 409

        clock[0] += 11
        assert client.post("/seed_session", json=_seed("s-0")).status_code == 200


def test_typed_seed_allows_runtime_session_routing_with_multiple_workers() -> None:
    server = _server(num_workers=2)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed()).status_code == 200
    assert len(server._sandboxes) == 1


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


def _request(session_id: str = "resources-session") -> SimpleNamespace:
    return SimpleNamespace(session={SESSION_ID_KEY: session_id})


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
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    server = _server(sandbox_config={"ttl_s": 600}, session_close_retry_window_seconds=10)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed("abandoned")).status_code == 200
        clock[0] += 601
        assert client.post("/seed_session", json=_seed("next")).status_code == 200
        server._sandboxes[0].stop.assert_awaited_once()
        assert "abandoned" not in server._session_id_to_sandbox and "abandoned" not in server._open_task_sessions
        assert client.post("/seed_session", json=_seed("abandoned")).status_code == 409, "it is fenced like a close"

        clock[0] += 11
        assert client.post("/close_session", json=_close("next")).status_code == 200
        assert "abandoned" not in server._task_sessions


@pytest.mark.parametrize("state", ["unknown", "closed", "missing"])
def test_unavailable_verify_is_masked_without_capturing_or_grading(state: str) -> None:
    server = _server()
    with TestClient(server.setup_webserver()) as client:
        if state != "unknown":
            assert client.post("/seed_session", json=_seed()).status_code == 200
            if state == "closed":
                cookies = dict(client.cookies)
                assert client.post("/close_session", json=_close()).status_code == 200
                client.cookies.clear()
                client.cookies.update(cookies)
            else:
                server._session_id_to_sandbox.pop("resources-session")
        response = client.post("/verify", json=_VERIFY)
        assert response.status_code == 200, response.text
        assert response.json()["mask_sample"] is True
        assert response.json()["failure_kind"] == SESSION_LOST
        if state == "closed":
            assert "closed" in response.json()["failure_reason"]
        server._capture.assert_not_awaited()
        server._grade.assert_not_awaited()


def test_golden_verification_does_not_need_a_seed() -> None:
    server = _server(is_verifying_golden_patch=True)
    with TestClient(server.setup_webserver()) as client:
        result = client.post("/verify", json=_VERIFY)
        assert result.status_code == 200, result.text
        assert result.json()["reward"] == 1.0
        server._grade.assert_awaited_once_with("golden")
        server._capture.assert_not_awaited()
        assert server._sandboxes == []


def test_invalid_task_data_is_rejected_without_assigning_session_or_starting_sandbox() -> None:
    server = _server()
    with TestClient(server.setup_webserver()) as client:
        seed = _seed()
        seed["task_data"] = {}
        result = client.post("/seed_session", json=seed)
        assert result.status_code == 422
        assert server._sandboxes == []
        assert not server._task_sessions


@pytest.mark.asyncio
async def test_verify_retry_reuses_capture_after_stop_and_grading_failures() -> None:
    server = _server()
    await server.seed_task_sandbox_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()), _Task)
    sandbox = server._sandboxes[0]
    sandbox.stop.side_effect = [RuntimeError("stop failed"), None]
    body = _VerifyRequest.model_validate(_VERIFY)
    with pytest.raises(RuntimeError, match="stop failed"):
        await server.verify(_request(), body)
    assert server._session_id_to_sandbox["resources-session"] is sandbox
    assert server._task_sessions["resources-session"].capture == "original patch"
    server._grade.assert_not_awaited()
    server._capture.return_value = "replacement patch must not be read"
    server._grade.side_effect = [RuntimeError("grading failed"), 1.0]
    with pytest.raises(RuntimeError, match="grading failed"):
        await server.verify(_request(), body)
    assert not server._session_id_to_sandbox
    result = await server.verify(_request(), body)
    assert result.reward == 1.0
    assert await server.verify(_request(), body) == result
    server._capture.assert_awaited_once()
    assert server._grade.await_count == 2
    assert all(call.args == ("original patch",) for call in server._grade.await_args_list)
    with pytest.raises(ResourcesSessionConflictError, match="input"):
        await server.verify(_request(), body.model_copy(update={"extra_retry_data": "different"}))
    await server.shutdown()


@pytest.mark.asyncio
async def test_close_waits_for_failed_verification_stop_and_keeps_retry_ownership() -> None:
    server = _server()
    await server.seed_task_sandbox_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()), _Task)
    sandbox = server._sandboxes[0]
    stopping, finish = asyncio.Event(), asyncio.Event()

    async def stop() -> None:
        if sandbox.stop.await_count == 1:
            stopping.set()
            await finish.wait()
        raise RuntimeError("provider unavailable")

    sandbox.stop.side_effect = stop
    async with asyncio.timeout(2):
        verification = asyncio.create_task(server.verify(_request(), _VerifyRequest.model_validate(_VERIFY)))
        await stopping.wait()
        close = asyncio.create_task(
            server.close_resources_session(_request(), ResourcesCloseSessionRequest.model_validate(_close()))
        )
        await asyncio.sleep(0)
        assert not close.done(), "close must wait for the verification's cleanup attempt"
        finish.set()
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await verification
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await close
    assert server._session_id_to_sandbox["resources-session"] is sandbox
    assert server._task_session_fence.closed_episode("resources-session") is None
    sandbox.stop.side_effect = None
    await server.close_resources_session(_request(), ResourcesCloseSessionRequest.model_validate(_close()))
    assert sandbox.stop.await_count == 3
    assert not server._session_id_to_sandbox


@pytest.mark.asyncio
async def test_expiry_failed_stop_retains_owner_until_explicit_close(monkeypatch: MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(sandbox_sessions, "monotonic", lambda: clock[0])
    server = _server()  # No provider TTL: cleanup ownership cannot rely on eventual provider deletion.
    await server.seed_task_sandbox_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()), _Task)
    sandbox = server._sandboxes[0]
    sandbox.stop.side_effect = RuntimeError("provider unavailable")
    clock[0] += sandbox_sessions.DEFAULT_SESSION_LIFETIME_S + 1
    await server._close_abandoned_task_sessions(clock[0])
    record = server._task_sessions["resources-session"]
    assert record.expired
    assert server._session_id_to_sandbox["resources-session"] is sandbox
    assert server._session_id_to_prepared["resources-session"] == "repo__repo-1"
    assert server._task_session_fence.closed_episode("resources-session") is None
    result = await server.verify(_request(), _VerifyRequest.model_validate(_VERIFY))
    assert result.mask_sample and result.failure_kind == SESSION_EXPIRED
    server._grade.assert_not_awaited()
    with pytest.raises(ResourcesSessionConflictError, match="expired"):
        await server.seed_task_sandbox_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()), _Task)
    sandbox.stop.side_effect = None
    await server.close_resources_session(_request(), ResourcesCloseSessionRequest.model_validate(_close()))
    assert not server._session_id_to_sandbox
    assert not server._session_id_to_prepared


@pytest.mark.asyncio
async def test_shutdown_preserves_task_identity_when_cleanup_fails() -> None:
    server = _server()
    seed = ResourcesSeedSessionRequest.model_validate(_seed())
    await server.seed_task_sandbox_session(_request(), seed, _Task)
    sandbox = server._sandboxes[0]
    sandbox.stop.side_effect = [RuntimeError("provider unavailable"), None]
    await server.shutdown()
    assert server._task_sessions["resources-session"].seed == seed
    assert server._session_id_to_sandbox["resources-session"] is sandbox
    await server.shutdown()
    assert not server._session_id_to_sandbox
    assert not server._task_sessions


@pytest.mark.asyncio
async def test_failed_verifier_stop_preserves_verdict_and_close_retries() -> None:
    server = _server()
    await server.seed_task_sandbox_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()), _Task)
    verifier = _sandbox("verifier")
    verifier.stop.side_effect = [RuntimeError("provider unavailable"), None]

    async def grade(patch: str) -> float:
        server._verifier_sandboxes["resources-session"] = verifier
        return 1.0

    server._grade.side_effect = grade
    body = _VerifyRequest.model_validate(_VERIFY)
    result = await server.verify(_request(), body)
    assert result.reward == 1.0
    assert server._verifier_sandboxes["resources-session"] is verifier
    assert await server.verify(_request(), body) == result
    server._grade.assert_awaited_once()
    await server.close_resources_session(_request(), ResourcesCloseSessionRequest.model_validate(_close()))
    assert verifier.stop.await_count == 2
    assert not server._verifier_sandboxes


@pytest.mark.asyncio
async def test_older_claim_release_consumer_keeps_ownership_during_concurrent_close() -> None:
    server = _server()
    await server.seed_task_sandbox_session(_request(), ResourcesSeedSessionRequest.model_validate(_seed()), _Task)
    server._claim_task_sandbox("resources-session")
    sandbox = server._session_id_to_sandbox.pop("resources-session")
    closing = asyncio.create_task(
        server.close_resources_session(_request(), ResourcesCloseSessionRequest.model_validate(_close()))
    )
    await asyncio.sleep(0)
    assert not closing.done(), "close must wait while an older consumer has popped the capture sandbox"
    sandbox.stop.side_effect = RuntimeError("provider unavailable")
    async with asyncio.timeout(2):
        await server._release_task_sandbox("resources-session", sandbox)
        with pytest.raises(RuntimeError, match="provider unavailable"):
            await closing
    assert server._session_id_to_sandbox["resources-session"] is sandbox
    assert server._task_session_fence.closed_episode("resources-session") is None
    sandbox.stop.side_effect = None
    await server.close_resources_session(_request(), ResourcesCloseSessionRequest.model_validate(_close()))
    assert not server._session_id_to_sandbox


# --- additional coverage: expiry, failed-seed leftovers, races, claim/release guards -------------------------------


def _seed_request(session_id: str = "resources-session") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest.model_validate(_seed(session_id))


def _close_request(session_id: str = "resources-session") -> ResourcesCloseSessionRequest:
    return ResourcesCloseSessionRequest.model_validate(_close(session_id))


@pytest.mark.asyncio
async def test_verify_after_a_successful_expiry_is_masked_as_expired_then_lost_when_the_fence_ends(
    monkeypatch: MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(sandbox_sessions, "monotonic", lambda: clock[0])
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    server = _server(session_close_retry_window_seconds=10)
    await server.seed_task_sandbox_session(_request(), _seed_request(), _Task)
    clock[0] += sandbox_sessions.DEFAULT_SESSION_LIFETIME_S + 1
    await server._close_abandoned_task_sessions(clock[0])
    server._sandboxes[0].stop.assert_awaited_once()

    body = _VerifyRequest.model_validate(_VERIFY)
    expired = await server.verify(_request(), body)
    assert expired.mask_sample and expired.failure_kind == SESSION_EXPIRED
    server._capture.assert_not_awaited()
    server._grade.assert_not_awaited()

    clock[0] += 11
    lost = await server.verify(_request(), body)
    assert lost.mask_sample and lost.failure_kind == SESSION_LOST
    server._grade.assert_not_awaited()
    assert "resources-session" not in server._task_sessions


async def _seed_whose_cleanup_fails(server: _Server, *, stop_results: list[Any]) -> MagicMock:
    """Seed a session whose preparation fails and whose cleanup stop then fails, so it leaves a record behind."""
    server._fail_preparation = True
    original_start = server._start_task_sandbox

    async def start_with_failing_cleanup(session_id: str, task: _Task) -> str:
        try:
            return await original_start(session_id, task)
        finally:
            server._sandboxes[-1].stop.side_effect = stop_results

    server._start_task_sandbox = start_with_failing_cleanup
    with pytest.raises(RuntimeError, match="preparation failed"):
        await server.seed_task_sandbox_session(_request(), _seed_request(), _Task)
    return server._sandboxes[0]


@pytest.mark.asyncio
async def test_failed_seed_with_failed_cleanup_is_still_closed_after_its_lifetime(monkeypatch: MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(sandbox_sessions, "monotonic", lambda: clock[0])
    server = _server()
    leaked = await _seed_whose_cleanup_fails(server, stop_results=[RuntimeError("provider unavailable"), None])
    assert server._session_id_to_sandbox["resources-session"] is leaked

    clock[0] += sandbox_sessions.DEFAULT_SESSION_LIFETIME_S + 1
    await server._close_abandoned_task_sessions(clock[0])
    assert leaked.stop.await_count == 2, "the lifetime sweep must retry the leaked sandbox of a failed seed"
    assert "resources-session" not in server._session_id_to_sandbox


@pytest.mark.asyncio
async def test_verify_of_a_session_whose_seed_never_finished_is_masked_not_graded() -> None:
    server = _server()
    await _seed_whose_cleanup_fails(server, stop_results=RuntimeError("provider unavailable"))
    assert server._task_sessions["resources-session"].response is None

    result = await server.verify(_request(), _VerifyRequest.model_validate(_VERIFY))
    assert result.mask_sample and result.failure_kind == SESSION_LOST
    assert "seeding" in result.failure_reason
    server._capture.assert_not_awaited()
    server._grade.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_seed_replay_is_rejected_when_the_seeded_sandbox_is_gone() -> None:
    server = _server()
    await server.seed_task_sandbox_session(_request(), _seed_request(), _Task)
    server._session_id_to_sandbox.pop("resources-session")
    with pytest.raises(ResourcesSessionConflictError, match="no longer available"):
        await server.seed_task_sandbox_session(_request(), _seed_request(), _Task)
    assert len(server._sandboxes) == 1


@pytest.mark.asyncio
async def test_concurrent_identical_verifies_grade_once_and_share_the_verdict() -> None:
    server = _server()
    await server.seed_task_sandbox_session(_request(), _seed_request(), _Task)
    gate = asyncio.Event()

    async def slow_grade(patch: str) -> float:
        await gate.wait()
        return 1.0

    server._grade.side_effect = slow_grade
    body = _VerifyRequest.model_validate(_VERIFY)
    first = asyncio.create_task(server.verify(_request(), body))
    second = asyncio.create_task(server.verify(_request(), body))
    await asyncio.sleep(0.01)
    assert not first.done() and not second.done()
    gate.set()
    results = await asyncio.gather(first, second)
    assert results[0] == results[1] and results[0].reward == 1.0
    server._capture.assert_awaited_once()
    server._grade.assert_awaited_once()
    server._sandboxes[0].stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_cancelled_verify_stops_its_grader_and_the_retry_grades_the_original_patch() -> None:
    server = _server()
    await server.seed_task_sandbox_session(_request(), _seed_request(), _Task)
    verifier = _sandbox("verifier")
    grading = asyncio.Event()

    async def hanging_grade(patch: str) -> float:
        server._verifier_sandboxes["resources-session"] = verifier
        grading.set()
        await asyncio.sleep(10)
        return 1.0

    server._grade.side_effect = hanging_grade
    body = _VerifyRequest.model_validate(_VERIFY)
    async with anyio.create_task_group() as group:
        group.start_soon(server.verify, _request(), body)
        await grading.wait()
        group.cancel_scope.cancel()  # the client went away mid-grading

    verifier.stop.assert_awaited_once()
    assert not server._verifier_sandboxes
    assert server._task_sessions["resources-session"].capture == "original patch"
    server._sandboxes[0].stop.assert_awaited_once()

    server._capture.return_value = "replacement patch must not be read"
    server._grade.side_effect = None
    server._grade.return_value = 1.0
    result = await server.verify(_request(), body)
    assert result.reward == 1.0
    server._grade.assert_awaited_with("original patch")
    server._capture.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_seed_replay_waits_for_an_inflight_verify_and_is_then_rejected() -> None:
    server = _server()
    await server.seed_task_sandbox_session(_request(), _seed_request(), _Task)
    gate = asyncio.Event()

    async def slow_grade(patch: str) -> float:
        await gate.wait()
        return 1.0

    server._grade.side_effect = slow_grade
    verifying = asyncio.create_task(server.verify(_request(), _VerifyRequest.model_validate(_VERIFY)))
    await asyncio.sleep(0.01)
    replay = asyncio.create_task(server.seed_task_sandbox_session(_request(), _seed_request(), _Task))
    await asyncio.sleep(0.01)
    assert not replay.done(), "the seed replay waits for the session lock that verify holds"
    gate.set()
    await verifying
    with pytest.raises(ResourcesSessionConflictError, match="started verification"):
        await replay
    assert len(server._sandboxes) == 1


@pytest.mark.asyncio
async def test_close_racing_the_expiry_sweep_stops_the_sandbox_once_and_fences_the_session(
    monkeypatch: MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(sandbox_sessions, "monotonic", lambda: clock[0])
    server = _server()
    await server.seed_task_sandbox_session(_request(), _seed_request(), _Task)
    clock[0] += sandbox_sessions.DEFAULT_SESSION_LIFETIME_S + 1
    # An unrelated request runs the sweep while the owner's close is in flight.
    closing = asyncio.create_task(server.close_resources_session(_request(), _close_request()))
    sweeping = asyncio.create_task(server.seed_task_sandbox_session(_request("other"), _seed_request("other"), _Task))
    await asyncio.gather(closing, sweeping)
    assert server._sandboxes[0].stop.await_count == 1
    assert server._task_session_fence.closed_episode("resources-session") is not None
    assert "resources-session" not in server._session_id_to_sandbox


@pytest.mark.asyncio
async def test_an_expired_session_that_is_being_verified_is_not_swept_and_does_not_block_other_requests(
    monkeypatch: MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(sandbox_sessions, "monotonic", lambda: clock[0])
    server = _server()
    await server.seed_task_sandbox_session(_request("busy"), _seed_request("busy"), _Task)
    gate = asyncio.Event()

    async def slow_grade(patch: str) -> float:
        await gate.wait()
        return 1.0

    server._grade.side_effect = slow_grade
    verifying = asyncio.create_task(server.verify(_request("busy"), _VerifyRequest.model_validate(_VERIFY)))
    await asyncio.sleep(0.01)
    clock[0] += sandbox_sessions.DEFAULT_SESSION_LIFETIME_S + 1
    try:
        async with asyncio.timeout(1):
            await server.seed_task_sandbox_session(_request("other"), _seed_request("other"), _Task)
        assert not verifying.done()
        assert not server._task_sessions["busy"].expired
    finally:
        gate.set()
    result = await verifying
    assert result.reward == 1.0 and not result.mask_sample


@pytest.mark.asyncio
async def test_claim_rejects_a_repeated_claim_and_a_closed_session() -> None:
    """_claim/_release is the API the SWE-bench consumer builds on; its guards need their own coverage."""
    server = _server()
    await server.seed_task_sandbox_session(_request(), _seed_request(), _Task)
    server._claim_task_sandbox("resources-session")
    with pytest.raises(ResourcesSessionConflictError, match="already verified"):
        server._claim_task_sandbox("resources-session")
    await server._release_task_sandbox("resources-session", server._session_id_to_sandbox.pop("resources-session"))
    await server.close_resources_session(_request(), _close_request())
    with pytest.raises(ResourcesSessionConflictError, match="closed"):
        server._claim_task_sandbox("resources-session")


# --- expiry retry pacing, unreleased claims and concurrent shutdown ---------------------------------------------


@pytest.mark.asyncio
async def test_failed_expiry_stop_is_retried_once_per_interval_not_by_every_request(monkeypatch: MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(sandbox_sessions, "monotonic", lambda: clock[0])
    server = _server()
    await server.seed_task_sandbox_session(_request(), _seed_request("stuck"), _Task)
    stuck = server._sandboxes[0]
    stuck.stop.side_effect = RuntimeError("provider unavailable")
    clock[0] += sandbox_sessions.DEFAULT_SESSION_LIFETIME_S + 1

    for index in range(3):
        await server.seed_task_sandbox_session(_request(), _seed_request(f"other-{index}"), _Task)
    assert stuck.stop.await_count == 1, "unrelated requests must not each retry the stuck session"
    assert server._task_sessions["stuck"].expired and "stuck" in server._session_id_to_sandbox

    clock[0] += sandbox_sessions.EXPIRY_RETRY_INTERVAL_S + 1
    await server.seed_task_sandbox_session(_request(), _seed_request("other-3"), _Task)
    assert stuck.stop.await_count == 2

    stuck.stop.side_effect = None
    await server.close_resources_session(_request("stuck"), _close_request("stuck"))
    assert stuck.stop.await_count == 3, "an explicit close does not wait for the retry interval"
    assert "stuck" not in server._session_id_to_sandbox


@pytest.mark.asyncio
async def test_requests_queued_behind_a_failing_expiry_stop_do_not_retry_it(monkeypatch: MonkeyPatch) -> None:
    clock = [1000.0]
    monkeypatch.setattr(sandbox_sessions, "monotonic", lambda: clock[0])
    server = _server()
    await server.seed_task_sandbox_session(_request(), _seed_request("stuck"), _Task)
    stuck = server._sandboxes[0]
    release = asyncio.Event()

    async def fail_after_release() -> None:
        await release.wait()
        raise RuntimeError("provider unavailable")

    stuck.stop.side_effect = fail_after_release
    clock[0] += sandbox_sessions.DEFAULT_SESSION_LIFETIME_S + 1
    seeds = [
        asyncio.create_task(server.seed_task_sandbox_session(_request(), _seed_request(f"other-{index}"), _Task))
        for index in range(4)
    ]
    await asyncio.sleep(0.05)
    release.set()
    await asyncio.gather(*seeds)
    assert stuck.stop.await_count == 1


@pytest.mark.asyncio
async def test_a_claim_that_is_never_released_does_not_keep_close_from_stopping_the_sandbox(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(sandbox_sessions, "STOP_TIMEOUT_S", 0.05)
    server = _server()
    await server.seed_task_sandbox_session(_request(), _seed_request(), _Task)
    server._claim_task_sandbox("resources-session")

    await server.close_resources_session(_request(), _close_request())
    server._sandboxes[0].stop.assert_awaited_once()
    assert not server._session_id_to_sandbox


@pytest.mark.asyncio
async def test_shutdown_stops_the_open_sessions_concurrently() -> None:
    server = _server()
    for index in range(5):
        await server.seed_task_sandbox_session(_request(), _seed_request(f"session-{index}"), _Task)
    stopping = peak = 0

    async def slow_stop() -> None:
        nonlocal stopping, peak
        stopping += 1
        peak = max(peak, stopping)
        await asyncio.sleep(0.05)
        stopping -= 1

    for sandbox in server._sandboxes:
        sandbox.stop.side_effect = slow_stop

    await server.shutdown()
    assert peak == 5, "one slow stop must not delay the others"
    assert all(sandbox.stop.await_count == 1 for sandbox in server._sandboxes)
    assert not server._session_id_to_sandbox and not server._task_sessions


@pytest.mark.asyncio
async def test_a_verify_that_claims_the_task_sandbox_and_fails_does_not_block_close(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(sandbox_sessions, "STOP_TIMEOUT_S", 30.0)
    server = _server()
    await server.seed_task_sandbox_session(_request(), _seed_request(), _Task)

    async def claim_then_fail(request: Request, body: _VerifyRequest) -> BaseVerifyResponse:
        server._claim_task_sandbox("resources-session")
        raise RuntimeError("grading failed")

    server._verify = claim_then_fail
    with pytest.raises(RuntimeError, match="grading failed"):
        await server.verify(_request(), _VerifyRequest.model_validate(_VERIFY))

    await asyncio.wait_for(server.close_resources_session(_request(), _close_request()), timeout=2)
    server._sandboxes[0].stop.assert_awaited_once()
