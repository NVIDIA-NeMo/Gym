# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed SWE-bench sessions preserve task ownership across retries and teardown."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import SESSION_LOST, EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    ResourcesSessionConflictError,
)
from resources_servers.swebench import app as mod
from resources_servers.swebench.tests.test_app import VERIFY_BODY


PATCH = "diff --git a/chart.js b/chart.js\n--- a/chart.js\n+++ b/chart.js\n@@ -1 +1 @@\n-bug\n+fixed\n"


def _sandbox() -> MagicMock:
    sandbox = MagicMock()
    sandbox._handle = SandboxHandle(sandbox_id="task-box", provider_name="test-provider", raw=None)

    async def execute(command, **kwargs):
        output = "/testbed\n" if command == "pwd" else "seed-head\n" if command == "git rev-parse HEAD" else PATCH
        return SandboxExecResult(return_code=0, stdout=output, stderr="")

    sandbox.exec = AsyncMock(side_effect=execute)
    sandbox.upload = AsyncMock()
    sandbox.serialize = AsyncMock(return_value={"sandbox_id": "task-box", "provider": "test-provider"})
    sandbox.stop = AsyncMock()
    return sandbox


def _seed(*, session_id: str = "session", task_id: str = "caller-row-0") -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EpisodeId(rollout_id="rollout", attempt=0),
        task_id=TaskId(taskset="swebench:multimodal", task_id=task_id),
        task_data=mod.SWEBenchInstanceRequest.model_validate(VERIFY_BODY).model_dump(),
    )


def _request(session_id: str = "session") -> SimpleNamespace:
    return SimpleNamespace(session={SESSION_ID_KEY: session_id})


def _close(session_id: str = "session") -> ResourcesCloseSessionRequest:
    return ResourcesCloseSessionRequest(resources_session_id=session_id, episode_id=_seed().episode_id)


@pytest.fixture
def session_server(monkeypatch: pytest.MonkeyPatch):
    server = mod.SwebenchResourcesServer(
        config=mod.SwebenchResourcesServerConfig(
            host="localhost",
            port=8000,
            name="swebench",
            entrypoint="app.py",
            num_workers=1,
            sandbox_provider="sandbox",
            sandbox_config={"ttl_s": 18000},
            apply_anti_cheating=False,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    task, verifier = _sandbox(), _sandbox()
    server._make_test_spec = MagicMock(
        return_value=SimpleNamespace(repo="repo", base_commit="base", instance_id="task")
    )

    async def create(test_spec, *, task_session_id=None, verifier_session_id=None):
        sandbox = verifier if verifier_session_id is not None else task
        if task_session_id is not None:
            server._session_id_to_sandbox[task_session_id] = sandbox
        if verifier_session_id is not None:
            server._verifier_sandboxes[verifier_session_id] = sandbox
        return sandbox

    server._create_sandbox = AsyncMock(side_effect=create)
    grader = AsyncMock(return_value={"completed": True, "resolved": True})
    monkeypatch.setattr(mod, "run_instance", grader)
    return server, task, verifier, grader


def test_typed_http_seed_reuses_sandbox_and_accepts_caller_assigned_task_id(session_server):
    server, task, _, _ = session_server
    with TestClient(server.setup_webserver()) as client:
        body = _seed().model_dump(mode="json")
        first = client.post("/seed_session", json=body)
        assert first.status_code == 200, first.text
        assert first.json()["resources_session_id"] == "session"
        assert first.json()["sandbox_access"] == {
            "connection": {
                "kind": "direct",
                "provider_config_ref": "sandbox",
                "descriptor": {"sandbox_id": "task-box", "provider": "test-provider"},
            },
            "workdir": "/testbed",
        }
        assert client.post("/seed_session", json=body).json() == first.json()
        server._create_sandbox.assert_awaited_once()
        task.serialize.assert_awaited_once_with(scope="operate")
        # A lost seed reply may leave the Environment Server without its cookie.
        client.cookies.clear()
        closed = client.post("/close_session", json=_close().model_dump(mode="json"))
        assert closed.status_code == 200, closed.text
        assert client.post("/close_session", json=_close().model_dump(mode="json")).json() == closed.json()
        assert client.post("/seed_session", json=body).status_code == 409
    task.stop.assert_awaited_once()


@pytest.mark.parametrize("field", ["episode_id", "task_id", "task_data"])
def test_conflicting_http_seed_does_not_replace_existing_session(session_server, field):
    server, task, _, _ = session_server
    with TestClient(server.setup_webserver()) as client:
        body = _seed().model_dump(mode="json")
        assert client.post("/seed_session", json=body).status_code == 200
        changed = deepcopy(body)
        if field == "episode_id":
            changed[field]["rollout_id"] = "other"
        elif field == "task_id":
            changed[field]["task_id"] = "other"
        else:
            changed[field]["problem_statement"] = "different task"
        assert client.post("/seed_session", json=changed).status_code == 409
        assert client.post("/seed_session", json=body).status_code == 200
        assert client.post("/close_session", json=_close().model_dump(mode="json")).status_code == 200
    server._create_sandbox.assert_awaited_once()
    task.stop.assert_awaited_once()


def test_invalid_task_is_rejected_before_creating_sandbox(session_server):
    server, _, _, _ = session_server
    with TestClient(server.setup_webserver()) as client:
        body = _seed().model_dump(mode="json")
        body["task_data"] = {}
        assert client.post("/seed_session", json=body).status_code == 422
    server._create_sandbox.assert_not_awaited()


@pytest.mark.parametrize("session_state", ["never_seeded", "closed", "lost"])
def test_http_verify_masks_unavailable_session_without_creating_sandbox(session_server, session_state):
    server, _, _, grader = session_server
    with TestClient(server.setup_webserver()) as client:
        if session_state != "never_seeded":
            assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
            if session_state == "closed":
                cookies = dict(client.cookies)
                assert client.post("/close_session", json=_close().model_dump(mode="json")).status_code == 200
                # Reuse the original cookie to exercise the retained closed-session fence.
                client.cookies.clear()
                client.cookies.update(cookies)
            else:
                server._session_id_to_sandbox.pop("session")
        server._create_sandbox.reset_mock()
        server._make_test_spec.reset_mock()
        response = client.post("/verify", json=VERIFY_BODY)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["mask_sample"] is True
        assert result["failure_kind"] == SESSION_LOST
        assert result["reward"] == 0.0
        assert result["evaluation_completed"] is False
        assert result["resolved"] is False
        assert result["model_patch"] is None
        assert result["model_patch_sha256"] is None
        assert result["eval_sandbox_start_time_taken"] == 0.0
        if session_state == "closed":
            assert "closed" in result["failure_reason"]
        server._make_test_spec.assert_not_called()
        server._create_sandbox.assert_not_awaited()
        grader.assert_not_awaited()


def test_http_golden_patch_verify_does_not_require_seed(session_server):
    server, _, _, grader = session_server
    server.config.is_verifying_golden_patch = True
    with TestClient(server.setup_webserver()) as client:
        response = client.post("/verify", json=VERIFY_BODY)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["evaluation_completed"] is True
        assert result["reward"] == 1
        assert result["mask_sample"] is False
        assert result["model_patch"] == VERIFY_BODY["patch"]
        assert grader.await_args.kwargs["pred"]["model_patch"] == VERIFY_BODY["patch"]
        grader.assert_awaited_once()
        server._create_sandbox.assert_awaited_once()


async def test_unknown_close_fences_late_seed_and_rejects_another_episode(session_server):
    server, _, _, _ = session_server
    await server.close_resources_session(_request(), _close())
    with pytest.raises(ResourcesSessionConflictError, match="closed"):
        await server.seed_session(_request(), _seed())
    with pytest.raises(ResourcesSessionConflictError, match="episode_id"):
        await server.close_resources_session(
            _request(),
            _close().model_copy(update={"episode_id": EpisodeId(rollout_id="other", attempt=0)}),
        )
    server._create_sandbox.assert_not_awaited()


async def test_failed_handoff_retains_failed_cleanup_for_close_retry(session_server):
    server, task, _, _ = session_server
    task.serialize.side_effect = RuntimeError("handoff failed")
    task.stop.side_effect = [RuntimeError("stop failed"), None]
    with pytest.raises(RuntimeError, match="handoff failed"):
        await server.seed_session(_request(), _seed())
    assert server._session_id_to_sandbox["session"] is task
    assert server._task_sessions["session"].response is None
    assert server._task_session_fence.closed_episode("session") is None
    await server.close_resources_session(_request(), _close())
    assert not server._session_id_to_sandbox
    assert not server._task_sessions
    assert task.stop.await_count == 2


async def test_failed_seed_can_retry_after_failed_cleanup(session_server):
    server, task, _, _ = session_server
    task.serialize.side_effect = [RuntimeError("handoff failed"), {"sandbox_id": "retry-box"}]
    task.stop.side_effect = [RuntimeError("stop failed"), None, None]
    with pytest.raises(RuntimeError, match="handoff failed"):
        await server.seed_session(_request(), _seed())
    result = await server.seed_session(_request(), _seed())
    assert result.sandbox_access.connection.descriptor == {"sandbox_id": "retry-box"}
    assert task.stop.await_count == 2
    assert server._create_sandbox.await_count == 2
    await server.close_resources_session(_request(), _close())


async def test_cancelled_seed_stops_task_before_exposing_access(session_server):
    server, task, _, _ = session_server
    task.serialize.side_effect = asyncio.CancelledError
    with pytest.raises(asyncio.CancelledError):
        await server.seed_session(_request(), _seed())
    task.stop.assert_awaited_once()
    assert not server._task_sessions
    assert not server._session_id_to_sandbox


@pytest.mark.parametrize(
    "pwd_result",
    [
        SandboxExecResult(return_code=1, stdout="/testbed", stderr=""),
        SandboxExecResult(return_code=0, stdout="/testbed", stderr="", error_type="timeout"),
        SandboxExecResult(return_code=0, stdout="relative/path", stderr=""),
    ],
)
async def test_invalid_workdir_stops_task_before_handoff(session_server, pwd_result):
    server, task, _, _ = session_server
    task.exec.side_effect = None
    task.exec.return_value = pwd_result
    with pytest.raises(RuntimeError, match="workdir"):
        await server.seed_session(_request(), _seed())
    task.stop.assert_awaited_once()
    task.serialize.assert_not_awaited()
    assert not server._task_sessions


async def test_close_failure_keeps_session_and_does_not_fence(session_server):
    server, task, _, _ = session_server
    await server.seed_session(_request(), _seed())
    task.stop.side_effect = [RuntimeError("stop failed"), None]
    with pytest.raises(RuntimeError, match="stop failed"):
        await server.close_resources_session(_request(), _close())
    assert server._task_session_fence.closed_episode("session") is None
    assert server._session_id_to_sandbox["session"] is task
    await server.close_resources_session(_request(), _close())
    assert server._task_session_fence.closed_episode("session") == _seed().episode_id


async def test_close_cleanup_survives_anyio_cancellation(session_server):
    server, task, _, _ = session_server
    await server.seed_session(_request(), _seed())
    stopped = False

    async def stop():
        nonlocal stopped
        await anyio.sleep(0)
        stopped = True

    task.stop.side_effect = stop
    with anyio.CancelScope() as scope:
        scope.cancel()
        # The caller has already acquired ownership of cleanup; cancellation must not interrupt provider stop.
        await server._stop_task_sandbox("session")
    assert stopped
    assert not server._session_id_to_sandbox


async def test_close_timeout_retains_owner_for_retry(session_server, monkeypatch):
    monkeypatch.setattr("resources_servers.swebench.sandbox_sessions.STOP_TIMEOUT_S", 0.05)
    server, task, _, _ = session_server
    await server.seed_session(_request(), _seed())

    async def blocked():
        await anyio.sleep_forever()

    task.stop.side_effect = blocked
    with pytest.raises(TimeoutError):
        await server.close_resources_session(_request(), _close())
    assert server._session_id_to_sandbox["session"] is task
    task.stop.side_effect = None
    await server.close_resources_session(_request(), _close())


async def test_verification_retry_reuses_patch_after_task_stop_failure(session_server):
    server, task, verifier, grader = session_server
    await server.seed_session(_request(), _seed())
    body = mod.SWEBenchVerifyRequest.model_validate(VERIFY_BODY)
    task.stop.side_effect = [RuntimeError("stop failed"), None]
    with pytest.raises(RuntimeError, match="stop failed"):
        await server.verify(_request(), body)
    assert server._task_sessions["session"].capture == PATCH
    grader.assert_not_awaited()
    extraction_calls = task.exec.await_count
    result = await server.verify(_request(), body)
    assert result.reward == 1 and result.model_patch == PATCH
    assert grader.await_args.kwargs["pred"]["model_patch"] == PATCH
    assert task.exec.await_count == extraction_calls
    assert "session" not in server._session_id_to_sandbox
    assert await server.verify(_request(), body) == result
    grader.assert_awaited_once()
    verifier.stop.assert_awaited_once()
    await server.close_resources_session(_request(), _close())
    assert not server._task_sessions


async def test_verification_rejects_changed_task_and_reseed(session_server):
    server, _, _, grader = session_server
    await server.seed_session(_request(), _seed())
    body = mod.SWEBenchVerifyRequest.model_validate(VERIFY_BODY)
    with pytest.raises(ResourcesSessionConflictError, match="task"):
        await server.verify(_request(), body.model_copy(update={"instance_id": "other"}))
    await server.verify(_request(), body)
    with pytest.raises(ResourcesSessionConflictError, match="verification"):
        await server.seed_session(_request(), _seed())
    with pytest.raises(ResourcesSessionConflictError, match="input"):
        await server.verify(
            _request(),
            body.model_copy(
                update={"responses_create_params": body.responses_create_params.model_copy(update={"input": "other"})}
            ),
        )
    grader.assert_awaited_once()
    await server.close_resources_session(_request(), _close())


async def test_verifier_failure_releases_verifier_and_can_retry_saved_patch(session_server):
    server, task, verifier, grader = session_server
    await server.seed_session(_request(), _seed())
    grader.side_effect = [RuntimeError("grading transport failed"), {"completed": True, "resolved": True}]
    body = mod.SWEBenchVerifyRequest.model_validate(VERIFY_BODY)
    with pytest.raises(RuntimeError, match="grading transport"):
        await server.verify(_request(), body)
    assert server._task_sessions["session"].capture == PATCH
    assert "session" not in server._verifier_sandboxes
    assert (await server.verify(_request(), body)).model_patch == PATCH
    task.stop.assert_awaited_once()
    assert verifier.stop.await_count == 2
    await server.close_resources_session(_request(), _close())


async def test_close_waits_for_verification_and_keeps_failed_verifier_teardown(session_server):
    server, _, verifier, grader = session_server
    await server.seed_session(_request(), _seed())
    started = asyncio.Event()
    finish = asyncio.Event()

    async def grade(**kwargs):
        started.set()
        await finish.wait()
        return {"completed": True, "resolved": True}

    grader.side_effect = grade
    verifier.stop.side_effect = [RuntimeError("stop failed"), RuntimeError("stop still failed"), None]
    async with asyncio.timeout(2):
        verification = asyncio.create_task(
            server.verify(_request(), mod.SWEBenchVerifyRequest.model_validate(VERIFY_BODY))
        )
        await started.wait()
        close = asyncio.create_task(server.close_resources_session(_request(), _close()))
        await asyncio.sleep(0)
        assert not close.done()
        finish.set()
        assert (await verification).reward == 1
        with pytest.raises(RuntimeError, match="stop still failed"):
            await close
    assert server._verifier_sandboxes["session"] is verifier
    assert server._task_session_fence.closed_episode("session") is None
    await server.close_resources_session(_request(), _close())
    assert not server._verifier_sandboxes


async def test_shutdown_stops_abandoned_task_and_retains_failure(session_server):
    server, task, _, _ = session_server
    await server.seed_session(_request(), _seed())
    task.stop.side_effect = [RuntimeError("shutdown stop failed"), None]
    await server.shutdown()
    assert server._session_id_to_sandbox["session"] is task
    await server.shutdown()
    assert not server._session_id_to_sandbox
    assert not server._task_sessions


def test_flat_seed_response_is_preserved(session_server):
    server, task, _, _ = session_server
    with TestClient(server.setup_webserver()) as client:
        result = client.post("/seed_session", json=_seed().task_data)
        assert result.status_code == 200, result.text
        assert result.json() == {"sandbox_handle": "task-box"}
        assert all(record.seed is None for record in server._task_sessions.values())
    task.stop.assert_awaited_once()
