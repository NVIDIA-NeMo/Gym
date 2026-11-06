# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed SWE-bench sessions preserve task ownership across retries and teardown."""

import asyncio
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import anyio
import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import SESSION_EXPIRED, SESSION_LOST, EpisodeId, TaskId
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

    async def create(test_spec, *, test_assets=(), task_session_id=None, verifier_session_id=None):
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


def test_episode_seed_over_http_reuses_sandbox_and_accepts_caller_assigned_task_id(session_server):
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
    assert server._task_session_fences.get("session") is None
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
    assert server._task_session_fences.get("session") is None
    assert server._session_id_to_sandbox["session"] is task
    await server.close_resources_session(_request(), _close())
    assert server._task_session_fences.get("session") == _seed().episode_id


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
    assert server._task_session_fences.get("session") is None
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


def _extraction_result(task: MagicMock, result: SandboxExecResult) -> None:
    """Make the patch-extraction command (neither `pwd` nor `git rev-parse HEAD`) return ``result``."""
    execute = task.exec.side_effect

    async def with_extraction(command, **kwargs):
        if command in ("pwd", "git rev-parse HEAD"):
            return await execute(command, **kwargs)
        return result

    task.exec.side_effect = with_extraction


def _start_through_the_real_create_sandbox(server, monkeypatch: pytest.MonkeyPatch, box: MagicMock) -> None:
    """Run the real ``_create_sandbox`` (the fixture stubs it) against a provider-level fake ``box``."""
    server._create_sandbox = type(server)._create_sandbox.__get__(server)
    server._make_test_spec.return_value = SimpleNamespace(
        repo="repo", base_commit="base", instance_id="task", instance_image_key="image:key"
    )
    monkeypatch.setattr(mod, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(mod, "resolve_provider_config", lambda *_: MagicMock())
    monkeypatch.setattr(mod, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(mod, "AsyncSandbox", MagicMock(return_value=box))


@pytest.mark.parametrize(
    "rev_parse",
    [
        SandboxExecResult(return_code=0, stdout="seed-head\n", stderr=""),
        SandboxExecResult(return_code=128, stdout="", stderr="fatal: not a git repository"),
        SandboxExecResult(return_code=0, stdout="\n", stderr=""),
    ],
    ids=["seed-head", "git-unavailable", "empty-head"],
)
async def test_patch_is_diffed_against_the_seed_head_in_the_agent_workdir(session_server, rev_parse):
    server, task, _, _ = session_server
    execute = task.exec.side_effect

    async def exec_with_rev_parse(command, **kwargs):
        return rev_parse if command == "git rev-parse HEAD" else await execute(command, **kwargs)

    task.exec.side_effect = exec_with_rev_parse
    await server.seed_session(_request(), _seed())
    await server.verify(_request(), mod.SWEBenchVerifyRequest.model_validate(VERIFY_BODY))
    command = task.exec.await_args_list[-1].args[0]
    expected_base = "seed-head" if rev_parse.stdout.strip() else "base"  # falls back to the task's base commit
    assert command.startswith("cd /testbed && git add -u ")
    assert command.endswith(f"cd /testbed && git --no-pager diff --cached {expected_base}")


async def test_seed_head_is_recorded_after_the_anti_cheat_scrub(session_server):
    server, task, _, _ = session_server
    server.config.apply_anti_cheating = True
    await server.seed_session(_request(), _seed())
    commands = [call.args[0] for call in task.exec.await_args_list]
    assert commands == [
        "pwd",
        "git reset --hard && WORKING_DIRECTORY=/testbed bash anti_cheat_setup.sh && rm anti_cheat_setup.sh",
        "git rev-parse HEAD",
    ]
    assert task.upload.await_args.args[1] == "/testbed/anti_cheat_setup.sh"


def test_agent_run_seed_cannot_replace_an_episode_session(session_server):
    server, task, _, _ = session_server
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        flat = client.post("/seed_session", json=_seed().task_data)
        assert flat.status_code == 409, flat.text
        assert "agent-run seed cannot replace an episode session" in flat.text
        task.stop.assert_not_awaited()
        server._create_sandbox.assert_awaited_once()
        assert client.post("/close_session", json=_close().model_dump(mode="json")).status_code == 200
        assert not server._session_id_to_seed_head
    task.stop.assert_awaited_once()


async def test_failed_flat_seed_stops_the_task_sandbox(session_server):
    server, task, _, _ = session_server
    task.exec.side_effect = None
    task.exec.return_value = SandboxExecResult(return_code=1, stdout="", stderr="no cwd")
    with pytest.raises(RuntimeError, match="workdir"):
        await server.seed_session(_request(), mod.SWEBenchSeedSessionRequest.model_validate(VERIFY_BODY))
    task.stop.assert_awaited_once()
    assert not server._session_id_to_sandbox and not server._session_id_to_seed_head


def test_http_verify_after_expiry_is_masked_as_expired(session_server, monkeypatch: pytest.MonkeyPatch):
    server, task, _, grader = session_server
    clock = [1000.0]
    monkeypatch.setattr("resources_servers.swebench.sandbox_sessions.monotonic", lambda: clock[0])
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed().model_dump(mode="json")).status_code == 200
        cookies = dict(client.cookies)
        clock[0] += 18000 + 1  # the fixture's sandbox_config.ttl_s
        # The next session call closes the abandoned one.
        assert client.post("/seed_session", json=_seed(session_id="next").model_dump(mode="json")).status_code == 200
        task.stop.assert_awaited_once()
        client.cookies.clear()
        client.cookies.update(cookies)
        server._create_sandbox.reset_mock()
        response = client.post("/verify", json=VERIFY_BODY)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["mask_sample"] is True and result["failure_kind"] == SESSION_EXPIRED
        assert result["reward"] == 0.0 and result["evaluation_completed"] is False
        assert "expired" in result["failure_reason"]
        server._create_sandbox.assert_not_awaited()
        grader.assert_not_awaited()


async def test_task_sandbox_that_fails_to_start_stays_owned_for_a_close_retry(session_server, monkeypatch):
    server, _, _, _ = session_server
    box = _sandbox()
    box.start_with_setup = AsyncMock(side_effect=RuntimeError("provider create failed"))
    box.stop.side_effect = [RuntimeError("stop failed"), None]
    _start_through_the_real_create_sandbox(server, monkeypatch, box)
    with pytest.raises(RuntimeError, match="provider create failed"):
        await server.seed_session(_request(), _seed())
    # The failed stop leaves the registered sandbox for the retried close.
    assert server._session_id_to_sandbox["session"] is box
    await server.close_resources_session(_request(), _close())
    assert box.stop.await_count == 2
    assert not server._session_id_to_sandbox and not server._task_sessions


async def test_verifier_sandbox_that_fails_to_start_is_stopped(session_server, monkeypatch):
    server, task, _, grader = session_server
    await server.seed_session(_request(), _seed())
    box = _sandbox()
    box.start_with_setup = AsyncMock(side_effect=RuntimeError("verifier create failed"))
    _start_through_the_real_create_sandbox(server, monkeypatch, box)
    body = mod.SWEBenchVerifyRequest.model_validate(VERIFY_BODY)
    with pytest.raises(RuntimeError, match="verifier create failed"):
        await server.verify(_request(), body)
    box.stop.assert_awaited_once()
    assert not server._verifier_sandboxes
    grader.assert_not_awaited()
    # The patch was saved before the task sandbox stopped, so the retry grades it without a task sandbox.
    task.stop.assert_awaited_once()
    server._create_sandbox = AsyncMock(side_effect=lambda *a, **k: _sandbox())
    result = await server.verify(_request(), body)
    assert result.reward == 1 and result.model_patch == PATCH


async def test_extraction_error_is_masked_released_and_replayed(session_server):
    server, task, _, grader = session_server
    await server.seed_session(_request(), _seed())
    _extraction_result(task, SandboxExecResult(return_code=0, stdout="", stderr="", error_type="timeout"))
    # The empty patch cannot apply; the agent's patch was never collected, so that is no verdict on it.
    grader.return_value = {"completed": False, "resolved": False, "patch_apply_failed": True, "error": "apply"}
    body = mod.SWEBenchVerifyRequest.model_validate(VERIFY_BODY)
    first = await server.verify(_request(), body)
    assert first.mask_sample is True and first.failure_kind == "verifier_error"
    assert first.failure_reason == "Failed to extract the model patch from the agent sandbox"
    assert first.model_patch is None and first.model_patch_sha256 is None
    task.stop.assert_awaited_once()
    assert grader.await_args.kwargs["pred"]["model_patch"] == ""
    assert await server.verify(_request(), body) == first
    grader.assert_awaited_once()
    await server.close_resources_session(_request(), _close())


def test_task_data_matches_the_instance_request():
    from resources_servers.swebench.task_data import TaskData

    wire = mod.SWEBenchInstanceRequest.model_fields
    assert set(TaskData.model_fields) == set(wire)
    # image_assets is the one optional field: only SWE-bench Multimodal rows carry it.
    assert {n for n, f in TaskData.model_fields.items() if f.is_required()} == set(wire) - {"image_assets"}
    assert not TaskData.model_fields["image_assets"].is_required()
    assert {n: f.annotation for n, f in TaskData.model_fields.items()} == {n: f.annotation for n, f in wire.items()}


async def test_episode_seed_withholds_the_test_assets_from_the_task_sandbox_and_verify_uploads_them(session_server):
    from resources_servers.swebench.tests.test_app import CHART_ROW, PNG_A, PNG_B

    server, _, _, _ = session_server
    seed = _seed().model_copy(update={"task_data": mod.SWEBenchInstanceRequest.model_validate(CHART_ROW).model_dump()})
    await server.seed_session(_request(), seed)
    await server.verify(_request(), mod.SWEBenchVerifyRequest.model_validate(CHART_ROW))

    seed_call, verify_call = server._create_sandbox.await_args_list
    assert "test_assets" not in seed_call.kwargs
    assert [asset.content for asset in verify_call.kwargs["test_assets"]] == [PNG_A, PNG_B]


EXAMPLE_ROWS = [
    json.loads(line) for line in (Path(mod.__file__).parent / "data/example.jsonl").read_text().splitlines()
]


@pytest.mark.parametrize("row", EXAMPLE_ROWS, ids=lambda row: row["instance_id"])
async def test_committed_rows_pass_the_episode_seed_and_verify_identity_check(session_server, row):
    server, _, _, grader = session_server
    task_data = {key: value for key, value in row.items() if key != "responses_create_params"}
    seed = _seed().model_copy(update={"task_data": task_data})
    await server.seed_session(_request(), seed)
    # The flat body single_agent_turn sends: the task's own fields plus the request and the agent's response.
    verify_body = (
        task_data | {"responses_create_params": row["responses_create_params"]} | {"response": VERIFY_BODY["response"]}
    )
    result = await server.verify(_request(), mod.SWEBenchVerifyRequest.model_validate(verify_body))
    assert result.instance_id == row["instance_id"] and result.reward == 1 and not result.mask_sample
    grader.assert_awaited_once()
    await server.close_resources_session(_request(), _close())


def test_flat_seeded_session_is_verified_from_its_own_sandbox(session_server):
    server, task, _, grader = session_server
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_seed().task_data).json() == {"sandbox_handle": "task-box"}
        response = client.post("/verify", json=VERIFY_BODY)
        assert response.status_code == 200, response.text
        result = response.json()
        assert result["reward"] == 1 and result["mask_sample"] is False and result["model_patch"] == PATCH
        assert grader.await_args.kwargs["pred"]["model_patch"] == PATCH
        assert task.exec.await_args_list[-1].args[0].endswith("cd /testbed && git --no-pager diff --cached seed-head")
        task.stop.assert_awaited_once()
        # The task sandbox is consumed: a second verify is masked as lost, not scored 0.
        again = client.post("/verify", json=VERIFY_BODY).json()
        assert again["mask_sample"] is True and again["failure_kind"] == SESSION_LOST
        grader.assert_awaited_once()


def test_a_failed_task_stop_after_capture_still_grades_a_flat_seeded_session(session_server):
    server, task, _, grader = session_server
    task.stop.side_effect = [RuntimeError("stop failed"), None]
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=_seed().task_data).status_code == 200
        graded = client.post("/verify", json=VERIFY_BODY)
        assert graded.status_code == 200, "a session with no saved patch and no close must not lose its grade"
        assert graded.json()["reward"] == 1 and graded.json()["mask_sample"] is False
        grader.assert_awaited_once()
        assert server._session_id_to_sandbox, "the unconfirmed stop keeps the sandbox for shutdown to retry"
    assert task.stop.await_count == 2 and not server._session_id_to_sandbox


async def test_a_git_failure_in_the_agents_repository_is_its_graded_zero_not_a_masked_row(session_server):
    server, task, _, grader = session_server
    await server.seed_session(_request(), _seed())
    # What `git diff --cached <seed head>` returns once the agent removed .git or pruned the seed commit.
    _extraction_result(task, SandboxExecResult(return_code=128, stdout="", stderr="fatal: not a git repository"))
    grader.return_value = {"completed": False, "resolved": False, "patch_apply_failed": True, "error": "apply"}
    result = await server.verify(_request(), mod.SWEBenchVerifyRequest.model_validate(VERIFY_BODY))
    assert result.reward == 0 and result.mask_sample is False
    assert grader.await_args.kwargs["pred"]["model_patch"] == ""
    task.stop.assert_awaited_once()


async def test_flat_seed_records_the_workdir_so_verify_does_not_ask_the_agents_sandbox_for_it(session_server):
    server, task, _, _ = session_server
    await server.seed_session(_request(), mod.SWEBenchSeedSessionRequest.model_validate(VERIFY_BODY))
    assert server._task_sessions["session"].workdir == "/testbed"
    await server.verify(_request(), mod.SWEBenchVerifyRequest.model_validate(VERIFY_BODY))
    pwd_calls = [call for call in task.exec.await_args_list if call.args[0] == "pwd"]
    assert len(pwd_calls) == 1, "only the seed resolves the workdir"


async def test_a_failed_flat_seed_keeps_its_own_error_when_the_cleanup_stop_fails(session_server):
    server, task, _, _ = session_server
    task.exec.side_effect = None
    task.exec.return_value = SandboxExecResult(return_code=1, stdout="", stderr="no cwd")
    task.stop.side_effect = [RuntimeError("provider close failed"), None]
    with pytest.raises(RuntimeError, match="workdir"):
        await server.seed_session(_request(), mod.SWEBenchSeedSessionRequest.model_validate(VERIFY_BODY))
    assert server._session_id_to_sandbox, "the failed stop keeps the handle for a retry"
    await server.shutdown()
    assert task.stop.await_count == 2 and not server._session_id_to_sandbox
