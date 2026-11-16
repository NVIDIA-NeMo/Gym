# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Both SWE consumers preserve task ownership and grading when using shared sessions."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.core import SESSION_LOST
from nemo_gym.sandbox import SandboxExecResult, SandboxHandle
from nemo_gym.server_runtime import ResourcesCloseSessionRequest, ResourcesSessionConflictError
from resources_servers.r2e_gym import app as r2e_app
from resources_servers.r2e_gym.tests import test_sessions as r2e_sessions
from resources_servers.swe_gym import app as swe_app
from resources_servers.swe_gym.tests import test_sessions as swe_sessions


@pytest.fixture(params=[(swe_app, swe_sessions, "SWEGym"), (r2e_app, r2e_sessions, "R2EGym")], ids=["swe", "r2e"])
def consumer(request, monkeypatch):
    module, fixtures, prefix = request.param
    server_type = getattr(module, f"{prefix}ResourcesServer")
    create_sandbox = server_type._create_sandbox
    server, sandboxes = fixtures._server(monkeypatch)
    return module, fixtures, server, sandboxes, create_sandbox


def close_request(fixtures):
    seed = fixtures._seed()
    return ResourcesCloseSessionRequest(resources_session_id=seed.resources_session_id, episode_id=seed.episode_id)


@pytest.mark.parametrize("state", ["unseeded", "missing", "closed"])
def test_unavailable_task_is_masked_without_a_grading_sandbox(consumer, state):
    _, fixtures, server, sandboxes, _ = consumer
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        if state != "unseeded":
            seeded = client.post("/seed_session", json=fixtures._seed().model_dump(mode="json"))
            assert seeded.status_code == 200, seeded.text
            if state == "missing":
                server._session_id_to_sandbox.pop(fixtures._seed().resources_session_id)
            else:
                cookies = dict(client.cookies)
                closed = client.post("/close_session", json=close_request(fixtures).model_dump(mode="json"))
                assert closed.status_code == 200, closed.text
                client.cookies.clear()
                client.cookies.update(cookies)
        before = len(sandboxes)
        verified = client.post("/verify", json=fixtures._VERIFY)
        assert verified.status_code == 200, verified.text
        result = verified.json()
        assert result["mask_sample"] and result["failure_kind"] == SESSION_LOST
        assert not result["evaluation_completed"] and result["reward"] == 0
        assert len(sandboxes) == before


def test_envelope_task_id_is_independent_but_rebinding_and_wrong_task_are_rejected(consumer):
    _, fixtures, server, sandboxes, _ = consumer
    seed = fixtures._seed().model_dump(mode="json")
    seed["task_id"]["task_id"] = "materialized-task-key"
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        response = client.post("/seed_session", json=seed)
        assert response.status_code == 200, response.text
        rebound = seed | {"task_id": seed["task_id"] | {"task_id": "another-task-key"}}
        assert client.post("/seed_session", json=rebound).status_code == 409
        wrong_task = fixtures._VERIFY | {"instance_id": "different-instance"}
        assert client.post("/verify", json=wrong_task).status_code == 409
        assert len(sandboxes) == 1
        response = client.post("/verify", json=fixtures._VERIFY)
        assert response.status_code == 200 and response.json()["reward"] == 1, response.text
        altered = fixtures._VERIFY | {"responses_create_params": {"input": "A different task"}}
        assert client.post("/verify", json=altered).status_code == 409
        assert len(sandboxes) == 2


def test_capture_survives_failed_task_teardown_and_preserves_capture_mode(consumer):
    module, fixtures, server, sandboxes, _ = consumer
    server.config.patch_capture_mode = "committed"
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=fixtures._seed().model_dump(mode="json")).status_code == 200
        task = sandboxes[0]
        task.stop.side_effect = [RuntimeError("stop unavailable"), None]
        assert client.post("/verify", json=fixtures._VERIFY).status_code == 500
        assert server._session_id_to_sandbox[fixtures._seed().resources_session_id] is task
        assert len(sandboxes) == 1
        retried = client.post("/verify", json=fixtures._VERIFY)
        assert retried.status_code == 200 and retried.json()["reward"] == 1, retried.text
        assert retried.json()["model_patch"] == "diff --git a/x b/x\n"
        assert module.capture_model_patch.await_count == 1
        assert module.capture_model_patch.await_args.kwargs["mode"] == "committed"
        assert task.stop.await_count == 2
        assert len(sandboxes) == 2


async def test_close_waits_for_verification_and_releases_each_sandbox_once(consumer, monkeypatch):
    module, fixtures, server, sandboxes, _ = consumer
    entered, release = asyncio.Event(), asyncio.Event()
    grade = module.run_verification

    async def blocked_grade(**kwargs):
        entered.set()
        await release.wait()
        return await grade(**kwargs)

    monkeypatch.setattr(module, "run_verification", blocked_grade)
    seed = fixtures._seed()
    request = Request({"type": "http", "session": {SESSION_ID_KEY: seed.resources_session_id}})
    await server.seed_session(request, seed)
    verify_type = swe_app.SWEGymVerifyRequest if module is swe_app else r2e_app.R2EGymVerifyRequest
    verifying = asyncio.create_task(server.verify(request, verify_type.model_validate(fixtures._VERIFY)))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        closing = asyncio.create_task(server.close_resources_session(request, close_request(fixtures)))
        await asyncio.sleep(0)
        assert not closing.done()
        sandboxes[0].stop.assert_awaited_once()
        sandboxes[1].stop.assert_not_awaited()
        release.set()
        assert (await verifying).reward == 1
        await closing
        assert [sandbox.stop.await_count for sandbox in sandboxes] == [1, 1]
        assert not server._session_id_to_sandbox and not server._verifier_sandboxes
    finally:
        release.set()
        await asyncio.gather(verifying, *([closing] if closing is not None else []), return_exceptions=True)
        await server.shutdown()


def test_partial_task_start_remains_owned_when_cleanup_fails(consumer, monkeypatch):
    module, fixtures, server, _, create_sandbox = consumer
    monkeypatch.setattr(type(server), "_create_sandbox", create_sandbox)
    sandbox = MagicMock()
    sandbox.start = AsyncMock(side_effect=RuntimeError("provider start outcome unknown"))
    sandbox.stop = AsyncMock(side_effect=[RuntimeError("stop unavailable"), None])
    sandbox.exec = AsyncMock(return_value=SandboxExecResult(return_code=0, stdout="", stderr=""))
    monkeypatch.setattr(module, "AsyncSandbox", lambda config: sandbox)
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {server.config.sandbox_provider: {"test": {}}})
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=fixtures._seed().model_dump(mode="json"))
        assert seeded.status_code == 500
        assert server._session_id_to_sandbox[fixtures._seed().resources_session_id] is sandbox
        closed = client.post("/close_session", json=close_request(fixtures).model_dump(mode="json"))
        assert closed.status_code == 200, closed.text
        assert not server._session_id_to_sandbox
    assert sandbox.stop.await_count == 2


def test_failed_close_keeps_task_owned_until_successful_retry(consumer):
    _, fixtures, server, sandboxes, _ = consumer
    body = close_request(fixtures).model_dump(mode="json")
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=fixtures._seed().model_dump(mode="json")).status_code == 200
        task = sandboxes[0]
        task.stop.side_effect = [RuntimeError("stop unavailable"), None]
        assert client.post("/close_session", json=body).status_code == 500
        assert server._session_id_to_sandbox[body["resources_session_id"]] is task
        assert client.post("/close_session", json=body).status_code == 200
        assert client.post("/close_session", json=body).status_code == 200
        assert not server._session_id_to_sandbox
        assert task.stop.await_count == 2


def test_seedless_golden_verification_keeps_hidden_files_in_the_grader_only(consumer):
    module, fixtures, server, sandboxes, _ = consumer
    server.config.is_verifying_golden_patch = True
    body = fixtures._VERIFY | {"patch": "diff --git a/fix.py b/fix.py\n"}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        verified = client.post("/verify", json=body)
        assert verified.status_code == 200 and verified.json()["reward"] == 1, verified.text
        assert verified.json()["patch_source"] == "golden"
        module.capture_model_patch.assert_not_awaited()
        assert len(sandboxes) == 1
        assert "diff --git a/fix.py b/fix.py\n" in sandboxes[0].initial_files.values()
        sandboxes[0].stop.assert_awaited_once()


def test_seed_task_never_receives_golden_or_hidden_verification_files(consumer):
    _, fixtures, server, sandboxes, _ = consumer
    seed = fixtures._seed().model_dump(mode="json")
    seed["task_data"] |= {"patch": "golden patch", "test_patch": "hidden tests"}
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=seed)
        assert seeded.status_code == 200, seeded.text
        assert sandboxes[0].initial_files == {}


# --- additional coverage: the real _create_sandbox, grading-attempt cleanup, agent-run seed vs episode session


def _use_real_create_sandbox(consumer, monkeypatch):
    """Run the consumer's real ``_create_sandbox`` over fake provider sandboxes, recording each one it starts."""
    module, fixtures, server, _, create_sandbox = consumer
    monkeypatch.setattr(type(server), "_create_sandbox", create_sandbox)
    created: list[MagicMock] = []

    def make_sandbox(config) -> MagicMock:
        sandbox = MagicMock()
        sandbox._handle = SandboxHandle(sandbox_id=f"real-{len(created) + 1}", provider_name="test", raw=None)
        sandbox.start = AsyncMock()
        sandbox.stop = AsyncMock()
        sandbox.serialize = AsyncMock(return_value={"sandbox_id": sandbox._handle.sandbox_id})
        sandbox.exec = AsyncMock(return_value=SandboxExecResult(return_code=0, stdout="", stderr=""))
        created.append(sandbox)
        return sandbox

    monkeypatch.setattr(module, "AsyncSandbox", make_sandbox)
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {server.config.sandbox_provider: {"test": {}}})
    return module, fixtures, server, created


def test_an_episode_session_stops_every_sandbox_the_real_create_sandbox_started(consumer, monkeypatch):
    _, fixtures, server, created = _use_real_create_sandbox(consumer, monkeypatch)
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        seeded = client.post("/seed_session", json=fixtures._seed().model_dump(mode="json"))
        assert seeded.status_code == 200, seeded.text
        verified = client.post("/verify", json=fixtures._VERIFY)
        assert verified.status_code == 200 and verified.json()["reward"] == 1, verified.text
        assert len(created) == 2, "the task sandbox and the grading sandbox"
        assert all(sandbox.stop.await_count == 1 for sandbox in created), [s.stop.await_count for s in created]
        assert not server._session_id_to_sandbox and not server._verifier_sandboxes
        closed = client.post("/close_session", json=close_request(fixtures).model_dump(mode="json"))
        assert closed.status_code == 200, closed.text
    assert all(sandbox.stop.await_count == 1 for sandbox in created)


def test_a_grading_sandbox_whose_start_fails_stays_owned_until_close_retries(consumer, monkeypatch):
    module, fixtures, server, created = _use_real_create_sandbox(consumer, monkeypatch)
    real_make = module.AsyncSandbox

    def make_sandbox(config):
        sandbox = real_make(config)
        if len(created) == 2:  # the first grading sandbox: its start outcome is unknown, its stop fails once
            sandbox.start = AsyncMock(side_effect=RuntimeError("provider start outcome unknown"))
            sandbox.stop = AsyncMock(side_effect=[RuntimeError("stop unavailable"), None])
        return sandbox

    monkeypatch.setattr(module, "AsyncSandbox", make_sandbox)
    server.config.inconclusive_verification_retries = 0
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=fixtures._seed().model_dump(mode="json")).status_code == 200
        client.post("/verify", json=fixtures._VERIFY)
        assert len(created) == 2
        # The attempt never started and its stop failed: the handle must stay owned so that close can retry it.
        assert server._verifier_sandboxes.get("resources-session") is created[1]
        closed = client.post("/close_session", json=close_request(fixtures).model_dump(mode="json"))
        assert closed.status_code == 200, closed.text
        assert created[1].stop.await_count == 2
        assert not server._verifier_sandboxes and not server._session_id_to_sandbox


def test_an_inconclusive_grading_attempt_stops_its_sandbox_before_the_retry_on_a_fresh_one(consumer, monkeypatch):
    module, fixtures, server, sandboxes, _ = consumer
    verification_result = type(module.run_verification.return_value)
    monkeypatch.setattr(
        module,
        "run_verification",
        AsyncMock(
            side_effect=[
                verification_result(False, False, False, None, "", "image pull failed"),
                verification_result(True, True, True, {}, "ok"),
            ]
        ),
    )
    server.config.inconclusive_verification_retries = 1
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=fixtures._seed().model_dump(mode="json")).status_code == 200
        verified = client.post("/verify", json=fixtures._VERIFY)
        assert verified.status_code == 200 and verified.json()["reward"] == 1, verified.text
        assert len(sandboxes) == 3, "task sandbox, inconclusive grading sandbox, retry grading sandbox"
        assert [sandbox.stop.await_count for sandbox in sandboxes] == [1, 1, 1]
        assert not server._session_id_to_sandbox and not server._verifier_sandboxes


def test_a_failed_stop_of_an_inconclusive_attempt_keeps_the_handle_and_close_retries_it(consumer, monkeypatch):
    module, fixtures, server, sandboxes, _ = consumer
    verification_result = type(module.run_verification.return_value)
    monkeypatch.setattr(
        module,
        "run_verification",
        AsyncMock(return_value=verification_result(False, False, False, None, "", "image pull failed")),
    )
    server.config.inconclusive_verification_retries = 1
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=fixtures._seed().model_dump(mode="json")).status_code == 200
        original_create = type(server)._create_sandbox

        async def create_with_failing_stop(self, body, files=None, **kwargs):
            sandbox = await original_create(self, body, files, **kwargs)
            if kwargs.get("verifier_session_id") is not None:
                sandbox.stop.side_effect = [RuntimeError("stop unavailable"), RuntimeError("still unavailable"), None]
            return sandbox

        monkeypatch.setattr(type(server), "_create_sandbox", create_with_failing_stop)
        verified = client.post("/verify", json=fixtures._VERIFY)
        assert verified.status_code == 500, "a stop that cannot confirm cleanup must not be swallowed"
        assert server._verifier_sandboxes["resources-session"] is sandboxes[1]
        closed = client.post("/close_session", json=close_request(fixtures).model_dump(mode="json"))
        assert closed.status_code == 200, closed.text
        assert sandboxes[1].stop.await_count == 3, "the loop's stop, the wrapper's retry in its finally, the close"
        assert not server._verifier_sandboxes


@pytest.mark.parametrize("closed_first", [False, True], ids=["open", "closed"])
async def test_an_agent_run_seed_cannot_replace_an_episode_session(consumer, closed_first):
    module, fixtures, server, sandboxes, _ = consumer
    agent_run_type = module.SWEGymSeedSessionRequest if module is swe_app else module.R2EGymSeedSessionRequest
    episode_seed = fixtures._seed()
    request = Request({"type": "http", "session": {SESSION_ID_KEY: episode_seed.resources_session_id}})
    await server.seed_session(request, episode_seed)
    if closed_first:
        await server.close_resources_session(request, close_request(fixtures))
        request.session[SESSION_ID_KEY] = episode_seed.resources_session_id  # the close dropped the cookie key
    with pytest.raises(ResourcesSessionConflictError, match="agent-run seed"):
        await server.seed_session(request, agent_run_type.model_validate(fixtures._ROW))
    assert len(sandboxes) == 1
    await server.shutdown()


# --- failed cleanup around verify and seed --------------------------------------------------------------------


def _request_types(module):
    prefix = "SWEGym" if module is swe_app else "R2EGym"
    return getattr(module, f"{prefix}SeedSessionRequest"), getattr(module, f"{prefix}VerifyRequest")


def test_a_failed_task_stop_after_capture_still_grades_a_session_seeded_through_run(consumer):
    _, fixtures, server, sandboxes, _ = consumer
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=fixtures._ROW).status_code == 200
        sandboxes[0].stop.side_effect = [RuntimeError("provider close failed"), None]
        verified = client.post("/verify", json=fixtures._VERIFY)
        assert verified.status_code == 200, "a session with no saved capture and no close must not lose its grade"
        assert verified.json()["reward"] == 1.0
        assert not verified.json()["mask_sample"]
        assert server._session_id_to_sandbox, "the unconfirmed stop keeps the sandbox for shutdown to retry"
    assert sandboxes[0].stop.await_count == 2 and not server._session_id_to_sandbox


def test_a_failed_task_stop_after_capture_keeps_an_episode_session_retryable_on_the_saved_patch(consumer):
    module, fixtures, server, sandboxes, _ = consumer
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/seed_session", json=fixtures._seed().model_dump(mode="json")).status_code == 200
        sandboxes[0].stop.side_effect = [RuntimeError("provider close failed"), None]
        failed = client.post("/verify", json=fixtures._VERIFY)
        assert failed.status_code == 500
        assert server._session_id_to_sandbox, "the handle stays registered for the retry"

        retried = client.post("/verify", json=fixtures._VERIFY)
        assert retried.status_code == 200, retried.text
        assert retried.json()["reward"] == 1.0 and retried.json()["model_patch"] == "diff --git a/x b/x\n"
        assert module.capture_model_patch.await_count == 1, "the retry reuses the saved patch"
        assert sandboxes[0].stop.await_count == 2


async def test_a_session_seeded_through_run_keeps_its_start_error_when_the_cleanup_stop_fails(consumer, monkeypatch):
    module, fixtures, server, sandboxes, _ = consumer
    agent_run_type, _ = _request_types(module)
    session_id = "cookie-session"

    async def start_then_fail(self, session_id, body):
        sandbox = MagicMock()
        sandbox.stop = AsyncMock(side_effect=[RuntimeError("provider close failed"), None])
        sandboxes.append(sandbox)
        self._session_id_to_sandbox[session_id] = sandbox
        raise RuntimeError("image start failed")

    monkeypatch.setattr(type(server), "_start_task_sandbox", start_then_fail)
    request = Request({"type": "http", "session": {SESSION_ID_KEY: session_id}})
    with pytest.raises(RuntimeError, match="image start failed"):
        await server.seed_session(request, agent_run_type.model_validate(fixtures._ROW))
    assert server._session_id_to_sandbox[session_id] is sandboxes[0], "the failed stop keeps the handle"
    await server.shutdown()
    assert sandboxes[0].stop.await_count == 2 and not server._session_id_to_sandbox


async def test_an_interrupted_grading_after_a_failed_extraction_grades_the_same_empty_patch_on_retry(
    consumer, monkeypatch
):
    module, fixtures, server, sandboxes, _ = consumer
    _, verify_type = _request_types(module)
    monkeypatch.setattr(module, "capture_model_patch", AsyncMock(side_effect=RuntimeError("git failed")))
    graded = module.run_verification.return_value
    grading_started = asyncio.Event()
    calls = 0

    async def grade_once_interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            grading_started.set()
            await asyncio.Event().wait()
        return graded

    monkeypatch.setattr(module, "run_verification", grade_once_interrupted)
    episode_seed = fixtures._seed()
    request = Request({"type": "http", "session": {SESSION_ID_KEY: episode_seed.resources_session_id}})
    await server.seed_session(request, episode_seed)
    body = verify_type.model_validate(fixtures._VERIFY)

    interrupted = asyncio.create_task(server.verify(request, body))
    await grading_started.wait()
    interrupted.cancel()
    with pytest.raises(asyncio.CancelledError):
        await interrupted
    assert not server._session_id_to_sandbox, "the task sandbox is gone, so only a saved capture can regrade"

    retried = await server.verify(request, body)
    assert not retried.mask_sample and retried.evaluation_completed
    assert "Failed to extract model patch" in retried.error
