# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment Server sessions: typed seed, the /verify cookie, verify replay, masked failures, close and shutdown."""

import asyncio
import logging
from time import monotonic
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientOSError, ServerDisconnectedError
from fastapi import HTTPException
from fastapi.testclient import TestClient
from pydantic import ValidationError

import nemo_gym.base_resources_server as base_resources_server
import nemo_gym.server_utils as server_utils
import resources_servers.scale_swe.app as scale_swe_app
import resources_servers.scale_swe.verification as verification
import resources_servers.swebench.patch_capture as patch_capture
from nemo_gym.base_resources_server import ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from nemo_gym.testing.session_conformance import check_resources_session_contract
from resources_servers.scale_swe.app import (
    ScaleSWEResourcesServer,
    ScaleSWEResourcesServerConfig,
    ScaleSWESeedSessionRequest,
    ScaleSWEVerifyRequest,
)
from resources_servers.scale_swe.verification import TEST_OUTPUT_BEGIN, VerificationResult
from resources_servers.swebench.patch_capture import CaptureCommandError, PatchCapture


ROW = {"instance_id": "task", "workdir": "/repo", "image_url": "image", "FAIL_TO_PASS": '["tests/test_x.py::test_a"]'}
EPISODE = EpisodeId(rollout_id="rollout")
# What `git rev-parse HEAD` prints in the task sandbox.
BASE_COMMIT = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"  # pragma: allowlist secret
RESPONSE = {
    "output": [],
    "id": "response",
    "created_at": 0,
    "model": "model",
    "object": "response",
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}


def ran_tests(*, resolved: bool) -> VerificationResult:
    """A verifier run that reached pytest, so its outcome is a verdict about the patch."""
    status = "PASSED" if resolved else "FAILED"
    return VerificationResult(
        completed=True,
        resolved=resolved,
        patch_applied=True,
        test_results={"resolved": resolved},
        test_output=f"{TEST_OUTPUT_BEGIN}\n{status} tests/test_x.py::test_a\n",
    )


RESOLVED = ran_tests(resolved=True)
UNRESOLVED = ran_tests(resolved=False)


class FakeSandbox:
    """A sandbox double, compared and hashed by identity like ``AsyncSandbox``."""

    def __init__(self, **overrides) -> None:
        self.exec = AsyncMock(
            return_value=SimpleNamespace(return_code=0, stdout=f"{BASE_COMMIT}\n", stderr="", error_type=None)
        )
        self.serialize = AsyncMock(return_value={"sandbox_id": "sb"})
        self.stop = AsyncMock()
        self._handle = SimpleNamespace(sandbox_id="sb")
        for name, value in overrides.items():
            setattr(self, name, value)


def config(**overrides) -> ScaleSWEResourcesServerConfig:
    return ScaleSWEResourcesServerConfig(
        host="",
        port=0,
        name="",
        entrypoint="",
        sandbox_provider="sandbox",
        sandbox_config={},
        apply_anti_cheating=False,
        **overrides,
    )


@pytest.fixture
def sandboxes() -> list[FakeSandbox]:
    """Every sandbox the server created, the task sandbox first."""
    return []


@pytest.fixture
def server(monkeypatch, sandboxes) -> ScaleSWEResourcesServer:
    server = ScaleSWEResourcesServer(config=config(), server_client=MagicMock(spec=ServerClient))

    async def create(body, files=None):
        sandboxes.append(FakeSandbox())
        return sandboxes[-1]

    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(side_effect=create))
    return server


@pytest.fixture
def grading(monkeypatch) -> SimpleNamespace:
    """Patch capture and the verifier run, so a test decides what each returns."""
    capture = AsyncMock(return_value=PatchCapture(patch="diff --git a/x b/x\n", mode="worktree", source="worktree"))
    run = AsyncMock(return_value=RESOLVED)
    monkeypatch.setattr(scale_swe_app, "capture_model_patch", capture)
    monkeypatch.setattr(scale_swe_app, "run_verification", run)
    return SimpleNamespace(capture=capture, run=run)


def use_verifier_exec(monkeypatch, server, sandboxes, result: SandboxExecResult) -> None:
    """Grade with the real run_verification, writing no logs, on verifier sandboxes whose exec returns ``result``."""

    async def run_without_logs(**kwargs):
        return await verification.run_verification(**{**kwargs, "log_dir": None})

    async def create(body, files=None):
        sandboxes.append(FakeSandbox())
        if files:  # A verifier sandbox.
            sandboxes[-1].exec = AsyncMock(return_value=result)
        return sandboxes[-1]

    monkeypatch.setattr(scale_swe_app, "run_verification", run_without_logs)
    server._create_sandbox.side_effect = create


def seed(session_id: str = "attempt", **task_data) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id=session_id,
        episode_id=EPISODE,
        task_id=TaskId(taskset="scale_swe", task_id="task"),
        task_data=ROW | task_data,
    )


def close(episode_id: EpisodeId = EPISODE, session_id: str = "attempt") -> ResourcesCloseSessionRequest:
    return ResourcesCloseSessionRequest(resources_session_id=session_id, episode_id=episode_id)


def verify_body(**task_data) -> ScaleSWEVerifyRequest:
    # What an Environment Server sends: the seeded row plus the model's request and response.
    return ScaleSWEVerifyRequest.model_validate(
        ROW | task_data | {"responses_create_params": {"input": "Fix the bug"}, "response": RESPONSE}
    )


def test_environment_server_session_contract(server):
    check_resources_session_contract(server.setup_webserver(), seed(), keeps_state=True)
    # No session record, and so no lock, outlives a close.
    assert not server._resources_sessions


@pytest.mark.parametrize("timeout", [0, -1, float("inf")])
def test_stop_timeout_must_be_positive_and_finite(timeout):
    with pytest.raises(ValidationError):
        config(sandbox_stop_timeout_seconds=timeout)


@pytest.mark.asyncio
async def test_typed_seed_returns_sandbox_access_and_binds_the_verify_cookie(server):
    request = SimpleNamespace(session={SESSION_ID_KEY: "cookie"})
    response = await server.seed_session(request, seed())

    assert request.session[SESSION_ID_KEY] == "attempt"
    assert server._session_id_to_base_commit["attempt"] == BASE_COMMIT
    assert response.sandbox_access.connection.provider_config_ref == "sandbox"
    assert response.sandbox_access.connection.descriptor == {"sandbox_id": "sb"}
    assert response.sandbox_access.workdir == "/repo"
    # A retried seed returns the same session rather than starting a second sandbox.
    assert await server.seed_session(request, seed()) == response
    server._create_sandbox.assert_awaited_once()
    with pytest.raises(HTTPException, match="different request"):
        await server.seed_session(request, seed(instance_id="other"))


@pytest.mark.asyncio
async def test_failed_seed_stops_its_sandbox(server):
    sandbox = FakeSandbox(serialize=AsyncMock(side_effect=OSError("provider unavailable")))
    server._create_sandbox.side_effect = None
    server._create_sandbox.return_value = sandbox

    with pytest.raises(OSError):
        await server.seed_session(SimpleNamespace(session={}), seed())
    sandbox.stop.assert_awaited_once()
    assert "attempt" not in server._session_id_to_sandbox
    assert not server._resources_sessions


def test_http_lifecycle_replays_the_first_verdict_until_close(server, grading, sandboxes):
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=seed().model_dump(mode="json")).status_code == 200
        first = client.post("/verify", json=verify_body().model_dump(mode="json"))
        assert first.status_code == 200, first.text
        assert first.json()["reward"] == 1.0
        assert first.json()["evaluation_completed"] is True
        assert first.json()["mask_sample"] is False
        replay = client.post("/verify", json=verify_body().model_dump(mode="json"))
        assert replay.status_code == 200
        assert replay.json() == first.json()
        closed = client.post("/close_session", json=close().model_dump(mode="json"))
        assert closed.status_code == 200
        assert client.post("/close_session", json=close().model_dump(mode="json")).json() == closed.json()
        assert client.post("/seed_session", json=seed().model_dump(mode="json")).status_code == 409
        assert client.post("/verify", json=verify_body().model_dump(mode="json")).status_code == 409
    # One capture and one grading: the replay never graded the stopped task sandbox's empty patch.
    grading.capture.assert_awaited_once()
    grading.run.assert_awaited_once()
    task_sandbox, verifier_sandbox = sandboxes
    task_sandbox.stop.assert_awaited_once()
    verifier_sandbox.stop.assert_awaited_once()
    assert not server._resources_sessions and not server._session_id_to_sandbox


@pytest.mark.parametrize("disconnect", [ServerDisconnectedError, ClientOSError])
async def test_lost_verdict_is_replayed_through_the_shared_http_retry(server, grading, monkeypatch, disconnect):
    responses = []
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=seed().model_dump(mode="json")).status_code == 200

        async def send(**kwargs):
            response = client.post("/verify", json=kwargs["json"])
            responses.append(response)
            if len(responses) == 1:
                assert response.status_code == 200
                raise disconnect("lost completed verdict")
            return response

        monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: SimpleNamespace(request=send))
        response = await server_utils._request_with_retries(
            "POST",
            "http://testserver/verify",
            _internal=True,
            _max_connection_retries=2,
            json=verify_body().model_dump(mode="json"),
        )
        assert [r.status_code for r in responses] == [200, 200]
        assert response.json() == responses[0].json()
        assert response.json()["reward"] == 1.0
    grading.capture.assert_awaited_once()
    grading.run.assert_awaited_once()


@pytest.mark.parametrize("field", ["response", "responses_create_params"])
async def test_verify_refuses_a_changed_request_and_keeps_the_first_verdict(server, grading, field):
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    result = await server.verify(request, verify_body())
    changed = verify_body()
    if field == "response":
        changed.response.id = "other-response"
    else:
        changed.responses_create_params.input = "Another prompt"
    with pytest.raises(HTTPException, match="first request") as error:
        await server.verify(request, changed)
    assert error.value.status_code == 409
    # A caller that mutates what it got back does not change what is replayed.
    result.reward = 0.0
    result.response.id = "mutated"
    replay = await server.verify(request, verify_body())
    assert replay.reward == 1.0
    assert replay.response.id == "response"
    grading.run.assert_awaited_once()


@pytest.mark.parametrize("field", ["instance_id", "image_url", "f2p_script"])
async def test_verify_refuses_another_task_without_touching_the_sandbox(server, grading, field):
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    with pytest.raises(HTTPException, match="seeded task"):
        await server.verify(request, verify_body(**{field: "other"}))
    grading.capture.assert_not_awaited()
    assert (await server.verify(request, verify_body())).reward == 1.0


async def test_concurrent_verification_replays_after_the_first_finishes(server, grading):
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    entered, release = asyncio.Event(), asyncio.Event()

    async def run(**kwargs):
        entered.set()
        await release.wait()
        return RESOLVED

    grading.run.side_effect = run
    first = asyncio.create_task(server.verify(request, verify_body()))
    await asyncio.wait_for(entered.wait(), timeout=5)
    second = asyncio.create_task(server.verify(request, verify_body()))
    await asyncio.sleep(0)
    assert not second.done()
    release.set()
    results = await asyncio.gather(first, second)
    assert results[0] == results[1]
    grading.run.assert_awaited_once()


@pytest.mark.parametrize("failure", [RuntimeError, asyncio.CancelledError])
async def test_interrupted_verification_requires_retrying_the_episode(server, monkeypatch, failure):
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    monkeypatch.setattr(server, "_verify", AsyncMock(side_effect=failure("verification interrupted")))
    with pytest.raises(failure):
        await server.verify(request, verify_body())
    with pytest.raises(HTTPException, match="retry the episode") as error:
        await server.verify(request, verify_body())
    assert error.value.status_code == 503
    server._verify.assert_awaited_once()
    await server.close_resources_session(close())
    assert "attempt" in server._closed_sessions


async def test_cancelled_verification_stops_the_task_sandbox_and_requires_retrying_the_episode(
    server, grading, sandboxes
):
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    entered = asyncio.Event()

    async def run(**kwargs):
        entered.set()
        await asyncio.Event().wait()

    grading.run.side_effect = run
    verifying = asyncio.create_task(server.verify(request, verify_body()))
    await asyncio.wait_for(entered.wait(), timeout=5)
    verifying.cancel()
    with pytest.raises(asyncio.CancelledError):
        await verifying
    task_sandbox, verifier_sandbox = sandboxes
    task_sandbox.stop.assert_awaited_once()
    verifier_sandbox.stop.assert_awaited_once()
    with pytest.raises(HTTPException, match="retry the episode"):
        await asyncio.wait_for(server.verify(request, verify_body()), timeout=5)
    grading.run.assert_awaited_once()


@pytest.mark.parametrize(
    ("failure", "failure_kind"),
    [
        ("capture", "verifier_error"),
        ("capture_provider_error", "verifier_error"),
        ("capture_timeout", "verifier_error"),
        ("no_base_commit", "verifier_error"),
        ("no_task_sandbox", "session_lost"),
        ("verifier_start", "provider_unavailable"),
        ("verifier_exec", "verifier_error"),
        ("test_patch", "verifier_error"),
    ],
)
async def test_verification_without_a_verdict_is_masked_and_not_kept(
    server, grading, sandboxes, failure, failure_kind
):
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    if failure == "capture":
        # The exec itself raised: the agent's sandbox could not be reached.
        grading.capture.side_effect = OSError("exec stream lost")
    elif failure == "capture_provider_error":
        grading.capture.side_effect = CaptureCommandError(
            "ended without a status", return_code=125, error_type="sandbox"
        )
    elif failure == "capture_timeout":
        grading.capture.side_effect = CaptureCommandError("timed out", return_code=124, error_type="timeout")
    elif failure == "no_base_commit":
        server._session_id_to_base_commit["attempt"] = ""
    elif failure == "no_task_sandbox":
        del server._session_id_to_sandbox["attempt"]
    elif failure == "verifier_start":
        server._create_sandbox.side_effect = OSError("no capacity")
    elif failure == "verifier_exec":
        grading.run.side_effect = OSError("exec stream lost")
    else:
        grading.run.return_value = VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=True,
            test_results=None,
            test_output="",
            error="held-out test patch did not apply",
            test_patch_failed=True,
        )

    result = await server.verify(request, verify_body())

    assert result.reward == 0.0
    assert result.evaluation_completed is False
    assert result.mask_sample is True
    assert result.failure_kind == failure_kind
    assert result.failure_reason and result.failure_reason == result.error
    if failure.startswith(("capture", "no_")):
        # Nothing was captured, so nothing was graded: no empty patch scored as a completed 0.
        assert result.failure_reason.startswith("Failed to extract model patch")
        grading.run.assert_not_awaited()
        assert len(sandboxes) == 1
    if failure != "no_task_sandbox":
        sandboxes[0].stop.assert_awaited_once()
    # Not kept as the session's result: a retry cannot grade the stopped sandbox, so it asks for another episode.
    with pytest.raises(HTTPException, match="retry the episode") as error:
        await server.verify(request, verify_body())
    assert error.value.status_code == 503
    await server.close_resources_session(close())


async def test_a_completed_zero_is_a_kept_measurement(server, grading):
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    grading.run.return_value = UNRESOLVED
    result = await server.verify(request, verify_body())
    assert result.reward == 0.0 and result.evaluation_completed is True
    assert result.mask_sample is False and result.failure_kind is None and result.failure_reason is None
    assert await server.verify(request, verify_body()) == result
    grading.run.assert_awaited_once()


@pytest.mark.parametrize("case", ["no_verdict", "capture_failed", "exec_error"])
async def test_cookie_path_verification_neither_masks_nor_replays(server, grading, sandboxes, monkeypatch, case):
    """An Agent's /run (the cookie path) masks nothing and replays nothing."""
    request = SimpleNamespace(session={SESSION_ID_KEY: "cookie"})
    await server.seed_session(request, ScaleSWESeedSessionRequest.model_validate(ROW))
    if case == "no_verdict":
        grading.run.return_value = VerificationResult(
            completed=False,
            resolved=False,
            patch_applied=True,
            test_results=None,
            test_output="",
            error="held-out test patch did not apply",
        )
    elif case == "capture_failed":
        grading.capture.side_effect = RuntimeError("git diff failed")
        grading.run.return_value = UNRESOLVED
    else:
        use_verifier_exec(
            monkeypatch,
            server,
            sandboxes,
            SandboxExecResult(stdout=None, stderr="", return_code=125, error_type="sandbox"),
        )

    result = await server.verify(request, verify_body())

    assert result.reward == 0.0
    assert result.mask_sample is False and result.failure_kind is None
    if case == "no_verdict":
        assert result.evaluation_completed is False
        assert result.error == "held-out test patch did not apply"
        assert grading.run.await_count == 2  # One inconclusive retry on a fresh sandbox.
    elif case == "exec_error":
        # A provider failure of the eval exec is graded like any other run.
        assert result.evaluation_completed is True
    else:
        # The patch that could not be captured is graded as an empty one.
        assert result.evaluation_completed is True
        assert result.error == "Failed to extract model patch: git diff failed"
        assert grading.run.await_args.kwargs["inputs"].patch == ""


@pytest.mark.asyncio
async def test_close_stops_the_sandbox_and_a_failed_stop_stays_retryable(server, grading):
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    sandbox = server._session_id_to_sandbox["attempt"]
    sandbox.stop.side_effect = [OSError("temporary"), None]

    with pytest.raises(OSError):
        await server.close_resources_session(close())
    assert server._session_id_to_sandbox["attempt"] is sandbox
    # The failed close is not recorded, but it still fences the session against a seed.
    assert "attempt" not in server._closed_sessions
    with pytest.raises(HTTPException, match="already closed"):
        await server.seed_session(request, seed())
    await server.close_resources_session(close())
    await server.close_resources_session(close())
    assert sandbox.stop.await_count == 2
    assert "attempt" not in server._session_id_to_sandbox
    assert "attempt" in server._closed_sessions
    with pytest.raises(HTTPException, match="already closed"):
        await server.seed_session(request, seed())
    with pytest.raises(HTTPException, match="closed"):
        await server.verify(request, verify_body())
    grading.capture.assert_not_awaited()


@pytest.mark.asyncio
async def test_close_for_another_episode_is_refused(server):
    await server.seed_session(SimpleNamespace(session={}), seed())

    with pytest.raises(HTTPException, match="episode_id"):
        await server.close_resources_session(close(EpisodeId(rollout_id="other")))
    server._session_id_to_sandbox["attempt"].stop.assert_not_awaited()
    await server.close_resources_session(close())
    with pytest.raises(HTTPException, match="episode_id"):
        await server.close_resources_session(close(EpisodeId(rollout_id="other")))


async def test_close_records_are_kept_for_the_retry_window_only(server, monkeypatch):
    now = 100.0
    monkeypatch.setattr(base_resources_server, "monotonic", lambda: now)
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    await server.close_resources_session(close())
    await server.close_resources_session(close(session_id="never-seeded"))
    assert not server._resources_sessions
    now = 399.0  # A retried close inside the window does not extend it.
    await server.close_resources_session(close())
    for session_id in ("attempt", "never-seeded"):
        with pytest.raises(HTTPException, match="already closed"):
            await server.seed_session(request, seed(session_id))
    now = 401.0
    assert len(server._closed_sessions) == 0
    # Past the window no worker remembers the close, so the ID can be seeded (see ClosedResourcesSessionRecords).
    await server.seed_session(request, seed("never-seeded"))
    assert "never-seeded" in server._session_id_to_sandbox


@pytest.mark.parametrize("close_waits", [True, False])
async def test_close_fences_a_seed_in_progress(server, sandboxes, close_waits):
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_create(body, files=None):
        entered.set()
        await release.wait()
        sandboxes.append(FakeSandbox())
        return sandboxes[-1]

    server._create_sandbox.side_effect = slow_create
    request = SimpleNamespace(session={})
    seeding = asyncio.create_task(server.seed_session(request, seed()))
    await asyncio.wait_for(entered.wait(), timeout=5)
    closing = asyncio.create_task(server.close_resources_session(close()))
    await asyncio.sleep(0)
    if not close_waits:
        # The caller gave up on the close; the fence it set still takes effect.
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
    release.set()
    with pytest.raises(HTTPException, match="closed during setup"):
        await seeding
    if close_waits:
        assert (await closing).resources_session_id == "attempt"
    (sandbox,) = sandboxes
    sandbox.stop.assert_awaited_once()
    assert "attempt" in server._closed_sessions
    assert not server._resources_sessions and not server._session_id_to_sandbox
    with pytest.raises(HTTPException, match="already closed"):
        await server.seed_session(request, seed())


async def test_close_that_arrives_during_verification_takes_effect_after_it(server, grading):
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    entered, release = asyncio.Event(), asyncio.Event()

    async def run(**kwargs):
        entered.set()
        await release.wait()
        return RESOLVED

    grading.run.side_effect = run
    verifying = asyncio.create_task(server.verify(request, verify_body()))
    await asyncio.wait_for(entered.wait(), timeout=5)
    closing = asyncio.create_task(server.close_resources_session(close()))
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    release.set()
    assert (await verifying).reward == 1.0
    assert "attempt" in server._closed_sessions
    assert not server._resources_sessions
    with pytest.raises(HTTPException, match="closed"):
        await server.verify(request, verify_body())


async def test_slow_stop_at_close_is_waited_for_again_never_cut_short(server):
    server.config.sandbox_stop_timeout_seconds = 0.01
    await server.seed_session(SimpleNamespace(session={}), seed())
    sandbox = server._session_id_to_sandbox["attempt"]
    release = asyncio.Event()
    cut_short = []

    async def slow_stop():
        try:
            await release.wait()
        except asyncio.CancelledError:
            cut_short.append(True)
            raise

    sandbox.stop.side_effect = slow_stop
    started = monotonic()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(server.close_resources_session(close()), timeout=5)
    assert monotonic() - started < 2
    assert server._session_id_to_sandbox["attempt"] is sandbox
    assert "attempt" not in server._closed_sessions
    release.set()
    await server.close_resources_session(close())
    # The retried close joined the stop that was still running instead of starting another.
    sandbox.stop.assert_awaited_once()
    assert not cut_short
    assert "attempt" in server._closed_sessions and not server._session_id_to_sandbox


@pytest.mark.parametrize("stop_failure", ["error", "slow"])
async def test_failed_seed_cleanup_is_bounded_and_keeps_the_sandbox_for_close(server, stop_failure):
    server.config.sandbox_stop_timeout_seconds = 0.01
    sandbox = FakeSandbox(exec=AsyncMock(side_effect=OSError("exec stream lost")))
    release = asyncio.Event()

    async def stop():
        if sandbox.stop.await_count == 1:
            if stop_failure == "slow":
                await release.wait()
                return
            raise OSError("provider unavailable")

    sandbox.stop.side_effect = stop
    server._create_sandbox.side_effect = None
    server._create_sandbox.return_value = sandbox

    started = monotonic()
    with pytest.raises(OSError, match="exec stream lost"):
        await asyncio.wait_for(server.seed_session(SimpleNamespace(session={}), seed()), timeout=5)
    assert monotonic() - started < 2
    assert server._session_id_to_sandbox["attempt"] is sandbox
    release.set()
    await server.close_resources_session(close())
    # A stop that failed is started again; a slow one is waited for again, not started twice.
    assert sandbox.stop.await_count == (2 if stop_failure == "error" else 1)
    assert "attempt" in server._closed_sessions and not server._session_id_to_sandbox


async def test_failed_stop_after_capture_leaves_the_sandbox_to_close(server, grading):
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    sandbox = server._session_id_to_sandbox["attempt"]
    sandbox.stop.side_effect = [OSError("provider unavailable"), None]

    result = await server.verify(request, verify_body())

    assert result.reward == 1.0 and result.evaluation_completed is True
    assert server._session_id_to_sandbox["attempt"] is sandbox
    await server.close_resources_session(close())
    assert sandbox.stop.await_count == 2
    assert not server._session_id_to_sandbox


async def test_shutdown_stops_every_environment_server_sandbox_even_when_one_stop_fails(server, sandboxes, caplog):
    server.config.sandbox_stop_timeout_seconds = 0.01
    app = server.setup_webserver()
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    for session_id in ("first", "second"):
        await server.seed_session(SimpleNamespace(session={}), seed(session_id))
    # An Agent's /run seeds a cookie session with the row itself.
    await server.seed_session(
        SimpleNamespace(session={SESSION_ID_KEY: "cookie"}), ScaleSWESeedSessionRequest.model_validate(ROW)
    )
    failing, slow, cookie_sandbox = sandboxes
    release = asyncio.Event()
    failing.stop.side_effect = OSError("provider unavailable")
    slow.stop.side_effect = release.wait

    started = monotonic()
    with caplog.at_level(logging.ERROR, logger=scale_swe_app.__name__):
        await asyncio.wait_for(lifespan.__aexit__(None, None, None), timeout=5)

    # Shutdown returned without raising, did not wait out the slow stop, and reached every Environment Server
    # sandbox. The cookie path's sandbox is left to the provider's TTL, as before.
    assert monotonic() - started < 2
    failing.stop.assert_awaited_once()
    slow.stop.assert_awaited_once()
    cookie_sandbox.stop.assert_not_awaited()
    # A sandbox whose stop failed or ran long stays owned; the provider TTL reclaims it.
    assert set(server._session_id_to_sandbox) == {"first", "second", "cookie"}
    assert sum("provider TTL is the backstop" in record.getMessage() for record in caplog.records) == 2
    # Shutdown never cut the slow stop short; it finishes on its own.
    release.set()
    await asyncio.wait_for(asyncio.gather(*server._stop_tasks.values()), timeout=5)


async def test_seed_in_progress_at_shutdown_stops_its_own_sandbox(server, sandboxes):
    app = server.setup_webserver()
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_create(body, files=None):
        entered.set()
        await release.wait()
        sandboxes.append(FakeSandbox())
        return sandboxes[-1]

    server._create_sandbox.side_effect = slow_create
    request = SimpleNamespace(session={})
    seeding = asyncio.create_task(server.seed_session(request, seed()))
    await asyncio.wait_for(entered.wait(), timeout=5)
    shutdown = asyncio.create_task(lifespan.__aexit__(None, None, None))
    await asyncio.sleep(0)
    assert not shutdown.done()  # It waits for the request in progress on the session.
    release.set()
    with pytest.raises(HTTPException, match="shutting down"):
        await asyncio.wait_for(seeding, timeout=5)
    await asyncio.wait_for(shutdown, timeout=5)
    (sandbox,) = sandboxes
    sandbox.stop.assert_awaited_once()
    assert not server._session_id_to_sandbox
    with pytest.raises(HTTPException, match="shutting down") as error:
        await server.seed_session(request, seed("late"))
    assert error.value.status_code == 503


@pytest.mark.parametrize(
    ("exec_result", "reason"),
    [
        pytest.param(
            # OpenSandbox, when the exec stream ends without a final status.
            SandboxExecResult(
                stdout=None,
                stderr="Command ended without a final status: no exit code, error or completion event",
                return_code=125,
                error_type="sandbox",
            ),
            "verifier exec ended without an exit status (sandbox)",
            id="provider_error",
        ),
        pytest.param(
            SandboxExecResult(
                stdout="bash: /tmp/nemo_gym_eval.sh: No such file or directory\n", stderr=None, return_code=127
            ),
            "eval script stopped before the tests started",
            id="never_reached_the_tests",
        ),
    ],
)
async def test_verifier_run_that_never_tested_the_patch_has_no_verdict(
    server, grading, sandboxes, monkeypatch, exec_result, reason
):
    use_verifier_exec(monkeypatch, server, sandboxes, exec_result)
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())

    result = await server.verify(request, verify_body())

    assert result.reward == 0.0 and result.evaluation_completed is False
    assert result.mask_sample is True and result.failure_kind == "verifier_error"
    assert result.failure_reason == reason
    assert len(sandboxes) == 3  # The task sandbox, then one retry on a fresh verifier sandbox.
    # Not kept: a retry asks for another episode instead of replaying a completed 0.
    with pytest.raises(HTTPException, match="retry the episode") as error:
        await server.verify(request, verify_body())
    assert error.value.status_code == 503


async def test_verifier_timeout_is_still_a_measured_zero(server, grading, sandboxes, monkeypatch):
    # A provider may return no output on timeout, and a test the patch made hang must still score 0.
    use_verifier_exec(
        monkeypatch,
        server,
        sandboxes,
        SandboxExecResult(stdout=None, stderr=None, return_code=124, error_type="timeout"),
    )
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())

    result = await server.verify(request, verify_body())

    assert result.reward == 0.0 and result.evaluation_completed is True
    assert result.mask_sample is False and result.failure_kind is None
    assert await server.verify(request, verify_body()) == result
    assert len(sandboxes) == 2


async def test_repository_the_agent_broke_scores_zero(server, grading, sandboxes, monkeypatch):
    monkeypatch.setattr(scale_swe_app, "capture_model_patch", patch_capture.capture_model_patch)
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    (task_sandbox,) = sandboxes
    lock_error = "fatal: Unable to create '/repo/.git/index.lock': File exists."

    async def stale_lock(command, **kwargs):
        # A git command the harness deadline interrupted left .git/index.lock behind.
        if "add -N ." in command:
            return SimpleNamespace(return_code=128, stdout="", stderr=lock_error, error_type=None)
        if "HEAD_SHA" in command:
            return SimpleNamespace(return_code=0, stdout=f"HEAD_SHA {BASE_COMMIT}\n", stderr="", error_type=None)
        return SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)

    task_sandbox.exec.side_effect = stale_lock

    result = await server.verify(request, verify_body())

    assert result.reward == 0.0 and result.evaluation_completed is True
    assert result.mask_sample is False and result.failure_kind is None
    assert result.failure_reason == f"Failed to extract model patch: git exited 128: {lock_error}"
    assert result.error == result.failure_reason
    # Nothing was graded, and the lock the agent left was not touched.
    grading.run.assert_not_awaited()
    assert len(sandboxes) == 1
    assert not any("index.lock" in call.args[0] for call in task_sandbox.exec.await_args_list)
    # A measured 0 is the session's verdict.
    assert await server.verify(request, verify_body()) == result


def slow_verifier_stops(server, stop) -> None:
    """Give every verifier sandbox the server creates from now on ``stop`` as its stop."""
    create_task_sandbox = server._create_sandbox.side_effect

    async def create(body, files=None):
        sandbox = await create_task_sandbox(body, files)
        if files:  # A verifier sandbox.
            sandbox.stop.side_effect = stop
        return sandbox

    server._create_sandbox.side_effect = create


async def test_slow_verifier_stop_is_left_to_close_never_cut_short(server, grading, sandboxes):
    server.config.sandbox_stop_timeout_seconds = 0.01
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    release = asyncio.Event()
    cut_short = []

    async def slow_stop():
        try:
            await release.wait()
        except asyncio.CancelledError:
            cut_short.append(True)
            raise

    slow_verifier_stops(server, slow_stop)

    result = await server.verify(request, verify_body())

    # The verdict does not wait for the verifier's stop, but the session keeps the verifier for close.
    assert result.reward == 1.0
    task_sandbox, verifier = sandboxes
    assert server._session_id_to_verifier_sandboxes == {"attempt": [verifier]}
    with pytest.raises(TimeoutError):
        await server.close_resources_session(close())
    assert "attempt" not in server._closed_sessions
    release.set()
    await server.close_resources_session(close())
    verifier.stop.assert_awaited_once()
    assert not cut_short
    assert not server._session_id_to_verifier_sandboxes and "attempt" in server._closed_sessions


async def test_failed_verifier_stop_is_retried_at_close(server, grading, sandboxes):
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    slow_verifier_stops(server, [OSError("provider unavailable"), None])

    assert (await server.verify(request, verify_body())).reward == 1.0
    task_sandbox, verifier = sandboxes
    assert server._session_id_to_verifier_sandboxes == {"attempt": [verifier]}
    await server.close_resources_session(close())

    assert verifier.stop.await_count == 2
    assert not server._session_id_to_verifier_sandboxes and "attempt" in server._closed_sessions


async def test_shutdown_retries_a_verifier_stop_that_failed(server, grading, sandboxes):
    app = server.setup_webserver()
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    request = SimpleNamespace(session={})
    await server.seed_session(request, seed())
    slow_verifier_stops(server, [OSError("provider unavailable"), None])
    await server.verify(request, verify_body())

    await asyncio.wait_for(lifespan.__aexit__(None, None, None), timeout=5)

    task_sandbox, verifier = sandboxes
    assert verifier.stop.await_count == 2
    assert not server._session_id_to_verifier_sandboxes and not server._session_id_to_sandbox


async def test_shutdown_stops_a_busy_session_after_a_bounded_wait(server, sandboxes):
    server.config.sandbox_stop_timeout_seconds = 0.01
    app = server.setup_webserver()
    lifespan = app.router.lifespan_context(app)
    await lifespan.__aenter__()
    await server.seed_session(SimpleNamespace(session={}), seed())
    session_lock = server._resources_sessions["attempt"].lock
    await session_lock.acquire()  # A request still in progress on the session.
    try:
        started = monotonic()
        await asyncio.wait_for(lifespan.__aexit__(None, None, None), timeout=5)
        assert monotonic() - started < 2
        sandboxes[0].stop.assert_awaited_once()
        assert not server._session_id_to_sandbox
    finally:
        session_lock.release()


@pytest.mark.parametrize(
    "rev_parse",
    [
        SimpleNamespace(return_code=128, stdout="HEAD\n", stderr="fatal: ambiguous argument 'HEAD'", error_type=None),
        SimpleNamespace(return_code=125, stdout=None, stderr="", error_type="sandbox"),
    ],
    ids=["no_commit", "provider_error"],
)
async def test_seed_without_a_base_commit_is_retryable_and_stops_its_sandbox(server, rev_parse):
    ok = SimpleNamespace(return_code=0, stdout=f"{BASE_COMMIT}\n", stderr="", error_type=None)
    sandbox = FakeSandbox(
        exec=AsyncMock(side_effect=lambda command, **kwargs: rev_parse if "rev-parse HEAD" in command else ok)
    )
    server._create_sandbox.side_effect = None
    server._create_sandbox.return_value = sandbox

    with pytest.raises(HTTPException, match="retry the episode") as error:
        await server.seed_session(SimpleNamespace(session={}), seed())

    assert error.value.status_code == 503
    sandbox.stop.assert_awaited_once()
    assert not server._session_id_to_sandbox and not server._resources_sessions


async def test_golden_patch_mode_refuses_typed_seeds(server):
    server.config.is_verifying_golden_patch = True

    with pytest.raises(HTTPException, match="Golden-patch mode") as error:
        await server.seed_session(SimpleNamespace(session={}), seed())

    assert error.value.status_code == 422
    server._create_sandbox.assert_not_awaited()
