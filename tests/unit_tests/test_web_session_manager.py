# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nemo_gym.web.api_models import WebResetRequest, WebSeedSessionRequest, WebStepRequest, WebStepResponse
from nemo_gym.web.models import (
    WebAction,
    WebArtifactRef,
    WebBenchmark,
    WebObservation,
    WebStepResult,
    WebTask,
    WebVerifierResult,
)
from nemo_gym.web.operation_runner import DirectWebOperationRunner
from nemo_gym.web.resource_config import WebResourcesServerConfig
from nemo_gym.web.session import (
    BenchmarkPreconditionError,
    CapacityUnavailableError,
    SessionConflictError,
    SessionNotFoundError,
)
from nemo_gym.web.session_manager import WebSessionManager
from nemo_gym.web.site_pool import LocalSiteLockPool, SiteLease, UnmanagedSitePool


class FakeBackend:
    def __init__(self, _config, session_id, _artifacts):
        self.session_id = session_id
        self.reset_calls = 0
        self.observe_calls = 0
        self.step_calls = 0
        self.evaluate_calls = 0
        self.close_calls = 0
        self.fail_reset = False
        self.fail_step = False
        self.fail_evaluate = False
        self.fail_close = False
        self.observation = WebObservation(url="about:blank")

    def reset(self, task: WebTask):
        self.reset_calls += 1
        if self.fail_reset:
            raise RuntimeError("reset failed")
        self.observation = WebObservation(url=f"https://example.test/{task.task_id}")
        return self.observation, {"reset_calls": self.reset_calls}

    def observe(self):
        self.observe_calls += 1
        return self.observation

    def step(self, action: WebAction):
        self.step_calls += 1
        if self.fail_step:
            raise RuntimeError("step failed")
        self.observation = WebObservation(url=f"https://example.test/step/{self.step_calls}")
        return WebStepResult(
            observation=self.observation,
            execution_ok=True,
            terminated=action.terminal,
            truncated=action.name == "truncate",
        )

    def evaluate(self, final_answer=None):
        del final_answer
        self.evaluate_calls += 1
        if self.fail_evaluate:
            raise RuntimeError("evaluate failed")
        return WebVerifierResult(reward=1.0, raw_score=1.0, task_success=True)

    def close(self):
        self.close_calls += 1
        if self.fail_close:
            raise RuntimeError("close failed")


class FakeSitePool:
    def __init__(self):
        self.acquired: list[SiteLease] = []
        self.released: list[tuple[SiteLease, bool]] = []

    async def acquire(self, session_id: str, task: WebTask) -> SiteLease:
        lease = SiteLease(
            lease_id=f"fake:{session_id}",
            isolated=True,
            metadata={"sites": task.sites},
        )
        self.acquired.append(lease)
        return lease

    async def release(self, lease: SiteLease, *, healthy: bool) -> None:
        self.released.append((lease, healthy))

    async def health(self) -> dict[str, Any]:
        return {"mode": "fake", "active_leases": len(self.acquired) - len(self.released)}


def _config(tmp_path, **updates: Any) -> WebResourcesServerConfig:
    values = {
        "name": "web",
        "host": "localhost",
        "port": 8000,
        "entrypoint": "app.py",
        "domain": "agent",
        "artifact_dir": str(tmp_path),
    }
    values.update(updates)
    return WebResourcesServerConfig(**values)


def _task(task_id: str = "0", benchmark: WebBenchmark = WebBenchmark.WEBARENA) -> WebTask:
    return WebTask(benchmark=benchmark, task_id=task_id, sites=["shopping"])


def _step(operation_id: str, *, name: str = "noop", terminal: bool = False) -> WebStepRequest:
    return WebStepRequest(
        operation_id=operation_id,
        action=WebAction(name=name, script=f"{name}()", terminal=terminal),
    )


def _manager(tmp_path, *, factory=FakeBackend, site_pool=None, **config_updates):
    backends: list[FakeBackend] = []

    def capture_factory(*args):
        backend = factory(*args)
        backends.append(backend)
        return backend

    manager = WebSessionManager(
        _config(tmp_path, **config_updates),
        backend_factory=capture_factory,
        site_pool=site_pool,
        operation_runner=DirectWebOperationRunner(),
    )
    return manager, backends


@pytest.mark.asyncio
async def test_session_lifecycle_caches_operations_and_results(tmp_path) -> None:
    pool = FakeSitePool()
    manager, backends = _manager(tmp_path, site_pool=pool)
    await manager.start()

    seed = await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))
    repeated_seed = await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))
    status = await manager.session_status("session-a")
    observed = await manager.observe("session-a")

    assert seed == repeated_seed
    assert seed.info == {
        "reset_calls": 1,
        "site_lease_id": "fake:session-a",
        "site_isolated": True,
        "site_lease_metadata": {"sites": ["shopping"]},
    }
    assert status.status == "ready"
    assert status.site_lease_id == "fake:session-a"
    assert observed.url.endswith("/0")

    first_step = await manager.step("session-a", _step("operation-1"))
    repeated_step = await manager.step("session-a", _step("operation-1"))
    assert first_step == repeated_step
    assert backends[0].step_calls == 1

    reset = await manager.reset_session("session-a", WebResetRequest(task=_task()))
    assert reset.info["reset_calls"] == 2
    assert manager._sessions["session-a"].operations == {}

    finished = await manager.step("session-a", _step("operation-2", terminal=True))
    assert finished.terminated is True
    assert manager._sessions["session-a"].status == "finished"

    first_evaluation = await manager.evaluate("session-a", "done")
    repeated_evaluation = await manager.evaluate("session-a", "ignored")
    assert first_evaluation == repeated_evaluation
    assert backends[0].evaluate_calls == 1
    with pytest.raises(SessionConflictError, match="already been evaluated"):
        await manager.step("session-a", _step("operation-3"))

    artifact = WebArtifactRef(uri="file:///recording.webm", mime_type="video/webm", size_bytes=1, sha256="0" * 64)
    manager._artifacts.recording_artifacts = lambda session_id: [artifact] if session_id == "session-a" else []
    assert await manager.recording_artifacts("session-a") == [artifact]

    health = await manager.health()
    assert health["sessions"] == 1
    assert health["creating"] == 0
    assert health["site_pool"]["mode"] == "fake"
    assert health["uptime_seconds"] >= 0

    assert await manager.close_session("session-a") is True
    assert await manager.close_session("session-a") is True
    assert backends[0].close_calls == 1
    assert pool.released == [(pool.acquired[0], True)]
    await manager.stop()
    assert manager._reaper_task is None


@pytest.mark.asyncio
async def test_admission_and_task_identity_guards(tmp_path) -> None:
    manager, _ = _manager(tmp_path, max_sessions=1)
    await manager.seed_session("session-a", WebSeedSessionRequest(task=_task("0")))

    with pytest.raises(CapacityUnavailableError, match="capacity is full"):
        await manager.seed_session("session-b", WebSeedSessionRequest(task=_task("1")))
    with pytest.raises(SessionConflictError, match="already owns"):
        await manager.seed_session("session-a", WebSeedSessionRequest(task=_task("1")))
    with pytest.raises(SessionNotFoundError):
        await manager.session_status("missing")

    await manager.close_session("session-a")
    manager._creating.add("session-c")
    with pytest.raises(SessionConflictError, match="already being created"):
        await manager.seed_session("session-c", WebSeedSessionRequest(task=_task("2")))
    manager._creating.clear()

    disabled, _ = _manager(tmp_path, allowed_benchmarks=[WebBenchmark.WEBVOYAGER])
    with pytest.raises(ValueError, match="disabled by server configuration"):
        await disabled.seed_session("session", WebSeedSessionRequest(task=_task()))
    await manager.stop()
    await disabled.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    ["error", "closing", "resetting", "stepping", "evaluating", "finished", "evaluated"],
)
async def test_seed_replay_rejects_non_ready_session_states(tmp_path, status: str) -> None:
    manager, _ = _manager(tmp_path)
    await manager.seed_session("session", WebSeedSessionRequest(task=_task()))
    manager._sessions["session"].status = status

    with pytest.raises(SessionConflictError, match=rf"cannot replay seed while status='{status}'"):
        await manager.seed_session("session", WebSeedSessionRequest(task=_task()))

    await manager.close_session("session")


@pytest.mark.asyncio
async def test_seed_precondition_failure_releases_backend_and_lease(tmp_path) -> None:
    pool = FakeSitePool()

    class MissingAssetBackend(FakeBackend):
        def reset(self, task: WebTask):
            del task
            raise ValueError("reference image is missing")

        def close(self):
            super().close()
            raise RuntimeError("cleanup also failed")

    manager, backends = _manager(tmp_path, factory=MissingAssetBackend, site_pool=pool)
    with pytest.raises(BenchmarkPreconditionError, match="reference image is missing"):
        await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))

    assert backends[0].close_calls == 1
    assert pool.released == [(pool.acquired[0], False)]
    assert (await manager.health())["creating"] == 0
    await manager.stop()


@pytest.mark.asyncio
async def test_seed_factory_failure_releases_acquired_lease(tmp_path) -> None:
    pool = FakeSitePool()

    def failing_factory(*_args):
        raise RuntimeError("factory failed")

    manager = WebSessionManager(
        _config(tmp_path),
        backend_factory=failing_factory,
        site_pool=pool,
        operation_runner=DirectWebOperationRunner(),
    )
    with pytest.raises(RuntimeError, match="factory failed"):
        await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))

    assert pool.released == [(pool.acquired[0], False)]
    assert manager._creating == set()
    await manager.stop()


@pytest.mark.asyncio
async def test_reset_step_evaluate_and_close_failures_mark_session_unhealthy(tmp_path) -> None:
    pool = FakeSitePool()
    manager, backends = _manager(tmp_path, site_pool=pool, max_sessions=3)

    await manager.seed_session("reset", WebSeedSessionRequest(task=_task("reset")))
    backends[0].fail_reset = True
    with pytest.raises(RuntimeError, match="reset failed"):
        await manager.reset_session("reset", WebResetRequest(task=_task("reset")))
    assert "reset" not in manager._sessions
    assert backends[0].close_calls == 1
    assert pool.released == [(pool.acquired[0], False)]

    await manager.seed_session("step", WebSeedSessionRequest(task=_task("step")))
    backends[1].fail_step = True
    with pytest.raises(RuntimeError, match="step failed"):
        await manager.step("step", _step("operation"))
    assert manager._sessions["step"].status == "error"

    await manager.seed_session("evaluate", WebSeedSessionRequest(task=_task("evaluate")))
    backends[2].fail_evaluate = True
    backends[2].fail_close = True
    with pytest.raises(RuntimeError, match="evaluate failed"):
        await manager.evaluate("evaluate")
    assert manager._sessions["evaluate"].status == "error"

    await manager.stop()
    assert [healthy for _lease, healthy in pool.released] == [False, False, False]


@pytest.mark.asyncio
async def test_cancelled_reset_immediately_releases_backend_and_lease(tmp_path) -> None:
    pool = FakeSitePool()
    manager, backends = _manager(tmp_path, site_pool=pool)
    await manager.seed_session("cancelled", WebSeedSessionRequest(task=_task("cancelled")))

    def cancel_reset(_task):
        raise asyncio.CancelledError

    backends[0].reset = cancel_reset
    with pytest.raises(asyncio.CancelledError):
        await manager.reset_session("cancelled", WebResetRequest(task=_task("cancelled")))

    assert "cancelled" not in manager._sessions
    assert backends[0].close_calls == 1
    assert pool.released == [(pool.acquired[0], False)]


@pytest.mark.asyncio
async def test_reset_requires_same_task_and_step_cache_is_bounded(tmp_path) -> None:
    manager, backends = _manager(tmp_path)
    await manager.seed_session("session", WebSeedSessionRequest(task=_task("0")))

    with pytest.raises(SessionConflictError, match="already owns"):
        await manager.reset_session("session", WebResetRequest(task=_task("other")))

    state = manager._sessions["session"]
    state.operations.update(
        (
            f"operation-{index}",
            WebStepResponse(
                operation_id=f"operation-{index}",
                observation=state.observation,
                execution_ok=True,
            ),
        )
        for index in range(128)
    )
    await manager.step("session", _step("operation-128", name="truncate"))

    assert len(state.operations) == 128
    assert "operation-0" not in state.operations
    assert "operation-1" in state.operations
    assert state.status == "finished"
    assert backends[0].step_calls == 1
    await manager.stop()


def test_site_pool_selection_uses_configured_mode(tmp_path) -> None:
    assert isinstance(WebSessionManager._make_site_pool(_config(tmp_path)), UnmanagedSitePool)
    assert isinstance(
        WebSessionManager._make_site_pool(_config(tmp_path, site_pool_mode="local_locks")),
        LocalSiteLockPool,
    )


@pytest.mark.asyncio
async def test_reaper_closes_expired_sessions(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    manager, _ = _manager(tmp_path)
    await manager.seed_session("stale", WebSeedSessionRequest(task=_task()))
    manager._sessions["stale"].last_access_at = 0
    manager.close_session = AsyncMock(return_value=True)
    sleep_calls = 0

    async def one_iteration(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", one_iteration)
    with pytest.raises(asyncio.CancelledError):
        await manager._reaper_loop()

    manager.close_session.assert_awaited_once_with("stale")
    await manager.stop()
