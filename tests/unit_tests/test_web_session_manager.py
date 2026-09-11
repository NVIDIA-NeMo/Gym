# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest

from nemo_gym.web.api_models import WebResetRequest, WebSeedSessionRequest, WebStepRequest, WebStepResponse
from nemo_gym.web.browser_session import BrowserSessionError, BrowserSessionHandle, BrowserSessionSpec
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
    def __init__(self, _config, session_id, _artifacts, browser_lease):
        self.session_id = session_id
        self.browser_lease = browser_lease
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


class FakeBrowserSessionProvider:
    name = "fake_browser"

    def __init__(self) -> None:
        self.acquired: list[BrowserSessionSpec] = []
        self.released: list[BrowserSessionHandle] = []

    async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle:
        self.acquired.append(spec)
        return BrowserSessionHandle(
            session_id=f"browser:{spec.metadata['rollout_session_id']}",
            provider_name=self.name,
            transport="agentenv",
            endpoint="https://agentenv.invalid/session",
        )

    async def release(self, handle: BrowserSessionHandle) -> None:
        self.released.append(handle)


class DelayedBrowserSessionProvider(FakeBrowserSessionProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.finish = asyncio.Event()

    async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle:
        self.acquired.append(spec)
        self.started.set()
        await self.finish.wait()
        return BrowserSessionHandle(
            session_id=f"browser:{spec.metadata['rollout_session_id']}",
            provider_name=self.name,
            transport="agentenv",
            endpoint="https://agentenv.invalid/session",
        )


class FailingHeartbeatBrowserSessionProvider(FakeBrowserSessionProvider):
    async def heartbeat(self, handle: BrowserSessionHandle) -> None:
        raise RuntimeError(f"lease {handle.session_id} expired")


class HangingHeartbeatBrowserSessionProvider(FakeBrowserSessionProvider):
    async def heartbeat(self, handle: BrowserSessionHandle) -> None:
        del handle
        await asyncio.Event().wait()


class LifecycleBrowserSessionProvider(FakeBrowserSessionProvider):
    def __init__(self) -> None:
        super().__init__()
        self.started = 0
        self.closed = 0
        self.heartbeats = 0

    async def start(self) -> None:
        self.started += 1

    async def close(self) -> None:
        self.closed += 1

    async def heartbeat(self, handle: BrowserSessionHandle) -> None:
        del handle
        self.heartbeats += 1


class FailingReleaseBrowserSessionProvider(FakeBrowserSessionProvider):
    async def release(self, handle: BrowserSessionHandle) -> None:
        raise RuntimeError(f"could not release {handle.session_id}")


class CancelThenReleaseBrowserSessionProvider(FakeBrowserSessionProvider):
    def __init__(self, *, fail_second: bool = False) -> None:
        super().__init__()
        self.release_started = asyncio.Event()
        self.release_calls = 0
        self.fail_second = fail_second

    async def release(self, handle: BrowserSessionHandle) -> None:
        self.release_calls += 1
        if self.release_calls == 1:
            self.release_started.set()
            await asyncio.Event().wait()
        if self.fail_second:
            raise RuntimeError("release retry failed")
        self.released.append(handle)


class BrowserErrorProvider(FakeBrowserSessionProvider):
    async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle:
        del spec
        raise BrowserSessionError("provider unavailable")


class InvalidHandleProvider(FakeBrowserSessionProvider):
    async def acquire(self, spec: BrowserSessionSpec):
        del spec
        return object()


class AnonymousHandleProvider(FakeBrowserSessionProvider):
    async def acquire(self, spec: BrowserSessionSpec) -> BrowserSessionHandle:
        del spec
        return BrowserSessionHandle(transport="agentenv")


class FailingSitePool(FakeSitePool):
    async def release(self, lease: SiteLease, *, healthy: bool) -> None:
        await super().release(lease, healthy=healthy)
        raise RuntimeError("site release failed")


class FailingCloseRunner(DirectWebOperationRunner):
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1
        raise RuntimeError("runner close failed")


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


def _manager(tmp_path, *, factory=FakeBackend, site_pool=None, browser_session_provider=None, **config_updates):
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
        browser_session_provider=browser_session_provider,
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
        "browser_lease_id": "session-a",
        "browser_provider": "local_process",
        "browser_transport": "local_process",
    }
    assert status.status == "ready"
    assert status.site_lease_id == "fake:session-a"
    assert status.browser_lease_id == "session-a"
    assert status.browser_provider == "local_process"
    assert status.browser_transport == "local_process"
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
async def test_browser_provider_lease_is_bound_to_backend_and_released_exactly_once(tmp_path) -> None:
    provider = FakeBrowserSessionProvider()
    manager, backends = _manager(tmp_path, browser_session_provider=provider)

    seed = await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))
    assert provider.acquired[0].metadata == {
        "rollout_session_id": "session-a",
        "benchmark": "webarena",
        "task_id": "0",
    }
    assert provider.acquired[0].lease_ttl_seconds == 900
    assert backends[0].browser_lease.session_id == "browser:session-a"
    assert seed.info["browser_provider"] == "fake_browser"
    assert seed.info["browser_transport"] == "agentenv"

    assert await manager.close_session("session-a") is True
    assert await manager.close_session("session-a") is True
    assert len(provider.released) == 1
    assert provider.released[0].session_id == "browser:session-a"


@pytest.mark.asyncio
async def test_late_provider_acquire_is_released_after_seed_timeout(tmp_path) -> None:
    provider = DelayedBrowserSessionProvider()
    pool = FakeSitePool()
    manager, _ = _manager(
        tmp_path,
        site_pool=pool,
        browser_session_provider=provider,
        browser_acquire_timeout_seconds=0.01,
    )

    with pytest.raises(CapacityUnavailableError, match="did not acquire"):
        await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))

    assert manager._creating == set()
    assert pool.released == [(pool.acquired[0], False)]
    cleanup_tasks = tuple(manager._late_browser_cleanup_tasks)
    assert len(cleanup_tasks) == 1
    provider.finish.set()
    await asyncio.gather(*cleanup_tasks)

    assert [handle.session_id for handle in provider.released] == ["browser:session-a"]
    assert (await manager.health())["browser_provider"]["active_leases"] == 0
    await manager.stop()


@pytest.mark.asyncio
async def test_cancelled_seed_releases_a_provider_handle_that_arrives_late(tmp_path) -> None:
    provider = DelayedBrowserSessionProvider()
    pool = FakeSitePool()
    manager, _ = _manager(tmp_path, site_pool=pool, browser_session_provider=provider)

    seed_task = asyncio.create_task(manager.seed_session("session-a", WebSeedSessionRequest(task=_task())))
    await provider.started.wait()
    seed_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await seed_task

    assert manager._creating == set()
    assert pool.released == [(pool.acquired[0], False)]
    cleanup_tasks = tuple(manager._late_browser_cleanup_tasks)
    assert len(cleanup_tasks) == 1
    provider.finish.set()
    await asyncio.gather(*cleanup_tasks)
    assert [handle.session_id for handle in provider.released] == ["browser:session-a"]
    await manager.stop()


@pytest.mark.asyncio
async def test_heartbeat_failure_limit_closes_and_releases_remote_session(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = FailingHeartbeatBrowserSessionProvider()
    manager, _ = _manager(
        tmp_path,
        browser_session_provider=provider,
        browser_heartbeat_failure_limit=1,
    )
    await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))
    sleep_calls = 0

    async def one_iteration(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", one_iteration)
    with pytest.raises(asyncio.CancelledError):
        await manager._browser_heartbeat_loop()

    assert "session-a" not in manager._sessions
    assert [handle.session_id for handle in provider.released] == ["browser:session-a"]


@pytest.mark.asyncio
async def test_hung_heartbeat_is_bounded_and_releases_remote_session(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = HangingHeartbeatBrowserSessionProvider()
    manager, _ = _manager(
        tmp_path,
        browser_session_provider=provider,
        browser_heartbeat_timeout_seconds=0.01,
        browser_heartbeat_failure_limit=1,
    )
    await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))
    sleep_calls = 0

    async def one_iteration(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", one_iteration)
    with pytest.raises(asyncio.CancelledError):
        await manager._browser_heartbeat_loop()

    assert "session-a" not in manager._sessions
    assert [handle.session_id for handle in provider.released] == ["browser:session-a"]


@pytest.mark.asyncio
async def test_provider_lifecycle_is_started_once_and_closed_on_stop(tmp_path) -> None:
    provider = LifecycleBrowserSessionProvider()
    manager, _ = _manager(tmp_path, browser_session_provider=provider)

    await manager.start()
    await manager.start()
    assert provider.started == 1

    await manager.stop()
    assert provider.closed == 1


@pytest.mark.asyncio
async def test_successful_heartbeat_clears_consecutive_failure_count(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = LifecycleBrowserSessionProvider()
    manager, _ = _manager(tmp_path, browser_session_provider=provider)
    await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))
    manager._sessions["session-a"].browser_heartbeat_failures = 2
    sleep_calls = 0

    async def one_iteration(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls > 1:
            raise asyncio.CancelledError

    monkeypatch.setattr(asyncio, "sleep", one_iteration)
    with pytest.raises(asyncio.CancelledError):
        await manager._browser_heartbeat_loop()

    assert provider.heartbeats == 1
    assert manager._sessions["session-a"].browser_heartbeat_failures == 0
    await manager.close_session("session-a")


@pytest.mark.asyncio
async def test_nonrenewable_provider_has_no_heartbeat_loop(tmp_path) -> None:
    manager, _ = _manager(tmp_path, browser_session_provider=FakeBrowserSessionProvider())
    assert await manager._browser_heartbeat_loop() is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "error_type", "message"),
    [
        (BrowserErrorProvider(), CapacityUnavailableError, "provider unavailable"),
        (InvalidHandleProvider(), TypeError, "invalid handle"),
    ],
)
async def test_provider_acquire_failures_release_site_capacity(
    tmp_path,
    provider,
    error_type,
    message: str,
) -> None:
    pool = FakeSitePool()
    manager, _ = _manager(
        tmp_path,
        site_pool=pool,
        browser_session_provider=provider,
    )

    with pytest.raises(error_type, match=message):
        await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))

    assert pool.released == [(pool.acquired[0], False)]
    assert manager._creating == set()
    await manager.stop()


@pytest.mark.asyncio
async def test_provider_fills_missing_handle_identity(tmp_path) -> None:
    provider = AnonymousHandleProvider()
    manager, backends = _manager(tmp_path, browser_session_provider=provider)

    response = await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))

    assert response.info["browser_lease_id"] == "session-a"
    assert response.info["browser_provider"] == provider.name
    assert backends[0].browser_lease.session_id == "session-a"
    await manager.close_session("session-a")


@pytest.mark.asyncio
async def test_close_attempts_all_cleanup_layers_after_independent_failures(tmp_path) -> None:
    provider = FakeBrowserSessionProvider()
    pool = FailingSitePool()
    runner = FailingCloseRunner()
    manager, backends = _manager(
        tmp_path,
        site_pool=pool,
        browser_session_provider=provider,
    )
    manager._shared_operation_runner = None
    manager._make_operation_runner = lambda _session_id: runner
    await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))
    backends[0].fail_close = True

    assert await manager.close_session("session-a") is True
    assert backends[0].close_calls == 1
    assert runner.close_calls == 1
    assert [handle.session_id for handle in provider.released] == ["browser:session-a"]
    assert pool.released == [(pool.acquired[0], False)]


@pytest.mark.asyncio
async def test_failed_seed_attempts_all_cleanup_layers_after_independent_failures(tmp_path) -> None:
    provider = FakeBrowserSessionProvider()
    pool = FailingSitePool()
    runner = FailingCloseRunner()

    class FailingResetAndCloseBackend(FakeBackend):
        def reset(self, task: WebTask):
            del task
            raise RuntimeError("reset failed")

        def close(self):
            super().close()
            raise RuntimeError("backend close failed")

    manager, backends = _manager(
        tmp_path,
        factory=FailingResetAndCloseBackend,
        site_pool=pool,
        browser_session_provider=provider,
    )
    manager._shared_operation_runner = None
    manager._make_operation_runner = lambda _session_id: runner

    with pytest.raises(RuntimeError, match="reset failed"):
        await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))

    assert backends[0].close_calls == 1
    assert runner.close_calls == 1
    assert [handle.session_id for handle in provider.released] == ["browser:session-a"]
    assert pool.released == [(pool.acquired[0], False)]
    assert manager._creating == set()


@pytest.mark.asyncio
async def test_browser_release_failure_is_reported_in_health(tmp_path) -> None:
    provider = FailingReleaseBrowserSessionProvider()
    manager, _ = _manager(tmp_path, browser_session_provider=provider)
    await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))

    await manager.close_session("session-a")
    browser_health = (await manager.health())["browser_provider"]

    assert browser_health["release_failures"] == 1
    assert browser_health["active_leases"] == 1
    assert "could not release" in browser_health["last_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_second", [False, True])
async def test_cancelled_browser_release_gets_one_bounded_retry(tmp_path, fail_second: bool) -> None:
    provider = CancelThenReleaseBrowserSessionProvider(fail_second=fail_second)
    manager, _ = _manager(tmp_path, browser_session_provider=provider)
    handle = BrowserSessionHandle(
        session_id="browser:session-a",
        provider_name=provider.name,
        transport="agentenv",
    )
    manager._browser_leases_acquired = 1
    release_task = asyncio.create_task(manager._release_browser_lease(handle))
    await provider.release_started.wait()

    release_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await release_task

    assert provider.release_calls == 2
    browser_health = (await manager.health())["browser_provider"]
    if fail_second:
        assert browser_health["release_failures"] == 1
        assert browser_health["released"] == 0
    else:
        assert [item.session_id for item in provider.released] == ["browser:session-a"]
        assert browser_health["released"] == 1


@pytest.mark.asyncio
async def test_repeated_seed_cancellation_keeps_failed_seed_cleanup_alive(tmp_path) -> None:
    provider = FakeBrowserSessionProvider()
    pool = FakeSitePool()
    reset_started = asyncio.Event()
    close_started = asyncio.Event()
    finish_close = asyncio.Event()

    class BlockingSeedCleanupRunner(DirectWebOperationRunner):
        async def run(self, operation, *args):
            if getattr(operation, "__name__", "") == "reset":
                reset_started.set()
                await asyncio.Event().wait()
            if getattr(operation, "__name__", "") == "close":
                close_started.set()
                await finish_close.wait()
            return operation(*args)

    manager = WebSessionManager(
        _config(tmp_path),
        backend_factory=FakeBackend,
        site_pool=pool,
        operation_runner=BlockingSeedCleanupRunner(),
        browser_session_provider=provider,
    )
    caller = asyncio.create_task(manager.seed_session("session-a", WebSeedSessionRequest(task=_task())))
    await reset_started.wait()
    caller.cancel()
    await close_started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert provider.released == []
    assert manager._creating == {"session-a"}
    finish_close.set()
    await asyncio.gather(*tuple(manager._failed_seed_cleanup_tasks))

    assert [handle.session_id for handle in provider.released] == ["browser:session-a"]
    assert pool.released == [(pool.acquired[0], False)]
    assert manager._creating == set()
    await manager.stop()


@pytest.mark.asyncio
async def test_cancelled_close_keeps_cleanup_alive_until_leases_are_released(tmp_path) -> None:
    provider = FakeBrowserSessionProvider()
    close_started = asyncio.Event()
    finish_close = asyncio.Event()

    class BlockingCloseRunner(DirectWebOperationRunner):
        async def run(self, operation, *args):
            if getattr(operation, "__name__", "") == "close":
                close_started.set()
                await finish_close.wait()
            return operation(*args)

    manager = WebSessionManager(
        _config(tmp_path),
        backend_factory=FakeBackend,
        operation_runner=BlockingCloseRunner(),
        browser_session_provider=provider,
    )
    await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))

    caller = asyncio.create_task(manager.close_session("session-a"))
    await close_started.wait()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller

    assert "session-a" not in manager._sessions
    assert provider.released == []
    finish_close.set()
    await asyncio.gather(*tuple(manager._session_cleanup_tasks.values()))
    assert [handle.session_id for handle in provider.released] == ["browser:session-a"]
    await manager.stop()


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
