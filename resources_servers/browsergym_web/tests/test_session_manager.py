# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading
from typing import Any

import pytest

from nemo_gym.web.models import (
    WebAction,
    WebBenchmark,
    WebObservation,
    WebStepResult,
    WebTask,
    WebVerifierResult,
)
from resources_servers.browsergym_web.config import BrowserGymWebResourcesServerConfig
from resources_servers.browsergym_web.models import WebSeedSessionRequest, WebStepRequest
from resources_servers.browsergym_web.session_manager import (
    BenchmarkPreconditionError,
    BrowserGymSessionManager,
    CapacityUnavailableError,
    SessionConflictError,
)


class FakeBackend:
    def __init__(self, _config, session_id, _artifacts):
        self.session_id = session_id
        self.step_calls = 0
        self.evaluate_calls = 0
        self.closed = False
        self.observation = WebObservation(url="about:blank")

    def reset(self, task: WebTask):
        self.observation = WebObservation(url=f"https://example.test/{task.task_id}")
        return self.observation, {"fake": True}

    def observe(self):
        return self.observation

    def step(self, action: WebAction):
        self.step_calls += 1
        self.observation = WebObservation(url=f"https://example.test/step/{self.step_calls}")
        return WebStepResult(observation=self.observation, execution_ok=True)

    def evaluate(self, final_answer=None):
        del final_answer
        self.evaluate_calls += 1
        return WebVerifierResult(reward=1.0, raw_score=1.0, task_success=True)

    def close(self):
        self.closed = True


class ThreadAffinityBackend(FakeBackend):
    call_threads: set[int] = set()

    @classmethod
    def _record_thread(cls):
        cls.call_threads.add(threading.get_ident())

    def reset(self, task: WebTask):
        self._record_thread()
        return super().reset(task)

    def observe(self):
        self._record_thread()
        return super().observe()

    def step(self, action: WebAction):
        self._record_thread()
        return super().step(action)

    def evaluate(self, final_answer=None):
        self._record_thread()
        return super().evaluate(final_answer)

    def close(self):
        self._record_thread()
        return super().close()


def _config(tmp_path, **updates: Any) -> BrowserGymWebResourcesServerConfig:
    values = {
        "name": "browsergym_web",
        "host": "localhost",
        "port": 8000,
        "entrypoint": "app.py",
        "domain": "agent",
        "artifact_dir": str(tmp_path),
    }
    values.update(updates)
    return BrowserGymWebResourcesServerConfig(**values)


def _task(task_id: str = "0") -> WebTask:
    return WebTask(benchmark=WebBenchmark.WEBARENA, task_id=task_id)


@pytest.mark.asyncio
async def test_session_lifecycle_is_idempotent(tmp_path):
    backends: list[FakeBackend] = []

    def factory(*args):
        backend = FakeBackend(*args)
        backends.append(backend)
        return backend

    manager = BrowserGymSessionManager(_config(tmp_path), backend_factory=factory)
    seed = await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))
    repeated = await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))

    assert seed.observation.url.endswith("/0")
    assert seed.info["site_isolated"] is False
    assert repeated.session_id == seed.session_id
    assert len(backends) == 1

    request = WebStepRequest(
        operation_id="operation-1",
        action=WebAction(name="noop", script="noop()"),
    )
    first_step = await manager.step("session-a", request)
    repeated_step = await manager.step("session-a", request)
    assert first_step == repeated_step
    assert backends[0].step_calls == 1

    first_eval = await manager.evaluate("session-a", "done")
    repeated_eval = await manager.evaluate("session-a", "done")
    assert first_eval == repeated_eval
    assert backends[0].evaluate_calls == 1

    assert await manager.close_session("session-a") is True
    assert backends[0].closed is True
    assert (await manager.health())["site_pool"]["active_leases"] == 0
    await manager.stop()


@pytest.mark.asyncio
async def test_capacity_and_session_identity_are_enforced(tmp_path):
    manager = BrowserGymSessionManager(_config(tmp_path, max_sessions=1), backend_factory=FakeBackend)
    await manager.seed_session("session-a", WebSeedSessionRequest(task=_task("0")))

    with pytest.raises(CapacityUnavailableError):
        await manager.seed_session("session-b", WebSeedSessionRequest(task=_task("1")))
    with pytest.raises(SessionConflictError):
        await manager.seed_session("session-a", WebSeedSessionRequest(task=_task("1")))

    await manager.close_session("session-a")
    await manager.stop()


@pytest.mark.asyncio
async def test_concurrent_sessions_keep_browser_calls_on_one_thread(tmp_path):
    ThreadAffinityBackend.call_threads = set()
    manager = BrowserGymSessionManager(
        _config(tmp_path, max_sessions=2),
        backend_factory=ThreadAffinityBackend,
    )
    event_loop_thread = threading.get_ident()

    await asyncio.gather(
        manager.seed_session("session-a", WebSeedSessionRequest(task=_task("0"))),
        manager.seed_session("session-b", WebSeedSessionRequest(task=_task("1"))),
    )
    await asyncio.gather(
        manager.step(
            "session-a",
            WebStepRequest(operation_id="a-1", action=WebAction(name="noop", script="noop()")),
        ),
        manager.step(
            "session-b",
            WebStepRequest(operation_id="b-1", action=WebAction(name="noop", script="noop()")),
        ),
    )
    await asyncio.gather(
        manager.evaluate("session-a"),
        manager.evaluate("session-b"),
    )
    await manager.stop()

    assert len(ThreadAffinityBackend.call_threads) == 1
    assert event_loop_thread not in ThreadAffinityBackend.call_threads


@pytest.mark.asyncio
async def test_cancelled_seed_releases_creating_capacity(tmp_path):
    class BlockingSitePool:
        def __init__(self):
            self.started = asyncio.Event()
            self.release_gate = asyncio.Event()

        async def acquire(self, session_id, task):
            del session_id, task
            self.started.set()
            await self.release_gate.wait()
            raise AssertionError("cancelled seed unexpectedly resumed")

        async def release(self, lease, *, healthy):
            del lease, healthy

        async def health(self):
            return {"mode": "test", "active_leases": 0}

    site_pool = BlockingSitePool()
    manager = BrowserGymSessionManager(
        _config(tmp_path, max_sessions=1),
        backend_factory=FakeBackend,
        site_pool=site_pool,
    )
    seed_task = asyncio.create_task(manager.seed_session("session-a", WebSeedSessionRequest(task=_task("0"))))
    await site_pool.started.wait()

    seed_task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await seed_task

    health = await manager.health()
    assert health["creating"] == 0
    assert health["sessions"] == 0
    await manager.stop()


@pytest.mark.asyncio
async def test_reset_value_error_is_a_terminal_benchmark_precondition(tmp_path):
    backends: list[FakeBackend] = []

    class MissingAssetBackend(FakeBackend):
        def reset(self, task: WebTask):
            del task
            raise ValueError("Could not download image: HTTP 404")

    def factory(*args):
        backend = MissingAssetBackend(*args)
        backends.append(backend)
        return backend

    manager = BrowserGymSessionManager(_config(tmp_path), backend_factory=factory)
    with pytest.raises(BenchmarkPreconditionError, match="Could not download image"):
        await manager.seed_session("session-a", WebSeedSessionRequest(task=_task()))

    health = await manager.health()
    assert health["creating"] == 0
    assert health["sessions"] == 0
    assert backends[0].closed is True
    await manager.stop()
