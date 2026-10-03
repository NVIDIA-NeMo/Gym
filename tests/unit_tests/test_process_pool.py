# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The warm process pool must bound everything it owns: queue, workers, deadlines, and descendants."""

import asyncio
import contextvars
import multiprocessing as mp
import os
import pickle
import threading
from typing import Any

import psutil
import pytest

from nemo_gym.process_pool import (
    PoolClosedError,
    PoolSaturatedError,
    ProcessPoolError,
    ResultSerializationError,
    TaskFailedError,
    TaskSerializationError,
    TaskTimeoutError,
    WarmProcessPool,
    WarmProcessPoolConfig,
    WorkerCrashedError,
    WorkerStartError,
    _ResultMessage,
    _TaskMessage,
    _wait_readable,
    _worker_main,
    worker_start_method_is_supported,
)
from tests.unit_tests import process_pool_tasks as tasks


def _descendants() -> list[psutil.Process]:
    return psutil.Process(os.getpid()).children(recursive=True)


def _live_descendants() -> list[psutil.Process]:
    """Live children of this process, excluding multiprocessing's own per-process daemons.

    The ``spawn`` and ``forkserver`` start methods keep a resource tracker (and a fork server)
    alive for the life of the parent interpreter. Those belong to :mod:`multiprocessing`, not to
    any one pool, so they are not leaks.
    """
    live = []
    for child in _descendants():
        try:
            zombie = child.status() == psutil.STATUS_ZOMBIE
            cmdline = "" if zombie else " ".join(child.cmdline())
        except psutil.NoSuchProcess:
            continue
        except (psutil.AccessDenied, psutil.ZombieProcess):
            zombie, cmdline = True, ""
        if not zombie and ("multiprocessing.resource_tracker" in cmdline or "multiprocessing.forkserver" in cmdline):
            continue
        # A zombie is a child nobody joined. That is a leak, so it stays in the list.
        live.append(child)
    return live


def _pool(**overrides: Any) -> WarmProcessPool:
    config = {
        "num_workers": 2,
        "max_pending": 16,
        "default_timeout_seconds": 10.0,
        "kill_grace_seconds": 0.5,
        "shutdown_grace_seconds": 1.0,
        "worker_start_timeout_seconds": 60.0,
    }
    config.update(overrides)
    return WarmProcessPool(WarmProcessPoolConfig(**config), name="test_pool")


@pytest.fixture
def no_leaked_children():
    before = {p.pid for p in _live_descendants()}
    yield
    after = {p.pid for p in _live_descendants()} - before
    assert not after, f"pool left descendants running: {sorted(after)}"


class TestNormalOperation:
    async def test_runs_functions_with_args_and_kwargs(self, no_leaked_children) -> None:
        async with _pool() as pool:
            assert await pool.run(tasks.square, 7) == 49
            assert await pool.run(tasks.add, 2, b=3) == 5
            assert pool.stats().completed == 2
            assert pool.stats().live_workers == 2

    async def test_many_concurrent_tasks_return_their_own_results(self, no_leaked_children) -> None:
        async with _pool(num_workers=3, max_pending=64) as pool:
            results = await asyncio.gather(*(pool.run(tasks.square, i) for i in range(60)))
        assert results == [i * i for i in range(60)]

    async def test_large_payloads_round_trip(self, no_leaked_children) -> None:
        payload = os.urandom(2 * 1024 * 1024)
        async with _pool(num_workers=1) as pool:
            assert await pool.run(tasks.echo_bytes, payload) == payload

    async def test_initializer_runs_once_per_worker_before_any_task(self, no_leaked_children) -> None:
        pool = WarmProcessPool(
            WarmProcessPoolConfig(num_workers=2, shutdown_grace_seconds=1.0),
            initializer=tasks.set_init_value,
            initargs=("warm",),
        )
        async with pool:
            assert await asyncio.gather(*(pool.run(tasks.read_init_value) for _ in range(4))) == ["warm"] * 4

    async def test_tasks_reuse_the_same_worker_processes(self, no_leaked_children) -> None:
        async with _pool(num_workers=1) as pool:
            pids = {await pool.run(tasks.pid) for _ in range(5)}
        assert len(pids) == 1

    async def test_timing_separates_queue_wait_from_execution(self, no_leaked_children) -> None:
        async with _pool(num_workers=1) as pool:
            blocker = asyncio.create_task(pool.run(tasks.spin_ms, 150))
            await asyncio.sleep(0.02)
            value, timing = await pool.run_with_timing(tasks.spin_ms, 40)
            await blocker
        assert value == 40
        assert timing.execution_seconds >= 0.04
        # The blocker occupied the only worker for ~150ms; this task waited most of that in the queue.
        assert timing.queue_wait_seconds > 0.05
        assert timing.worker_pid > 0
        stats = pool.stats()
        assert stats.queue_wait_seconds_max >= timing.queue_wait_seconds
        assert stats.execution_seconds_total >= 0.19

    @pytest.mark.skipif(not worker_start_method_is_supported("forkserver"), reason="forkserver unavailable here")
    async def test_forkserver_start_method_works(self, no_leaked_children) -> None:
        async with _pool(num_workers=1, start_method="forkserver") as pool:
            assert await pool.run(tasks.square, 3) == 9

    def test_fork_is_not_a_permitted_start_method(self) -> None:
        with pytest.raises(ValueError):
            WarmProcessPoolConfig(start_method="fork")

    async def test_callers_request_context_survives_a_run(self, no_leaked_children) -> None:
        # Nothing non-picklable crosses into the child, and the caller's contextvars (rollout
        # correlation ids live there) are intact when the result comes back.
        rollout_id: contextvars.ContextVar[str] = contextvars.ContextVar("rollout_id")
        rollout_id.set("rollout-42")
        async with _pool(num_workers=1) as pool:
            assert await pool.run(tasks.square, 3) == 9
        assert rollout_id.get() == "rollout-42"


class TestSerialization:
    async def test_unpicklable_task_fails_at_the_call_site_without_reaching_a_worker(self, no_leaked_children) -> None:
        async with _pool(num_workers=1) as pool:
            with pytest.raises(TaskSerializationError):
                await pool.run(lambda: 1)
            stats = pool.stats()
        assert stats.task_serialization_errors == 1
        assert stats.submitted == 0

    async def test_unpicklable_result_is_reported_distinctly(self, no_leaked_children) -> None:
        async with _pool(num_workers=1) as pool:
            with pytest.raises(ResultSerializationError):
                await pool.run(tasks.return_unpicklable)
            # The worker survives a bad result and keeps serving.
            assert await pool.run(tasks.square, 2) == 4
            assert pool.stats().result_serialization_errors == 1
            assert pool.stats().restarts == 0

    async def test_task_exception_arrives_with_type_message_and_child_traceback(self, no_leaked_children) -> None:
        async with _pool(num_workers=1) as pool:
            with pytest.raises(TaskFailedError) as info:
                await pool.run(tasks.raise_value_error, "bad input")
            assert await pool.run(tasks.square, 3) == 9
        assert info.value.error_type == "ValueError"
        assert info.value.message == "bad input"
        assert "raise_value_error" in info.value.child_traceback
        assert pool.stats().failed == 1


class TestQueueBounds:
    async def test_saturated_queue_fails_fast_with_zero_queue_timeout(self, no_leaked_children) -> None:
        async with _pool(num_workers=1, max_pending=1) as pool:
            in_flight = asyncio.create_task(pool.run(tasks.sleep_then, "a", 0.4))
            await asyncio.sleep(0.05)
            queued = asyncio.create_task(pool.run(tasks.sleep_then, "b", 0.0))
            await asyncio.sleep(0.05)
            with pytest.raises(PoolSaturatedError):
                await pool.run(tasks.square, 1, queue_timeout=0)
            assert await asyncio.gather(in_flight, queued) == ["a", "b"]
        stats = pool.stats()
        assert stats.saturation_rejections == 1
        assert stats.saturation_waits >= 1

    async def test_saturated_queue_raises_after_queue_timeout(self, no_leaked_children) -> None:
        async with _pool(num_workers=1, max_pending=1) as pool:
            in_flight = asyncio.create_task(pool.run(tasks.sleep_then, "a", 0.5))
            await asyncio.sleep(0.05)
            queued = asyncio.create_task(pool.run(tasks.sleep_then, "b", 0.0))
            await asyncio.sleep(0.05)
            with pytest.raises(PoolSaturatedError):
                await pool.run(tasks.square, 1, queue_timeout=0.05)
            await asyncio.gather(in_flight, queued)

    async def test_saturated_queue_waits_when_no_queue_timeout(self, no_leaked_children) -> None:
        async with _pool(num_workers=1, max_pending=1) as pool:
            first = asyncio.create_task(pool.run(tasks.sleep_then, "a", 0.2))
            await asyncio.sleep(0.05)
            second = asyncio.create_task(pool.run(tasks.sleep_then, "b", 0.0))
            await asyncio.sleep(0.05)
            third = await pool.run(tasks.square, 4, queue_timeout=None)
            assert await asyncio.gather(first, second) == ["a", "b"]
        assert third == 16

    async def test_pending_and_live_worker_counts_stay_within_bounds_under_load(self, no_leaked_children) -> None:
        max_pending, num_workers = 8, 3
        observed_pending, observed_children = [], []
        async with _pool(num_workers=num_workers, max_pending=max_pending) as pool:

            async def sample() -> None:
                while True:
                    observed_pending.append(pool.stats().pending)
                    observed_children.append(len(_live_descendants()))
                    await asyncio.sleep(0.005)

            sampler = asyncio.create_task(sample())
            results = await asyncio.gather(*(pool.run(tasks.spin_ms, 5) for _ in range(120)))
            sampler.cancel()
        assert results == [5] * 120
        assert max(observed_pending) <= max_pending
        assert max(observed_children) <= num_workers
        assert pool.stats().live_workers == 0


class TestDeadlines:
    async def test_timeout_kills_and_replaces_the_worker(self, no_leaked_children) -> None:
        async with _pool(num_workers=1) as pool:
            first_pid = await pool.run(tasks.pid)
            with pytest.raises(TaskTimeoutError):
                await pool.run(tasks.busy_loop_forever, timeout=0.2)
            second_pid = await pool.run(tasks.pid)
        assert first_pid != second_pid
        stats = pool.stats()
        assert stats.timeouts == 1
        assert stats.restarts == 1

    async def test_worker_ignoring_sigterm_is_still_replaced_via_sigkill(self, no_leaked_children) -> None:
        async with _pool(num_workers=1, kill_grace_seconds=0.2) as pool:
            with pytest.raises(TaskTimeoutError):
                await pool.run(tasks.busy_loop_ignoring_sigterm, timeout=0.2)
            assert await pool.run(tasks.square, 5) == 25
        assert pool.stats().restarts == 1

    async def test_a_stuck_worker_does_not_corrupt_other_workers_results(self, no_leaked_children) -> None:
        async with _pool(num_workers=2, max_pending=64) as pool:
            stuck = asyncio.create_task(pool.run(tasks.busy_loop_forever, timeout=0.3))
            await asyncio.sleep(0.05)
            healthy = await asyncio.gather(*(pool.run(tasks.square, i) for i in range(30)))
            with pytest.raises(TaskTimeoutError):
                await stuck
        assert healthy == [i * i for i in range(30)]

    async def test_no_deadline_when_timeout_is_none(self, no_leaked_children) -> None:
        async with _pool(num_workers=1, default_timeout_seconds=0.05) as pool:
            assert await pool.run(tasks.sleep_then, "slow", 0.15, timeout=None) == "slow"


class TestCancellation:
    async def test_cancelling_an_in_flight_task_frees_the_slot_by_replacing_the_worker(
        self, no_leaked_children
    ) -> None:
        async with _pool(num_workers=1) as pool:
            first_pid = await pool.run(tasks.pid)
            running = asyncio.create_task(pool.run(tasks.busy_loop_forever, timeout=None))
            await asyncio.sleep(0.1)
            running.cancel()
            with pytest.raises(asyncio.CancelledError):
                await running
            # The next task must not wait behind the cancelled one.
            second_pid = await asyncio.wait_for(pool.run(tasks.pid), timeout=10.0)
        assert first_pid != second_pid
        stats = pool.stats()
        assert stats.cancelled_in_flight == 1
        assert stats.restarts == 1

    async def test_cancelling_a_queued_task_never_dispatches_it(self, no_leaked_children) -> None:
        async with _pool(num_workers=1, max_pending=4) as pool:
            blocker = asyncio.create_task(pool.run(tasks.sleep_then, "a", 0.3))
            await asyncio.sleep(0.05)
            queued = asyncio.create_task(pool.run(tasks.square, 2))
            await asyncio.sleep(0.02)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert await blocker == "a"
            await pool.run(tasks.square, 1)
        stats = pool.stats()
        assert stats.cancelled_pending == 1
        assert stats.restarts == 0


class TestWorkerCrash:
    async def test_crashed_worker_fails_the_task_and_is_replaced(self, no_leaked_children) -> None:
        async with _pool(num_workers=1) as pool:
            with pytest.raises(WorkerCrashedError) as info:
                await pool.run(tasks.hard_exit, 3)
            assert await pool.run(tasks.square, 6) == 36
        assert info.value.exitcode == 3
        stats = pool.stats()
        assert stats.crashes == 1
        assert stats.restarts == 1


class TestRecycling:
    async def test_worker_is_retired_after_max_tasks(self, no_leaked_children) -> None:
        async with _pool(num_workers=1, max_tasks_per_worker=2) as pool:
            pids = [await pool.run(tasks.pid) for _ in range(5)]
            assert pool.stats().live_workers == 1
        assert pids[0] == pids[1]
        assert pids[1] != pids[2]
        assert pids[2] == pids[3]
        assert pids[3] != pids[4]
        assert pool.stats().recycles == 2
        assert pool.stats().restarts == 0


class TestWorkerDeathBetweenTasks:
    async def test_worker_killed_while_idle_fails_the_next_task_and_is_replaced(self, no_leaked_children) -> None:
        async with _pool(num_workers=1) as pool:
            victim = await pool.run(tasks.pid)
            os.kill(victim, 9)
            # Let the kernel close the dead child's end of the pipe before we write to it.
            for _ in range(100):
                if not psutil.pid_exists(victim) or psutil.Process(victim).status() == psutil.STATUS_ZOMBIE:
                    break
                await asyncio.sleep(0.01)
            with pytest.raises(WorkerCrashedError):
                await pool.run(tasks.square, 2)
            assert await pool.run(tasks.square, 2) == 4
        assert pool.stats().crashes == 1
        assert pool.stats().restarts == 1

    def test_is_alive_is_false_for_a_closed_process_handle(self) -> None:
        from nemo_gym.process_pool import _Worker

        process = mp.get_context("spawn").Process(target=int)
        process.start()
        process.join(5)
        process.close()
        assert _Worker(slot=0, process=process, conn=None).is_alive() is False


class TestTaskUnpicklableInWorker:
    async def test_function_importable_only_in_the_parent_is_a_serialization_error(
        self, tmp_path, monkeypatch, no_leaked_children
    ) -> None:
        # Spawned workers copy sys.path at start. A module added to the parent's path afterwards
        # pickles fine here and fails to import there, which is exactly the case this path exists for.
        module_dir = tmp_path / "late_module"
        module_dir.mkdir()
        (module_dir / "late_only.py").write_text("def triple(x):\n    return 3 * x\n")
        async with _pool(num_workers=1) as pool:
            monkeypatch.syspath_prepend(str(module_dir))
            import late_only  # noqa: PLC0415

            with pytest.raises(TaskSerializationError, match="could not be unpickled in worker"):
                await pool.run(late_only.triple, 2)
            assert await pool.run(tasks.square, 2) == 4
        assert pool.stats().task_serialization_errors == 1
        assert pool.stats().restarts == 0


class TestStartup:
    async def test_initializer_that_never_reports_ready_is_a_start_error(self, no_leaked_children) -> None:
        pool = WarmProcessPool(
            WarmProcessPoolConfig(num_workers=1, worker_start_timeout_seconds=1.0, kill_grace_seconds=0.2),
            initializer=tasks.hanging_initializer,
        )
        with pytest.raises(WorkerStartError, match="did not report ready"):
            await pool.start()

    async def test_initializer_that_exits_is_a_start_error(self, no_leaked_children) -> None:
        pool = WarmProcessPool(WarmProcessPoolConfig(num_workers=1), initializer=tasks.exiting_initializer)
        with pytest.raises(WorkerStartError, match="exited during startup"):
            await pool.start()

    async def test_failing_initializer_raises_with_the_child_traceback_and_leaves_nothing(
        self, no_leaked_children
    ) -> None:
        pool = WarmProcessPool(
            WarmProcessPoolConfig(num_workers=2, worker_start_timeout_seconds=30.0),
            initializer=tasks.failing_initializer,
        )
        with pytest.raises(WorkerStartError, match="initializer exploded on purpose"):
            await pool.start()
        assert pool.stats().live_workers == 0

    async def test_unpicklable_initializer_fails_in_the_parent(self, no_leaked_children) -> None:
        pool = WarmProcessPool(WarmProcessPoolConfig(num_workers=1), initializer=lambda: None)
        with pytest.raises(WorkerStartError):
            await pool.start()

    async def test_run_before_start_is_a_programming_error(self) -> None:
        pool = _pool()
        with pytest.raises(RuntimeError):
            await pool.run(tasks.square, 1)
        await pool.aclose()

    async def test_start_is_idempotent(self, no_leaked_children) -> None:
        pool = _pool(num_workers=1)
        await pool.start()
        await pool.start()
        assert pool.stats().live_workers == 1
        await pool.aclose()

    async def test_concurrent_starts_spawn_the_workers_once(self, no_leaked_children) -> None:
        pool = _pool(num_workers=2)
        await asyncio.gather(pool.start(), pool.start(), pool.start())
        assert pool.stats().live_workers == 2
        assert len(_live_descendants()) == 2
        await pool.aclose()

    async def test_start_after_close_is_rejected(self, no_leaked_children) -> None:
        pool = _pool(num_workers=1)
        await pool.aclose()
        with pytest.raises(PoolClosedError):
            await pool.start()


class TestShutdown:
    async def test_close_kills_in_flight_work_after_the_grace_period(self, no_leaked_children) -> None:
        pool = _pool(num_workers=1, shutdown_grace_seconds=0.2)
        await pool.start()
        running = asyncio.create_task(pool.run(tasks.busy_loop_forever, timeout=None))
        await asyncio.sleep(0.1)
        await pool.aclose()
        with pytest.raises(PoolClosedError):
            await running
        assert not _live_descendants()

    async def test_close_fails_queued_tasks(self, no_leaked_children) -> None:
        pool = _pool(num_workers=1, shutdown_grace_seconds=0.1)
        await pool.start()
        blocker = asyncio.create_task(pool.run(tasks.busy_loop_forever, timeout=None))
        await asyncio.sleep(0.05)
        queued = asyncio.create_task(pool.run(tasks.square, 3))
        await asyncio.sleep(0.02)
        await pool.aclose()
        with pytest.raises(PoolClosedError):
            await queued
        with pytest.raises(PoolClosedError):
            await blocker

    async def test_submission_waiting_for_a_queue_slot_fails_when_the_pool_closes(self, no_leaked_children) -> None:
        pool = _pool(num_workers=1, max_pending=1, shutdown_grace_seconds=0.1)
        await pool.start()
        blocker = asyncio.create_task(pool.run(tasks.busy_loop_forever, timeout=None))
        await asyncio.sleep(0.05)
        queued = asyncio.create_task(pool.run(tasks.square, 1))
        await asyncio.sleep(0.02)
        # The queue is full, so this one is parked inside queue.put() when the pool closes.
        waiting = asyncio.create_task(pool.run(tasks.square, 2))
        await asyncio.sleep(0.02)
        await pool.aclose()
        for task in (blocker, queued, waiting):
            with pytest.raises(PoolClosedError):
                await task

    async def test_replacement_is_skipped_when_the_pool_is_already_closing(self, no_leaked_children) -> None:
        pool = _pool(num_workers=1, shutdown_grace_seconds=3.0)
        await pool.start()
        running = asyncio.create_task(pool.run(tasks.busy_loop_forever, timeout=0.3))
        await asyncio.sleep(0.05)
        # The shutdown grace outlasts the deadline, so the timeout fires while the pool is closing.
        # The slot must reap the worker and stop, not spawn a replacement into a closing pool.
        await pool.aclose()
        with pytest.raises(TaskTimeoutError):
            await running
        stats = pool.stats()
        assert stats.live_workers == 0
        assert stats.restarts == 1

    async def test_closing_during_a_replacement_of_a_sigterm_ignoring_worker_leaves_nothing(
        self, no_leaked_children
    ) -> None:
        # The deadline fires during shutdown, so the slot is inside the terminate-then-kill sequence
        # for a worker that ignores SIGTERM when its loop is cancelled. That worker must still die.
        pool = _pool(num_workers=1, shutdown_grace_seconds=3.0, kill_grace_seconds=1.5)
        await pool.start()
        running = asyncio.create_task(pool.run(tasks.busy_loop_ignoring_sigterm, timeout=0.3))
        await asyncio.sleep(0.05)
        await pool.aclose()
        with pytest.raises(TaskTimeoutError):
            await running
        assert pool.stats().live_workers == 0

    async def test_run_after_close_is_rejected(self, no_leaked_children) -> None:
        pool = _pool(num_workers=1)
        await pool.start()
        await pool.aclose()
        with pytest.raises(PoolClosedError):
            await pool.run(tasks.square, 1)
        await pool.aclose()  # idempotent

    async def test_concurrent_closes_all_wait_for_the_reap(self, no_leaked_children) -> None:
        pool = _pool(num_workers=2, shutdown_grace_seconds=0.2)
        await pool.start()
        running = asyncio.create_task(pool.run(tasks.busy_loop_forever, timeout=None))
        await asyncio.sleep(0.05)
        await asyncio.gather(pool.aclose(), pool.aclose(), pool.aclose())
        # Every caller returned only after the workers were gone.
        assert not _live_descendants()
        with pytest.raises(PoolClosedError):
            await running

    async def test_close_waits_for_a_finishing_task(self, no_leaked_children) -> None:
        pool = _pool(num_workers=1, shutdown_grace_seconds=2.0)
        await pool.start()
        running = asyncio.create_task(pool.run(tasks.sleep_then, "done", 0.2))
        await asyncio.sleep(0.05)
        await pool.aclose()
        assert await running == "done"


class TestResilience:
    async def test_a_failed_respawn_is_retried_on_the_next_task(self, no_leaked_children, monkeypatch) -> None:
        pool = _pool(num_workers=1)
        await pool.start()
        real_spawn = pool._spawn_worker
        failures_left = [1]

        async def flaky_spawn(slot):
            if failures_left[0]:
                failures_left[0] -= 1
                raise WorkerStartError("simulated: no more pids")
            return await real_spawn(slot)

        monkeypatch.setattr(pool, "_spawn_worker", flaky_spawn)
        with pytest.raises(WorkerCrashedError):
            await pool.run(tasks.hard_exit, 2)  # the replacement spawn fails; the slot is left empty
        assert pool.stats().live_workers == 0
        assert await asyncio.wait_for(pool.run(tasks.square, 5), timeout=30) == 25  # retried on demand
        assert pool.stats().live_workers == 1
        assert pool.stats().spawn_failures == 1
        await pool.aclose()

    async def test_a_persistently_failing_respawn_fails_the_task_and_keeps_the_slot_alive(
        self, no_leaked_children, monkeypatch
    ) -> None:
        pool = _pool(num_workers=1, kill_grace_seconds=0.05)
        await pool.start()
        real_spawn = pool._spawn_worker
        failures_left = [2]

        async def flaky_spawn(slot):
            if failures_left[0]:
                failures_left[0] -= 1
                raise WorkerStartError("simulated: no more pids")
            return await real_spawn(slot)

        monkeypatch.setattr(pool, "_spawn_worker", flaky_spawn)
        with pytest.raises(WorkerCrashedError):
            await pool.run(tasks.hard_exit, 2)
        with pytest.raises(WorkerStartError):
            await pool.run(tasks.square, 1)  # second spawn failure is charged to this task
        assert await asyncio.wait_for(pool.run(tasks.square, 6), timeout=30) == 36
        assert pool.stats().spawn_failures == 2
        await pool.aclose()

    async def test_a_parent_side_bug_fails_one_task_and_replaces_the_worker(
        self, no_leaked_children, monkeypatch
    ) -> None:
        import nemo_gym.process_pool as module

        pool = _pool(num_workers=1)
        await pool.start()
        real_loads = module.pickle.loads
        explode = [True]

        def broken_loads(raw):
            if explode[0]:
                explode[0] = False
                raise RuntimeError("corrupt reply")
            return real_loads(raw)

        monkeypatch.setattr(module.pickle, "loads", broken_loads)
        with pytest.raises(ProcessPoolError, match="internal error"):
            await pool.run(tasks.square, 2)
        assert await pool.run(tasks.square, 2) == 4
        assert pool.stats().restarts == 1
        await pool.aclose()


class TestWorkerLoopInProcess:
    """Drive ``_worker_main`` in a thread over a real pipe so the child-side code is exercised here."""

    def _run_worker(self, initializer=None, initargs=()) -> tuple[Any, threading.Thread]:
        parent, child = mp.get_context("spawn").Pipe(duplex=True)
        thread = threading.Thread(
            target=_worker_main,
            args=(child, initializer, initargs),
            kwargs={"install_signal_handlers": False},
            daemon=True,
        )
        thread.start()
        return parent, thread

    def _send_task(self, conn: Any, task_id: int, fn: Any, *args: Any, **kwargs: Any) -> _ResultMessage:
        conn.send_bytes(pickle.dumps(_TaskMessage(task_id=task_id, fn=fn, args=args, kwargs=kwargs)))
        return pickle.loads(conn.recv_bytes())

    def test_ready_then_results_then_shutdown(self) -> None:
        parent, thread = self._run_worker(tasks.set_init_value, ("x",))
        assert pickle.loads(parent.recv_bytes()) == ("ready",)
        ok = self._send_task(parent, 1, tasks.square, 9)
        assert (ok.task_id, ok.ok, ok.value) == (1, True, 81)
        failed = self._send_task(parent, 2, tasks.raise_value_error, "boom")
        assert (failed.ok, failed.error_type, failed.error_message) == (False, "ValueError", "boom")
        assert "boom" in failed.child_traceback
        unpicklable = self._send_task(parent, 3, tasks.return_unpicklable)
        assert unpicklable.ok is False and unpicklable.result_unpicklable is True
        parent.send_bytes(b"\x00shutdown")
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_garbage_task_bytes_are_reported_as_a_task_serialization_failure(self) -> None:
        parent, thread = self._run_worker()
        assert pickle.loads(parent.recv_bytes()) == ("ready",)
        parent.send_bytes(b"not a pickle")
        result = pickle.loads(parent.recv_bytes())
        assert result.ok is False and result.task_unpicklable is True
        parent.close()
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_failing_initializer_reports_and_exits(self) -> None:
        parent, thread = self._run_worker(tasks.failing_initializer)
        kind, tb = pickle.loads(parent.recv_bytes())
        assert kind == "init_error"
        assert "exploded on purpose" in tb
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_parent_closing_the_pipe_ends_the_loop(self) -> None:
        parent, thread = self._run_worker()
        assert pickle.loads(parent.recv_bytes()) == ("ready",)
        parent.close()
        thread.join(timeout=5)
        assert not thread.is_alive()

    def test_parent_vanishing_mid_task_ends_the_loop_quietly(self) -> None:
        parent, thread = self._run_worker()
        assert pickle.loads(parent.recv_bytes()) == ("ready",)
        parent.send_bytes(pickle.dumps(_TaskMessage(task_id=1, fn=tasks.sleep_then, args=("late", 0.2), kwargs={})))
        parent.close()
        thread.join(timeout=5)
        assert not thread.is_alive()


class TestWaitReadable:
    async def test_returns_false_on_timeout_and_true_on_data(self) -> None:
        a, b = mp.get_context("spawn").Pipe(duplex=True)
        assert await _wait_readable(a, timeout=0.05) is False
        b.send_bytes(b"x")
        assert await _wait_readable(a, timeout=1.0) is True
        assert a.recv_bytes() == b"x"
        a.close()
        b.close()

    async def test_returns_false_when_the_until_event_fires_first(self) -> None:
        a, b = mp.get_context("spawn").Pipe(duplex=True)
        stop = asyncio.Event()
        asyncio.get_running_loop().call_later(0.02, stop.set)
        assert await _wait_readable(a, timeout=5.0, until=stop) is False
        a.close()
        b.close()
