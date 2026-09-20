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
"""Benchmark the ways a resources server can run CPU-bound verification off its event loop.

Strategies compared, all driven from one asyncio loop the way ``verify()`` would be:

``inline``          call the function on the event loop (what most verifiers do today)
``to_thread``       ``asyncio.to_thread`` (shares the GIL with the loop)
``fork_per_task``   fork a child per call and poll ``is_alive`` (what ``math_with_judge`` does today)
``pool``            :class:`nemo_gym.process_pool.WarmProcessPool`

For each cell the harness records completions per second, queue-wait and execution
percentiles, event-loop lag (a heartbeat that expects to wake every 10 ms), CPU and RSS of
the process tree, bytes moved, worker restarts and timeouts, and how many descendants are
still alive after shutdown.

Fault injection (``--inject``) makes a fraction of tasks hang, crash, return an unpicklable
value, or get cancelled, to show that the pool stays bounded while the others do not.

Example, a quick local matrix::

    python scripts/benchmark_process_pool.py --tasks 400 --task-ms 1 10 100 \\
        --payload-kb 1 100 --workers 1 4 --concurrency 32 256 --out results/pool_bench.jsonl

The full matrix from the tracking issue (task 1 ms to 1 s, payload 1 KB to 2 MB, workers 1 to
32, concurrency 1 to 8,192) is meant for a Linux box with cores to spare and takes a while.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import multiprocessing as mp
import os
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import psutil

from nemo_gym.process_pool import (
    ProcessPoolError,
    TaskTimeoutError,
    WarmProcessPool,
    WarmProcessPoolConfig,
)


# ---------------------------------------------------------------------------
# The work. Pure Python CPU spin so the GIL is held, plus a payload that must round-trip.
# ---------------------------------------------------------------------------


def spin(task_ms: float, payload: bytes, mode: str = "ok") -> bytes:
    if mode == "hang":
        while True:
            pass
    if mode == "crash":
        os._exit(7)
    deadline = time.perf_counter() + task_ms / 1000.0
    while time.perf_counter() < deadline:
        pass
    if mode == "unpicklable":
        return threading.Lock()  # type: ignore[return-value]
    return payload


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


class Strategy:
    name = "base"

    async def start(self) -> None: ...

    async def run(self, task_ms: float, payload: bytes, mode: str, timeout: float) -> tuple[Optional[bytes], str]:
        raise NotImplementedError

    async def aclose(self) -> None: ...

    def restarts(self) -> int:
        return 0


class Inline(Strategy):
    name = "inline"

    async def run(self, task_ms, payload, mode, timeout):
        if mode in ("hang", "crash"):
            # Inline has no way to bound these; running them would take the harness down.
            return None, "unbounded"
        try:
            return spin(task_ms, payload, mode), "ok"
        except Exception as exc:  # noqa: BLE001
            return None, type(exc).__name__


class ToThread(Strategy):
    name = "to_thread"

    async def run(self, task_ms, payload, mode, timeout):
        if mode in ("hang", "crash"):
            return None, "unbounded"
        try:
            return await asyncio.to_thread(spin, task_ms, payload, mode), "ok"
        except Exception as exc:  # noqa: BLE001
            return None, type(exc).__name__


def _fork_target(task_ms: float, payload: bytes, mode: str, conn: Any) -> None:
    try:
        conn.send(spin(task_ms, payload, mode))
    except BaseException as exc:  # noqa: BLE001
        with contextlib.suppress(Exception):
            conn.send(exc)
    finally:
        conn.close()


class ForkPerTask(Strategy):
    """The lifecycle ``math_with_judge`` implements today: fork, poll every 50 ms, terminate, kill, join."""

    name = "fork_per_task"

    def __init__(self, max_concurrency: int) -> None:
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._ctx = mp.get_context("fork")

    async def run(self, task_ms, payload, mode, timeout):
        async with self._semaphore:
            parent, child = self._ctx.Pipe(duplex=False)
            process = self._ctx.Process(target=_fork_target, args=(task_ms, payload, mode, child))
            process.start()
            child.close()
            loop = asyncio.get_running_loop()
            deadline = loop.time() + timeout
            value: Any = None
            received = False
            try:
                while process.is_alive():
                    # Drain as soon as data is ready. A child sending more than the pipe buffer
                    # (64 KB on Linux) blocks until the parent reads, so waiting for exit first
                    # would deadlock. The real math_with_judge only ever sent a small tuple.
                    if not received and parent.poll():
                        # poll() is also true at EOF: a child that died before sending anything.
                        try:
                            value = parent.recv()
                            received = True
                        except EOFError:
                            break
                    if loop.time() >= deadline:
                        process.terminate()
                        await asyncio.sleep(0.05)
                        if process.is_alive():
                            process.kill()
                        process.join(timeout=1.0)
                        return None, "timeout"
                    await asyncio.sleep(0.05)
                process.join(timeout=0)
                if not received and parent.poll():
                    with contextlib.suppress(EOFError):
                        value = parent.recv()
                        received = True
                if process.exitcode != 0 or not received:
                    return None, "crash"
                if isinstance(value, BaseException):
                    return None, type(value).__name__
                return value, "ok"
            finally:
                parent.close()


class Pool(Strategy):
    name = "pool"

    def __init__(self, workers: int, max_pending: int) -> None:
        self._pool = WarmProcessPool(
            WarmProcessPoolConfig(
                num_workers=workers,
                max_pending=max_pending,
                default_timeout_seconds=None,
                kill_grace_seconds=0.5,
                shutdown_grace_seconds=2.0,
            ),
            name="bench_pool",
        )

    async def start(self) -> None:
        await self._pool.start()

    async def run(self, task_ms, payload, mode, timeout):
        try:
            return await self._pool.run(spin, task_ms, payload, mode, timeout=timeout), "ok"
        except TaskTimeoutError:
            return None, "timeout"
        except ProcessPoolError as exc:
            return None, type(exc).__name__

    async def aclose(self) -> None:
        await self._pool.aclose()

    def restarts(self) -> int:
        return self._pool.stats().restarts


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------


@dataclass
class Cell:
    strategy: str
    task_ms: float
    payload_kb: int
    workers: int
    concurrency: int
    inject: str
    inject_fraction: float
    tasks: int
    wall_seconds: float = 0.0
    completions_per_s: float = 0.0
    ok: int = 0
    outcomes: dict[str, int] = field(default_factory=dict)
    latency_ms_p50: float = 0.0
    latency_ms_p95: float = 0.0
    latency_ms_p99: float = 0.0
    loop_lag_ms_p50: float = 0.0
    loop_lag_ms_p95: float = 0.0
    loop_lag_ms_max: float = 0.0
    cpu_percent_tree: float = 0.0
    rss_mb_tree_peak: float = 0.0
    payload_mb_moved: float = 0.0
    restarts: int = 0
    descendants_after_close: int = 0
    note: str = ""


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, int(round(q * (len(values) - 1))))
    return values[idx]


def _tree_rss_mb() -> float:
    me = psutil.Process()
    total = me.memory_info().rss
    for child in me.children(recursive=True):
        with contextlib.suppress(psutil.Error):
            total += child.memory_info().rss
    return total / 2**20


def _live_descendants() -> int:
    count = 0
    for child in psutil.Process().children(recursive=True):
        with contextlib.suppress(psutil.Error):
            cmd = " ".join(child.cmdline())
            if child.status() != psutil.STATUS_ZOMBIE and "multiprocessing." not in cmd:
                count += 1
    return count


async def _heartbeat(lags: list[float], stop: asyncio.Event, period: float = 0.01) -> None:
    loop = asyncio.get_running_loop()
    while not stop.is_set():
        expected = loop.time() + period
        await asyncio.sleep(period)
        lags.append(max(0.0, loop.time() - expected) * 1000.0)


async def run_cell(cell: Cell, strategy: Strategy, timeout: float, seed: int) -> Cell:
    payload = os.urandom(cell.payload_kb * 1024)
    rng = random.Random(seed)
    modes = ["ok"] * cell.tasks
    if cell.inject != "none":
        for i in rng.sample(range(cell.tasks), k=max(1, int(cell.tasks * cell.inject_fraction))):
            modes[i] = cell.inject

    await strategy.start()
    lags: list[float] = []
    stop = asyncio.Event()
    hb = asyncio.create_task(_heartbeat(lags, stop))
    me = psutil.Process()
    me.cpu_percent(None)
    for child in me.children(recursive=True):
        with contextlib.suppress(psutil.Error):
            child.cpu_percent(None)
    rss_peak = _tree_rss_mb()

    latencies: list[float] = []
    outcomes: dict[str, int] = {}
    semaphore = asyncio.Semaphore(cell.concurrency)

    async def one(mode: str) -> None:
        nonlocal rss_peak
        async with semaphore:
            t0 = time.perf_counter()
            if mode == "cancel":
                task = asyncio.ensure_future(strategy.run(cell.task_ms, payload, "ok", timeout))
                await asyncio.sleep(min(0.002, cell.task_ms / 4000.0))
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
                outcome = "cancelled"
            else:
                _, outcome = await strategy.run(cell.task_ms, payload, mode, timeout)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            outcomes[outcome] = outcomes.get(outcome, 0) + 1
            if len(latencies) % 50 == 0:
                rss_peak = max(rss_peak, _tree_rss_mb())

    start = time.perf_counter()
    await asyncio.gather(*(one(m) for m in modes))
    wall = time.perf_counter() - start
    stop.set()
    await hb

    cpu = me.cpu_percent(None)
    for child in me.children(recursive=True):
        with contextlib.suppress(psutil.Error):
            cpu += child.cpu_percent(None)
    cell.restarts = strategy.restarts()
    await strategy.aclose()
    await asyncio.sleep(0.2)

    cell.wall_seconds = round(wall, 3)
    cell.completions_per_s = round(cell.tasks / wall, 1) if wall else 0.0
    cell.ok = outcomes.get("ok", 0)
    cell.outcomes = outcomes
    cell.latency_ms_p50 = round(_pct(latencies, 0.50), 2)
    cell.latency_ms_p95 = round(_pct(latencies, 0.95), 2)
    cell.latency_ms_p99 = round(_pct(latencies, 0.99), 2)
    cell.loop_lag_ms_p50 = round(_pct(lags, 0.50), 2)
    cell.loop_lag_ms_p95 = round(_pct(lags, 0.95), 2)
    cell.loop_lag_ms_max = round(max(lags) if lags else 0.0, 2)
    cell.cpu_percent_tree = round(cpu, 1)
    cell.rss_mb_tree_peak = round(rss_peak, 1)
    cell.payload_mb_moved = round(2 * cell.payload_kb * cell.ok / 1024, 2)
    cell.descendants_after_close = _live_descendants()
    return cell


def make_strategy(name: str, workers: int, concurrency: int) -> Strategy:
    if name == "inline":
        return Inline()
    if name == "to_thread":
        return ToThread()
    if name == "fork_per_task":
        return ForkPerTask(max_concurrency=workers)
    if name == "pool":
        return Pool(workers=workers, max_pending=max(concurrency, 1))
    raise ValueError(name)


def print_table(cells: list[Cell]) -> None:
    header = (
        f"{'strategy':14s} {'ms':>6s} {'KB':>5s} {'wk':>3s} {'conc':>5s} {'inject':>11s} "
        f"{'ok/N':>9s} {'compl/s':>8s} {'p50ms':>8s} {'p99ms':>9s} {'lag p95':>8s} {'lag max':>8s} "
        f"{'cpu%':>6s} {'rssMB':>7s} {'restart':>7s} {'left':>4s}"
    )
    print(header)
    print("-" * len(header))
    for c in cells:
        inject = c.inject if c.inject == "none" else f"{c.inject}@{c.inject_fraction:.0%}"
        print(
            f"{c.strategy:14s} {c.task_ms:6.0f} {c.payload_kb:5d} {c.workers:3d} {c.concurrency:5d} {inject:>11s} "
            f"{c.ok:4d}/{c.tasks:<4d} {c.completions_per_s:8.1f} {c.latency_ms_p50:8.1f} {c.latency_ms_p99:9.1f} "
            f"{c.loop_lag_ms_p95:8.1f} {c.loop_lag_ms_max:8.1f} {c.cpu_percent_tree:6.0f} {c.rss_mb_tree_peak:7.0f} "
            f"{c.restarts:7d} {c.descendants_after_close:4d}"
        )


async def main_async(args: argparse.Namespace) -> None:
    out = Path(args.out) if args.out else None
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
    cells: list[Cell] = []
    for strategy_name in args.strategies:
        for task_ms in args.task_ms:
            for payload_kb in args.payload_kb:
                for workers in args.workers if strategy_name in ("pool", "fork_per_task") else [1]:
                    for concurrency in args.concurrency:
                        for inject in args.inject:
                            if strategy_name in ("inline", "to_thread") and inject in ("hang", "crash"):
                                continue
                            cell = Cell(
                                strategy=strategy_name,
                                task_ms=task_ms,
                                payload_kb=payload_kb,
                                workers=workers,
                                concurrency=concurrency,
                                inject=inject,
                                inject_fraction=args.inject_fraction,
                                tasks=args.tasks,
                            )
                            strategy = make_strategy(strategy_name, workers, concurrency)
                            cell = await run_cell(cell, strategy, timeout=args.timeout, seed=args.seed)
                            cells.append(cell)
                            print(
                                f"{cell.strategy:14s} ms={task_ms:<5.0f} kb={payload_kb:<5d} wk={workers:<3d} "
                                f"conc={concurrency:<5d} inject={inject:<11s} -> {cell.completions_per_s:8.1f}/s "
                                f"lag p95 {cell.loop_lag_ms_p95:6.1f}ms ok {cell.ok}/{cell.tasks} left {cell.descendants_after_close}",
                                file=sys.stderr,
                                flush=True,
                            )
                            if out:
                                with out.open("a") as f:
                                    f.write(json.dumps(asdict(cell)) + "\n")
    print()
    print_table(cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--strategies", nargs="+", default=["inline", "to_thread", "fork_per_task", "pool"])
    parser.add_argument("--task-ms", nargs="+", type=float, default=[1, 10, 100, 1000])
    parser.add_argument("--payload-kb", nargs="+", type=int, default=[1, 100, 2048])
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 4, 16, 32])
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 32, 256, 8192])
    parser.add_argument("--tasks", type=int, default=512, help="tasks per cell")
    parser.add_argument("--timeout", type=float, default=5.0, help="per-task deadline in seconds")
    parser.add_argument(
        "--inject", nargs="+", default=["none"], choices=["none", "hang", "crash", "unpicklable", "cancel"]
    )
    parser.add_argument("--inject-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None, help="append one JSON line per cell here")
    args = parser.parse_args()
    if "fork_per_task" in args.strategies and "fork" not in mp.get_all_start_methods():
        args.strategies = [s for s in args.strategies if s != "fork_per_task"]
        print("fork start method unavailable; skipping fork_per_task", file=sys.stderr)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
