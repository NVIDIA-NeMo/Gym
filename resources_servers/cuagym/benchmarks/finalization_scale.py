# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only synthetic benchmark for CUA-Gym finalization and admission.

This intentionally excludes browsers, vLLM, and the remote Hub. It isolates the
single-process JSON/finalization path and the queueing behavior that precedes
those external systems.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import resource
import time
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import AsyncIterator, Literal

import orjson

from resources_servers.webarena.app import TerminalOperationRegistry
from resources_servers.webarena.schemas import (
    WebArenaFinalizeRequest,
    WebArenaFinalizeResponse,
)


Mode = Literal["legacy", "finalize", "finalize_admitted"]
Shape = Literal["sync", "async"]


@dataclass
class _Counters:
    active: int = 0
    waiting: int = 0
    peak_active: int = 0
    peak_waiting: int = 0
    evaluations: int = 0

    def queued(self) -> None:
        self.waiting += 1
        self.peak_waiting = max(self.peak_waiting, self.waiting)

    def admitted(self) -> None:
        self.waiting -= 1
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)

    def completed(self) -> None:
        self.active -= 1


@dataclass
class _RequestResult:
    latency_s: float
    request_bytes: int
    response_bytes: int
    target: int
    completed_at_s: float


@dataclass
class BenchmarkResult:
    shape: Shape
    mode: Mode
    requested_concurrency: int
    admitted_concurrency: int
    episodes: int
    wall_time_s: float
    episodes_per_s: float
    latency_p50_s: float
    latency_p95_s: float
    latency_p99_s: float
    target_0_ready_s: float
    mean_request_bytes: float
    mean_response_bytes: float
    peak_active: int
    peak_waiting: int
    evaluation_calls: int
    duplicate_evaluations: int
    leases_left: int
    errors: int
    rss_peak_gib: float


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil((percentile / 100) * len(ordered)) - 1),
    )
    return ordered[index]


def _rss_peak_gib() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**2)


@asynccontextmanager
async def _admit(
    semaphore: asyncio.Semaphore | None,
    counters: _Counters,
) -> AsyncIterator[None]:
    counters.queued()
    if semaphore is None:
        counters.admitted()
        try:
            yield
        finally:
            counters.completed()
        return

    async with semaphore:
        counters.admitted()
        try:
            yield
        finally:
            counters.completed()


def _legacy_payload(env_id: str, filler: str) -> dict:
    return {
        "verify_request_id": f"verify-{env_id}",
        "responses_create_params": {"input": [{"role": "user", "content": "task"}]},
        "response": {
            "env_id": env_id,
            "output": [{"type": "message", "content": filler}],
            "metadata": {"full_rollout_history": filler},
        },
    }


async def _run_case(
    *,
    mode: Mode,
    shape: Shape,
    episodes: int,
    requested_concurrency: int,
    admission_cap: int,
    payload_bytes: int,
    evaluation_delay_s: float,
) -> BenchmarkResult:
    if shape == "sync":
        targets = [0] * episodes
    else:
        midpoint = episodes // 2
        targets = [0] * midpoint + [1] * (episodes - midpoint)

    admitted_concurrency = (
        min(requested_concurrency, admission_cap) if mode == "finalize_admitted" else requested_concurrency
    )
    semaphore = asyncio.Semaphore(admitted_concurrency)
    counters = _Counters()
    registry = TerminalOperationRegistry(ttl_seconds=60)
    leases = {f"env-{index}" for index in range(episodes)}
    filler = "x" * payload_bytes
    benchmark_started = time.perf_counter()

    async def evaluate_once(
        env_id: str,
        finalize_request_id: str,
    ) -> WebArenaFinalizeResponse:
        counters.evaluations += 1
        await asyncio.sleep(evaluation_delay_s)
        leases.discard(env_id)
        return WebArenaFinalizeResponse(
            env_id=env_id,
            finalize_request_id=finalize_request_id,
            reward=1.0,
            evaluator_details={"message": "synthetic"},
            termination_status="success",
            answer="done",
        )

    async def run_episode(index: int, target: int) -> _RequestResult:
        env_id = f"env-{index}"
        started = time.perf_counter()
        async with _admit(semaphore, counters):
            if mode == "legacy":
                request_payload = _legacy_payload(env_id, filler)
                encoded_request = orjson.dumps(request_payload)
                orjson.loads(encoded_request)
                counters.evaluations += 1
                await asyncio.sleep(evaluation_delay_s)
                leases.discard(env_id)
                response_payload = {
                    **request_payload,
                    "reward": 1.0,
                    "evaluator_details": {"message": "synthetic"},
                }
                encoded_response = orjson.dumps(response_payload)
                orjson.loads(encoded_response)
            else:
                finalize_request_id = f"finalize-{env_id}"
                request = WebArenaFinalizeRequest(
                    env_id=env_id,
                    finalize_request_id=finalize_request_id,
                )
                encoded_request = orjson.dumps(request.model_dump(mode="json"))
                orjson.loads(encoded_request)
                response = await registry.execute(
                    env_id,
                    finalize_request_id,
                    lambda: evaluate_once(env_id, finalize_request_id),
                )
                encoded_response = orjson.dumps(response.model_dump(mode="json"))
                orjson.loads(encoded_response)

        completed = time.perf_counter()
        return _RequestResult(
            latency_s=completed - started,
            request_bytes=len(encoded_request),
            response_bytes=len(encoded_response),
            target=target,
            completed_at_s=completed - benchmark_started,
        )

    errors = 0
    results: list[_RequestResult] = []
    for start in range(0, episodes, requested_concurrency):
        window = [
            asyncio.create_task(run_episode(index, targets[index]))
            for index in range(start, min(start + requested_concurrency, episodes))
        ]
        completed = await asyncio.gather(*window, return_exceptions=True)
        for item in completed:
            if isinstance(item, BaseException):
                errors += 1
            else:
                results.append(item)

    wall_time_s = time.perf_counter() - benchmark_started
    await registry.close()
    latencies = [result.latency_s for result in results]
    target_0_completions = [result.completed_at_s for result in results if result.target == 0]
    return BenchmarkResult(
        shape=shape,
        mode=mode,
        requested_concurrency=requested_concurrency,
        admitted_concurrency=admitted_concurrency,
        episodes=episodes,
        wall_time_s=wall_time_s,
        episodes_per_s=len(results) / wall_time_s if wall_time_s else 0.0,
        latency_p50_s=_percentile(latencies, 50),
        latency_p95_s=_percentile(latencies, 95),
        latency_p99_s=_percentile(latencies, 99),
        target_0_ready_s=max(target_0_completions, default=0.0),
        mean_request_bytes=(sum(result.request_bytes for result in results) / len(results) if results else 0.0),
        mean_response_bytes=(sum(result.response_bytes for result in results) / len(results) if results else 0.0),
        peak_active=counters.peak_active,
        peak_waiting=counters.peak_waiting,
        evaluation_calls=counters.evaluations,
        duplicate_evaluations=max(0, counters.evaluations - episodes),
        leases_left=len(leases),
        errors=errors,
        rss_peak_gib=_rss_peak_gib(),
    )


def _parser() -> argparse.ArgumentParser:
    cpu_command = """\
CPU allocation example:
  NEMORL=/path/to/nemo-rl
  CONTAINER=/path/to/nemo_rl.sqsh
  srun -A <account> -p cpu -t 01:00:00 \\
    -N1 --exclusive --container-image=\"$CONTAINER\" \\
    --container-mounts=\"${NEMORL}:/opt/nemo-rl\" \\
    --container-workdir=/opt/nemo-rl bash -lc \\
    'PYTHONPATH=/opt/nemo-rl/3rdparty/Gym-workspace/Gym \\
      /opt/nemo_rl_venv/bin/python \\
      3rdparty/Gym-workspace/Gym/resources_servers/cuagym/benchmarks/finalization_scale.py'
"""
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=cpu_command,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--episodes", type=int, default=512)
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",
        default=[64, 128, 192, 256, 384],
    )
    parser.add_argument("--admission-cap", type=int, default=256)
    parser.add_argument("--payload-bytes", type=int, default=64 * 1024)
    parser.add_argument("--evaluation-delay-s", type=float, default=0.01)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/cuagym-finalization-scale.jsonl"),
    )
    return parser


async def _main(args: argparse.Namespace) -> None:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.admission_cap <= 0:
        raise ValueError("--admission-cap must be positive")
    if any(value <= 0 for value in args.concurrency):
        raise ValueError("--concurrency values must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as output:
        for shape in ("sync", "async"):
            for concurrency in args.concurrency:
                for mode in ("legacy", "finalize", "finalize_admitted"):
                    result = await _run_case(
                        mode=mode,
                        shape=shape,
                        episodes=args.episodes,
                        requested_concurrency=concurrency,
                        admission_cap=args.admission_cap,
                        payload_bytes=args.payload_bytes,
                        evaluation_delay_s=args.evaluation_delay_s,
                    )
                    record = asdict(result)
                    output.write(orjson.dumps(record) + b"\n")
                    print(
                        f"{shape:5s} {mode:18s} concurrency={concurrency:3d} "
                        f"admitted={result.admitted_concurrency:3d} "
                        f"throughput={result.episodes_per_s:8.1f}/s "
                        f"p99={result.latency_p99_s:.3f}s "
                        f"target0={result.target_0_ready_s:.3f}s "
                        f"bytes={result.mean_request_bytes:.0f}/{result.mean_response_bytes:.0f} "
                        f"errors={result.errors} leases={result.leases_left}"
                    )


if __name__ == "__main__":
    asyncio.run(_main(_parser().parse_args()))
