# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load-test dataset reward programs against a remote CUA-Gym deployment."""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import math
import os
import random
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import orjson

from resources_servers.cuagym.browser_worker import _best_effort_reset_hub_state
from resources_servers.cuagym.episode_code import app_url, parse_reward, run_code, substitute
from resources_servers.cuagym.schemas import task_info_from_row
from resources_servers.webarena.schemas import WebArenaTaskRow


@dataclass(frozen=True)
class Episode:
    index: int
    task_id: str
    app_dir: str
    sid: str
    setup_code: str | None
    reward_code: str


@dataclass
class ProgramResult:
    phase: str
    concurrency: int
    episode_index: int
    task_id: str
    app_dir: str
    duration_s: float
    queue_wait_s: float
    returncode: int | None
    reward: float | None
    timed_out: bool
    error: str | None


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, math.ceil((percentile / 100) * len(ordered)) - 1),
    )
    return ordered[index]


def _load_episodes(
    dataset: Path,
    *,
    count: int,
    seed: int,
    hub_url: str,
    base_port: int,
) -> list[Episode]:
    rows = [orjson.loads(line) for line in dataset.read_bytes().splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No rows found in {dataset}")
    rng = random.Random(seed)
    sampled = list(rows)
    rng.shuffle(sampled)
    if count > len(sampled):
        sampled.extend(rng.choice(rows) for _ in range(count - len(sampled)))
    sampled = sampled[:count]

    episodes = []
    run_id = uuid.uuid4().hex[:12]
    for index, row in enumerate(sampled):
        task = WebArenaTaskRow.model_validate(row["verifier_metadata"]["task"])
        info = task_info_from_row(task)
        sid = f"reward-load-{run_id}-{index:04d}"
        episodes.append(
            Episode(
                index=index,
                task_id=task.task_id,
                app_dir=info.app_dir,
                sid=sid,
                setup_code=(
                    substitute(
                        info.initial_setup,
                        sid=sid,
                        hub_base_url=hub_url,
                        base_port=base_port,
                    )
                    if info.initial_setup
                    else None
                ),
                reward_code=substitute(
                    info.eval_reward_code,
                    sid=sid,
                    hub_base_url=hub_url,
                    base_port=base_port,
                ),
            )
        )
    return episodes


async def _run_programs(
    episodes: list[Episode],
    *,
    phase: str,
    concurrency: int,
    timeout_s: float,
    executor: concurrent.futures.ThreadPoolExecutor,
) -> list[ProgramResult]:
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()
    started = time.perf_counter()

    async def run_episode(episode: Episode) -> ProgramResult:
        queued_at = time.perf_counter()
        async with semaphore:
            admitted_at = time.perf_counter()
            code = episode.setup_code if phase == "setup" else episode.reward_code
            if not code:
                return ProgramResult(
                    phase=phase,
                    concurrency=concurrency,
                    episode_index=episode.index,
                    task_id=episode.task_id,
                    app_dir=episode.app_dir,
                    duration_s=0.0,
                    queue_wait_s=admitted_at - queued_at,
                    returncode=0,
                    reward=None,
                    timed_out=False,
                    error=None,
                )
            try:
                result = await loop.run_in_executor(
                    executor,
                    partial(
                        run_code,
                        code,
                        timeout_seconds=timeout_s,
                        extra_env=(
                            {
                                "CUA_GYM_AGENT_ANSWER": "",
                                "CUA_GYM_AGENT_STATUS": "success",
                            }
                            if phase == "reward"
                            else None
                        ),
                    ),
                )
                return ProgramResult(
                    phase=phase,
                    concurrency=concurrency,
                    episode_index=episode.index,
                    task_id=episode.task_id,
                    app_dir=episode.app_dir,
                    duration_s=time.perf_counter() - admitted_at,
                    queue_wait_s=admitted_at - queued_at,
                    returncode=result.returncode,
                    reward=parse_reward(result.stdout) if phase == "reward" else None,
                    timed_out=False,
                    error=(None if result.returncode == 0 else (result.stderr or result.stdout)[-500:]),
                )
            except subprocess.TimeoutExpired:
                return ProgramResult(
                    phase=phase,
                    concurrency=concurrency,
                    episode_index=episode.index,
                    task_id=episode.task_id,
                    app_dir=episode.app_dir,
                    duration_s=time.perf_counter() - admitted_at,
                    queue_wait_s=admitted_at - queued_at,
                    returncode=None,
                    reward=None,
                    timed_out=True,
                    error=f"timed out after {timeout_s}s",
                )
            except Exception as error:
                return ProgramResult(
                    phase=phase,
                    concurrency=concurrency,
                    episode_index=episode.index,
                    task_id=episode.task_id,
                    app_dir=episode.app_dir,
                    duration_s=time.perf_counter() - admitted_at,
                    queue_wait_s=admitted_at - queued_at,
                    returncode=None,
                    reward=None,
                    timed_out=False,
                    error=f"{type(error).__name__}: {error}",
                )

    results = await asyncio.gather(*(run_episode(episode) for episode in episodes))
    durations = [result.duration_s for result in results]
    failures = [result for result in results if result.timed_out or result.error is not None]
    wall_time_s = time.perf_counter() - started
    print(
        f"{phase:6s} concurrency={concurrency:3d} episodes={len(results):3d} "
        f"wall={wall_time_s:7.2f}s rate={len(results) / wall_time_s:7.2f}/s "
        f"p50={_percentile(durations, 50):6.2f}s "
        f"p95={_percentile(durations, 95):6.2f}s "
        f"p99={_percentile(durations, 99):6.2f}s "
        f"failures={len(failures)} timeouts={sum(r.timed_out for r in results)}"
    )
    return results


async def _cleanup(
    episodes: list[Episode],
    *,
    hub_url: str,
    base_port: int,
    concurrency: int,
    executor: concurrent.futures.ThreadPoolExecutor,
) -> None:
    semaphore = asyncio.Semaphore(concurrency)
    loop = asyncio.get_running_loop()

    async def reset(episode: Episode) -> None:
        async with semaphore:
            base_url = app_url(episode.app_dir, hub_url, base_port)
            await loop.run_in_executor(
                executor,
                _best_effort_reset_hub_state,
                base_url,
                episode.sid,
            )

    await asyncio.gather(*(reset(episode) for episode in episodes))


def _summary(results: list[ProgramResult]) -> dict[str, Any]:
    durations = [result.duration_s for result in results]
    failures = [result for result in results if result.timed_out or result.error is not None]
    return {
        "record_type": "summary",
        "phase": results[0].phase if results else None,
        "concurrency": results[0].concurrency if results else None,
        "episodes": len(results),
        "p50_s": _percentile(durations, 50),
        "p95_s": _percentile(durations, 95),
        "p99_s": _percentile(durations, 99),
        "max_s": max(durations, default=0.0),
        "failures": len(failures),
        "timeouts": sum(result.timed_out for result in results),
        "missing_reward": sum(result.phase == "reward" and result.reward is None for result in results),
        "apps": dict(Counter(result.app_dir for result in results)),
    }


async def _wait_for_barrier(
    barrier_dir: Path,
    *,
    rank: int,
    participants: int,
    timeout_s: float,
) -> None:
    barrier_dir.mkdir(parents=True, exist_ok=True)
    (barrier_dir / f"{rank}.ready").write_text("ready\n")
    deadline = time.monotonic() + timeout_s
    while len(list(barrier_dir.glob("*.ready"))) < participants:
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Timed out waiting for {participants} barrier participants in {barrier_dir}")
        await asyncio.sleep(0.1)
    print(f"Barrier ready: rank={rank} participants={participants}")


async def _main(args: argparse.Namespace) -> None:
    os.environ["CUA_GYM_ENDPOINTS_YAML"] = str(args.endpoints)
    episodes = _load_episodes(
        args.dataset,
        count=args.episodes,
        seed=args.seed,
        hub_url=args.hub_url,
        base_port=args.base_port,
    )
    print(f"Loaded {len(episodes)} episodes: {dict(Counter(e.app_dir for e in episodes))}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[ProgramResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.setup_concurrency, *args.concurrency)) as executor:
        try:
            setup_results = await _run_programs(
                episodes,
                phase="setup",
                concurrency=args.setup_concurrency,
                timeout_s=args.setup_timeout_s,
                executor=executor,
            )
            all_results.extend(setup_results)
            ready_indices = {
                result.episode_index for result in setup_results if not result.timed_out and result.error is None
            }
            ready = [episode for episode in episodes if episode.index in ready_indices]
            if len(ready) != len(episodes):
                print(f"WARNING: testing rewards for {len(ready)}/{len(episodes)} successfully initialized episodes")
            if args.barrier_dir is not None:
                await _wait_for_barrier(
                    args.barrier_dir,
                    rank=args.barrier_rank,
                    participants=args.barrier_participants,
                    timeout_s=args.barrier_timeout_s,
                )
            for concurrency in args.concurrency:
                all_results.extend(
                    await _run_programs(
                        ready,
                        phase="reward",
                        concurrency=concurrency,
                        timeout_s=args.reward_timeout_s,
                        executor=executor,
                    )
                )
        finally:
            await _cleanup(
                episodes,
                hub_url=args.hub_url,
                base_port=args.base_port,
                concurrency=args.setup_concurrency,
                executor=executor,
            )

    with args.output.open("wb") as output:
        for result in all_results:
            output.write(orjson.dumps({"record_type": "request", **asdict(result)}) + b"\n")
        grouped: dict[tuple[str, int], list[ProgramResult]] = {}
        for result in all_results:
            grouped.setdefault((result.phase, result.concurrency), []).append(result)
        for results in grouped.values():
            output.write(orjson.dumps(_summary(results)) + b"\n")
    print(f"Wrote {args.output}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="CUA-Gym task jsonl (e.g. a webarena *.cc.jsonl training pool)",
    )
    parser.add_argument(
        "--endpoints",
        type=Path,
        default=Path("/opt/nemo-rl/examples/nemo_gym/webarena_endpoints.yaml"),
    )
    parser.add_argument("--hub-url", required=True, help="CUA-Gym hub base URL, e.g. http://<hub-host>")
    parser.add_argument("--base-port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=384)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[64, 128, 256, 384])
    parser.add_argument("--setup-concurrency", type=int, default=64)
    parser.add_argument("--setup-timeout-s", type=float, default=90)
    parser.add_argument("--reward-timeout-s", type=float, default=120)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--barrier-dir", type=Path)
    parser.add_argument("--barrier-rank", type=int, default=0)
    parser.add_argument("--barrier-participants", type=int, default=1)
    parser.add_argument("--barrier-timeout-s", type=float, default=300)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/cuagym-reward-concurrency.jsonl"),
    )
    return parser


if __name__ == "__main__":
    asyncio.run(_main(_parser().parse_args()))
