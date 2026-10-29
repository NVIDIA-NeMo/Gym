# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A checkpoint of many rollouts in flight across many workers, restored onto other worker counts.

The ``/run`` relay drives Simple Agent and the stateful counter resources server, against the fake inference backend.
Each rollout seeds a counter session, increments it by 1 and then by 2 in two tool calls, and verifies it is 6.
Every rollout's second model call is held, so the checkpoint finds each counter at 4,
held by whichever resources worker created it, while the agent holds each rollout's own ``/run`` episode.
After a kill of every Gym process, the servers restart with other worker counts, the checkpoint is restored,
and each rollout continues as its next attempt.
Every rollout must either have finished before the checkpoint or continue and score 1.0,
and once the sessions' close window has passed, no server may keep a claim or a session.

Opt in with ``NEMO_GYM_CHECKPOINT_MULTIWORKER_STRESS`` set to the number of rollouts in flight;
``NEMO_GYM_CHECKPOINT_STRESS_REPORT`` names a JSON file for the numbers.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest
from checkpoint_deployment import COUNTER_SCRIPT, Deployment, GymHTTP, counter_row
from test_checkpoint_e2e import gym_http_client, lift_connection_limit, reward_of, wait_until  # noqa: F401

from nemo_gym.core import EpisodeId
from nemo_gym.server_runtime._checkpoint import coordination


ROLLOUTS = os.environ.get("NEMO_GYM_CHECKPOINT_MULTIWORKER_STRESS")
# Short, so the claims of finished sessions end within the test.
CLOSE_WINDOW_SECONDS = 5.0
# Model calls of a rollout that is never interrupted: two tool calls, then the final answer.
CALLS_PER_ROLLOUT = 3

pytestmark = pytest.mark.skipif(
    not ROLLOUTS, reason="set NEMO_GYM_CHECKPOINT_MULTIWORKER_STRESS to the number of rollouts in flight"
)


async def calls_reached(deployment: Deployment, calls: int) -> bool:
    return (await deployment.backend_call_count())["calls"] == calls


async def nothing_left(deployment: Deployment) -> bool:
    """Whether the resources and agent servers keep no claim, session, or request in flight."""
    for name in ("resources", "agent"):
        if sum(deployment.claims(name).values()):
            return False
        status = await deployment.status(name)
        counts = status["report"]["counts"]
        if counts.get("sessions") or counts.get("inflight") or status.get("unclaimed"):
            return False
    return True


@pytest.mark.parametrize(
    ("workers_before", "workers_after"),
    [
        ({"resources": 8, "agent": 4}, {"resources": 4, "agent": 8}),
        ({"resources": 4, "agent": 8}, {"resources": 8, "agent": 4}),
    ],
    ids=["resources8-agent4-to-resources4-agent8", "resources4-agent8-to-resources8-agent4"],
)
async def test_a_checkpoint_of_rollouts_across_many_workers_continues_on_other_worker_counts(
    tmp_path: Path, workers_before: dict[str, int], workers_after: dict[str, int]
) -> None:
    count = int(ROLLOUTS)
    early = max(1, count // 10)
    deployment = Deployment("counter", tmp_path, extra_config={"global_aiohttp_connector_limit_per_host": 16384})
    for name in ("resources", "agent"):
        deployment.configure(
            name, num_workers=workers_before[name], session_close_retry_window_seconds=CLOSE_WINDOW_SECONDS
        )
    timings: dict[str, float] = {}
    report: dict[str, Any] = {
        "rollouts": count,
        "finished_before": early,
        "before": workers_before,
        "after": workers_after,
    }
    try:
        deployment.start_backend()
        deployment.start_gym()
        deployment.backend("/_ctl/script", COUNTER_SCRIPT)
        await lift_connection_limit()
        async with GymHTTP(deployment.url("environment"), timeout=1800) as http:
            # Rollouts that finish before the checkpoint.
            finished = await asyncio.gather(*(http.post("/run", json=counter_row(f"early-{i}")) for i in range(early)))
            # Every later model call waits; each rollout's first call is let through once all of them arrived,
            # so every rollout is held at its second call, with its counter at 1 and its first tool result delivered.
            deployment.backend("/_ctl/gate", {})
            rollout_ids = [f"stress-{i}" for i in range(count)]
            started = time.monotonic()
            runs = [asyncio.create_task(http.post("/run", json=counter_row(rollout_id))) for rollout_id in rollout_ids]
            base = early * CALLS_PER_ROLLOUT
            await wait_until(lambda: calls_reached(deployment, base + count), timeout=900)
            deployment.backend("/_ctl/step", {"calls": count})
            await wait_until(lambda: calls_reached(deployment, base + 2 * count), timeout=900)
            timings["dispatch_to_held"] = time.monotonic() - started
            report["claims_before"] = {name: dict(deployment.claims(name)) for name in ("resources", "agent")}

            participants = await deployment.participants()
            clock = time.monotonic()
            prepared = await coordination.prepare(participants, "stress", deadline_ts=time.time() + 600)
            timings["prepare"] = time.monotonic() - clock
            assert prepared.prepared, prepared.blockers()
            clock = time.monotonic()
            replies = await coordination.commit(
                participants,
                "stress",
                str(tmp_path / "ckpt"),
                [EpisodeId(rollout_id=rollout_id) for rollout_id in rollout_ids],
                deadline_ts=time.time() + 600,
            )
            timings["commit"] = time.monotonic() - clock
            # The agent holds each rollout's own /run episode; the counters are resources sessions.
            exported = replies["agent"]["episode_ids"]

            clock = time.monotonic()
            deployment.crash_gym()
            for run in runs:
                run.cancel()
            for name in ("resources", "agent"):
                deployment.configure(name, num_workers=workers_after[name])
            deployment.backend("/_ctl/release", {})
            deployment.start_gym()
            timings["restart"] = time.monotonic() - clock
            participants = await deployment.participants()
            clock = time.monotonic()
            await coordination.restore(
                participants,
                "stress-restore",
                str(tmp_path / "ckpt"),
                [EpisodeId.from_capture_key(key) for key in exported],
                deadline_ts=time.time() + 600,
            )
            timings["restore"] = time.monotonic() - clock
            await coordination.resume(participants, "stress-restore", deadline_ts=time.time() + 120)
            restored_claims = dict(deployment.claims("resources"))

            clock = time.monotonic()
            replacements = await asyncio.gather(
                *(http.post("/run", json=counter_row(key, attempt=1)) for key in exported)
            )
            timings["replacements"] = time.monotonic() - clock
            calls = (await deployment.backend_call_count())["calls"]

        clock = time.monotonic()
        try:
            await wait_until(lambda: nothing_left(deployment), timeout=CLOSE_WINDOW_SECONDS + 60)
        except TimeoutError:
            pass
        timings["until_nothing_left"] = time.monotonic() - clock
        left = {name: dict(deployment.claims(name)) for name in ("resources", "agent")}
        statuses = {name: (await deployment.status(name))["report"]["counts"] for name in ("resources", "agent")}
    finally:
        deployment.stop()

    early_rewards = [reward_of(reply.json()) for reply in finished]
    rewards = [reward_of(reply.json()) for reply in replacements]
    resources_records = sum(1 for _ in (tmp_path / "ckpt" / "gym" / "resources").rglob("records-*.jsonl"))
    report.update(
        {
            "exported": len(exported),
            "finished_before_scored_1": sum(reward == 1.0 for reward in early_rewards),
            "continued_scored_1": sum(reward == 1.0 for reward in rewards),
            "resources_workers_holding_sessions_before": len(report["claims_before"]["resources"]),
            "resources_workers_holding_sessions_after_restore": len(restored_claims),
            "restored_session_claims": sum(restored_claims.values()),
            "model_calls": calls,
            "claims_left": left,
            "counts_left": statuses,
            "timings_s": {name: round(value, 2) for name, value in timings.items()},
            "resources_record_files": resources_records,
        }
    )
    print(f"\n[checkpoint stress] {json.dumps(report)}")
    path = os.environ.get("NEMO_GYM_CHECKPOINT_STRESS_REPORT")
    if path:
        reports = json.loads(Path(path).read_text()) if Path(path).exists() else []
        Path(path).write_text(json.dumps(reports + [report], indent=1))

    # Every rollout finished before the checkpoint or was exported, and every one scores 1.0.
    assert early_rewards == [1.0] * early
    assert sorted(exported) == sorted(rollout_ids)
    assert rewards == [1.0] * count
    # The sessions were spread over several resources workers before, and over every worker after the restore.
    assert len(report["claims_before"]["resources"]) > 1
    assert len(restored_claims) == workers_after["resources"]
    assert sum(restored_claims.values()) == count
    # A continued rollout makes only the calls it still needs: the held second call again, and the final answer.
    assert calls == early * CALLS_PER_ROLLOUT + 2 * count + (CALLS_PER_ROLLOUT - 1) * count
    # Nothing is left: no claim, no session, no request in flight.
    assert left == {"resources": {}, "agent": {}}
    assert all(not counts.get("sessions") and not counts.get("inflight") for counts in statuses.values())
