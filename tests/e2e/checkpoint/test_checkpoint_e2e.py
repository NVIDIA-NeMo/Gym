# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial-rollout checkpointing across real Gym processes, driven by Gym's coordination functions.

Every scenario checkpoints a rollout mid-episode, usually kills every Gym process, restarts them, restores,
and runs the replacement attempt. The fake backend records every model call, so each test can tell
exactly which work was continued and which was redone.

These tests start real server processes and take up to a minute each. They are skipped unless
``NEMO_GYM_CHECKPOINT_E2E=1``::

    NEMO_GYM_CHECKPOINT_E2E=1 pytest tests/e2e/checkpoint -x

The real-model smoke test also needs an OpenAI-compatible vLLM server with tool calling and a reasoning
parser, for example::

    vllm serve Qwen/Qwen3-0.6B --port 8000 --enable-auto-tool-choice --tool-call-parser hermes \\
        --reasoning-parser qwen3
    NEMO_GYM_CHECKPOINT_E2E=1 NEMO_GYM_CHECKPOINT_VLLM_URL=http://127.0.0.1:8000/v1 \\
        pytest tests/e2e/checkpoint -k real_model
"""

import asyncio
import collections
import itertools
import json
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any, Optional

import pytest
from checkpoint_deployment import (
    CAPTURE_CONTROL_TOKEN,
    COUNTER_SCRIPT,
    TOKEN,
    Deployment,
    GymHTTP,
    Reply,
    counter_row,
    weather_episode,
    weather_row,
)

import nemo_gym.client.http_client
import nemo_gym.server_utils as server_utils
from nemo_gym.core import EpisodeId
from nemo_gym.server_runtime._checkpoint import coordination


# The policy model server runs in one process, or as a coordinator plus uvicorn workers.
POLICY_WORKERS = pytest.mark.parametrize("policy_workers", [1, 2])
# So do the environment, agent, and resources servers.
SERVER_WORKERS = pytest.mark.parametrize("server_workers", [1, 2], ids=["servers1", "servers2"])

pytestmark = pytest.mark.skipif(
    os.environ.get("NEMO_GYM_CHECKPOINT_E2E") != "1",
    reason="starts real server processes; set NEMO_GYM_CHECKPOINT_E2E=1 to run",
)


@pytest.fixture(autouse=True)
async def gym_http_client() -> AsyncIterator[None]:
    """Coordination calls go through Gym's global aiohttp client, which is bound to this test's loop."""
    nemo_gym.client.http_client.set_global_aiohttp_client(server_utils.GlobalAIOHTTPAsyncClientConfig())
    yield
    await nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT.close()
    nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT = None
    # Coordination uses the reserved control pool, created on first use in this test's loop.
    if nemo_gym.client.http_client._GLOBAL_AIOHTTP_CONTROL_CLIENT is not None:
        await nemo_gym.client.http_client._GLOBAL_AIOHTTP_CONTROL_CLIENT.close()
        nemo_gym.client.http_client._GLOBAL_AIOHTTP_CONTROL_CLIENT = None


async def lift_connection_limit() -> None:
    """Replace this test's global aiohttp client with one that opens as many connections per server as it needs.

    Each ``/run`` holds a connection until its episode replies,
    so a pool smaller than the rollouts in flight would leave the rest unsent.
    """
    await nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT.close()
    nemo_gym.client.http_client._GLOBAL_AIOHTTP_CLIENT = None
    nemo_gym.client.http_client.set_global_aiohttp_client(
        server_utils.GlobalAIOHTTPAsyncClientConfig(global_aiohttp_connector_limit_per_host=0)
    )


@pytest.fixture
def deploy(tmp_path: Path) -> Iterator[Callable[..., Deployment]]:
    deployments: list[Deployment] = []

    def start(topology: str, **options: Any) -> Deployment:
        deployment = Deployment(topology, tmp_path, **options)
        deployments.append(deployment)
        deployment.start_backend()
        deployment.start_gym()
        return deployment

    yield start
    for deployment in deployments:
        deployment.stop()


async def wait_until(predicate: Callable[[], bool | Awaitable[bool]], timeout: float = 30) -> None:
    """Poll ``predicate``, which may be a coroutine function, until it holds."""
    deadline = time.time() + timeout
    while True:
        holds = predicate()
        if not isinstance(holds, bool):
            holds = await holds
        if holds:
            return
        if time.time() > deadline:
            raise TimeoutError("condition not reached")
        await asyncio.sleep(0.1)


async def wait_for_calls(deployment: Deployment, calls: int, *, timeout: float = 30) -> None:
    """Wait until the backend has received ``calls`` model calls, polling its count without blocking the loop."""

    async def reached() -> bool:
        return (await deployment.backend_call_count())["calls"] == calls

    await wait_until(reached, timeout=timeout)


async def returned(deployment: Deployment, calls: int) -> bool:
    return (await deployment.backend_call_count())["returned"] >= calls


async def every_rollout_held_at_its_second_call(deployment: Deployment, rollouts: int) -> bool:
    """Whether every rollout's first call returned and its second arrived, with the per-rollout hold in place.

    The hold lets only each rollout's first call return, so ``rollouts`` returned calls are every first call,
    and twice as many received include every second one.
    """
    counts = await deployment.backend_call_count()
    return counts["returned"] == rollouts and counts["calls"] == 2 * rollouts


def deadline(seconds: float = 20) -> float:
    return time.time() + seconds


async def environment_preparing(deployment: Deployment) -> bool:
    return (await deployment.status("environment"))["phase"] == "preparing"


async def checkpoint(
    deployment: Deployment,
    checkpoint_dir: Path,
    rollout_ids: list[str],
    *,
    attempt: int = 0,
    checkpoint_id: str = "c1",
) -> None:
    """Prepare and commit, continuing ``rollout_ids``; the caller resumes or crashes afterwards."""
    participants = await deployment.participants()
    prepared = await coordination.prepare(participants, checkpoint_id, deadline_ts=deadline())
    assert prepared.prepared, prepared.blockers()
    episode_ids = [EpisodeId(rollout_id=rollout_id, attempt=attempt) for rollout_id in rollout_ids]
    await coordination.commit(participants, checkpoint_id, str(checkpoint_dir), episode_ids, deadline_ts=deadline())


async def crash_and_restore(
    deployment: Deployment,
    checkpoint_dir: Path,
    rollout_ids: list[str],
    *,
    skip_kinds: tuple[str, ...] = (),
    attempt: int = 0,
    restore_id: str = "r1",
    server_workers: Optional[int] = None,
) -> None:
    """Crash, restart, and restore; ``server_workers`` restarts the servers with a different worker count."""
    deployment.crash_gym()
    if server_workers is not None:
        deployment.set_server_workers(server_workers)
    if not deployment.external_inference:
        deployment.backend("/_ctl/release", {})
    deployment.slow_verify_flag.unlink(missing_ok=True)
    deployment.slow_verify_log.write_text("")
    deployment.start_gym()
    participants = await deployment.participants()
    restoring = coordination.Participants(
        client=participants.client,
        auth_token=participants.auth_token,
        members=tuple(member for member in participants.members if member.kind not in skip_kinds),
    )
    episode_ids = [EpisodeId(rollout_id=rollout_id, attempt=attempt) for rollout_id in rollout_ids]
    await coordination.restore(restoring, restore_id, str(checkpoint_dir), episode_ids, deadline_ts=deadline())
    await coordination.resume(participants, restore_id, deadline_ts=deadline())


async def capture_manifest(deployment: Deployment, rollout_id: str) -> Any:
    """The policy model server's capture manifest for ``rollout_id``."""
    return await GymHTTP(deployment.url("policy_model")).get(
        f"/training-token-capture/control/rollouts/{rollout_id}/manifest",
        headers={"authorization": f"Bearer {CAPTURE_CONTROL_TOKEN}"},
    )


def records_file(participant_dir: Path) -> Path:
    """A participant's records file, which is named after its digest."""
    [path] = participant_dir.glob("records-*.jsonl")
    return path


def reward_of(row: dict) -> Optional[float]:
    """A flat row carries its reward at the top level.

    An episode record from the typed ``single_agent_turn`` request path nests it in its result.
    """
    if "reward" in row:
        return row["reward"]
    return (row.get("result") or {}).get("reward")


async def checkpoint_replies(deployment: Deployment, checkpoint_dir: Path, rollout_ids: list[str]) -> dict[str, Any]:
    """Prepare and commit as ``checkpoint`` does, and return each participant's commit reply."""
    participants = await deployment.participants()
    prepared = await coordination.prepare(participants, "c1", deadline_ts=deadline())
    assert prepared.prepared, prepared.blockers()
    episode_ids = [EpisodeId(rollout_id=rollout_id) for rollout_id in rollout_ids]
    return await coordination.commit(participants, "c1", str(checkpoint_dir), episode_ids, deadline_ts=deadline())


def session_records(checkpoint_dir: Path, kind: str) -> list[str]:
    [path] = (checkpoint_dir / "gym" / kind).rglob("records-*.jsonl")
    return path.read_text().splitlines()


def environment_record(checkpoint_dir: Path) -> dict:
    [line] = records_file(checkpoint_dir / "gym/environment/environment").read_text().splitlines()
    return json.loads(line)


@POLICY_WORKERS
@SERVER_WORKERS
async def test_session_episode_continues_from_its_last_boundary_after_a_crash(
    deploy, tmp_path: Path, policy_workers: int, server_workers: int
) -> None:
    deployment = deploy("sessions", policy_workers=policy_workers, server_workers=server_workers)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=weather_episode("crash-1")))
        await wait_for_calls(deployment, 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["crash-1"])
        await crash_and_restore(deployment, tmp_path / "ckpt", ["crash-1"])
        first.cancel()

        replacement = await http.post("/run", json=weather_episode("crash-1", attempt=1))

    calls = deployment.backend_calls()
    # Call 0 was delivered and its tool ran before the checkpoint; call 1 was never delivered.
    # The replacement continues after the tool result, so it makes one call and never repeats call 0.
    assert [call["n_messages"] for call in calls] == [2, 4, 4]
    assert replacement.json()["result"]["reward"] == 1.0


@POLICY_WORKERS
@SERVER_WORKERS
async def test_a_second_crash_before_the_replacement_starts_still_continues(
    deploy, tmp_path: Path, policy_workers: int, server_workers: int
) -> None:
    deployment = deploy("sessions", policy_workers=policy_workers, server_workers=server_workers)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=weather_episode("twice-1")))
        await wait_for_calls(deployment, 2)
        await checkpoint(deployment, tmp_path / "ckpt1", ["twice-1"])
        await crash_and_restore(deployment, tmp_path / "ckpt1", ["twice-1"])
        first.cancel()
        # The next checkpoint lands before the framework re-dispatches attempt 1, and Gym crashes again.
        await checkpoint(deployment, tmp_path / "ckpt2", ["twice-1"], attempt=1, checkpoint_id="c2")
        await crash_and_restore(deployment, tmp_path / "ckpt2", ["twice-1"], attempt=1, restore_id="r2")

        replacement = await http.post("/run", json=weather_episode("twice-1", attempt=2))

    # Attempt 2 still continues after the tool result: it makes one call and never repeats call 0.
    assert [call["n_messages"] for call in deployment.backend_calls()] == [2, 4, 4]
    assert replacement.json()["result"]["reward"] == 1.0


@POLICY_WORKERS
async def test_a_checkpoint_restores_again_after_its_replacement_made_calls(
    deploy, tmp_path: Path, policy_workers: int
) -> None:
    deployment = deploy("counter", token_capture=True, policy_workers=policy_workers)
    deployment.backend("/_ctl/script", COUNTER_SCRIPT)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=counter_row("again-1")))
        await wait_for_calls(deployment, 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["again-1"])
        await crash_and_restore(deployment, tmp_path / "ckpt", ["again-1"])
        first.cancel()
        # Attempt 1 commits model calls to its capture ledger, then Gym crashes before the next checkpoint.
        await http.post("/run", json=counter_row("again-1", attempt=1))
        dead_calls = len(deployment.backend_calls())
        await crash_and_restore(deployment, tmp_path / "ckpt", ["again-1"], restore_id="r2")

        replacement = await http.post("/run", json=counter_row("again-1", attempt=1))

    manifest = (await capture_manifest(deployment, "again-1-a1")).json()
    admissions = [call["admission"] for call in deployment.backend_calls()]
    chain = [record["model_call_id"] for record in manifest["records"]]
    dead = {admission["model_call_id"] for admission in admissions[2:dead_calls]}

    # The second run of attempt 1 continues from the checkpoint's boundary, as the first did,
    # and none of the dead run's calls are part of its lineage.
    assert replacement.json()["reward"] == 1.0
    assert admissions[dead_calls]["parent_call_id"] == admissions[0]["model_call_id"]
    assert chain[0] == admissions[0]["model_call_id"]
    assert dead and not dead & set(chain)
    assert not manifest.get("failures")


@SERVER_WORKERS
async def test_legacy_run_continues_from_its_last_boundary_after_a_crash(
    deploy, tmp_path: Path, server_workers: int
) -> None:
    deployment = deploy("relay", server_workers=server_workers)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=weather_row("legacy-1")))
        await wait_for_calls(deployment, 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["legacy-1"])
        await crash_and_restore(deployment, tmp_path / "ckpt", ["legacy-1"])
        first.cancel()

        replacement = await http.post("/run", json=weather_row("legacy-1", attempt=1))

    assert [call["n_messages"] for call in deployment.backend_calls()] == [1, 3, 3]
    assert replacement.json()["reward"] == 1.0


@pytest.mark.parametrize("restore_resources", [True, False])
@SERVER_WORKERS
async def test_resources_state_is_restored_with_the_episode(
    deploy, tmp_path: Path, restore_resources: bool, server_workers: int
) -> None:
    deployment = deploy("counter", server_workers=server_workers)
    deployment.backend("/_ctl/script", COUNTER_SCRIPT)
    # The first increment runs before the checkpoint; the model call for the second is held.
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=counter_row("counter-1")))
        await wait_for_calls(deployment, 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["counter-1"])
        skip = () if restore_resources else ("resources",)
        await crash_and_restore(deployment, tmp_path / "ckpt", ["counter-1"], skip_kinds=skip)
        first.cancel()

        replacement = await http.post("/run", json=counter_row("counter-1", attempt=1))

    # Only the restored counter (3 + 1) reaches the expected 6 after the second increment.
    # The negative controls show the restore matters.
    if restore_resources:
        assert replacement.status_code == 200, replacement.text
        assert replacement.json()["reward"] == 1.0
    else:
        # The continued episode increments a fresh counter to 2 and verifies normally, with no reward.
        # With several workers, nothing claims the old cookie's session,
        # so the worker that serves its first request claims it, and the rest follow, as with one worker.
        assert replacement.status_code == 200, replacement.text
        assert replacement.json()["reward"] == 0.0


@POLICY_WORKERS
async def test_token_lineage_continues_across_a_crash(deploy, tmp_path: Path, policy_workers: int) -> None:
    deployment = deploy("counter", token_capture=True, policy_workers=policy_workers)
    deployment.backend("/_ctl/script", COUNTER_SCRIPT)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=counter_row("counter-1")))
        await wait_for_calls(deployment, 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["counter-1"])
        await crash_and_restore(deployment, tmp_path / "ckpt", ["counter-1"])
        first.cancel()
        replacement = await http.post("/run", json=counter_row("counter-1", attempt=1))

    manifest = (await capture_manifest(deployment, "counter-1-a1")).json()
    admissions = [call["admission"] for call in deployment.backend_calls()]
    chain = [record["model_call_id"] for record in manifest["records"]]

    # Call 1 was never delivered,
    # so its replacement (call 2) continues from call 0 and call 1 is not part of the continued lineage.
    assert replacement.json()["reward"] == 1.0
    assert admissions[2]["capture_key"] == "counter-1-a1"
    assert admissions[2]["parent_call_id"] == admissions[0]["model_call_id"]
    assert chain[0] == admissions[0]["model_call_id"]
    assert admissions[1]["model_call_id"] not in chain
    assert not manifest.get("failures")


@POLICY_WORKERS
async def test_an_in_flight_generation_is_cut_and_its_prefix_continued(
    deploy, tmp_path: Path, policy_workers: int
) -> None:
    deployment = deploy("sessions", token_capture=True, generation_cuts=True, policy_workers=policy_workers)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=weather_episode("cut-1")))
        await wait_for_calls(deployment, 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["cut-1"])
        cut_call = deployment.backend_calls()[1]["admission"]["model_call_id"]
        cut_rounds = deployment.backend("/_ctl/cuts")
        await crash_and_restore(deployment, tmp_path / "ckpt", ["cut-1"])
        first.cancel()
        replacement = await http.post("/run", json=weather_episode("cut-1", attempt=1))

    after = [call for call in deployment.backend_calls() if call["admission"]["capture_key"] == "cut-1-a1"]
    assert cut_rounds == [[cut_call]]
    assert [call.get("continued_from") for call in after] == [cut_call]
    assert replacement.json()["result"]["reward"] == 1.0


@POLICY_WORKERS
@SERVER_WORKERS
async def test_a_checkpoint_without_a_crash_parks_and_then_resumes_the_episode(
    deploy, tmp_path: Path, policy_workers: int, server_workers: int
) -> None:
    deployment = deploy("sessions", policy_workers=policy_workers, server_workers=server_workers)
    deployment.backend("/_ctl/hold", {"after_calls": 0})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        run = asyncio.create_task(http.post("/run", json=weather_episode("park-1")))
        await wait_for_calls(deployment, 1)
        participants = await deployment.participants()
        prepared = await coordination.prepare(participants, "c1", deadline_ts=deadline())
        # The generation finishes while Gym is prepared: the model server holds its response.
        deployment.backend("/_ctl/release", {})
        # The backend has answered: the policy model server now holds the response, and nothing else moves.
        await wait_until(lambda: returned(deployment, 1))
        await asyncio.sleep(1.0)
        parked = not run.done() and len(deployment.backend_calls()) == 1
        await coordination.resume(participants, "c1", deadline_ts=deadline())
        response = await run

    assert prepared.prepared
    assert parked
    assert response.json()["result"]["reward"] == 1.0
    assert len(deployment.backend_calls()) == 2


@pytest.mark.parametrize("topology", ["sessions", "counter"])
@pytest.mark.parametrize(
    ("workers_before", "workers_after"),
    [(1, 1), (2, 2), (1, 2), (2, 1), (2, 4)],
    ids=["servers1to1", "servers2to2", "servers1to2", "servers2to1", "servers2to4"],
)
async def test_concurrent_sessions_across_workers_all_continue_after_a_crash(
    deploy, tmp_path: Path, topology: str, workers_before: int, workers_after: int
) -> None:
    """Many rollouts in flight at once, their sessions spread over every worker, all continue after a crash.

    The servers may restart with a different worker count than they were checkpointed with.
    """
    count = 16
    deployment = deploy(topology, server_workers=workers_before)
    if topology == "counter":
        deployment.backend("/_ctl/script", COUNTER_SCRIPT)
        rows = {f"many-{index}": counter_row(f"many-{index}") for index in range(count)}
        # Sessions topology: the turn loop's agent sessions.
        # Counter: the resources sessions holding each counter.
        # The relay keeps no episode state; the agent holds each /run episode.
        session_kind, episode_owner, calls_per_rollout = "resources", "agent", 3
    else:
        rows = {f"many-{index}": weather_episode(f"many-{index}") for index in range(count)}
        session_kind, episode_owner, calls_per_rollout = "agent", "environment", 2
    # Every rollout's first model call returns; every later call is held.
    # The first calls are released only once all of them have arrived.
    # Otherwise a rollout answered early could send its second call before another rollout sent its first.
    deployment.backend("/_ctl/gate", {})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        runs = [asyncio.create_task(http.post("/run", json=row)) for row in rows.values()]
        await wait_for_calls(deployment, count, timeout=60)
        deployment.backend("/_ctl/step", {"calls": count})
        await wait_for_calls(deployment, 2 * count, timeout=60)
        replies = await checkpoint_replies(deployment, tmp_path / "ckpt", list(rows))
        # Each worker holds the sessions it created, as its claims say.
        holders = deployment.claims(session_kind)
        await crash_and_restore(deployment, tmp_path / "ckpt", list(rows), server_workers=workers_after)
        for run in runs:
            run.cancel()
        if topology == "counter":
            replacements = [counter_row(rollout_id, attempt=1) for rollout_id in rows]
        else:
            replacements = [weather_episode(rollout_id, attempt=1) for rollout_id in rows]
        results = await asyncio.gather(*(http.post("/run", json=row) for row in replacements))

    rewards = [reward_of(result.json()) for result in results]
    assert replies[episode_owner]["episode_ids"] == sorted(rows)
    # With one worker nothing is claimed; with several, the sessions come from every worker.
    assert len(holders) == (0 if workers_before == 1 else workers_before)
    assert len(session_records(tmp_path / "ckpt", session_kind)) == count
    assert rewards == [1.0] * count
    # Between them, the rollouts had one call delivered each before the crash.
    # Every continued rollout makes only the calls it still needed,
    # so none repeats a delivered call or restarts from its input.
    assert len(deployment.backend_calls()) == 2 * count + (calls_per_rollout - 1) * count


@pytest.mark.parametrize("verify", ["replay", "wait"])
async def test_verification_in_flight_is_replayed_or_waited_for(deploy, tmp_path: Path, verify: str) -> None:
    deployment = deploy("slow", checkpoint_verify=verify)
    deployment.slow_verify_flag.touch()
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=weather_episode("verify-1")))
        await wait_until(lambda: deployment.verifications() == 1)
        participants = await deployment.participants()
        early = await coordination.prepare(participants, "c1", deadline_ts=deadline(1))
        if verify == "wait":
            deployment.slow_verify_flag.unlink()
        await checkpoint(deployment, tmp_path / "ckpt", ["verify-1"])
        record = environment_record(tmp_path / "ckpt")
        await crash_and_restore(deployment, tmp_path / "ckpt", ["verify-1"])
        first.cancel()
        replacement = await http.post("/run", json=weather_episode("verify-1", attempt=1))

    assert replacement.json()["result"]["reward"] == 1.0
    assert len(deployment.backend_calls()) == 2
    if verify == "replay":
        # A replay step does not hold the checkpoint; the restored episode runs verification again.
        assert early.prepared
        assert record["boundary"]["next"] == "verify"
        assert deployment.verifications() == 1
    else:
        # A wait step holds the checkpoint until it finishes; its result is restored, never re-run.
        assert early.blockers() == {"environment": ["verify-1"]}
        assert record["boundary"]["next"] == "return"
        assert deployment.verifications() == 0


@pytest.mark.parametrize("crash", [False, True])
async def test_an_episode_that_finishes_during_prepare_is_neither_lost_nor_redone(
    deploy, tmp_path: Path, crash: bool
) -> None:
    deployment = deploy("slow")
    deployment.slow_verify_flag.touch()
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        run = asyncio.create_task(http.post("/run", json=weather_episode("done-1")))
        await wait_until(lambda: deployment.verifications() == 1)
        participants = await deployment.participants()
        preparing = asyncio.create_task(coordination.prepare(participants, "c1", deadline_ts=deadline()))
        # The environment server has closed admission before verification finishes.
        await wait_until(lambda: environment_preparing(deployment))
        deployment.slow_verify_flag.unlink()
        prepared = await preparing
        await asyncio.sleep(0.5)
        # No episode finishes while Gym is prepared: the result waits at the episode's last boundary.
        delivered_during_checkpoint = run.done()
        await coordination.commit(
            participants, "c1", str(tmp_path / "ckpt"), [EpisodeId(rollout_id="done-1")], deadline_ts=deadline()
        )
        record = environment_record(tmp_path / "ckpt")
        if crash:
            await crash_and_restore(deployment, tmp_path / "ckpt", ["done-1"])
            run.cancel()
            response = await http.post("/run", json=weather_episode("done-1", attempt=1))
        else:
            await coordination.resume(participants, "c1", deadline_ts=deadline())
            response = await run

    assert prepared.prepared
    assert not delivered_during_checkpoint
    assert record["boundary"]["next"] == "return"
    assert response.json()["result"]["reward"] == 1.0
    assert len(deployment.backend_calls()) == 2
    assert deployment.verifications() == (0 if crash else 1)


@pytest.mark.skipif(
    not os.environ.get("NEMO_GYM_CHECKPOINT_VLLM_URL"), reason="set NEMO_GYM_CHECKPOINT_VLLM_URL to a vLLM server"
)
@POLICY_WORKERS
@SERVER_WORKERS
async def test_real_model_rollouts_are_either_finished_or_continued_after_a_crash(
    deploy, tmp_path: Path, policy_workers: int, server_workers: int
) -> None:
    deployment = deploy(
        "sessions",
        inference_url=os.environ["NEMO_GYM_CHECKPOINT_VLLM_URL"],
        model_name=os.environ.get("NEMO_GYM_CHECKPOINT_VLLM_MODEL", "Qwen/Qwen3-0.6B"),
        policy_workers=policy_workers,
        server_workers=server_workers,
    )
    rollout_ids = [f"real-{index}" for index in range(64)]
    async with GymHTTP(deployment.url("environment"), timeout=300) as http:
        # Identical rollouts sent together also finish together on a fast server,
        # so the checkpoint would find none in flight.
        # Spaced out, and no longer sent once the first finishes, most are mid-episode.
        runs = {}
        for rollout_id in rollout_ids:
            if any(run.done() for run in runs.values()):
                break
            runs[rollout_id] = asyncio.create_task(http.post("/run", json=weather_episode(rollout_id)))
            await asyncio.sleep(0.02)
        rollout_ids = list(runs)
        # Checkpoint while the model is mid-generation for most rollouts.
        await wait_until(lambda: any(run.done() for run in runs.values()), timeout=120)
        participants = await deployment.participants()
        prepared = await coordination.prepare(participants, "c1", deadline_ts=deadline(120))
        # Which rollouts finished is the environment server's to say, not the client's:
        # a reply the server already sent can still be on its way when the client looks.
        # Its commit exports every episode it still holds, so the ones it leaves out had finished before prepare.
        replies = await coordination.commit(
            participants,
            "c1",
            str(tmp_path / "ckpt"),
            [EpisodeId(rollout_id=rollout_id) for rollout_id in rollout_ids],
            deadline_ts=deadline(),
        )
        continued = replies["environment"]["episode_ids"]
        finished = set(rollout_ids) - set(continued)
        # An episode the checkpoint holds cannot reply until resume, which never comes before the crash.
        replied_while_prepared = [rollout_id for rollout_id in continued if runs[rollout_id].done()]
        first_results = [(await runs[rollout_id]).json() for rollout_id in sorted(finished)]
        records = [
            json.loads(line)
            for line in records_file(tmp_path / "ckpt/gym/environment/environment").read_text().splitlines()
        ]
        next_step = {record["episode_id"]["rollout_id"]: record["boundary"]["next"] for record in records}
        next_steps = sorted(next_step.values())
        agent_sessions = {
            json.loads(line)["episode_id"]["rollout_id"]
            for line in records_file(tmp_path / "ckpt/gym/agent/agent").read_text().splitlines()
        }
        await crash_and_restore(deployment, tmp_path / "ckpt", continued)
        for rollout_id in continued:
            runs[rollout_id].cancel()
        replacements = await asyncio.gather(
            *(http.post("/run", json=weather_episode(rollout_id, attempt=1)) for rollout_id in continued)
        )

    continued_results = [response.json() for response in replacements]
    print(f"finished before the checkpoint: {len(finished)}; continued: {len(continued)}; next steps: {next_steps}")
    print("rewards before:", [result["result"]["reward"] for result in first_results])
    print("rewards continued:", [result.get("result", {}).get("reward") for result in continued_results])

    assert prepared.prepared
    assert continued, "the checkpoint caught no rollout mid-episode"
    assert not replied_while_prepared
    # An episode checkpointed inside its agent step continues the agent's own session, not the step.
    # The episode holds that session until its close_agent step finishes,
    # so an episode parked before close_agent is exported with its session too.
    holding = {rollout_id for rollout_id, step in next_step.items() if step in ("invoke_agent", "close_agent")}
    assert holding <= agent_sessions
    # An agent seed still in flight at prepare is exported as well, with its episode parked before the seed step.
    assert agent_sessions <= holding | {rollout_id for rollout_id, step in next_step.items() if step == "seed"}
    assert all(response.status_code == 200 and response.json().get("failure") is None for response in replacements)


@pytest.mark.skipif(
    not os.environ.get("NEMO_GYM_CHECKPOINT_VLLM_URL"), reason="set NEMO_GYM_CHECKPOINT_VLLM_URL to a vLLM server"
)
@POLICY_WORKERS
@SERVER_WORKERS
async def test_real_model_restarts_keep_running_through_a_checkpoint_and_start_over_after_a_crash(
    deploy, tmp_path: Path, policy_workers: int, server_workers: int
) -> None:
    deployment = deploy(
        "mixed",
        inference_url=os.environ["NEMO_GYM_CHECKPOINT_VLLM_URL"],
        model_name=os.environ.get("NEMO_GYM_CHECKPOINT_VLLM_MODEL", "Qwen/Qwen3-0.6B"),
        policy_workers=policy_workers,
        server_workers=server_workers,
    )
    environment = GymHTTP(deployment.url("environment"), timeout=300)
    restarting = GymHTTP(deployment.url("restart_environment"), timeout=300)
    async with environment, restarting:
        # Alternate between the saved environment and the restart-only one, spaced out as in the test above.
        runs: dict[str, asyncio.Task] = {}
        for index in range(64):
            if any(run.done() for run in runs.values()):
                break
            client, rollout_id = (
                (environment, f"saved-{index}") if index % 2 == 0 else (restarting, f"restart-{index}")
            )
            runs[rollout_id] = asyncio.create_task(client.post("/run", json=weather_episode(rollout_id)))
            await asyncio.sleep(0.02)
        participants = await deployment.participants()
        prepared = await coordination.prepare(participants, "c1", deadline_ts=deadline(120))
        in_flight = [EpisodeId(rollout_id=rollout_id) for rollout_id, run in runs.items() if not run.done()]
        continued = prepared.continued(in_flight)
        restarts = prepared.restarts()
        await coordination.commit(participants, "c1", str(tmp_path / "ckpt"), continued, deadline_ts=deadline())
        saved = [episode.rollout_id for episode in continued]
        await crash_and_restore(deployment, tmp_path / "ckpt", saved)
        for run in runs.values():
            run.cancel()
        # The restarts' first attempts are refused everywhere before they start over from their input.
        participants = await deployment.participants()
        await coordination.retire(
            participants,
            "retire-restarts",
            sorted(restarts, key=lambda episode: episode.capture_key),
            deadline_ts=deadline(),
        )
        replacements = await asyncio.gather(
            *(environment.post("/run", json=weather_episode(rollout_id, attempt=1)) for rollout_id in saved),
            *(
                restarting.post("/run", json=weather_episode(episode.rollout_id, attempt=1))
                for episode in sorted(restarts, key=lambda episode: episode.capture_key)
            ),
        )

    print(f"continued: {len(saved)}; restarted: {len(restarts)}")
    print("rewards:", [response.json().get("result", {}).get("reward") for response in replacements])
    assert prepared.prepared, prepared.blockers()
    assert saved and restarts, "the checkpoint caught no saved or no restart rollout mid-episode"
    # Every restart-only rollout still in flight is a restart, and nothing else is.
    # A restart keeps running through the checkpoint,
    # so one prepare reported may have finished by the time the test lists what is in flight.
    reported = {episode.rollout_id for episode in restarts}
    running = {episode.rollout_id for episode in in_flight if episode.rollout_id.startswith("restart-")}
    assert running <= reported <= {rollout_id for rollout_id in runs if rollout_id.startswith("restart-")}
    assert all(rollout_id.startswith("saved-") for rollout_id in saved)
    assert all(response.status_code == 200 and response.json().get("failure") is None for response in replacements)


@pytest.mark.skipif(
    not os.environ.get("NEMO_GYM_CHECKPOINT_SCALE"), reason="set NEMO_GYM_CHECKPOINT_SCALE to a rollout count"
)
@POLICY_WORKERS
@SERVER_WORKERS
async def test_checkpoint_at_scale(deploy, tmp_path: Path, policy_workers: int, server_workers: int) -> None:
    """Checkpoint thousands of in-flight rollouts, crash, restore, and finish every one of them."""
    count = int(os.environ["NEMO_GYM_CHECKPOINT_SCALE"])
    # At most this many rollouts open connections at once,
    # as a training framework bounds the episodes it has in flight.
    # A busy server worker on uvloop accepts about one new connection per event-loop iteration.
    # A larger burst waits in the accept queue, and a session close can outlast the environment's cleanup timeout,
    # so the test would measure connection setup rather than the checkpoint.
    in_flight = int(os.environ.get("NEMO_GYM_CHECKPOINT_SCALE_IN_FLIGHT", "500"))
    # The training framework raises Gym's per-host connection limit the same way.
    deployment = deploy(
        "sessions",
        policy_workers=policy_workers,
        server_workers=server_workers,
        token_capture=True,
        generation_cuts=True,
        extra_config={"global_aiohttp_connector_limit_per_host": 16384},
    )
    # Each rollout's first model call returns and its second is held,
    # so the checkpoint finds every rollout at the same point: the tool result delivered,
    # the next generation in flight.
    deployment.backend("/_ctl/hold", {"after_calls_per_rollout": 1})
    rollout_ids = [f"scale-{index}" for index in range(count)]
    timings: dict[str, float] = {}
    await lift_connection_limit()
    async with GymHTTP(deployment.url("environment"), timeout=900) as http:
        started = time.monotonic()
        runs: list[asyncio.Task] = []
        # Dispatch in waves, and start the next wave once every rollout so far is held at its second call.
        # A held rollout keeps its connections open until the checkpoint, so it must not count against the bound:
        # a semaphore held until each run returns would stop dispatch at the bound and never reach the checkpoint.
        for wave in itertools.batched(rollout_ids, in_flight):
            runs.extend(
                asyncio.create_task(http.post("/run", json=weather_episode(rollout_id))) for rollout_id in wave
            )
            await wait_until(lambda: every_rollout_held_at_its_second_call(deployment, len(runs)), timeout=600)
        timings["dispatch_to_held"] = time.monotonic() - started

        participants = await deployment.participants()
        clock = time.monotonic()
        # Prepare stage by stage, in coordination's order, to see where the time goes.
        # Each stage must prepare on its own; the full prepare afterwards finds every stage already prepared.
        stages = {}
        for kind in coordination.PREPARE_ORDER:
            stage = coordination.Participants(
                client=participants.client,
                auth_token=participants.auth_token,
                members=tuple(member for member in participants.members if member.kind == kind),
            )
            stage_clock = time.monotonic()
            stages[kind] = await coordination.prepare(stage, "scale", deadline_ts=deadline(300))
            timings[f"prepare_{kind}"] = time.monotonic() - stage_clock
        prepared = await coordination.prepare(participants, "scale", deadline_ts=deadline(300))
        timings["prepare"] = time.monotonic() - clock
        for kind, staged in stages.items():
            assert staged.prepared, (kind, staged.blockers())
        assert prepared.prepared, prepared.blockers()
        clock = time.monotonic()
        replies = await coordination.commit(
            participants,
            "scale",
            str(tmp_path / "ckpt"),
            [EpisodeId(rollout_id=rollout_id) for rollout_id in rollout_ids],
            deadline_ts=deadline(300),
        )
        timings["commit"] = time.monotonic() - clock
        exported = replies["environment"]["episode_ids"]
        # The environment server says which rollouts it still held; one it left out had finished.
        finished = set(rollout_ids) - set(exported)
        retained = replies["policy_model"]["staging_keys"]

        clock = time.monotonic()
        deployment.crash_gym()
        deployment.backend("/_ctl/release", {})
        deployment.start_gym()
        timings["restart"] = time.monotonic() - clock
        for run in runs:
            run.cancel()
        participants = await deployment.participants()
        clock = time.monotonic()
        await coordination.restore(
            participants,
            "scale-restore",
            str(tmp_path / "ckpt"),
            [EpisodeId.from_capture_key(key) for key in exported],
            deadline_ts=deadline(300),
        )
        timings["restore"] = time.monotonic() - clock
        clock = time.monotonic()
        await coordination.resume(participants, "scale-restore", deadline_ts=deadline(120))
        timings["resume"] = time.monotonic() - clock

        # A replacement runs to its end, so a semaphore bounds the replacements in flight.
        replacing = asyncio.Semaphore(in_flight)

        async def replace(key: str) -> Reply:
            async with replacing:
                return await http.post("/run", json=weather_episode(key, attempt=1))

        clock = time.monotonic()
        replacements = await asyncio.gather(*(replace(key) for key in exported))
        timings["replacements"] = time.monotonic() - clock
        cuts = deployment.backend("/_ctl/cuts")
        continued = sum(1 for call in deployment.backend_calls() if call.get("continued_from"))

    sizes = {
        kind: sum(f.stat().st_size for f in (tmp_path / "ckpt" / "gym" / kind).rglob("records-*.jsonl"))
        for kind in ("environment", "agent", "model", "resources")
    }
    results = [response.json() for response in replacements]
    rewards = [(result.get("result") or {}).get("reward") for result in results]
    failures = collections.Counter(
        str(result.get("failure") or result)[:160] for result, reward in zip(results, rewards) if reward != 1.0
    )
    print(
        f"\nscale={count} in_flight={in_flight} policy_workers={policy_workers} server_workers={server_workers} finished_before={len(finished)} exported={len(exported)}"
        f"\ntimings_s={ {name: round(value, 2) for name, value in timings.items()} }"
        f"\nrecords_bytes={sizes} report={prepared.replies['policy_model']['report']['counts']}"
        f"\ncut_rounds={len(cuts)} prefixes_cut={sum(len(round_) for round_ in cuts)} calls_continued={continued}"
        f"\nstaging_keys_retained={len(retained)}"
        f"\nreplacement_failures={sum(failures.values())} {failures.most_common(5)}"
    )

    # Every rollout was held at its second call, so none finished and the checkpoint exported every one.
    assert sorted(exported) == sorted(rollout_ids)
    assert rewards == [1.0] * len(exported)


@SERVER_WORKERS
async def test_a_retired_episode_stops_everywhere_and_leaves_nothing_behind(
    deploy, tmp_path: Path, server_workers: int
) -> None:
    deployment = deploy("sessions", token_capture=True, server_workers=server_workers)
    # The episode's first model call is answered; the second is held, so the episode is mid-flight.
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        run = asyncio.create_task(http.post("/run", json=weather_episode("gone-1")))
        await wait_for_calls(deployment, 2)
        participants = await deployment.participants()
        # A retire whose deadline has passed only refuses the attempt: its resources session is marked,
        # and counted as retired until a retire frees it, on whichever worker holds it.
        late_retire = await GymHTTP(deployment.url("resources"), timeout=10).post(
            "/ng-control/v1/checkpoint/retire",
            json={"checkpoint_id": "mark", "deadline_ts": time.time() - 1, "episode_ids": [{"rollout_id": "gone-1"}]},
            headers={"authorization": f"Bearer {TOKEN}"},
        )
        marked = await deployment.status("resources")

        await coordination.retire(participants, "retire", [EpisodeId(rollout_id="gone-1")], deadline_ts=deadline())

        calls_at_retire = len(deployment.backend_calls())
        await wait_until(run.done, timeout=10)
        # Nothing of the attempt is left to answer the held call, or to make another one.
        deployment.backend("/_ctl/release", {})
        await asyncio.sleep(1)
        servers = ("environment", "agent", "resources", "policy_model")

        async def statuses() -> dict[str, dict]:
            return {name: await deployment.status(name) for name in servers}

        after_retire = await statuses()
        # A duplicate of the retired attempt's /run, such as a client retry, is refused until the rollout is forgotten.
        late = await http.post("/run", json=weather_episode("gone-1"))
        await coordination.forget(participants, "forget", ["gone-1"], deadline_ts=deadline())
        after_forget = await statuses()
        manifest = await capture_manifest(deployment, "gone-1")

    assert len(deployment.backend_calls()) == calls_at_retire
    assert after_retire["environment"]["report"]["counts"]["episodes"] == 0
    assert after_retire["agent"]["report"]["counts"]["sessions"] == 0
    assert after_retire["resources"]["report"]["counts"]["sessions"] == 0
    assert after_retire["policy_model"]["report"]["counts"]["inflight"] == 0
    # Only the rollout's refusal remains, one entry per server, until the controller forgets it.
    assert {name: status["retired_rollouts"] for name, status in after_retire.items()} == dict.fromkeys(servers, 1)
    assert "checkpoint_stale_attempt" in late.text
    assert {name: status["retired_rollouts"] for name, status in after_forget.items()} == dict.fromkeys(servers, 0)
    assert late_retire.json()["error"]["code"] == "checkpoint_deadline_exceeded"
    # The marked session refuses the episode's cleanup, and stays counted as retired until the forget.
    assert marked["retired_sessions"] == 1 and after_retire["resources"]["retired_sessions"] == 1
    assert after_forget["resources"]["retired_sessions"] == 0
    # The checkpoint retire also retired the attempt's capture ledger.
    assert manifest.status_code == 410 and "is retired" in manifest.json()["detail"]


async def _mixed_checkpoint(deployment: Deployment, checkpoint_dir: Path) -> coordination.PrepareResult:
    """With one continued and one restart episode in flight, checkpoint everything that can be continued."""
    participants = await deployment.participants()
    prepared = await coordination.prepare(participants, "c1", deadline_ts=deadline())
    # The restart episode's resources server cannot capture it, yet it does not hold the checkpoint up.
    assert prepared.prepared, prepared.blockers()
    in_flight = [EpisodeId(rollout_id="continued"), EpisodeId(rollout_id="restart")]
    await coordination.commit(
        participants, "c1", str(checkpoint_dir), prepared.continued(in_flight), deadline_ts=deadline()
    )
    return prepared


@SERVER_WORKERS
async def test_a_restart_episode_never_holds_up_a_checkpoint_and_starts_over_after_a_crash(
    deploy, tmp_path: Path, server_workers: int
) -> None:
    deployment = deploy("mixed", server_workers=server_workers)
    # Every model call waits, so both episodes are in flight at the checkpoint.
    deployment.backend("/_ctl/hold", {"after_calls": 0})
    environment = GymHTTP(deployment.url("environment"), timeout=120)
    restarting = GymHTTP(deployment.url("restart_environment"), timeout=120)
    async with environment, restarting:
        first = asyncio.create_task(environment.post("/run", json=weather_episode("continued")))
        second = asyncio.create_task(restarting.post("/run", json=weather_episode("restart")))
        await wait_for_calls(deployment, 2)
        prepared = await _mixed_checkpoint(deployment, tmp_path / "ckpt")
        await crash_and_restore(deployment, tmp_path / "ckpt", ["continued"])
        first.cancel()
        second.cancel()
        # The controller refuses the restart's crashed attempt everywhere, then starts it over as its next attempt.
        participants = await deployment.participants()
        await coordination.retire(
            participants, "retire-restarts", [EpisodeId(rollout_id="restart")], deadline_ts=deadline()
        )
        continued = await environment.post("/run", json=weather_episode("continued", attempt=1))
        restarted = await restarting.post("/run", json=weather_episode("restart", attempt=1))

    assert prepared.restarts() == {EpisodeId(rollout_id="restart")}
    # Only the continued episode is in the checkpoint.
    assert environment_record(tmp_path / "ckpt")["episode_id"]["rollout_id"] == "continued"
    assert records_file(tmp_path / "ckpt/gym/environment/restart_environment").read_text() == ""
    assert continued.json()["result"]["reward"] == 1.0
    assert restarted.json()["result"]["reward"] == 1.0


@SERVER_WORKERS
async def test_a_restart_episode_keeps_running_through_a_checkpoint_without_a_crash(
    deploy, tmp_path: Path, server_workers: int
) -> None:
    deployment = deploy("mixed", server_workers=server_workers)
    deployment.backend("/_ctl/hold", {"after_calls": 0})
    environment = GymHTTP(deployment.url("environment"), timeout=120)
    restarting = GymHTTP(deployment.url("restart_environment"), timeout=120)
    async with environment, restarting:
        first = asyncio.create_task(environment.post("/run", json=weather_episode("continued")))
        second = asyncio.create_task(restarting.post("/run", json=weather_episode("restart")))
        await wait_for_calls(deployment, 2)
        await _mixed_checkpoint(deployment, tmp_path / "ckpt")
        await coordination.resume(await deployment.participants(), "c1", deadline_ts=deadline())
        deployment.backend("/_ctl/release", {})
        continued, restarted = await asyncio.gather(first, second)

    # Nothing crashed, so neither episode lost any work: each finishes as its first attempt.
    assert continued.json()["result"]["reward"] == 1.0
    assert restarted.json()["result"]["reward"] == 1.0
    assert restarted.json()["episode_id"]["attempt"] == 0


MCP_TOKEN_HEADER = "X-NeMo-Gym-Session-Token"


async def mcp_reply(http: GymHTTP, token: str, name: str, arguments: dict) -> Reply:
    """Call a tool as a CLI agent harness does: the MCP session token only, never the session cookie."""
    return await http.post(
        "/mcp",
        headers={"accept": "application/json, text/event-stream", MCP_TOKEN_HEADER: token},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}},
    )


async def mcp_call(http: GymHTTP, token: str, name: str, arguments: dict) -> dict:
    response = await mcp_reply(http, token, name, arguments)
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result.get("isError") is not True, result
    return json.loads(result["content"][0]["text"])


async def test_mcp_tool_calls_wait_out_a_checkpoint_and_a_restore_hands_their_sessions_to_new_tokens(
    deploy, tmp_path: Path
) -> None:
    deployment = deploy("counter", server_workers=2, resources_mcp=True)
    rollout_ids = [f"mcp-{index}" for index in range(8)]
    # Gym's aiohttp client keeps no cookies, so each MCP call names its session by the token alone.
    async with GymHTTP(deployment.url("resources"), timeout=60) as http:
        tokens, cookies = [], []
        for index, rollout_id in enumerate(rollout_ids):
            seeded = await http.post(f"/ng-rollout/{rollout_id}/seed_session", json={"initial_count": index})
            tokens.append(seeded.json()["mcp"]["headers"][MCP_TOKEN_HEADER])
            cookies.append(seeded.cookies)
            await mcp_call(http, tokens[-1], "increment_counter", {"count": 10})

        participants = await deployment.participants()
        prepared = await coordination.prepare(participants, "c1", deadline_ts=deadline())
        assert prepared.prepared, prepared.blockers()
        # Calls made while the checkpoint is open wait on whichever worker owns the session, instead of failing.
        waiting = [asyncio.create_task(mcp_call(http, token, "increment_counter", {"count": 100})) for token in tokens]
        await asyncio.sleep(1)
        assert not any(task.done() for task in waiting)
        episode_ids = [EpisodeId(rollout_id=rollout_id) for rollout_id in rollout_ids]
        await coordination.commit(participants, "c1", str(tmp_path / "ckpt"), episode_ids, deadline_ts=deadline())
        await coordination.resume(participants, "c1", deadline_ts=deadline())
        await asyncio.gather(*waiting)

        # Gym restarts with three workers per server.
        await crash_and_restore(deployment, tmp_path / "ckpt", rollout_ids, server_workers=3)
        # A harness of the crashed attempt that outlived the crash still holds its token, which names that attempt.
        stale = [await mcp_reply(http, token, "increment_counter", {"count": 1000}) for token in tokens]
        # Each replacement seeds under its own attempt with the session's cookie, which claims the restored session.
        # The seed reply carries a token that names the new attempt.
        restored = []
        for rollout_id, cookie in zip(rollout_ids, cookies):
            reseeded = await http.post(
                f"/ng-rollout/{rollout_id}-a1/seed_session",
                json={"initial_count": 0},
                headers={"cookie": "; ".join(f"{name}={value}" for name, value in cookie.items())},
            )
            token = reseeded.json()["mcp"]["headers"][MCP_TOKEN_HEADER]
            restored.append((await mcp_call(http, token, "get_counter_value", {}))["count"])

    assert all(
        reply.status_code == 409 and reply.json()["error"]["code"] == "checkpoint_stale_attempt" for reply in stale
    )
    # The checkpoint holds the counts from before the waiting calls ran, and no stale call changed them.
    assert restored == [index + 10 for index in range(len(rollout_ids))]
