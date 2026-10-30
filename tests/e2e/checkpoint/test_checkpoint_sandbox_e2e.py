# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoints of resources sessions that own sandboxes, across real Gym processes and a fake sandbox backend.

The notes server's sessions each own a sandbox in the fake sandbox backend, which outlives a Gym crash.
A commit snapshots the sandboxes, a restore forks them, and a release deletes the snapshots of checkpoints the
controller no longer keeps. Skipped unless ``NEMO_GYM_CHECKPOINT_E2E=1``, like the rest of the suite.
"""

import asyncio
import os
from pathlib import Path

import pytest
from checkpoint_deployment import NOTES_SCRIPT, GymHTTP, notes_episode, notes_row
from test_checkpoint_e2e import (  # noqa: F401 (fixtures)
    checkpoint,
    crash_and_restore,
    deadline,
    deploy,
    gym_http_client,
    reward_of,
    wait_for_calls,
)

from nemo_gym._checkpoint import coordination


pytestmark = pytest.mark.skipif(
    os.environ.get("NEMO_GYM_CHECKPOINT_E2E") != "1",
    reason="starts real server processes; set NEMO_GYM_CHECKPOINT_E2E=1 to run",
)

TOPOLOGIES = pytest.mark.parametrize("topology", ["sandbox_sessions", "sandbox_sessions_legacy"])


def request_for(topology: str, rollout_id: str, attempt: int = 0) -> dict:
    return (
        notes_row(rollout_id, attempt) if topology == "sandbox_sessions_legacy" else notes_episode(rollout_id, attempt)
    )


def running(state: dict) -> list[dict]:
    return [box for box in state["boxes"].values() if box["state"] == "running"]


@TOPOLOGIES
async def test_a_checkpoint_snapshots_the_session_sandbox_and_the_episode_continues(
    deploy, tmp_path: Path, topology: str
) -> None:
    deployment = deploy(topology)
    deployment.backend("/_ctl/script", NOTES_SCRIPT)
    # The first model call returns; the second waits, after the first note is in the sandbox.
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        run = asyncio.create_task(http.post("/run", json=request_for(topology, "notes-1")))
        await wait_for_calls(deployment, 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["notes-1"])
        participants = await deployment.participants()
        await coordination.resume(participants, "c1", deadline_ts=deadline())
        deployment.backend("/_ctl/release", {})
        reply = await run

    state = deployment.sandbox_state()
    [snapshot] = state["snapshots"].values()
    assert reward_of(reply.json()) == 1.0
    # The snapshot holds what the sandbox held at the commit; the sandbox went on and was released after verify.
    assert snapshot["files"]["notes"] == "one\n"
    assert running(state) == []
    assert list((tmp_path / "gym-state" / "resources" / "snapshots").iterdir())[0].name == "c1"


@TOPOLOGIES
async def test_after_a_crash_the_replacement_continues_in_a_fork_of_the_snapshot(
    deploy, tmp_path: Path, topology: str
) -> None:
    deployment = deploy(topology)
    deployment.backend("/_ctl/script", NOTES_SCRIPT)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=request_for(topology, "notes-2")))
        await wait_for_calls(deployment, 2)
        await checkpoint(deployment, tmp_path / "ckpt", ["notes-2"])
        crashed = {box["id"] for box in running(deployment.sandbox_state())}
        await crash_and_restore(deployment, tmp_path / "ckpt", ["notes-2"])
        first.cancel()
        replacement = await http.post("/run", json=request_for(topology, "notes-2", attempt=1))

    state = deployment.sandbox_state()
    forks = [box for box in state["boxes"].values() if box["from_snapshot"] is not None]
    assert reward_of(replacement.json()) == 1.0, replacement.text
    # The replacement appended its note in the fork, which held the note from before the crash.
    [fork] = forks
    assert fork["files"]["notes"] == "one\ntwo\n"
    assert fork["state"] == "stopped", "verify released the fork once it captured the notes"
    # The crashed process's sandbox was superseded by the fork and stopped by the restore.
    assert all(state["boxes"][sandbox_id]["state"] == "stopped" for sandbox_id in crashed)
    # The model made only the call the crash lost.
    assert len(deployment.backend_calls()) == 4


async def test_a_release_deletes_the_snapshots_of_checkpoints_no_longer_kept(deploy, tmp_path: Path) -> None:
    deployment = deploy("sandbox_sessions")
    deployment.backend("/_ctl/script", NOTES_SCRIPT)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        run = asyncio.create_task(http.post("/run", json=notes_episode("notes-3")))
        await wait_for_calls(deployment, 2)
        participants = await deployment.participants()
        for checkpoint_id in ("c1", "c2"):
            await checkpoint(deployment, tmp_path / checkpoint_id, ["notes-3"], checkpoint_id=checkpoint_id)
            await coordination.resume(participants, checkpoint_id, deadline_ts=deadline())
        before = set(deployment.sandbox_state()["snapshots"])
        failures = await coordination.release(participants, "c2", ["c2"], deadline_ts=deadline())
        await asyncio.sleep(1)
        kept = set(deployment.sandbox_state()["snapshots"])
        failures_all = await coordination.release(participants, "c2", [], deadline_ts=deadline())
        await asyncio.sleep(1)
        deployment.backend("/_ctl/release", {})
        await run

    assert failures == {} and failures_all == {}
    assert len(before) == 2 and len(kept) == 1 and kept < before
    assert deployment.sandbox_state()["snapshots"] == {}
    assert not any((tmp_path / "gym-state" / "resources" / "snapshots").iterdir())


async def test_on_a_provider_without_snapshots_the_episode_restarts_and_commits_still_succeed(
    deploy, tmp_path: Path
) -> None:
    deployment = deploy("sandbox_sessions", sandbox_snapshots=False)
    deployment.backend("/_ctl/script", NOTES_SCRIPT)
    deployment.backend("/_ctl/hold", {"after_calls": 1})
    async with GymHTTP(deployment.url("environment"), timeout=120) as http:
        first = asyncio.create_task(http.post("/run", json=notes_episode("notes-4")))
        await wait_for_calls(deployment, 2)
        status = await deployment.status("resources")
        participants = await deployment.participants()
        prepared = await coordination.prepare(participants, "c1", deadline_ts=deadline())
        scope = prepared.continued([])
        await coordination.commit(participants, "c1", str(tmp_path / "ckpt"), scope, deadline_ts=deadline())
        await crash_and_restore(deployment, tmp_path / "ckpt", [])
        first.cancel()
        replacement = await http.post("/run", json=notes_episode("notes-4", attempt=1))

    assert status["mode"] == "restart_only"
    assert prepared.prepared and "notes-4" in {episode.capture_key for episode in prepared.restarts()}
    # Started over from its input in a new sandbox.
    assert reward_of(replacement.json()) == 1.0
    assert deployment.sandbox_state()["snapshots"] == {}
