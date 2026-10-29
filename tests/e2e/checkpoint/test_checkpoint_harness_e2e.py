# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoints of a sandboxed harness through the Environment Server, across real Gym processes.

The fake harness agent runs a harness under the real process supervisor, in a sandbox the workspace owner server
lends; sandboxes are directories on this machine whose snapshots fork with the same paths. The harness makes one model
call per step and writes one line per step, and verification captures the lines once. Skipped unless
``NEMO_GYM_CHECKPOINT_E2E=1``, like the rest of the suite.

| Scenario | Pass when |
| --- | --- |
| E1 commit, then resume, while the harness runs | paused in place, no relaunch, the episode finishes |
| E2 the pause fails | stopped, relaunched after resume, the episode finishes |
| E3 crash at invoke_agent, then restore | fork; both sessions re-seeded; the harness continues in the fork |
| E4 crash at close_agent, then restore | re-seeded, then closed with the evidence intact |
| E5 crash during verify after the capture, then restore | the session's data alone was exported; grading runs again |
| E6 the harness finishes as the park arrives | its result is kept, and the episode is at close_agent |
| E7 release | snapshots of dropped checkpoints deleted |
| E8 a provider without snapshots | the episode restarts; commits succeed; status says restart_only |
"""

import asyncio
import json
import os
from pathlib import Path

import pytest
from checkpoint_deployment import GymHTTP, harness_episode, harness_row
from test_checkpoint_e2e import (  # noqa: F401 (fixtures)
    checkpoint,
    crash_and_restore,
    deadline,
    deploy,
    environment_record,
    gym_http_client,
    wait_for_calls,
)

from nemo_gym._checkpoint import coordination


pytestmark = pytest.mark.skipif(
    os.environ.get("NEMO_GYM_CHECKPOINT_E2E") != "1",
    reason="starts real server processes; set NEMO_GYM_CHECKPOINT_E2E=1 to run",
)

STEPS = 4


def setup_backend(deployment) -> None:
    # Every model call is answered with a final message; the harness only counts them.
    deployment.backend("/_ctl/script", {"tool_calls": []})


def boxes(deployment) -> list[dict]:
    return [json.loads(path.read_text()) for path in (deployment.work_dir / "sandboxes" / "boxes").glob("*.json")]


def snapshots(deployment) -> list[str]:
    root = deployment.work_dir / "sandboxes" / "snapshots"
    return sorted(path.name for path in root.iterdir()) if root.exists() else []


def result(reply) -> dict:
    """The verify result, from a typed episode reply or a flat row's."""
    body = reply.json()
    if "result" not in body and "reward" in body:
        return body
    assert body.get("result") is not None, body
    return body["result"]


TOPOLOGIES = pytest.mark.parametrize("topology", ["harness_sessions", "harness_sessions_legacy"])


def request_for(topology: str, rollout_id: str, attempt: int = 0) -> dict:
    if topology == "harness_sessions_legacy":
        return harness_row(rollout_id, attempt, steps=STEPS)
    return harness_episode(rollout_id, attempt, steps=STEPS)


@TOPOLOGIES
async def test_e1_a_checkpoint_pauses_the_harness_in_place_and_the_episode_finishes(
    deploy, tmp_path: Path, topology: str
) -> None:
    deployment = deploy(topology, harness={"steps": STEPS})
    setup_backend(deployment)
    deployment.backend("/_ctl/hold", {"after_calls": 2})
    async with GymHTTP(deployment.url("environment"), timeout=180) as http:
        run = asyncio.create_task(http.post("/run", json=request_for(topology, "e1")))
        await wait_for_calls(deployment, 3)
        await checkpoint(deployment, tmp_path / "ckpt", ["e1"])
        await coordination.resume(await deployment.participants(), "c1", deadline_ts=deadline())
        deployment.backend("/_ctl/release", {})
        reply = await run

    assert result(reply)["reward"] == 1.0
    assert result(reply)["response"]["metadata"]["harness_launches"] == "1", "paused in place, not relaunched"
    assert len(snapshots(deployment)) == 1
    assert len(deployment.backend_calls()) == STEPS


async def test_e2_a_pause_that_fails_stops_the_harness_and_relaunches_it(deploy, tmp_path: Path) -> None:
    deployment = deploy("harness_sessions", harness={"steps": STEPS, "fail_pauses": 1})
    setup_backend(deployment)
    deployment.backend("/_ctl/hold", {"after_calls": 2})
    async with GymHTTP(deployment.url("environment"), timeout=180) as http:
        run = asyncio.create_task(http.post("/run", json=harness_episode("e2", steps=STEPS)))
        await wait_for_calls(deployment, 3)
        await checkpoint(deployment, tmp_path / "ckpt", ["e2"])
        await coordination.resume(await deployment.participants(), "c1", deadline_ts=deadline())
        deployment.backend("/_ctl/release", {})
        reply = await run

    assert result(reply)["reward"] == 1.0
    assert result(reply)["response"]["metadata"]["harness_launches"] == "2", "stopped, then relaunched once"
    # The held third call was abandoned with the stopped launch and made again by the relaunch.
    assert len(deployment.backend_calls()) == STEPS + 1


@TOPOLOGIES
async def test_e3_after_a_crash_at_invoke_agent_the_harness_continues_in_a_fork(
    deploy, tmp_path: Path, topology: str
) -> None:
    deployment = deploy(topology, harness={"steps": STEPS})
    setup_backend(deployment)
    deployment.backend("/_ctl/hold", {"after_calls": 2})
    async with GymHTTP(deployment.url("environment"), timeout=180) as http:
        first = asyncio.create_task(http.post("/run", json=request_for(topology, "e3")))
        await wait_for_calls(deployment, 3)
        await checkpoint(deployment, tmp_path / "ckpt", ["e3"])
        before = {box["id"] for box in boxes(deployment)}
        await crash_and_restore(deployment, tmp_path / "ckpt", ["e3"])
        first.cancel()
        replacement = await http.post("/run", json=request_for(topology, "e3", attempt=1))

    assert result(replacement)["reward"] == 1.0, replacement.text
    forks = [box for box in boxes(deployment) if box["from_snapshot"] is not None]
    assert len(forks) == 1, "the restore forked the snapshot once"
    assert all(box["state"] == "stopped" for box in boxes(deployment) if box["id"] in before), "S1 superseded"
    # Two calls were delivered before the crash and the held one was lost: the replacement makes only the rest.
    assert len(deployment.backend_calls()) == STEPS + 1
    assert result(replacement)["response"]["metadata"]["harness_launches"] == "2"


async def test_e4_e6_a_harness_that_finishes_as_the_park_arrives_keeps_its_result_through_a_crash(
    deploy, tmp_path: Path
) -> None:
    deployment = deploy("harness_sessions", harness={"steps": STEPS, "finish_on_park": True})
    setup_backend(deployment)
    async with GymHTTP(deployment.url("environment"), timeout=180) as http:
        first = asyncio.create_task(http.post("/run", json=harness_episode("e4", steps=STEPS)))
        # Every step is done; the harness waits to exit, and finishes when the checkpoint asks it to park.
        await wait_for_calls(deployment, STEPS)
        await checkpoint(deployment, tmp_path / "ckpt", ["e4"])
        record = environment_record(tmp_path / "ckpt")
        await crash_and_restore(deployment, tmp_path / "ckpt", ["e4"])
        first.cancel()
        replacement = await http.post("/run", json=harness_episode("e4", attempt=1, steps=STEPS))

    assert record["boundary"]["next"] == "close_agent", "the reply arrived during the checkpoint"
    assert result(replacement)["reward"] == 1.0, replacement.text
    observations = result(replacement)["ng_agent_observations"]
    assert observations["source"] == "fake_harness" and observations["records"], "the close kept the evidence"
    assert len(deployment.backend_calls()) == STEPS, "nothing was run again"


async def test_e5_after_a_crash_during_grading_the_captured_notes_are_graded_again(deploy, tmp_path: Path) -> None:
    deployment = deploy("harness_sessions", harness={"steps": STEPS, "verify_delay_s": 30})
    setup_backend(deployment)
    async with GymHTTP(deployment.url("environment"), timeout=180) as http:
        first = asyncio.create_task(http.post("/run", json=harness_episode("e5", steps=STEPS)))
        await wait_for_calls(deployment, STEPS)

        async def capturing() -> bool:
            return all(box["state"] == "stopped" for box in boxes(deployment))

        while not await capturing():
            await asyncio.sleep(0.2)
        await checkpoint(deployment, tmp_path / "ckpt", ["e5"])
        record = environment_record(tmp_path / "ckpt")
        # The restarted resources server grades at once.
        deployment.configure("resources", verify_delay_s=0)
        await crash_and_restore(deployment, tmp_path / "ckpt", ["e5"])
        first.cancel()
        replacement = await http.post("/run", json=harness_episode("e5", attempt=1, steps=STEPS))

    assert record["boundary"]["next"] == "verify"
    assert snapshots(deployment) == [], "a session past its capture is exported without a sandbox"
    assert result(replacement)["reward"] == 1.0, replacement.text


async def test_e7_a_release_deletes_the_snapshots_of_checkpoints_no_longer_kept(deploy, tmp_path: Path) -> None:
    deployment = deploy("harness_sessions", harness={"steps": STEPS})
    setup_backend(deployment)
    deployment.backend("/_ctl/hold", {"after_calls": 2})
    async with GymHTTP(deployment.url("environment"), timeout=180) as http:
        run = asyncio.create_task(http.post("/run", json=harness_episode("e7", steps=STEPS)))
        await wait_for_calls(deployment, 3)
        participants = await deployment.participants()
        for checkpoint_id in ("c1", "c2"):
            await checkpoint(deployment, tmp_path / checkpoint_id, ["e7"], checkpoint_id=checkpoint_id)
            await coordination.resume(participants, checkpoint_id, deadline_ts=deadline())
        two = snapshots(deployment)
        assert await coordination.release(participants, "c2", ["c2"], deadline_ts=deadline()) == {}
        await asyncio.sleep(1)
        one = snapshots(deployment)
        assert await coordination.release(participants, "c2", [], deadline_ts=deadline()) == {}
        await asyncio.sleep(1)
        deployment.backend("/_ctl/release", {})
        reply = await run

    assert len(two) == 2 and len(one) == 1 and set(one) < set(two)
    assert snapshots(deployment) == []
    assert result(reply)["reward"] == 1.0


async def test_e8_on_a_provider_without_snapshots_the_episode_restarts(deploy, tmp_path: Path) -> None:
    deployment = deploy("harness_sessions", harness={"steps": STEPS}, sandbox_snapshots=False)
    setup_backend(deployment)
    deployment.backend("/_ctl/hold", {"after_calls": 2})
    async with GymHTTP(deployment.url("environment"), timeout=180) as http:
        first = asyncio.create_task(http.post("/run", json=harness_episode("e8", steps=STEPS)))
        await wait_for_calls(deployment, 3)
        status = await deployment.status("resources")
        participants = await deployment.participants()
        prepared = await coordination.prepare(participants, "c1", deadline_ts=deadline())
        await coordination.commit(participants, "c1", str(tmp_path / "ckpt"), [], deadline_ts=deadline())
        await crash_and_restore(deployment, tmp_path / "ckpt", [])
        first.cancel()
        replacement = await http.post("/run", json=harness_episode("e8", attempt=1, steps=STEPS))

    assert status["mode"] == "restart_only"
    assert prepared.prepared and "e8" in {episode.capture_key for episode in prepared.restarts()}
    assert result(replacement)["reward"] == 1.0
    assert snapshots(deployment) == []
