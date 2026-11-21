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
| E4/E6 the harness finishes as the park arrives, crash at close_agent, then restore | its result is kept; re-seeded, then closed with the evidence intact |
| E5 crash during verify after the capture, then restore | the session's data alone was exported; grading runs again |
| E7 release | snapshots of dropped checkpoints deleted, the harness paused in place twice |
| E8 a provider without snapshots | the episode restarts in a new sandbox; commits succeed; status says restart_only |
| E9 a checkpoint of a restored episode, the older one released, crash, restore | continues in a fork of the second |
| E10 a second crash before the replacement attempt starts | the restored, unclaimed sessions are checkpointed again |
"""

import asyncio
import json
import os
import shutil
import sys
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
    wait_until,
)

from nemo_gym.server_runtime._checkpoint import coordination


pytestmark = [
    pytest.mark.skipif(
        os.environ.get("NEMO_GYM_CHECKPOINT_E2E") != "1",
        reason="starts real server processes; set NEMO_GYM_CHECKPOINT_E2E=1 to run",
    ),
    # The local sandboxes find their processes in /proc and run their commands, and the supervisor, in bash.
    pytest.mark.skipif(sys.platform != "linux" or shutil.which("bash") is None, reason="needs Linux /proc and bash"),
]

STEPS = 4


def setup_backend(deployment) -> None:
    # Every model call is answered with a final message; the harness only counts them.
    deployment.backend("/_ctl/script", {"tool_calls": []})


def boxes(deployment) -> list[dict]:
    return [json.loads(path.read_text()) for path in (deployment.work_dir / "sandboxes" / "boxes").glob("*.json")]


def snapshots(deployment) -> list[str]:
    root = deployment.work_dir / "sandboxes" / "snapshots"
    return sorted(path.name for path in root.iterdir()) if root.exists() else []


def harness_process_state(deployment) -> str:
    """The state letter in ``/proc/<pid>/stat`` of the one harness process the deployment runs."""
    [box] = boxes(deployment)
    [pid_file] = Path(box["path"]).glob("harness/*/harness.pid")
    return Path(f"/proc/{int(pid_file.read_text())}/stat").read_text().rsplit(")", 1)[1].split()[0]


def launches(reply) -> str:
    return result(reply)["response"]["metadata"]["harness_launches"]


def snapshot_notes(deployment, snapshot_id: str) -> str:
    return (deployment.work_dir / "sandboxes" / "snapshots" / snapshot_id / "tree" / "work" / "notes").read_text()


async def all_sandboxes_stopped(deployment) -> None:
    """Wait until every sandbox is stopped: a restore stops the sandboxes it replaced in the background."""
    await wait_until(
        lambda: bool(boxes(deployment)) and all(box["state"] == "stopped" for box in boxes(deployment)), timeout=60
    )


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
        await wait_for_calls(deployment, 3, timeout=120)
        await checkpoint(deployment, tmp_path / "ckpt", ["e1"])
        # The backend holds the harness's third call, so only the process state tells a pause from a wait.
        assert harness_process_state(deployment) in {"T", "t"}, "the harness is stopped during the checkpoint"
        await coordination.resume(await deployment.participants(), "c1", deadline_ts=deadline())
        await wait_until(lambda: harness_process_state(deployment) not in {"T", "t"})  # continued by the resume
        deployment.backend("/_ctl/release", {})
        reply = await run

    assert result(reply)["reward"] == 1.0
    assert launches(reply) == "1", "paused in place, not relaunched"
    assert len(snapshots(deployment)) == 1
    assert len(deployment.backend_calls()) == STEPS


async def test_e2_a_pause_that_fails_stops_the_harness_and_relaunches_it(deploy, tmp_path: Path) -> None:
    deployment = deploy("harness_sessions", harness={"steps": STEPS, "fail_pauses": 1})
    setup_backend(deployment)
    deployment.backend("/_ctl/hold", {"after_calls": 2})
    async with GymHTTP(deployment.url("environment"), timeout=180) as http:
        run = asyncio.create_task(http.post("/run", json=harness_episode("e2", steps=STEPS)))
        await wait_for_calls(deployment, 3, timeout=120)
        await checkpoint(deployment, tmp_path / "ckpt", ["e2"])
        await coordination.resume(await deployment.participants(), "c1", deadline_ts=deadline())
        deployment.backend("/_ctl/release", {})
        reply = await run

    assert result(reply)["reward"] == 1.0
    assert launches(reply) == "2", "stopped, then relaunched once"
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
        await wait_for_calls(deployment, 3, timeout=120)
        await checkpoint(deployment, tmp_path / "ckpt", ["e3"])
        before = {box["id"] for box in boxes(deployment)}
        await crash_and_restore(deployment, tmp_path / "ckpt", ["e3"])
        first.cancel()
        replacement = await http.post("/run", json=request_for(topology, "e3", attempt=1))
        # S1 was superseded by the fork, and the fork was stopped once its notes were captured.
        await all_sandboxes_stopped(deployment)

    assert len(before) == 1
    assert result(replacement)["reward"] == 1.0, replacement.text
    forks = [box for box in boxes(deployment) if box["from_snapshot"] is not None]
    assert len(forks) == 1 and len(boxes(deployment)) == 2, "the restore forked the snapshot once"
    # Two calls were delivered before the crash and the held one was lost: the replacement makes only the rest.
    assert len(deployment.backend_calls()) == STEPS + 1
    assert launches(replacement) == "2"


async def test_e4_e6_a_harness_that_finishes_as_the_park_arrives_keeps_its_result_through_a_crash(
    deploy, tmp_path: Path
) -> None:
    deployment = deploy("harness_sessions", harness={"steps": STEPS, "finish_on_park": True})
    setup_backend(deployment)
    async with GymHTTP(deployment.url("environment"), timeout=180) as http:
        first = asyncio.create_task(http.post("/run", json=harness_episode("e4", steps=STEPS)))
        # Every step is done; the harness waits to exit, and finishes when the checkpoint asks it to park.
        await wait_for_calls(deployment, STEPS, timeout=120)
        await checkpoint(deployment, tmp_path / "ckpt", ["e4"])
        record = environment_record(tmp_path / "ckpt")
        await crash_and_restore(deployment, tmp_path / "ckpt", ["e4"])
        first.cancel()
        replacement = await http.post("/run", json=harness_episode("e4", attempt=1, steps=STEPS))

    assert record["boundary"]["next"] == "close_agent", "the reply arrived during the checkpoint"
    assert result(replacement)["reward"] == 1.0, replacement.text
    observations = result(replacement)["ng_agent_observations"]
    assert observations["source"] == "fake_harness", "the close kept the evidence"
    assert [record["status"] for record in observations["records"]] == ["completed"]
    assert len(deployment.backend_calls()) == STEPS, "nothing was run again"


async def test_e5_after_a_crash_during_grading_the_captured_notes_are_graded_again(deploy, tmp_path: Path) -> None:
    # The grading outlasts the checkpoint however busy the machine is; the crash ends it.
    deployment = deploy("harness_sessions", harness={"steps": STEPS, "verify_delay_s": 600})
    setup_backend(deployment)
    async with GymHTTP(deployment.url("environment"), timeout=180) as http:
        first = asyncio.create_task(http.post("/run", json=harness_episode("e5", steps=STEPS)))
        await wait_for_calls(deployment, STEPS, timeout=120)
        # Verify captured the notes and stopped the sandbox, and now grades.
        await all_sandboxes_stopped(deployment)
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
    assert len(deployment.backend_calls()) == STEPS, "nothing was run again"
    assert [box["from_snapshot"] for box in boxes(deployment)] == [None], "no sandbox was made after the capture"


async def test_e7_a_release_deletes_the_snapshots_of_checkpoints_no_longer_kept(deploy, tmp_path: Path) -> None:
    deployment = deploy("harness_sessions", harness={"steps": STEPS})
    setup_backend(deployment)
    deployment.backend("/_ctl/hold", {"after_calls": 2})
    async with GymHTTP(deployment.url("environment"), timeout=180) as http:
        run = asyncio.create_task(http.post("/run", json=harness_episode("e7", steps=STEPS)))
        await wait_for_calls(deployment, 3, timeout=120)
        participants = await deployment.participants()
        for checkpoint_id in ("c1", "c2"):
            await checkpoint(deployment, tmp_path / checkpoint_id, ["e7"], checkpoint_id=checkpoint_id)
            await coordination.resume(participants, checkpoint_id, deadline_ts=deadline())
        two = snapshots(deployment)
        assert await coordination.release(participants, "c2", ["c2"], deadline_ts=deadline()) == {}
        await wait_until(lambda: len(snapshots(deployment)) == 1, timeout=60)
        one = snapshots(deployment)
        assert await coordination.release(participants, "c2", [], deadline_ts=deadline()) == {}
        await wait_until(lambda: snapshots(deployment) == [], timeout=60)
        deployment.backend("/_ctl/release", {})
        reply = await run

    assert len(two) == 2 and len(one) == 1 and set(one) < set(two)
    assert result(reply)["reward"] == 1.0
    assert launches(reply) == "1", "paused in place twice"
    assert len(deployment.backend_calls()) == STEPS


async def test_e8_on_a_provider_without_snapshots_the_episode_restarts(deploy, tmp_path: Path) -> None:
    deployment = deploy("harness_sessions", harness={"steps": STEPS}, sandbox_snapshots=False)
    setup_backend(deployment)
    deployment.backend("/_ctl/hold", {"after_calls": 2})
    async with GymHTTP(deployment.url("environment"), timeout=180) as http:
        first = asyncio.create_task(http.post("/run", json=harness_episode("e8", steps=STEPS)))
        await wait_for_calls(deployment, 3, timeout=120)
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
    assert len(deployment.backend_calls()) == 3 + STEPS, "the replacement started over from its input"
    assert [box["from_snapshot"] for box in boxes(deployment)] == [None, None], "in a new sandbox, not a fork"


async def test_e9_a_checkpoint_of_a_restored_episode_restores_after_the_older_checkpoint_is_released(
    deploy, tmp_path: Path
) -> None:
    deployment = deploy("harness_sessions", harness={"steps": STEPS})
    setup_backend(deployment)
    deployment.backend("/_ctl/hold", {"after_calls": 2})
    async with GymHTTP(deployment.url("environment"), timeout=300) as http:
        first = asyncio.create_task(http.post("/run", json=harness_episode("e9", steps=STEPS)))
        await wait_for_calls(deployment, 3, timeout=120)
        await checkpoint(deployment, tmp_path / "c1", ["e9"])
        await crash_and_restore(deployment, tmp_path / "c1", ["e9"])
        first.cancel()
        # The restore released the hold. Three calls were made; attempt 1 makes the fourth, and its fifth is held.
        deployment.backend("/_ctl/hold", {"after_calls": 4})
        second = asyncio.create_task(http.post("/run", json=harness_episode("e9", attempt=1, steps=STEPS)))
        await wait_for_calls(deployment, 5, timeout=120)
        await checkpoint(deployment, tmp_path / "c2", ["e9"], attempt=1, checkpoint_id="c2")
        participants = await deployment.participants()
        await coordination.resume(participants, "c2", deadline_ts=deadline())
        two = snapshots(deployment)
        # The controller published c2 and dropped c1, whose snapshot the running fork came from.
        failures = await coordination.release(participants, "c2", ["c2"], deadline_ts=deadline())
        await wait_until(lambda: len(snapshots(deployment)) == 1, timeout=60)
        kept = snapshots(deployment)
        kept_notes = snapshot_notes(deployment, kept[0])
        await crash_and_restore(deployment, tmp_path / "c2", ["e9"], attempt=1, restore_id="r2")
        second.cancel()
        replacement = await http.post("/run", json=harness_episode("e9", attempt=2, steps=STEPS))
        await all_sandboxes_stopped(deployment)

    assert failures == {}
    assert len(two) == 2 and set(kept) < set(two)
    assert kept_notes == "step-0\nstep-1\nstep-2\n", "the snapshot kept is the second checkpoint's"
    assert result(replacement)["reward"] == 1.0, replacement.text
    # Calls 3 and 5 were held and lost with their crashes: attempt 2 makes only the last step's call.
    assert len(deployment.backend_calls()) == STEPS + 2
    assert launches(replacement) == "3"
    forks = [box for box in boxes(deployment) if box["from_snapshot"] is not None]
    assert {box["from_snapshot"] for box in forks} == set(two), "each restore forked its own checkpoint's snapshot"
    assert len(boxes(deployment)) == 3


async def test_e10_a_second_crash_before_the_replacement_starts_still_continues_in_a_fork(
    deploy, tmp_path: Path
) -> None:
    deployment = deploy("harness_sessions", harness={"steps": STEPS})
    setup_backend(deployment)
    deployment.backend("/_ctl/hold", {"after_calls": 2})
    async with GymHTTP(deployment.url("environment"), timeout=300) as http:
        first = asyncio.create_task(http.post("/run", json=harness_episode("e10", steps=STEPS)))
        await wait_for_calls(deployment, 3, timeout=120)
        await checkpoint(deployment, tmp_path / "c1", ["e10"])
        await crash_and_restore(deployment, tmp_path / "c1", ["e10"])
        first.cancel()
        # The restore stops the sandbox it replaced in the background, best effort: a crash before that is done
        # would leave it running, which is not what this scenario is about.
        await wait_until(lambda: any(box["state"] == "stopped" for box in boxes(deployment)), timeout=60)
        # The next checkpoint lands before attempt 1 is dispatched: the agent session is restored but not
        # re-bound, and no seed has claimed the fork. Then Gym crashes again.
        await checkpoint(deployment, tmp_path / "c2", ["e10"], attempt=1, checkpoint_id="c2")
        await crash_and_restore(deployment, tmp_path / "c2", ["e10"], attempt=1, restore_id="r2")
        replacement = await http.post("/run", json=harness_episode("e10", attempt=2, steps=STEPS))
        await all_sandboxes_stopped(deployment)

    assert result(replacement)["reward"] == 1.0, replacement.text
    # Two calls were delivered before the first crash and nothing ran in between.
    assert len(deployment.backend_calls()) == STEPS + 1
    assert launches(replacement) == "2", "the launch the first crash ended, and one after the second restore"
    forks = [box for box in boxes(deployment) if box["from_snapshot"] is not None]
    assert len(forks) == 2, "each restore forked once"
