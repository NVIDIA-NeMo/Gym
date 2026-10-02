# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from pathlib import Path

import pytest

from nemo_gym._checkpoint.control import (
    CheckpointRequest,
    CommitRequest,
    ParticipantController,
    RestoreRequest,
    RetireRequest,
)
from nemo_gym._checkpoint.environment import EnvironmentParticipant, EpisodeRecord, task_digest
from nemo_gym._checkpoint.errors import AdmissionClosedError, ControlError, StaleAttemptError
from nemo_gym.episode_types import EpisodeId


TASK = {"task_id": "t"}
# A direct call to close_admission stands in for a prepare with a far deadline.
CLOSE = CheckpointRequest(checkpoint_id="direct", deadline_ts=time.time() + 3600)


def request(checkpoint_id: str = "c1", **extra) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}


def controller_for(participant: EnvironmentParticipant) -> ParticipantController:
    return ParticipantController(participant, instance_name="env", lease_grace_seconds=60)


async def test_a_wait_step_blocks_prepare_until_its_result_is_recorded() -> None:
    participant = EnvironmentParticipant()
    controller = controller_for(participant)
    episode_id = EpisodeId(rollout_id="r")
    verified = asyncio.Event()

    async def episode() -> None:
        participant.begin(episode_id, TASK, None)
        await participant.boundary(episode_id, {"next": "verify"})
        async with participant.step(episode_id, "wait"):
            await verified.wait()
        await participant.boundary(episode_id, {"next": "close", "reward": 1.0})
        await participant.end(episode_id)

    task = asyncio.create_task(episode())
    await asyncio.sleep(0.01)
    missed = await controller.prepare(CheckpointRequest(checkpoint_id="c1", deadline_ts=time.time() + 0.1))
    verified.set()
    prepared = await controller.prepare(CheckpointRequest(**request()))
    [record] = participant.export_records(None)
    await controller.resume(CheckpointRequest(**request()))
    await task

    assert missed["report"]["blockers"] == ["r"]
    assert prepared["phase"] == "prepared"
    assert record.boundary == {"next": "close", "reward": 1.0}


async def test_a_replay_step_does_not_block_and_its_result_waits_at_the_next_boundary() -> None:
    participant = EnvironmentParticipant()
    controller = controller_for(participant)
    episode_id = EpisodeId(rollout_id="r")
    judged = asyncio.Event()
    progress: list[str] = []

    async def episode() -> None:
        participant.begin(episode_id, TASK, None)
        await participant.boundary(episode_id, {"next": "verify", "response": "answer"})
        async with participant.step(episode_id, "replay"):
            await judged.wait()
        progress.append("verified")
        await participant.boundary(episode_id, {"next": "close", "reward": 1.0})
        progress.append("closed")
        await participant.end(episode_id)

    task = asyncio.create_task(episode())
    await asyncio.sleep(0.01)
    prepared = await controller.prepare(CheckpointRequest(**request()))
    [record] = participant.export_records(None)
    judged.set()
    await asyncio.sleep(0.01)
    held = list(progress)
    await controller.resume(CheckpointRequest(**request()))
    await task

    assert prepared["phase"] == "prepared"
    assert record.boundary == {"next": "verify", "response": "answer"}
    assert held == ["verified"]
    assert progress == ["verified", "closed"]


async def test_new_episodes_are_refused_while_closed_but_restored_ones_start() -> None:
    participant = EnvironmentParticipant()
    await participant.close_admission(CLOSE)
    with pytest.raises(AdmissionClosedError):
        participant.begin(EpisodeId(rollout_id="new"), TASK, None)

    fresh = EnvironmentParticipant()
    fresh.restore_records(
        [
            EpisodeRecord(
                episode_id=EpisodeId(rollout_id="r", attempt=2), task_digest=task_digest(TASK), boundary={"k": 1}
            )
        ]
    )
    # The participant controller fences every restored attempt.
    fresh.attempts.retire(EpisodeId(rollout_id="r", attempt=2))
    await fresh.close_admission(CLOSE)
    fresh.begin(EpisodeId(rollout_id="r", attempt=3), TASK, None)

    assert fresh.continuation(EpisodeId(rollout_id="r", attempt=3)) == {"k": 1}
    assert fresh.continuation(EpisodeId(rollout_id="r", attempt=3)) is None
    with pytest.raises(StaleAttemptError):
        fresh.begin(EpisodeId(rollout_id="r", attempt=2), TASK, None)


async def test_restored_state_must_match_the_replacement_task() -> None:
    participant = EnvironmentParticipant()
    participant.restore_records(
        [EpisodeRecord(episode_id=EpisodeId(rollout_id="r"), task_digest=task_digest({"task_id": "a"}), boundary={})]
    )

    with pytest.raises(ControlError, match="different task"):
        participant.begin(EpisodeId(rollout_id="r", attempt=1), {"task_id": "b"}, None)


@pytest.mark.parametrize("parked_by", ["boundary", "replay_step"])
async def test_checkpoint_pause_does_not_count_against_a_parked_episode_deadline(parked_by: str) -> None:
    participant = EnvironmentParticipant()
    episode_id = EpisodeId(rollout_id="r")
    finished = asyncio.Event()

    async def episode() -> None:
        async with asyncio.timeout(0.3) as deadline:
            participant.begin(episode_id, TASK, deadline)
            if parked_by == "boundary":
                await asyncio.sleep(0.05)
                await participant.boundary(episode_id, {"next": "x"})
            else:
                async with participant.step(episode_id, "replay"):
                    await asyncio.sleep(0.5)
        finished.set()

    task = asyncio.create_task(episode())
    await asyncio.sleep(0.01)
    await participant.close_admission(CLOSE)
    await asyncio.sleep(0.45)
    await participant.open_admission()
    await task

    assert finished.is_set()


async def test_commit_restore_round_trip_and_retire_after_restore(tmp_path: Path) -> None:
    source = EnvironmentParticipant()
    source_controller = controller_for(source)
    episode_id = EpisodeId(rollout_id="r", attempt=1)
    release = asyncio.Event()

    async def episode() -> None:
        source.begin(episode_id, TASK, None)
        await source.boundary(episode_id, {"next": "invoke_agent", "handles": {"agent_session_id": "a"}})
        async with source.step(episode_id, "replay"):
            await release.wait()

    task = asyncio.create_task(episode())
    await asyncio.sleep(0)
    await source_controller.prepare(CheckpointRequest(**request()))
    await source_controller.commit(CommitRequest(**request(checkpoint_dir=str(tmp_path))))
    release.set()
    await source_controller.resume(CheckpointRequest(**request()))
    await task

    restored = EnvironmentParticipant()
    controller = controller_for(restored)
    await controller.restore(RestoreRequest(**request("r1", checkpoint_dir=str(tmp_path), episode_ids=[episode_id])))
    await controller.resume(CheckpointRequest(**request("r1")))
    status = controller.status()
    await controller.retire(RetireRequest(**request("cleanup", episode_ids=[{"rollout_id": "r", "attempt": 2}])))

    assert status["restored_pending"] == ["r-a2"]
    assert controller.status()["restored_pending"] == []


async def test_a_restored_episode_not_yet_restarted_survives_the_next_checkpoint(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    episode_id = EpisodeId(rollout_id="r")
    source = EnvironmentParticipant()
    source_controller = controller_for(source)
    release = asyncio.Event()

    async def episode() -> None:
        source.begin(episode_id, TASK, None)
        await source.boundary(episode_id, {"next": "verify"})
        async with source.step(episode_id, "replay"):
            await release.wait()

    task = asyncio.create_task(episode())
    await asyncio.sleep(0)
    await source_controller.prepare(CheckpointRequest(**request()))
    await source_controller.commit(CommitRequest(**request(checkpoint_dir=str(first))))
    task.cancel()

    # Restore and resume, then checkpoint again before the replacement attempt starts.
    restored = EnvironmentParticipant()
    controller = controller_for(restored)
    await controller.restore(RestoreRequest(**request("r1", checkpoint_dir=str(first), episode_ids=[episode_id])))
    await controller.resume(CheckpointRequest(**request("r1")))
    await controller.prepare(CheckpointRequest(**request("c2")))
    committed = await controller.commit(CommitRequest(**request("c2", checkpoint_dir=str(second))))

    again = EnvironmentParticipant()
    again_controller = controller_for(again)
    replacement = EpisodeId(rollout_id="r", attempt=1)
    await again_controller.restore(
        RestoreRequest(**request("r2", checkpoint_dir=str(second), episode_ids=[replacement]))
    )
    await again_controller.resume(CheckpointRequest(**request("r2")))
    again.begin(EpisodeId(rollout_id="r", attempt=2), TASK, None)

    assert committed["episode_ids"] == ["r-a1"]
    assert again.continuation(EpisodeId(rollout_id="r", attempt=2)) == {"next": "verify"}


async def test_a_retire_during_final_cleanup_does_not_interrupt_it() -> None:
    from nemo_gym.base_environment_server import CleanupContext
    from tests.unit_tests.test_environment_server import _environment_server, _EnvironmentServer, _request

    cleanup_started, cleanup_finished = asyncio.Event(), asyncio.Event()

    class SlowCleanupServer(_EnvironmentServer):
        async def run(self, request, cleanup: CleanupContext):
            async def close_sessions() -> None:
                cleanup_started.set()
                await asyncio.sleep(0.2)
                cleanup_finished.set()

            cleanup.register_cleanup("sessions", close_sessions)
            return await super().run(request, cleanup)

    config = _environment_server().config.model_copy(update={"cleanup_timeout_seconds": 5})
    server = SlowCleanupServer(config=config, server_client=_environment_server().server_client)
    server._checkpoint = EnvironmentParticipant()
    controller = controller_for(server._checkpoint)
    episode = _request()
    run = asyncio.create_task(server.run_request(episode))
    await cleanup_started.wait()

    # A controller retiring a straggler: the episode still counts as running until its cleanup ends.
    await controller.retire(RetireRequest(**request(episode_ids=[episode.episode_id.model_dump()])))
    await run

    assert cleanup_finished.is_set()
