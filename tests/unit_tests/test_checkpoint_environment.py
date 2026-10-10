# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import time
from pathlib import Path

import pytest

from nemo_gym._checkpoint.control import (
    CheckpointRequest,
    CommitRequest,
    ParticipantControlPlane,
    RestoreRequest,
    RetireRequest,
)
from nemo_gym._checkpoint.environment import EnvironmentParticipant, EpisodeRecord, task_digest
from nemo_gym._checkpoint.errors import AdmissionClosedError, ControlError
from nemo_gym.episode_types import EpisodeId


TASK = {"task_id": "t"}
# A direct call to close_admission stands in for a prepare with a far deadline.
CLOSE = CheckpointRequest(checkpoint_id="direct", deadline_ts=time.time() + 3600)


def request(checkpoint_id: str = "c1", **extra) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}


def controller_for(participant: EnvironmentParticipant) -> ParticipantControlPlane:
    return ParticipantControlPlane(participant, instance_name="env", lease_grace_seconds=60)


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
    await fresh.close_admission(CLOSE)
    # Not before the resume that follows the restore; the restored record waits for the replacement.
    with pytest.raises(AdmissionClosedError):
        fresh.begin(EpisodeId(rollout_id="r", attempt=3), TASK, None)
    assert fresh.status_extra()["restored_pending"] == ["r-a3"]
    await fresh.open_admission()
    # After that resume, a new checkpoint that closes admission still admits the replacement.
    await fresh.close_admission(CLOSE)
    fresh.begin(EpisodeId(rollout_id="r", attempt=3), TASK, None)

    assert fresh.continuation(EpisodeId(rollout_id="r", attempt=3)) == {"k": 1}
    assert fresh.continuation(EpisodeId(rollout_id="r", attempt=3)) is None
    # Only a restored replacement starts while closed; any other attempt is a new episode.
    with pytest.raises(AdmissionClosedError):
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
    started, closed, release, finished = asyncio.Event(), asyncio.Event(), asyncio.Event(), asyncio.Event()
    deadlines: list[asyncio.Timeout] = []

    def start_deadline() -> None:
        # The 0.3 s deadline starts right before the episode is parked, with no await in between,
        # so only the checkpoint pause can outlast it, however slow the loop is.
        deadlines[0].reschedule(asyncio.get_running_loop().time() + 0.3)

    async def episode() -> None:
        async with asyncio.timeout(None) as deadline:
            deadlines.append(deadline)
            participant.begin(episode_id, TASK, deadline)
            if parked_by == "boundary":
                started.set()
                await closed.wait()
                start_deadline()
                await participant.boundary(episode_id, {"next": "x"})
            else:
                async with participant.step(episode_id, "replay"):
                    started.set()
                    await release.wait()
        finished.set()

    task = asyncio.create_task(episode())
    await started.wait()
    if parked_by == "replay_step":
        start_deadline()
    await participant.close_admission(CLOSE)
    closed.set()
    await asyncio.sleep(0.45)
    await participant.open_admission()
    release.set()
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


async def test_a_commit_that_no_longer_continues_a_restored_episode_retires_it(tmp_path: Path) -> None:
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
    await source_controller.commit(CommitRequest(**request(checkpoint_dir=str(tmp_path / "first"))))
    task.cancel()

    restored = EnvironmentParticipant()
    controller = controller_for(restored)
    await controller.restore(
        RestoreRequest(**request("r1", checkpoint_dir=str(tmp_path / "first"), episode_ids=[episode_id]))
    )
    await controller.resume(CheckpointRequest(**request("r1")))
    await controller.prepare(CheckpointRequest(**request("c2")))
    # The controller continues nothing from this checkpoint, so attempt 1 will never start.
    committed = await controller.commit(
        CommitRequest(**request("c2", checkpoint_dir=str(tmp_path / "second"), episode_ids=[]))
    )
    await controller.resume(CheckpointRequest(**request("c2")))

    # Outside the scope, so not exported, and retired after the write.
    assert committed["episode_ids"] == []
    assert controller.status()["restored_pending"] == []
    # Retired without a lasting refusal, since attempt 1 never ran here: a later /run for it starts from its input.
    restored.begin(EpisodeId(rollout_id="r", attempt=1), TASK, None)
    assert restored.continuation(EpisodeId(rollout_id="r", attempt=1)) is None


async def test_a_retire_replies_only_after_the_episode_and_its_cleanup_have_ended() -> None:
    from nemo_gym.base_environment_server import CleanupContext
    from tests.unit_tests.test_environment_server import _environment_server, _EnvironmentServer, _request

    sessions_closed = asyncio.Event()
    running = asyncio.Event()

    class LongEpisodeServer(_EnvironmentServer):
        async def run(self, request, cleanup: CleanupContext):
            async def close_sessions() -> None:
                await asyncio.sleep(0.1)
                sessions_closed.set()

            cleanup.register_cleanup("sessions", close_sessions)
            running.set()
            await asyncio.sleep(60)

    config = _environment_server().config.model_copy(update={"cleanup_timeout_seconds": 5})
    server = LongEpisodeServer(config=config, server_client=_environment_server().server_client)
    server._checkpoint = EnvironmentParticipant()
    controller = controller_for(server._checkpoint)
    episode = _request()
    run = asyncio.create_task(server.run_request(episode))
    await running.wait()

    await controller.retire(RetireRequest(**request("retire", episode_ids=[episode.episode_id.model_dump()])))

    # The retire replied after the episode stopped and its sessions were closed, not before.
    assert sessions_closed.is_set() and run.done()
    assert len(server._checkpoint.retired) == 1


async def test_an_episode_without_a_boundary_is_exported_and_restarts_from_its_input(tmp_path: Path) -> None:
    episode_id = EpisodeId(rollout_id="r")
    source = EnvironmentParticipant()
    source_controller = controller_for(source)
    release = asyncio.Event()

    async def episode() -> None:
        source.begin(episode_id, TASK, None)
        # The first step, a replayable seed, runs before any boundary was recorded.
        async with source.step(episode_id, "replay"):
            await release.wait()

    task = asyncio.create_task(episode())
    await asyncio.sleep(0)
    prepared = await source_controller.prepare(CheckpointRequest(**request()))
    committed = await source_controller.commit(
        CommitRequest(**request(checkpoint_dir=str(tmp_path), episode_ids=[episode_id]))
    )
    task.cancel()

    restored = EnvironmentParticipant()
    await controller_for(restored).restore(
        RestoreRequest(**request("r1", checkpoint_dir=str(tmp_path), episode_ids=[episode_id]))
    )
    await restored.open_admission()
    restored.begin(EpisodeId(rollout_id="r", attempt=1), TASK, None)

    assert prepared["phase"] == "prepared" and committed["episode_ids"] == ["r"]
    assert restored.continuation(EpisodeId(rollout_id="r", attempt=1)) is None


async def test_an_episode_whose_retire_was_cut_short_blocks_the_next_prepare(tmp_path: Path) -> None:
    episode_id = EpisodeId(rollout_id="r")
    participant = EnvironmentParticipant()
    controller = controller_for(participant)
    release, stepped = asyncio.Event(), asyncio.Event()

    async def episode() -> None:
        participant.begin(episode_id, TASK, None)
        try:
            await stepped.wait()
            await participant.boundary(episode_id, {"step": 1})
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            # Slow final cleanup, like closing sessions.
            await release.wait()
            raise
        finally:
            await participant.end(episode_id)

    task = asyncio.create_task(episode())
    await asyncio.sleep(0)
    prepare = asyncio.create_task(controller.prepare(CheckpointRequest(**request())))
    stepped.set()
    assert (await prepare)["phase"] == "prepared"
    await controller.resume(CheckpointRequest(**request()))
    retire = RetireRequest(**request("retire", episode_ids=[episode_id.model_dump()]))
    with pytest.raises(ControlError, match="deadline"):
        await controller.retire(retire.model_copy(update={"deadline_ts": time.time() + 0.05}))

    # A new checkpoint before the retire is retried: the episode is still being stopped, so it blocks, not exported.
    prepared = await controller.prepare(CheckpointRequest(**{**request("c2"), "deadline_ts": time.time() + 0.1}))
    assert prepared["phase"] == "preparing" and prepared["report"]["blockers"] == ["r"]
    with pytest.raises(ControlError, match="phase preparing"):
        await controller.commit(CommitRequest(**request("c2", checkpoint_dir=str(tmp_path))))

    await controller.resume(CheckpointRequest(**request("c2")))
    release.set()
    await controller.retire(retire)
    assert task.done() and participant.steps.keys() == []


async def test_readiness_is_answered_by_the_blocker_count_without_listing_episodes() -> None:
    participant = EnvironmentParticipant()
    for index in range(3):
        participant.begin(EpisodeId(rollout_id=f"r{index}"), TASK, None)

    assert participant.ready() is False and participant.readiness().blocker_count == 3
    for index in range(3):
        await participant.end(EpisodeId(rollout_id=f"r{index}"))
    assert participant.ready() is True and participant.readiness().ready is True


async def test_an_episode_marked_restart_never_holds_up_prepare_and_is_left_out_of_the_commit(tmp_path: Path) -> None:
    from nemo_gym._checkpoint.errors import RestartInScopeError

    participant = EnvironmentParticipant()
    controller = controller_for(participant)
    restart, continued = EpisodeId(rollout_id="r"), EpisodeId(rollout_id="s")
    release = asyncio.Event()

    async def restarting_episode() -> None:
        participant.begin(restart, TASK, None)
        await participant.boundary(restart, {"next": "invoke_agent"})
        # Its agent cannot be captured, so its own boundaries are useless after a crash.
        await participant.mark_restart(restart)
        async with participant.step(restart, "wait"):
            await release.wait()

    tasks = [asyncio.create_task(restarting_episode())]
    await asyncio.sleep(0)
    participant.begin(continued, TASK, None)
    preparing = asyncio.create_task(controller.prepare(CheckpointRequest(**request())))
    await asyncio.sleep(0.01)
    # The continued episode parks at its next boundary; the restart, still in a wait step, never holds prepare up.
    tasks.append(asyncio.create_task(participant.boundary(continued, {"next": "verify"})))
    prepared = await preparing
    with pytest.raises(RestartInScopeError):
        await controller.commit(
            CommitRequest(**request(checkpoint_dir=str(tmp_path), episode_ids=[restart, continued]))
        )
    committed = await controller.commit(
        CommitRequest(**request(checkpoint_dir=str(tmp_path), episode_ids=[continued]))
    )
    await controller.resume(CheckpointRequest(**request()))
    release.set()
    await asyncio.gather(*tasks)

    assert prepared["phase"] == "prepared" and prepared["report"]["restarts"] == ["r"]
    assert committed["episode_ids"] == ["s"]


async def test_a_replay_step_that_raises_during_a_checkpoint_starts_over_after_a_crash(tmp_path: Path) -> None:
    participant = EnvironmentParticipant()
    controller = controller_for(participant)
    episode_id = EpisodeId(rollout_id="r")
    fail = asyncio.Event()
    cleaning_up = asyncio.Event()
    cleaned_up = asyncio.Event()

    async def episode() -> None:
        participant.begin(episode_id, TASK, None)
        await participant.boundary(episode_id, {"next": "invoke_agent", "handles": "sessions"})
        try:
            async with participant.step(episode_id, "replay"):
                await fail.wait()
                raise RuntimeError("the agent failed")
        except RuntimeError:
            # The episode's /run now fails, and its cleanup may close the sessions the boundary names.
            participant.finishing(episode_id)
            cleaning_up.set()
            await cleaned_up.wait()
        await participant.end(episode_id)

    task = asyncio.create_task(episode())
    await asyncio.sleep(0.01)
    prepared = await controller.prepare(CheckpointRequest(**request()))
    fail.set()
    await cleaning_up.wait()
    ready = participant.readiness().ready
    committed = await controller.commit(CommitRequest(**request(checkpoint_dir=str(tmp_path))))
    [record] = participant.export_records(None)
    await controller.resume(CheckpointRequest(**request()))
    cleaned_up.set()
    await task

    # One failing episode does not fail the checkpoint, and it is never continued from the failed step's boundary.
    assert prepared["phase"] == "prepared" and ready
    assert committed["episode_ids"] == ["r"]
    assert record.boundary is None
