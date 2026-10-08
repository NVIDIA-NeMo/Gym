# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import Body, FastAPI
from pydantic import JsonValue

from nemo_gym._checkpoint.agent import (
    AgentSessionParticipant,
    AgentSessionRecord,
    RestartOnlyTrackingMiddleware,
    RestoredAgentSession,
)
from nemo_gym._checkpoint.control import (
    CheckpointRequest,
    CommitRequest,
    ParticipantControlPlane,
    RestoreRequest,
    RetireRequest,
)
from nemo_gym._checkpoint.errors import AdmissionClosedError, ControlError, StaleAttemptError
from nemo_gym._checkpoint.participant_workers import COORDINATOR_SOCKET_ENV
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import ServerClient


class Hooks:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, JsonValue]] = {}
        self.restored: list[RestoredAgentSession] = []
        self.retired: list[str] = []

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        return {session_key: self.sessions.get(session_key, {}) for session_key in session_keys}

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        self.restored.extend(sessions)

    async def retire_agent_session(self, session_key: str) -> None:
        self.retired.append(session_key)


# A direct call to close_admission stands in for a prepare with a far deadline.
CLOSE = CheckpointRequest(checkpoint_id="direct", deadline_ts=time.time() + 3600)


def control(checkpoint_id: str = "c1", **extra) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}


async def test_activation_parks_at_its_next_boundary_and_resumes() -> None:
    participant = AgentSessionParticipant(Hooks())
    controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    step_done = asyncio.Event()
    progress: list[int] = []

    async def loop() -> None:
        async with participant.activation("s", episode_id) as activation:
            for step in range(3):
                await step_done.wait()
                step_done.clear()
                progress.append(step)
                await activation.boundary(lambda step=step: {"step": step})

    task = asyncio.create_task(loop())
    prepare = asyncio.create_task(controller.prepare(CheckpointRequest(**control())))
    await asyncio.sleep(0.05)
    assert not prepare.done()

    step_done.set()
    prepared = await prepare
    records = await participant.export(None)
    await controller.resume(CheckpointRequest(**control()))
    for _ in range(2):
        await asyncio.sleep(0.01)
        step_done.set()
    await task

    assert prepared["phase"] == "prepared"
    assert [(record.session_key, record.boundary) for record in records] == [("s", {"step": 0})]
    assert progress == [0, 1, 2]


async def test_activation_invoked_while_closed_parks_before_doing_work() -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    await participant.close_admission(CLOSE)

    with pytest.raises(AdmissionClosedError):
        participant.open_session("other", EpisodeId(rollout_id="other"))

    async def loop() -> None:
        async with participant.activation("s", episode_id) as activation:
            await activation.boundary(lambda: {"step": 0})

    task = asyncio.create_task(loop())
    await asyncio.sleep(0.01)
    assert participant.readiness().counts["at_boundary"] == 1
    await participant.open_admission()
    await task


async def test_legacy_run_steps_block_prepare_only_when_waited_on() -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")

    async with participant.legacy_run("run:r", episode_id) as run:
        assert run.continuation is None
        # Between steps, before any boundary, the episode must reach a boundary first.
        assert participant.readiness().blockers == ["r"]
        await run.boundary({"next": "verify"})
        async with run.step("wait"):
            assert participant.readiness().blockers == ["r"]
        async with run.step("replay"):
            assert participant.readiness().ready
            [record] = await participant.export(None)

    assert participant.readiness().ready
    assert record.episode == {"next": "verify"}


async def test_retire_stops_the_activation_before_it_replies() -> None:
    hooks = Hooks()
    participant = AgentSessionParticipant(hooks)
    controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)

    async def stuck() -> None:
        async with participant.activation("s", episode_id):
            await asyncio.sleep(60)

    task = asyncio.create_task(stuck())
    await asyncio.sleep(0.01)
    missed = await controller.prepare(CheckpointRequest(checkpoint_id="c1", deadline_ts=time.time() + 0.05))
    # A straggler is retired after the checkpoint is abandoned.
    await controller.resume(CheckpointRequest(**control()))
    await controller.retire(controller_retire_request(episode_id))
    stopped = task.done()
    prepared = await controller.prepare(CheckpointRequest(**control("c2")))

    assert missed["phase"] == "preparing"
    assert stopped and task.cancelled()
    assert prepared["phase"] == "prepared"
    assert hooks.retired == ["s"]
    # The attempt stays refused until the controller forgets the rollout.
    assert len(participant.retired) == 1


def controller_retire_request(episode_id: EpisodeId) -> RetireRequest:
    return RetireRequest(**control("retire", episode_ids=[episode_id.model_dump()]))


async def test_restore_installs_sessions_under_the_next_attempt(tmp_path: Path) -> None:
    hooks = Hooks()
    hooks.sessions["s"] = {"cookies": {"a": "b"}}
    source = AgentSessionParticipant(hooks)
    source_controller = ParticipantControlPlane(source, instance_name="agent", lease_grace_seconds=60)
    episode_id = EpisodeId(rollout_id="r", attempt=1)
    source.open_session("s", episode_id)

    reached = asyncio.Event()

    async def loop() -> None:
        async with source.activation("s", episode_id) as activation:
            await reached.wait()
            await activation.boundary(lambda: {"step": 4})

    task = asyncio.create_task(loop())
    await asyncio.sleep(0)
    prepare = asyncio.create_task(source_controller.prepare(CheckpointRequest(**control())))
    await asyncio.sleep(0.01)
    reached.set()
    await prepare
    await source_controller.commit(CommitRequest(**control(checkpoint_dir=str(tmp_path))))
    await source_controller.resume(CheckpointRequest(**control()))
    await task

    restored_hooks = Hooks()
    restored = AgentSessionParticipant(restored_hooks)
    controller = ParticipantControlPlane(restored, instance_name="agent", lease_grace_seconds=60)
    await controller.restore(RestoreRequest(**control("r1", checkpoint_dir=str(tmp_path), episode_ids=[episode_id])))
    await controller.resume(CheckpointRequest(**control("r1")))
    next_id = EpisodeId(rollout_id="r", attempt=2)

    assert restored_hooks.restored == [RestoredAgentSession("s", next_id, {"cookies": {"a": "b"}})]
    with pytest.raises(StaleAttemptError):
        async with restored.activation("s", episode_id):
            pass
    async with restored.activation("s", next_id) as activation:
        assert activation.continuation == {"step": 4}
    async with restored.activation("s", next_id) as activation:
        assert activation.continuation is None


async def test_awaiting_a_model_call_counts_as_parked_at_the_last_boundary() -> None:
    participant = AgentSessionParticipant(Hooks())
    controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    response = asyncio.Event()

    async def loop() -> None:
        async with participant.activation("s", episode_id) as activation:
            await activation.boundary(lambda: {"step": 1})
            async with activation.awaiting_model():
                await response.wait()
            await activation.boundary(lambda: {"step": 2})

    task = asyncio.create_task(loop())
    await asyncio.sleep(0.01)
    prepared = await controller.prepare(CheckpointRequest(**control()))
    [record] = await participant.export(None)
    response.set()
    await asyncio.sleep(0.01)
    # The call returned during the checkpoint, so the activation parks at its next boundary.
    parked_after_call = participant.readiness().counts["at_boundary"]
    await controller.resume(CheckpointRequest(**control()))
    await task

    assert prepared["phase"] == "prepared"
    assert record.boundary == {"step": 1}
    assert parked_after_call == 1


class _UnsupportedAgent(SimpleResponsesAPIAgent):
    """An agent with no session hooks: its rollouts can only restart."""

    started: Any = None
    release: Any = None

    async def responses(self, body=...):
        raise NotImplementedError

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        self.started.set()
        await self.release.wait()
        raise NotImplementedError

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        return AgentSessionState(request=body)


def _unsupported_agent_app() -> tuple[_UnsupportedAgent, Any]:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    agent = _UnsupportedAgent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent"), server_client=client
    )
    agent.started, agent.release = asyncio.Event(), asyncio.Event()
    return agent, agent.setup_webserver()


async def test_an_agent_without_session_hooks_reports_its_rollouts_as_restarts_that_keep_running(
    tmp_path: Path,
) -> None:
    agent, app = _unsupported_agent_app()
    row = {"responses_create_params": {"input": "x"}, "_ng_rollout_id": "r"}

    def control(checkpoint_id: str, **extra: Any) -> dict:
        return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agent", headers={"Authorization": "Bearer t"}
    ) as http:
        run = asyncio.create_task(http.post("/run", json=row))
        await agent.started.wait()
        # Nothing of this agent is in the checkpoint: its rollout does not hold prepare up.
        prepared = await http.post("/ng-control/v1/checkpoint/prepare", json=control("c1"))
        unattributed = await http.post("/run", json={"responses_create_params": {"input": "x"}})
        refused = await http.post(
            "/ng-control/v1/checkpoint/commit",
            json=control("c1", checkpoint_dir=str(tmp_path), episode_ids=[{"rollout_id": "r"}]),
        )
        committed = await http.post(
            "/ng-control/v1/checkpoint/commit", json=control("c1", checkpoint_dir=str(tmp_path), episode_ids=[])
        )
        await http.post("/ng-control/v1/checkpoint/resume", json=control("c1"))
        # The rollout kept running through the checkpoint.
        running = not run.done()
        await http.post("/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}]))
        # The retire replied after the /run it retired had stopped.
        stopped = run.done()
        with pytest.raises(BaseException):
            await run

    assert prepared.json()["phase"] == "prepared"
    assert prepared.json()["report"]["restarts"] == ["r"] and prepared.json()["report"]["blockers"] == []
    assert unattributed.json()["error"]["code"] == "rollout_id_required"
    assert refused.json()["error"]["code"] == "restart_in_scope"
    assert committed.status_code == 200
    assert running and stopped
    assert await agent._restart_only.export(None) == []


async def test_an_agent_without_session_hooks_says_on_its_seed_reply_that_the_episode_restarts() -> None:
    agent, app = _unsupported_agent_app()
    seed = {
        "agent_session_id": "s",
        "episode_id": {"rollout_id": "r"},
        "task_id": {"taskset": "t", "task_id": "0"},
    }
    control = {"checkpoint_id": "c1", "deadline_ts": time.time() + 5}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agent", headers={"Authorization": "Bearer t"}
    ) as http:
        await http.post("/ng-control/v1/checkpoint/prepare", json=control)
        # A seed during a checkpoint waits, so the restarts the checkpoint reported do not change.
        participant = agent._restart_only
        waiting_for_resume = asyncio.Event()
        wait_open = participant.wait_open

        async def signalling_wait_open() -> None:
            waiting_for_resume.set()
            await wait_open()

        participant.wait_open = signalling_wait_open
        seeding = asyncio.create_task(http.post("/v1/agent_sessions", json=seed))
        await asyncio.wait_for(waiting_for_resume.wait(), 5)
        # The seed asks to wait whether or not a checkpoint is open; give one that did not wait time to finish.
        await asyncio.sleep(0.05)
        waiting = not seeding.done()
        await http.post("/ng-control/v1/checkpoint/resume", json=control)
        seeded = await seeding

    assert waiting
    assert seeded.headers.get("x-ng-checkpoint-restart") == "1"


async def test_restored_sessions_not_yet_claimed_survive_the_next_checkpoint(tmp_path: Path) -> None:
    second = tmp_path / "second"
    episode_id = EpisodeId(rollout_id="r")
    restored_records = [
        AgentSessionRecord(
            session_key="native", episode_id=episode_id, session={"k": 1}, boundary={"step": 3}, episode=None
        ),
        AgentSessionRecord(
            session_key="run:r2",
            episode_id=EpisodeId(rollout_id="r2"),
            session={},
            boundary={"step": 1},
            episode={"next": "verify"},
        ),
    ]
    source = AgentSessionParticipant(Hooks())
    await source.install(restored_records, [record.episode_id for record in restored_records])
    await source.open_admission()
    controller = ParticipantControlPlane(source, instance_name="agent", lease_grace_seconds=60)
    # A checkpoint lands after the restore but before either replacement attempt starts.
    await controller.prepare(CheckpointRequest(**control("c2")))
    committed = await controller.commit(CommitRequest(**control("c2", checkpoint_dir=str(second))))

    again = AgentSessionParticipant(Hooks())
    again_controller = ParticipantControlPlane(again, instance_name="agent", lease_grace_seconds=60)
    await again_controller.restore(
        RestoreRequest(
            **control(
                "r2",
                checkpoint_dir=str(second),
                episode_ids=[EpisodeId(rollout_id="r", attempt=1), EpisodeId(rollout_id="r2", attempt=1)],
            )
        )
    )
    await again_controller.resume(CheckpointRequest(**control("r2")))

    assert committed["episode_ids"] == ["r-a1", "r2-a1"]
    async with again.activation("native", EpisodeId(rollout_id="r", attempt=2)) as activation:
        assert activation.continuation == {"step": 3}
    async with again.legacy_run("run:r2", EpisodeId(rollout_id="r2", attempt=2)) as legacy_run:
        assert legacy_run.continuation == {"next": "verify"}


def test_with_several_workers_the_main_process_coordinates_an_agent_without_session_hooks(monkeypatch) -> None:
    monkeypatch.delenv(COORDINATOR_SOCKET_ENV, raising=False)
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    agent = _UnsupportedAgent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent", num_workers=2),
        server_client=client,
    )

    app = agent.setup_webserver()
    coordinator = app.state.nemo_gym_checkpoint_coordinator
    socket_path = os.environ[COORDINATOR_SOCKET_ENV]
    os.unlink(socket_path)

    # The main process serves no requests: its workers forward control calls to this coordinator.
    assert coordinator.socket_path == socket_path
    assert coordinator.participant.kind == "agent"
    assert coordinator.participant.expected_workers == 2
    assert not [route for route in app.routes if getattr(route, "path", "").startswith("/ng-control/")]


async def test_a_session_woken_by_resume_stays_parked_if_a_new_checkpoint_closes_first() -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    progressed = asyncio.Event()

    async def loop() -> None:
        async with participant.activation("s", episode_id) as activation:
            await activation.boundary(lambda: {"step": 1})
            progressed.set()

    await participant.close_admission(CLOSE)
    task = asyncio.create_task(loop())
    await asyncio.sleep(0.01)
    # Resume and a new close before the parked session gets to run again.
    await participant.open_admission()
    await participant.close_admission(CLOSE)
    await asyncio.sleep(0.05)
    parked_through_second_checkpoint = not progressed.is_set() and participant.readiness().ready
    await participant.open_admission()
    await asyncio.wait_for(task, 5)

    assert parked_through_second_checkpoint


async def test_an_agent_that_leaves_a_session_out_of_its_export_fails_the_commit() -> None:
    class ForgetfulHooks(Hooks):
        async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
            return {}

    participant = AgentSessionParticipant(ForgetfulHooks())
    participant.open_session("s", EpisodeId(rollout_id="r"))

    with pytest.raises(ControlError, match="did not export sessions"):
        await participant.export(None)


async def test_closing_a_session_stops_its_running_activation() -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    model_calls = []

    async def activation() -> None:
        async with participant.activation("s", episode_id):
            while True:
                model_calls.append("call")
                await asyncio.sleep(0.01)

    task = asyncio.create_task(activation())
    await asyncio.sleep(0.05)
    # The environment server gave up on the activation and closed the session from its cleanup.
    await participant.close_session("s")
    done, _ = await asyncio.wait([task], timeout=5)
    task.cancel()
    calls_at_close = len(model_calls)
    await asyncio.sleep(0.05)

    assert done and task.cancelled()
    assert len(model_calls) == calls_at_close
    assert not participant.has_session("s")


async def test_a_commit_that_no_longer_continues_a_restored_session_retires_it(tmp_path: Path) -> None:
    hooks = Hooks()
    participant = AgentSessionParticipant(hooks)
    await participant.install(
        [
            AgentSessionRecord(
                session_key="kept", episode_id=EpisodeId(rollout_id="k"), session={}, boundary=None, episode=None
            ),
            AgentSessionRecord(
                session_key="dropped", episode_id=EpisodeId(rollout_id="d"), session={}, boundary=None, episode=None
            ),
        ],
        [EpisodeId(rollout_id="k"), EpisodeId(rollout_id="d")],
    )
    await participant.open_admission()
    controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
    await controller.prepare(CheckpointRequest(**control("c2")))
    # The controller continues only rollout k from this checkpoint.
    await controller.commit(
        CommitRequest(
            **control("c2", checkpoint_dir=str(tmp_path), episode_ids=[EpisodeId(rollout_id="k", attempt=1)])
        )
    )

    assert hooks.retired == ["dropped"]
    assert participant.has_session("kept") and not participant.has_session("dropped")


async def test_a_retire_cut_short_keeps_the_session_until_its_activation_stops() -> None:
    hooks = Hooks()
    participant = AgentSessionParticipant(hooks)
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    release = asyncio.Event()

    async def slow_to_stop() -> None:
        async with participant.activation("s", episode_id):
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                # Cleanup that outlives the cancellation.
                await release.wait()
                raise

    task = asyncio.create_task(slow_to_stop())
    await asyncio.sleep(0.01)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(participant.retire(episode_id), 0.05)
    tracked_after_timeout = participant.has_session("s")

    retry = asyncio.create_task(participant.retire(episode_id))
    await asyncio.sleep(0.01)
    waiting = not retry.done()
    release.set()
    await asyncio.wait_for(retry, 5)

    assert tracked_after_timeout and waiting
    assert task.cancelled() and not participant.has_session("s")
    assert hooks.retired == ["s"]


async def test_a_session_whose_retire_hook_was_cut_short_stays_blocking_until_a_retry_frees_it() -> None:
    release = asyncio.Event()

    class SlowRetireHooks(Hooks):
        async def retire_agent_session(self, session_key: str) -> None:
            await release.wait()
            await super().retire_agent_session(session_key)

    hooks = SlowRetireHooks()
    participant = AgentSessionParticipant(hooks)
    episode_id = EpisodeId(rollout_id="r")
    # Between activations: no task is running, so the retire goes straight to freeing the session.
    participant.open_session("s", episode_id)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(participant.retire(episode_id), 0.05)

    report = participant.readiness()
    exported = await participant.export(None)
    release.set()
    await participant.retire(episode_id)

    assert not report.ready and report.blockers == ["r"]
    assert exported == []
    assert hooks.retired == ["s"] and not participant.has_session("s")


async def test_a_seed_arriving_during_an_open_checkpoint_waits_for_resume() -> None:
    participant = AgentSessionParticipant(Hooks())
    await participant.close_admission(CLOSE)

    async def seed() -> None:
        async with participant.seeding("late", EpisodeId(rollout_id="r")):
            participant.open_session("late", EpisodeId(rollout_id="r"), seed=True)

    # The environment server's seeding step is replayable: its boundary stays before the seed until resume.
    seeding = asyncio.create_task(seed())
    await asyncio.sleep(0.01)
    waited = not seeding.done() and not participant.has_session("late") and participant.readiness().ready
    with pytest.raises(AdmissionClosedError):
        # A new session that is not a seed, such as a legacy /run, is still refused.
        participant.open_session("legacy", EpisodeId(rollout_id="other"))
    await participant.open_admission()
    await asyncio.wait_for(seeding, 5)

    assert waited and participant.has_session("late")


async def test_a_seed_in_progress_when_admission_closes_holds_up_prepare_and_is_exported() -> None:
    participant = AgentSessionParticipant(Hooks())
    built = asyncio.Event()

    async def seed() -> None:
        async with participant.seeding("s", EpisodeId(rollout_id="r")):
            # The agent builds the session's state, which can take a while, then registers it.
            await built.wait()
            participant.open_session("s", EpisodeId(rollout_id="r"), seed=True)

    seeding = asyncio.create_task(seed())
    await asyncio.sleep(0.01)
    await participant.close_admission(CLOSE)
    blocked = participant.readiness()
    built.set()
    await asyncio.wait_for(seeding, 5)

    assert not blocked.ready and blocked.blockers == ["r"]
    assert participant.readiness().ready
    # The environment may record its boundary after the seed before the commit, so the session is exported.
    assert [record.session_key for record in await participant.export(None)] == ["s"]


async def _until(condition: Any) -> None:
    async with asyncio.timeout(5):
        while not condition():
            await asyncio.sleep(0.01)


async def test_overlapping_seeds_of_one_session_each_hold_up_prepare_until_they_finish() -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    releases = [asyncio.Event(), asyncio.Event()]
    inside: list[int] = []

    async def seed(number: int) -> None:
        # A retried seed request enters while the first is still building the session.
        async with participant.seeding("s", episode_id):
            inside.append(number)
            await releases[number].wait()
            participant.open_session("s", episode_id, seed=True)

    seeds = [asyncio.create_task(seed(0)), asyncio.create_task(seed(1))]
    await _until(lambda: len(inside) == 2)
    releases[0].set()
    await seeds[0]
    while_second_builds = participant.readiness()
    second_blocks = not participant.ready()
    releases[1].set()
    await seeds[1]

    assert not while_second_builds.ready and while_second_builds.blockers == ["r"] and second_blocks
    assert participant.ready() and participant.has_session("s")


async def test_a_restore_is_refused_while_a_seed_is_in_progress() -> None:
    participant = AgentSessionParticipant(Hooks())
    entered, built = asyncio.Event(), asyncio.Event()

    async def seed() -> None:
        async with participant.seeding("s", EpisodeId(rollout_id="r")):
            entered.set()
            await built.wait()

    seeding = asyncio.create_task(seed())
    await asyncio.wait_for(entered.wait(), 5)
    with pytest.raises(ControlError, match="seeds in progress"):
        await participant.install([], [])
    built.set()
    await seeding

    await participant.install([], [])


async def test_a_closed_session_holds_up_prepare_until_its_activation_has_stopped() -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    started, cleaned_up = asyncio.Event(), asyncio.Event()

    async def activation() -> None:
        async with participant.activation("s", episode_id):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                # Cleanup that outlives the cancellation, such as a tool call that must be released.
                await cleaned_up.wait()
                raise

    task = asyncio.create_task(activation())
    await asyncio.wait_for(started.wait(), 5)
    closing = asyncio.create_task(participant.close_session("s"))
    await _until(lambda: task.cancelling())
    report = participant.readiness()
    ready = participant.ready()
    cleaned_up.set()
    await asyncio.wait_for(closing, 5)

    assert not report.ready and report.blockers == ["r"] and not ready
    assert task.cancelled() and not participant.has_session("s") and participant.ready()


async def test_a_close_after_a_retire_cut_short_does_not_cancel_the_activation_again() -> None:
    hooks = Hooks()
    participant = AgentSessionParticipant(hooks)
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    started, cleaned_up = asyncio.Event(), asyncio.Event()
    cancellations: list[str] = []

    async def activation() -> None:
        async with participant.activation("s", episode_id):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellations.append("first")
                try:
                    await cleaned_up.wait()
                except asyncio.CancelledError:
                    cancellations.append("again")
                raise

    task = asyncio.create_task(activation())
    await asyncio.wait_for(started.wait(), 5)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(participant.retire(episode_id), 0.05)
    closing = asyncio.create_task(participant.close_session("s"))
    await _until(lambda: participant._sessions["s"].closing)
    cleaned_up.set()
    await asyncio.wait_for(closing, 5)
    # The retired session stays until a retried retire frees it.
    left_for_the_retire = participant.has_session("s")
    await participant.retire(episode_id)

    assert cancellations == ["first"] and task.cancelled()
    assert left_for_the_retire
    assert hooks.retired == ["s"] and not participant.has_session("s")


async def test_a_retire_during_a_close_does_not_cancel_the_activation_again() -> None:
    hooks = Hooks()
    participant = AgentSessionParticipant(hooks)
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    started, cleaned_up = asyncio.Event(), asyncio.Event()
    cancellations: list[str] = []

    async def activation() -> None:
        async with participant.activation("s", episode_id):
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellations.append("first")
                try:
                    await cleaned_up.wait()
                except asyncio.CancelledError:
                    cancellations.append("again")
                raise

    task = asyncio.create_task(activation())
    await asyncio.wait_for(started.wait(), 5)
    closing = asyncio.create_task(participant.close_session("s"))
    await _until(lambda: task.cancelling())
    retiring = asyncio.create_task(participant.retire(episode_id))
    await _until(lambda: "s" not in participant._sessions or participant._sessions["s"].retired)
    cleaned_up.set()
    await asyncio.wait_for(asyncio.gather(closing, retiring), 5)

    assert cancellations == ["first"] and task.cancelled()
    assert hooks.retired == ["s"] and not participant.has_session("s")


async def test_a_restored_legacy_run_claimed_at_verify_is_no_longer_pending_and_survives_a_commit(
    tmp_path: Path,
) -> None:
    hooks = Hooks()
    participant = AgentSessionParticipant(hooks)
    restored_id = EpisodeId(rollout_id="r")
    await participant.install(
        [
            AgentSessionRecord(
                session_key="run:r", episode_id=restored_id, session={}, boundary=None, episode={"next": "verify"}
            )
        ],
        [restored_id],
    )
    await participant.open_admission()
    controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
    verifying, verified = asyncio.Event(), asyncio.Event()

    async def run() -> None:
        async with participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=1)) as legacy_run:
            assert legacy_run.continuation == {"next": "verify"}
            await legacy_run.boundary({"next": "verify"})
            async with legacy_run.step("replay"):
                verifying.set()
                await verified.wait()

    task = asyncio.create_task(run())
    await asyncio.wait_for(verifying.wait(), 5)
    pending = await participant.restored_pending()
    await controller.prepare(CheckpointRequest(**control("c2")))
    # The controller's scope leaves the rollout out; the claimed /run must not be retired for that.
    await controller.commit(CommitRequest(**control("c2", checkpoint_dir=str(tmp_path), episode_ids=[])))
    running = not task.done()
    await controller.resume(CheckpointRequest(**control("c2")))
    verified.set()
    await asyncio.wait([task], timeout=5)

    assert pending == []
    assert running and hooks.retired == []
    assert task.done() and not task.cancelled()


async def test_ready_does_not_build_the_readiness_report(monkeypatch: pytest.MonkeyPatch) -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("idle", EpisodeId(rollout_id="idle"))
    participant.open_session("s", episode_id)

    def built() -> None:
        raise AssertionError("ready() built the readiness report")

    monkeypatch.setattr(participant, "readiness", built)
    idle = participant.ready()
    async with participant.activation("s", episode_id):
        running = participant.ready()
    async with participant.seeding("seeding", EpisodeId(rollout_id="other")):
        seeding = participant.ready()

    assert idle and not running and not seeding


async def test_model_calls_and_activations_notify_only_while_a_checkpoint_is_open() -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    notified: list[bool] = []
    notify = participant.notify

    async def counting() -> None:
        notified.append(participant.accepting)
        await notify()

    participant.notify = counting
    async with participant.activation("s", episode_id) as activation:
        await activation.boundary(lambda: {"step": 1})
        async with activation.awaiting_model():
            pass
    outside_a_checkpoint = list(notified)
    await participant.close_admission(CLOSE)
    # An activation that ends while a checkpoint is open wakes the prepare that waits for it.
    async with participant.activation("s", episode_id):
        pass

    assert outside_a_checkpoint == []
    assert notified == [False]


class _SandboxAgent(SimpleResponsesAPIAgent):
    """An agent whose seed awaits while it builds the session, as one that starts a sandbox does."""

    checkpoint_sessions_supported = True
    building: Any = None
    release: Any = None
    events: Any = None

    async def responses(self, body=...):
        raise NotImplementedError

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        raise NotImplementedError

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        self.building.set()
        await self.release.wait()
        return AgentSessionState(request=body)

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        return {session_key: {} for session_key in session_keys}

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        pass

    async def retire_agent_session(self, session_key: str) -> None:
        self.events.append(f"freed {session_key}")
        self._free_session(session_key)


async def test_a_retire_waits_for_a_seed_in_progress_which_frees_what_it_built() -> None:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    agent = _SandboxAgent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent"), server_client=client
    )
    agent.building, agent.release, agent.events = asyncio.Event(), asyncio.Event(), []
    app = agent.setup_webserver()
    participant = agent.checkpoint_participant
    seed = {"agent_session_id": "s", "episode_id": {"rollout_id": "r"}, "task_id": {"taskset": "t", "task_id": "0"}}
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agent", headers={"Authorization": "Bearer t"}
    ) as http:
        seeding = asyncio.create_task(http.post("/v1/agent_sessions", json=seed))
        await asyncio.wait_for(agent.building.wait(), 5)

        async def retire() -> httpx.Response:
            reply = await http.post(
                "/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}])
            )
            agent.events.append("retire replied")
            return reply

        retiring = asyncio.create_task(retire())
        await _until(lambda: len(participant.retired) == 1)
        # The seed is still building its sandbox, so the retire cannot reply yet.
        done, _ = await asyncio.wait([retiring], timeout=0.2)
        agent.release.set()
        seeded = await asyncio.wait_for(seeding, 5)
        retired = await asyncio.wait_for(retiring, 5)

    assert not done
    assert retired.status_code == 200
    assert seeded.status_code == 409 and seeded.json()["error"]["code"] == "stale_attempt"
    assert agent.events == ["freed s", "retire replied"]
    assert "s" not in agent._session_records and not participant.has_session("s") and participant.ready()


class _OwnAppAgent(_UnsupportedAgent):
    """An agent whose setup_webserver builds its own app without the base one."""

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        app.post("/run")(self.run)
        return app


async def test_an_agent_that_builds_its_own_app_still_reports_its_runs_as_restarts(tmp_path: Path) -> None:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    agent = _OwnAppAgent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent"), server_client=client
    )
    agent.started, agent.release = asyncio.Event(), asyncio.Event()
    # As the server framework does with whatever app setup_webserver returns.
    app = agent.setup_webserver()
    agent.setup_checkpoint_participant(app)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agent", headers={"Authorization": "Bearer t"}
    ) as http:
        run = asyncio.create_task(
            http.post("/run", json={"responses_create_params": {"input": "x"}, "_ng_rollout_id": "r"})
        )
        await asyncio.wait_for(agent.started.wait(), 5)
        prepared = await http.post("/ng-control/v1/checkpoint/prepare", json=control("c1"))
        unattributed = await asyncio.wait_for(http.post("/run", json={"responses_create_params": {"input": "x"}}), 5)
        await http.post("/ng-control/v1/checkpoint/resume", json=control("c1"))
        await http.post("/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}]))
        stopped = run.done()
        agent.release.set()
        with pytest.raises(BaseException):
            await asyncio.wait_for(run, 5)

    assert prepared.json()["phase"] == "prepared"
    assert prepared.json()["report"]["restarts"] == ["r"] and prepared.json()["report"]["blockers"] == []
    assert unattributed.json()["error"]["code"] == "rollout_id_required"
    assert stopped


def test_the_checkpoint_participant_is_installed_once_per_app() -> None:
    agent, app = _unsupported_agent_app()
    participant = agent._restart_only
    # The server framework installs it again for any app setup_webserver returns.
    agent.setup_checkpoint_participant(app)

    assert agent._restart_only is participant
    assert [entry.cls for entry in app.user_middleware].count(RestartOnlyTrackingMiddleware) == 1
