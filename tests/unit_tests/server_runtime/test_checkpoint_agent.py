# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import Body, FastAPI
from pydantic import JsonValue

from nemo_gym.base_responses_api_agent import (
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client import ServerClient
from nemo_gym.config import BaseServerConfig
from nemo_gym.core import EpisodeId
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse, RestoredAgentSession
from nemo_gym.server_runtime._checkpoint.agent import (
    AgentSessionParticipant,
    AgentSessionRecord,
    RestartOnlyTrackingMiddleware,
)
from nemo_gym.server_runtime._checkpoint.control import (
    CheckpointRequest,
    CommitRequest,
    ParticipantControlPlane,
    RestoreRequest,
    RetireRequest,
)
from nemo_gym.server_runtime._checkpoint.errors import AdmissionClosedError, ControlError, StaleAttemptError
from nemo_gym.server_runtime._checkpoint.participant_workers import COORDINATOR_SOCKET_ENV
from tests.unit_tests.server_runtime.test_checkpoint_resources import SEED


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


async def test_commit_reply_names_the_episodes_agent_runs_drive(tmp_path: Path) -> None:
    """An agent session an environment server started belongs to that server; the agent owns only its own /run."""
    participant = AgentSessionParticipant(Hooks())
    controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
    run_id, session_id = EpisodeId(rollout_id="run"), EpisodeId(rollout_id="session", attempt=1)
    participant.open_session("session", session_id)
    verifying, session_parked, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def own_run() -> None:
        async with participant.legacy_run("run:run", run_id) as run:
            await run.boundary({"next": "verify"})
            async with run.step("replay"):
                verifying.set()
                await release.wait()

    async def environment_session() -> None:
        async with participant.activation("session", session_id) as activation:
            await session_parked.wait()
            await activation.boundary(lambda: {"step": 2})

    tasks = [asyncio.create_task(own_run()), asyncio.create_task(environment_session())]
    await asyncio.wait_for(verifying.wait(), 5)
    prepare = asyncio.create_task(controller.prepare(CheckpointRequest(**control())))
    await asyncio.sleep(0.01)
    session_parked.set()
    await prepare
    reply = await controller.commit(CommitRequest(**control(checkpoint_dir=str(tmp_path))))
    await controller.resume(CheckpointRequest(**control()))
    release.set()
    await asyncio.wait_for(asyncio.gather(*tasks), 5)

    assert reply["episode_ids"] == ["run", "session-a1"]
    assert reply["run_episode_ids"] == ["run"]


async def test_a_restored_run_episode_not_yet_reclaimed_is_still_named_in_the_commit_reply(tmp_path: Path) -> None:
    """A checkpoint between a restore and the replacement /run must still credit the episode to its agent's own /run."""
    source = AgentSessionParticipant(Hooks())
    source_controller = ParticipantControlPlane(source, instance_name="agent", lease_grace_seconds=60)
    episode_id = EpisodeId(rollout_id="run")
    verifying, release = asyncio.Event(), asyncio.Event()

    async def own_run() -> None:
        async with source.legacy_run("run:run", episode_id) as run:
            await run.boundary({"next": "verify"})
            async with run.step("replay"):
                verifying.set()
                await release.wait()

    task = asyncio.create_task(own_run())
    await asyncio.wait_for(verifying.wait(), 5)
    await source_controller.prepare(CheckpointRequest(**control()))
    await source_controller.commit(CommitRequest(**control(checkpoint_dir=str(tmp_path / "first"))))
    await source_controller.resume(CheckpointRequest(**control()))
    release.set()
    await asyncio.wait_for(task, 5)

    restored = AgentSessionParticipant(Hooks())
    controller = ParticipantControlPlane(restored, instance_name="agent", lease_grace_seconds=60)
    await controller.restore(
        RestoreRequest(**control("r1", checkpoint_dir=str(tmp_path / "first"), episode_ids=[episode_id]))
    )
    await controller.resume(CheckpointRequest(**control("r1")))
    replacement = EpisodeId(rollout_id="run", attempt=1)
    # No replacement /run has claimed the session yet.
    assert await restored.unclaimed() == [replacement]
    await controller.prepare(CheckpointRequest(**control("c2")))
    reply = await controller.commit(
        CommitRequest(**control("c2", checkpoint_dir=str(tmp_path / "second"), episode_ids=[replacement]))
    )

    assert reply["episode_ids"] == ["run-a1"]
    assert reply["run_episode_ids"] == ["run-a1"]


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


@pytest.mark.parametrize(
    "declarations, error",
    [
        ({"checkpoint_mode": "stateless"}, "checkpoint_mode='stateless'"),
        ({"checkpoint_mode": "exportd"}, "checkpoint_mode='exportd'"),
    ],
)
def test_an_agent_with_an_unknown_checkpoint_declaration_fails_at_startup(
    declarations: dict[str, Any], error: str
) -> None:
    annotations = {name: ClassVar[Any] for name in declarations}
    agent_type = type("Declared", (_UnsupportedAgent,), {**declarations, "__annotations__": annotations})
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    agent = agent_type(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent"), server_client=client
    )

    with pytest.raises(ValueError, match=re.escape(error)):
        agent.setup_webserver()


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
    assert unattributed.json()["error"]["code"] == "checkpoint_rollout_id_required"
    assert refused.json()["error"]["code"] == "checkpoint_restart_in_scope"
    assert committed.status_code == 200
    assert running and stopped
    assert await agent._restart_only.export(None) == []


async def test_status_reports_an_agents_checkpoint_mode() -> None:
    """A controller can refuse, before any checkpoint, an agent whose rollouts would start over after a crash."""
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    exported = _SandboxAgent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent"), server_client=client
    )
    _, restart_only_app = _unsupported_agent_app()
    statuses = {}
    for mode, app in (("exported", exported.setup_webserver()), ("restart_only", restart_only_app)):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://agent", headers={"Authorization": "Bearer t"}
        ) as http:
            statuses[mode] = (await http.get("/ng-control/v1/checkpoint/status")).json()

    # The same key and values a resources server reports, matching the agent's declared checkpoint_mode.
    assert statuses["exported"]["kind"] == statuses["restart_only"]["kind"] == "agent"
    assert statuses["exported"]["mode"] == exported.checkpoint_mode == "exported"
    assert statuses["restart_only"]["mode"] == "restart_only"


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
            session_key="agent-session", episode_id=episode_id, session={"k": 1}, boundary={"step": 3}, episode=None
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
    async with again.activation("agent-session", EpisodeId(rollout_id="r", attempt=2)) as activation:
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
    thread = coordinator._thread
    try:
        # The main process serves no requests: its workers forward control calls to this coordinator.
        assert coordinator.socket_path == socket_path
        assert coordinator.participant.kind == "agent"
        assert coordinator.participant.expected_workers == 2
        assert not [route for route in app.routes if getattr(route, "path", "").startswith("/ng-control/")]
    finally:
        coordinator.stop()

    # Nothing of the coordinator outlives the test: its thread, its event loop and its socket are gone.
    assert not thread.is_alive() and not os.path.exists(socket_path)


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


async def test_a_restored_legacy_run_claimed_at_verify_is_no_longer_unclaimed_and_survives_a_commit(
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
    pending = await participant.unclaimed()
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

    checkpoint_mode = "exported"
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
    assert seeded.status_code == 409 and seeded.json()["error"]["code"] == "checkpoint_stale_attempt"
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
    assert unattributed.json()["error"]["code"] == "checkpoint_rollout_id_required"
    assert stopped


def test_the_checkpoint_participant_is_installed_once_per_app() -> None:
    agent, app = _unsupported_agent_app()
    participant = agent._restart_only
    # The server framework installs it again for any app setup_webserver returns.
    agent.setup_checkpoint_participant(app)

    assert agent._restart_only is participant
    assert [entry.cls for entry in app.user_middleware].count(RestartOnlyTrackingMiddleware) == 1


async def test_after_a_restore_a_late_request_of_the_replaced_attempt_is_refused(tmp_path: Path) -> None:
    from nemo_gym.server_runtime._checkpoint.agent import RestartOnlyAgentParticipant
    from nemo_gym.server_runtime._checkpoint.store import write_participant_state

    record = AgentSessionRecord(
        episode_id=EpisodeId(rollout_id="r"), session_key="s", session={}, boundary={"step": 1}
    )
    write_participant_state(
        tmp_path, kind="agent", instance="agent", checkpoint_id="c0", records=[record.to_json_record()]
    )
    write_participant_state(tmp_path / "none", kind="agent", instance="agent", checkpoint_id="c0", records=[])
    # Neither participant holds a record of "unrecorded", and the restart-only agent holds none at all.
    scope = [{"rollout_id": "r"}, {"rollout_id": "unrecorded"}]
    sessions = AgentSessionParticipant(Hooks())
    restart_only = RestartOnlyAgentParticipant()
    for participant, directory in ((sessions, tmp_path), (restart_only, tmp_path / "none")):
        controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
        await controller.restore(RestoreRequest(**control("r1", checkpoint_dir=str(directory), episode_ids=scope)))
        await controller.resume(CheckpointRequest(**control("r1")))

    # A harness of a crashed attempt that outlived the crash opens a session or sends a request.
    for late in (EpisodeId(rollout_id="r"), EpisodeId(rollout_id="unrecorded")):
        with pytest.raises(StaleAttemptError):
            sessions.open_session(f"late-{late.rollout_id}", late)
        with pytest.raises(StaleAttemptError):
            restart_only.admit(late.capture_key)
    sessions.open_session("new", EpisodeId(rollout_id="unrecorded", attempt=1))
    assert restart_only.admit("r-a1") == "r-a1"


async def test_live_sessions_outside_the_commit_scope_are_not_exported() -> None:
    participant = AgentSessionParticipant(Hooks())
    participant.open_session("s", EpisodeId(rollout_id="r"))
    participant.open_session("t", EpisodeId(rollout_id="q"))

    # Another participant reported q as a restart, so the coordinator commits r alone.
    records = await participant.export([EpisodeId(rollout_id="r")])

    assert [record.session_key for record in records] == ["s"]


_SEED = {"agent_session_id": "s", "episode_id": {"rollout_id": "r"}, "task_id": {"taskset": "t", "task_id": "0"}}


async def test_a_restart_only_seed_that_arrives_after_its_rollout_was_retired_is_refused() -> None:
    agent, app = _unsupported_agent_app()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agent", headers={"Authorization": "Bearer t"}
    ) as http:
        retired = await http.post(
            "/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}])
        )
        # The environment server sent this seed just before its own retire.
        seeded = await http.post("/v1/agent_sessions", json=_SEED)

    assert retired.status_code == 200
    assert seeded.status_code == 409 and seeded.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert "s" not in agent._session_records


class _RestartOnlySandboxAgent(_UnsupportedAgent):
    """A restart-only agent whose seed awaits while it starts a sandbox, and whose close stops it."""

    building: Any = None
    events: Any = None

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        if body.agent_session_id == "slow":
            self.building.set()
            await self.release.wait()
        return AgentSessionState(request=body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        self.events.append(f"stopped {state.request.agent_session_id}")
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id)


async def test_a_restart_only_retire_frees_its_rollout_s_sessions_and_waits_for_a_seed_in_progress() -> None:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    agent = _RestartOnlySandboxAgent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent"), server_client=client
    )
    agent.building, agent.release, agent.events = asyncio.Event(), asyncio.Event(), []
    app = agent.setup_webserver()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://agent", headers={"Authorization": "Bearer t"}
    ) as http:
        live = await http.post("/v1/agent_sessions", json=_SEED | {"agent_session_id": "live"})
        # Each seed is a different episode's, with no session cookie of its own yet.
        http.cookies.clear()
        other = await http.post(
            "/v1/agent_sessions", json=_SEED | {"agent_session_id": "other", "episode_id": {"rollout_id": "q"}}
        )
        http.cookies.clear()
        seeding = asyncio.create_task(http.post("/v1/agent_sessions", json=_SEED | {"agent_session_id": "slow"}))
        await asyncio.wait_for(agent.building.wait(), 5)

        async def retire() -> httpx.Response:
            reply = await http.post(
                "/ng-control/v1/checkpoint/retire", json=control("retire", episode_ids=[{"rollout_id": "r"}])
            )
            agent.events.append("retire replied")
            return reply

        retiring = asyncio.create_task(retire())
        await _until(lambda: len(agent._restart_only.retired) == 1)
        # The seed is still starting its sandbox, so the retire cannot reply yet.
        done, _ = await asyncio.wait([retiring], timeout=0.2)
        agent.release.set()
        seeded = await asyncio.wait_for(seeding, 5)
        retired = await asyncio.wait_for(retiring, 5)

    assert live.status_code == other.status_code == 200
    assert not done
    assert retired.status_code == 200
    assert seeded.status_code == 409 and seeded.json()["error"]["code"] == "checkpoint_stale_attempt"
    assert sorted(agent.events[:2]) == ["stopped live", "stopped slow"] and agent.events[2:] == ["retire replied"]
    assert set(agent._session_records) == {"other"}


async def test_an_activation_of_a_restart_an_earlier_stage_reported_neither_blocks_nor_is_exported_until_resume(
    tmp_path: Path,
) -> None:
    """A prepare asked again names r as a restart; r's activation, still between boundaries, then stops blocking.

    It is not exported, even by a commit without a scope.
    The restarts last for the checkpoint they were named in: the next checkpoint waits for r again.
    """
    from nemo_gym.server_runtime._checkpoint.control import PrepareRequest

    participant = AgentSessionParticipant(Hooks())
    controller = ParticipantControlPlane(participant, instance_name="agent", lease_grace_seconds=60)
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    in_tool_call, released = asyncio.Event(), asyncio.Event()

    async def activation() -> None:
        async with participant.activation("s", episode_id) as running:
            await running.boundary(lambda: {"step": 0})
            in_tool_call.set()
            await released.wait()
            await running.boundary(lambda: {"step": 1})

    task = asyncio.create_task(activation())
    await asyncio.wait_for(in_tool_call.wait(), 5)
    missed = await controller.prepare(PrepareRequest(**control(deadline_ts=time.time() + 0.2)))
    prepared = await controller.prepare(PrepareRequest(**control(restarts=["r"])))
    commit = await controller.commit(CommitRequest(**control(checkpoint_dir=str(tmp_path))))
    await controller.resume(CheckpointRequest(**control()))
    next_checkpoint = await controller.prepare(PrepareRequest(**control("c2", deadline_ts=time.time() + 0.2)))
    released.set()
    await controller.resume(CheckpointRequest(**control("c2")))
    await asyncio.wait_for(task, 5)

    assert missed["phase"] == "preparing" and missed["report"]["blockers"] == ["r"]
    assert prepared["phase"] == "prepared"
    assert commit["episode_ids"] == []
    assert next_checkpoint["phase"] == "preparing" and next_checkpoint["report"]["blockers"] == ["r"]


class _EpisodeClient(ServerClient):
    """Serve an environment server's calls with in-process agent and resources servers.

    Resources calls reach the resources server's app under the episode's rollout prefix.
    Agent calls drive the agent participant the way an agent server does.
    Each activation records a boundary, makes one resources tool call, and records another boundary.
    """

    resources: Any
    agent: Any
    mcp_grant: bool = False
    session_key: str = ""
    tool_calls: int = 0

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> Any:
        from nemo_gym.core import current_rollout_id

        body = kwargs.get("json") or {}
        rollout = current_rollout_id()
        if server_name == "resources":
            if url_path == "/seed_session":
                reply = await self.resources.post(f"/ng-rollout/{rollout}/seed_session", json=SEED)
                payload: dict[str, Any] = {"resources_session_id": body["resources_session_id"]}
                if self.mcp_grant:
                    payload["resources_tools"] = {"server_name": "resources", "headers": {"token": "t"}}
                return _EpisodeReply(payload, reply.headers, dict(reply.cookies))
            reply = await self.resources.post(f"/ng-rollout/{rollout}{url_path}", json=body)
            return _EpisodeReply(reply.json(), reply.headers, {})
        episode_id = EpisodeId.from_capture_key(rollout)
        if url_path == "/v1/agent_sessions":
            self.session_key = body["agent_session_id"]
            async with self.agent.seeding(body["agent_session_id"], episode_id):
                self.agent.open_session(body["agent_session_id"], episode_id, seed=True)
            return _EpisodeReply({"agent_session_id": body["agent_session_id"]}, {}, {"agent": "a"})
        if url_path == "/v1/agent_sessions/close":
            await self.agent.close_session(body["agent_session_id"])
            return _EpisodeReply({"agent_session_id": body["agent_session_id"]}, {}, {})
        async with self.agent.activation(self.session_key, episode_id) as activation:
            await activation.boundary(lambda: {"step": 0})
            self.tool_calls += 1
            await self.resources.post(f"/ng-rollout/{rollout}/increment")
            await activation.boundary(lambda: {"step": 1})
        return _EpisodeReply(_RESPONSE, {}, {})

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


class _EpisodeReply:
    ok = True

    def __init__(self, payload: dict, headers: Any, cookies: dict[str, str]) -> None:
        self._body = json.dumps(payload).encode()
        self.headers = headers
        self.cookies = {name: MagicMock(value=value) for name, value in cookies.items()}

    async def read(self) -> bytes:
        return self._body


_RESPONSE = {
    "id": "response",
    "created_at": 0,
    "model": "model",
    "object": "response",
    "output": [],
    "tool_choice": "auto",
    "parallel_tool_calls": True,
    "tools": [],
}


@pytest.mark.parametrize("source", ["restart_only_resources_seed", "mcp_grant"])
async def test_an_episode_the_environment_reports_as_a_restart_does_not_hold_up_the_agent_or_resources_stage(
    source: str, tmp_path: Path
) -> None:
    """The environment server reports r as a restart while r's activation is inside a slow resources tool call.

    The agent and resources stages leave r out of their blockers, so prepare does not wait for the tool call.
    The commit leaves r out everywhere, and r keeps running and finishes after resume.
    """
    from omegaconf import OmegaConf

    from environment_servers.single_agent_turn.app import (
        SingleAgentTurnEnvironmentServer,
        SingleAgentTurnEnvironmentServerConfig,
    )
    from nemo_gym.config import AgentServerRef, ResourcesServerRef
    from nemo_gym.core import TaskId
    from nemo_gym.server_runtime import SingleAgentTurnRequest
    from nemo_gym.server_runtime._checkpoint import coordination
    from nemo_gym.server_runtime._checkpoint.control import install_participant
    from nemo_gym.server_runtime._checkpoint.environment import EnvironmentParticipant
    from nemo_gym.server_runtime.episode_types import MaterializedTask
    from nemo_gym.server_runtime.single_agent_turn_types import SingleAgentTurnTaskInput
    from tests.unit_tests.server_runtime.test_checkpoint_coordination import InProcessClient, Recorder
    from tests.unit_tests.server_runtime.test_checkpoint_resources import RestartOnlyServer, make_server, until

    resources_server, resources = (
        make_server(RestartOnlyServer) if source == "restart_only_resources_seed" else make_server()
    )
    resources_server.gate = asyncio.Event()
    agent = AgentSessionParticipant(Hooks())
    environment = EnvironmentParticipant(mode="exported")
    # The control routes and the data routes are one app, which holds one participant.
    apps: dict[str, Any] = {"resources": resources._transport.app}
    for name, participant in [("environment", environment), ("agent", agent), ("policy", Recorder("model", [], "p"))]:
        apps[name] = FastAPI()
        install_participant(apps[name], participant, auth_token="t", instance_name=name, lease_grace_seconds=60)
    global_config = OmegaConf.create(
        {
            "resources": {"resources_servers": {"test": {"host": "resources", "port": 8000, "entrypoint": "app.py"}}},
            "agent": {"responses_api_agents": {"test": {"host": "agent", "port": 8001, "entrypoint": "app.py"}}},
            "environment": {"environment_servers": {"test": {}}},
            "policy": {"responses_api_models": {"test": {}}},
        }
    )
    episode_client = _EpisodeClient(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        resources=resources,
        agent=agent,
        mcp_grant=source == "mcp_grant",
    )
    environment_server = SingleAgentTurnEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            name="environment",
            host="environment",
            port=8002,
            entrypoint="app.py",
            resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
            agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
            cleanup_timeout_seconds=10,
            resources_tool_transports=["mcp"] if source == "mcp_grant" else [],
        ),
        server_client=episode_client,
    )
    environment_server._checkpoint = environment
    request = SingleAgentTurnRequest(
        episode_id=EpisodeId(rollout_id="r"),
        task=MaterializedTask(
            task_id=TaskId(taskset="test:source", task_id="task"),
            task_input=SingleAgentTurnTaskInput(responses_create_params={"input": "task"}, task_data={}),
        ),
    )
    control_client = InProcessClient(
        head_server_config=BaseServerConfig(host="head", port=1), global_config_dict=global_config, apps=apps
    )

    async with resources:
        running = asyncio.create_task(environment_server.run_request(request))
        # Seeds also count as in flight; wait for the activation's held tool call before checkpointing.
        await until(lambda: episode_client.tool_calls == 1 and resources_server._checkpoint.inflight == 1)
        participants = await coordination.discover(control_client, auth_token="t")
        prepared = await coordination.prepare(participants, "c1", deadline_ts=time.time() + 2)
        scope = prepared.continued([request.episode_id])
        commits = (
            await coordination.commit(participants, "c1", str(tmp_path), scope, deadline_ts=time.time() + 5)
            if prepared.prepared
            else {}
        )
        in_tool_call_through_the_commit = not running.done() and episode_client.tool_calls == 1
        resources_server.gate.set()
        await coordination.resume(participants, "c1", deadline_ts=time.time() + 5)
        response = await asyncio.wait_for(running, 5)

    assert prepared.prepared, prepared.blockers()
    assert prepared.restarts() == {EpisodeId(rollout_id="r")}
    assert scope == []
    assert {server: reply["episode_ids"] for server, reply in commits.items()} == {
        "environment": [],
        "policy": [],
        "agent": [],
        "resources": [],
    }
    assert in_tool_call_through_the_commit
    assert response.result.reward == 1.0
