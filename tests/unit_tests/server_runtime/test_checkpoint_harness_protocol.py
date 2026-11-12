# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The checkpoint protocol for harnesses that borrow a sandbox: release, park signals, re-binding, restarts."""

import asyncio
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi import Body
from pydantic import JsonValue

from nemo_gym.agents.base_responses_api_agent import (
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.client.server_client import ServerClient
from nemo_gym.core.episode_types import EpisodeId
from nemo_gym.server_runtime._checkpoint.agent import AgentSessionParticipant, RestoredAgentSession
from nemo_gym.server_runtime._checkpoint.control import (
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    CommitRequest,
    ParticipantControlPlane,
    PrepareReport,
    PrepareRequest,
    ReleaseRequest,
    RestoreRequest,
)
from nemo_gym.server_runtime._checkpoint.errors import (
    ControlError,
    InvalidPhaseError,
    StaleAttemptError,
    StaleCheckpointError,
)
from nemo_gym.server_runtime._checkpoint.settings import CheckpointSettings, checkpoint_settings
from nemo_gym.server_runtime._checkpoint.steps import (
    CHECKPOINT_REBINDS_TOOL_ACCESSES_HEADER,
    CHECKPOINT_RESTART_HEADER,
    seed_rebinds_tool_accesses,
)
from nemo_gym.server_runtime.base_resources_server import BaseRunRequest, BaseVerifyResponse
from tests.unit_tests.server_runtime.test_checkpoint_participant_workers import (  # noqa: F401 (fixtures)
    agent_worker,
    coordinated,
    lost_coordinators,
    socket_dir,
)


CLOSE = CheckpointRequest(checkpoint_id="direct", deadline_ts=time.time() + 3600)


def control(checkpoint_id: str = "c1", **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + 5, **extra}


class Hooks:
    def __init__(self) -> None:
        self.exported: list[tuple[list[str], str]] = []
        self.released: list[frozenset[str]] = []

    async def export_agent_sessions(
        self, session_keys: list[str], *, checkpoint_id: str
    ) -> dict[str, dict[str, JsonValue]]:
        self.exported.append((list(session_keys), checkpoint_id))
        return {key: {"key": key} for key in session_keys}

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        pass

    async def retire_agent_session(self, session_key: str) -> None:
        pass

    async def delete_external_state(self, kept: frozenset[str]) -> None:
        self.released.append(kept)


class _Releasing(CheckpointParticipant):
    kind = "resources"
    mode = "exported"
    record_model = CheckpointRecord

    def __init__(self) -> None:
        super().__init__()
        self.released: list[frozenset[str]] = []
        self.exported_for: list[str] = []

    async def close_admission(self, request: CheckpointRequest) -> None:
        pass

    async def open_admission(self) -> None:
        pass

    def readiness(self) -> PrepareReport:
        return PrepareReport(ready=True)

    async def retire(self, episode_id: EpisodeId) -> None:
        pass

    async def export(self, episode_ids: Optional[list[EpisodeId]], *, checkpoint_id: str) -> list[CheckpointRecord]:
        self.exported_for.append(checkpoint_id)
        return []

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        raise NotImplementedError

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        pass

    async def release(self, kept: frozenset[str]) -> None:
        self.released.append(kept)


# -- release and the checkpoint ID ---------------------------------------------------------------------------


async def test_release_is_admitted_only_while_no_checkpoint_is_open(tmp_path: Path) -> None:
    participant = _Releasing()
    controller = ParticipantControlPlane(participant, instance_name="res", lease_grace_seconds=60)
    await controller.prepare(CheckpointRequest(**control("c2")))
    with pytest.raises(InvalidPhaseError):
        await controller.release(ReleaseRequest(**control("c1", kept=["c1"])))
    await controller.commit(CommitRequest(**control("c2", checkpoint_dir=str(tmp_path))))
    with pytest.raises(InvalidPhaseError):
        # The checkpoint just written is not published until resume: a list computed before it would drop it.
        await controller.release(ReleaseRequest(**control("c1", kept=["c1"])))
    await controller.resume(CheckpointRequest(**control("c2")))
    reply = await controller.release(ReleaseRequest(**control("c2", kept=["c1", "c2"])))

    assert reply == {"released": True}
    assert participant.released == [frozenset({"c1", "c2"})]
    # The commit's export learned the ID of the checkpoint it wrote.
    assert participant.exported_for == ["c2"]


async def test_a_release_delayed_past_a_later_checkpoint_is_refused_as_stale(tmp_path: Path) -> None:
    participant = _Releasing()
    controller = ParticipantControlPlane(participant, instance_name="res", lease_grace_seconds=60)
    for checkpoint_id in ("c1", "c2"):
        await controller.prepare(CheckpointRequest(**control(checkpoint_id)))
        await controller.commit(CommitRequest(**control(checkpoint_id, checkpoint_dir=str(tmp_path / checkpoint_id))))
        await controller.resume(CheckpointRequest(**control(checkpoint_id)))

    # c1's release, sent before c2 existed and delivered after c2 resumed, would delete what only c2 keeps.
    with pytest.raises(StaleCheckpointError, match="committed or restored a later checkpoint"):
        await controller.release(ReleaseRequest(**control("c1", kept=["c1"])))
    # One this process never resumed dates from before it started, so it is older than the checkpoint it holds.
    with pytest.raises(StaleCheckpointError, match="committed or restored a later checkpoint"):
        await controller.release(ReleaseRequest(**control("c0", kept=["c0"])))
    current = await controller.release(ReleaseRequest(**control("c2", kept=["c2"])))
    again = await controller.release(ReleaseRequest(**control("c2", kept=["c2"])))
    end_of_run = await controller.release(ReleaseRequest(**control("c2", kept=[])))

    assert current == again == end_of_run == {"released": True}
    assert participant.released == [frozenset({"c2"}), frozenset({"c2"}), frozenset()]


async def test_an_aborted_checkpoint_does_not_make_the_release_of_the_one_before_it_stale(tmp_path: Path) -> None:
    participant = _Releasing()
    controller = ParticipantControlPlane(participant, instance_name="res", lease_grace_seconds=60)
    await controller.prepare(CheckpointRequest(**control("c1")))
    await controller.commit(CommitRequest(**control("c1", checkpoint_dir=str(tmp_path))))
    await controller.resume(CheckpointRequest(**control("c1")))
    # c2 is prepared and aborted, and c3 never reaches this participant: neither kept anything.
    await controller.prepare(CheckpointRequest(**control("c2")))
    await controller.resume(CheckpointRequest(**control("c2")))
    await controller.resume(CheckpointRequest(**control("c3")))

    # The release that ends the run is admitted whether it names the last published checkpoint or the last resumed.
    for checkpoint_id in ("c1", "c2", "c3"):
        assert await controller.release(ReleaseRequest(**control(checkpoint_id, kept=[]))) == {"released": True}
    assert participant.released == [frozenset()] * 3


async def test_a_release_from_before_a_restore_is_refused_as_stale(tmp_path: Path) -> None:
    first = ParticipantControlPlane(_Releasing(), instance_name="res", lease_grace_seconds=60)
    for checkpoint_id in ("c1", "c2"):
        await first.prepare(CheckpointRequest(**control(checkpoint_id)))
        await first.commit(CommitRequest(**control(checkpoint_id, checkpoint_dir=str(tmp_path / checkpoint_id))))
        await first.resume(CheckpointRequest(**control(checkpoint_id)))
    # The process restarts and restores c2: it no longer remembers resuming c1.
    participant = _Releasing()
    controller = ParticipantControlPlane(participant, instance_name="res", lease_grace_seconds=60)
    before_restore = await controller.release(ReleaseRequest(**control("c1", kept=["c1", "c2"])))
    await controller.restore(RestoreRequest(**control("c2", checkpoint_dir=str(tmp_path / "c2"), episode_ids=[])))
    await controller.resume(CheckpointRequest(**control("c2")))

    with pytest.raises(StaleCheckpointError, match="committed or restored a later checkpoint"):
        await controller.release(ReleaseRequest(**control("c1", kept=["c1"])))
    assert before_restore == await controller.release(ReleaseRequest(**control("c2", kept=["c2"])))
    assert participant.released == [frozenset({"c1", "c2"}), frozenset({"c2"})]


def test_a_release_request_names_valid_checkpoint_ids() -> None:
    with pytest.raises(ValueError):
        ReleaseRequest(**control("c1", kept=["../escape"]))
    assert ReleaseRequest(**control("c1", kept=[])).kept == []


def test_state_dir_round_trips_through_the_checkpoint_block() -> None:
    settings = checkpoint_settings(
        {"checkpoint": {"enabled": True, "control_auth_token": "t", "state_dir": "/lustre/run/state"}}
    )
    assert settings.state_dir == "/lustre/run/state"
    assert CheckpointSettings(enabled=True, control_auth_token="t").state_dir is None


async def test_agent_export_and_release_reach_the_agent_s_hooks() -> None:
    hooks = Hooks()
    participant = AgentSessionParticipant(hooks)
    participant.open_session("s", EpisodeId(rollout_id="r"))
    await participant.export(None, checkpoint_id="c7")
    await participant.release(frozenset({"c7"}))

    assert hooks.exported == [(["s"], "c7")]
    assert hooks.released == [frozenset({"c7"})]


async def test_release_reaches_every_worker(socket_dir: str) -> None:  # noqa: F811
    released: list[frozenset[str]] = []

    def releasing_worker() -> tuple[AgentSessionParticipant, Any]:
        participant, app = agent_worker()
        participant.hooks.delete_external_state = lambda kept: _record(released, kept)
        return participant, app

    async with coordinated(socket_dir, releasing_worker) as deployment:
        await deployment.controller.release(ReleaseRequest(**control("c3", kept=["c3", "c2"])))

    assert released == [frozenset({"c2", "c3"})] * 2


async def _record(released: list[frozenset[str]], kept: frozenset[str]) -> None:
    released.append(kept)


async def test_release_fails_unless_every_expected_worker_answered(socket_dir: str) -> None:  # noqa: F811
    released: list[frozenset[str]] = []

    def releasing_worker() -> tuple[AgentSessionParticipant, Any]:
        participant, app = agent_worker()
        participant.hooks.delete_external_state = lambda kept: _record(released, kept)
        return participant, app

    async with coordinated(socket_dir, releasing_worker, workers=1, expected=2) as deployment:
        # A worker that has not joined may hold state the release would delete: the controller releases again.
        with pytest.raises(ControlError, match="1 of 2 agent workers answered release"):
            await deployment.controller.release(ReleaseRequest(**control("c3", kept=["c3"])))
        await deployment.join()
        await deployment.controller.release(ReleaseRequest(**control("c3", kept=["c3"])))

    # The worker that was there ran the hook both times, and the late one once.
    assert released == [frozenset({"c3"})] * 3


def test_an_agent_participant_refuses_an_export_hook_that_cannot_take_the_checkpoint_id() -> None:
    class _HooksWithoutTheId(Hooks):
        async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
            return {}

    # Code that builds the participant around its own hooks learns it here, not at the first commit.
    with pytest.raises(ValueError, match="_HooksWithoutTheId.export_agent_sessions must accept .*checkpoint_id"):
        AgentSessionParticipant(_HooksWithoutTheId())


async def test_status_reports_whether_an_agent_s_checkpoints_keep_state_outside_gym() -> None:
    class _KeepsSnapshots(_ExportingAgent):
        async def delete_external_state(self, kept: frozenset[str]) -> None:
            pass

    async def reported(agent_type: type[_ExportingAgent]) -> bool:
        async with _http(_borrowing_agent(agent_type=agent_type)) as http:
            return (await http.get("/ng-control/v1/checkpoint/status")).json()["keeps_external_state"]

    # Only an agent that overrides the hook keeps such state: a controller that never releases can tell.
    assert await reported(_KeepsSnapshots) is True
    assert await reported(_ExportingAgent) is False
    # A participant built around its own hooks keeps such state when they include the hook.
    assert AgentSessionParticipant(Hooks()).keeps_external_state


# -- the park signal ------------------------------------------------------------------------------------------


async def test_a_park_request_wakes_an_activation_beside_its_harness_and_resume_clears_it() -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    events: list[str] = []

    async def harness_activation() -> None:
        async with participant.activation("s", episode_id) as activation:
            for checkpoint in range(2):
                await activation.wait_park_requested()
                events.append(f"park {checkpoint}")
                await activation.boundary(lambda checkpoint=checkpoint: {"paused": checkpoint})
                events.append(f"resumed {checkpoint}")
                assert not activation.park_requested

    task = asyncio.create_task(harness_activation())
    await asyncio.sleep(0.01)
    assert events == [] and not participant.ready()
    for checkpoint in range(2):
        await participant.close_admission(CLOSE)
        await _until(lambda: participant.ready())
        [record] = await participant.export(None, checkpoint_id=f"c{checkpoint}")
        assert record.boundary == {"paused": checkpoint}
        await participant.open_admission()
        await _until(lambda checkpoint=checkpoint: f"resumed {checkpoint}" in events)
    await task

    assert events == ["park 0", "resumed 0", "park 1", "resumed 1"]


async def test_an_activation_that_starts_while_a_checkpoint_is_open_is_asked_to_park_at_once() -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    await participant.close_admission(CLOSE)

    async with participant.activation("s", episode_id) as activation:
        assert activation.park_requested
        await asyncio.wait_for(activation.wait_park_requested(), 1)
        parked = asyncio.create_task(activation.boundary(lambda: {"step": 0}))
        await _until(lambda: participant.ready())
        await participant.open_admission()
        await parked


@pytest.mark.parametrize("ending", ["retire", "close"])
async def test_a_retire_or_close_wakes_an_activation_waiting_for_a_park_request_and_refuses_it(ending: str) -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    waits: list[asyncio.Task] = []

    async def activation() -> None:
        async with participant.activation("s", episode_id) as running:
            # A task of its own, which the activation's cancellation does not reach: only the wake-up ends it.
            waits.append(asyncio.create_task(running.wait_park_requested()))
            await asyncio.wait(waits)

    task = asyncio.create_task(activation())
    await _until(lambda: waits)
    if ending == "retire":
        await participant.mark_retired([episode_id])
        await participant.retire(episode_id)
    else:
        await participant.close_session("s")

    assert task.done()
    with pytest.raises(StaleAttemptError):
        await asyncio.wait_for(waits[0], 1)


async def test_a_session_whose_episode_restarts_is_never_asked_to_park() -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id, restart=True)
    asked = asyncio.Event()

    async def harness_activation() -> None:
        async with participant.activation("s", episode_id) as activation:
            await activation.wait_park_requested()
            asked.set()

    task = asyncio.create_task(harness_activation())
    await participant.close_admission(CLOSE)
    report = participant.readiness()
    await asyncio.sleep(0.05)
    asked_during_checkpoint = asked.is_set()
    await participant.open_admission()
    await asyncio.sleep(0.01)
    task.cancel()

    # The running restart neither blocks the checkpoint nor is exported, and it is reported as a restart.
    assert report.ready and report.restarts == ["r"]
    assert await participant.export(None, checkpoint_id="c") == []
    assert not asked_during_checkpoint and not asked.is_set()


async def test_an_activation_queued_for_a_slot_does_not_hold_up_a_checkpoint() -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    slot = asyncio.Semaphore(0)
    after_slot: list[str] = []

    async def activation() -> None:
        async with participant.activation("s", episode_id) as running:
            async with running.queued():
                await slot.acquire()
            after_slot.append("running")

    task = asyncio.create_task(activation())
    await asyncio.sleep(0.01)
    await participant.close_admission(CLOSE)
    ready_while_queued = participant.ready()
    # A slot granted while the checkpoint is open: the activation stays parked until resume.
    slot.release()
    await asyncio.sleep(0.05)
    held = list(after_slot)
    await participant.open_admission()
    await task

    assert ready_while_queued
    assert held == []
    assert after_slot == ["running"]


async def test_park_requests_reach_the_activations_of_every_worker(socket_dir: str) -> None:  # noqa: F811
    async with coordinated(socket_dir, agent_worker) as deployment:
        waits = []
        for index, link in enumerate(deployment.links):
            episode_id = EpisodeId(rollout_id=f"r{index}")
            link.participant.open_session(f"s{index}", episode_id)
            waits.append(asyncio.create_task(_park_once(link.participant, f"s{index}", episode_id)))
        await asyncio.sleep(0.01)
        prepared = await deployment.controller.prepare(CheckpointRequest(**control()))
        await deployment.controller.resume(CheckpointRequest(**control()))
        await asyncio.wait_for(asyncio.gather(*waits), 5)

    assert prepared["phase"] == "prepared"


async def test_restarts_an_earlier_stage_reported_are_not_asked_to_park_on_any_worker(socket_dir: str) -> None:  # noqa: F811
    async with coordinated(socket_dir, agent_worker) as deployment:
        asked: list[str] = []
        harnesses = []
        for index, link in enumerate(deployment.links):
            episode_id = EpisodeId(rollout_id=f"r{index}")
            link.participant.open_session(f"s{index}", episode_id)
            harnesses.append(
                asyncio.create_task(_record_park_requests(link.participant, f"s{index}", episode_id, asked))
            )
        await asyncio.sleep(0.01)
        # The environment stage reported both episodes as restarts; every worker learns that before it closes.
        prepared = await deployment.controller.prepare(PrepareRequest(**control(restarts=["r0", "r1"])))
        await asyncio.sleep(0.05)
        asked_during_checkpoint = list(asked)
        await deployment.controller.resume(CheckpointRequest(**control()))
        for harness in harnesses:
            harness.cancel()
        await asyncio.gather(*harnesses, return_exceptions=True)

    assert prepared["phase"] == "prepared"
    assert asked_during_checkpoint == []


async def _record_park_requests(
    participant: AgentSessionParticipant, session_key: str, episode_id: EpisodeId, asked: list[str]
) -> None:
    async with participant.activation(session_key, episode_id) as activation:
        await activation.wait_park_requested()
        asked.append(session_key)
        await activation.boundary(lambda: {"paused": True})


async def _park_once(participant: AgentSessionParticipant, session_key: str, episode_id: EpisodeId) -> None:
    async with participant.activation(session_key, episode_id) as activation:
        await activation.wait_park_requested()
        await activation.boundary(lambda: {"paused": True})


async def _until(condition: Any) -> None:
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


# -- re-binding a restored session and per-session restarts -------------------------------------------------


class _ExportingAgent(SimpleResponsesAPIAgent):
    """An exported agent that keeps every hook default except its session state's export and restore."""

    checkpoint_mode = "exported"
    rebinds: bool = True
    rebound: Any = None

    async def responses(self, body=...):
        raise NotImplementedError

    async def run(self, body: BaseRunRequest = Body()) -> BaseVerifyResponse:
        raise NotImplementedError

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        return AgentSessionState(request=body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> Any:
        raise NotImplementedError

    async def export_agent_sessions(
        self, session_keys: list[str], *, checkpoint_id: str
    ) -> dict[str, dict[str, JsonValue]]:
        return {key: {"request": self._session_state(key).request.model_dump(mode="json")} for key in session_keys}

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        for session in sessions:
            request = AgentSeedSessionRequest.model_validate(
                session.session["request"] | {"episode_id": session.episode_id.model_dump(mode="json")}
            )
            self._install_restored_session(session.session_key, AgentSessionState(request=request))

    async def retire_agent_session(self, session_key: str) -> None:
        self._free_session(session_key)


class _BorrowingAgent(_ExportingAgent):
    """An exported agent whose sessions borrow a sandbox, optionally re-binding a restored one."""

    rebinds_tool_accesses = True

    def _seed_restarts(self, body: AgentSeedSessionRequest) -> bool:
        # A session without a borrowed sandbox runs in one the agent owns, which no checkpoint captures.
        return body.sandbox_access is None

    async def _rebind_restored_agent_session_state(
        self, state: AgentSessionState, body: AgentSeedSessionRequest
    ) -> AgentSessionState:
        if not self.rebinds:
            return await super()._rebind_restored_agent_session_state(state, body)
        self.rebound.append((state.request.sandbox_access, body.sandbox_access))
        return state


def _borrowing_agent(*, rebinds: bool = True, agent_type: type[_ExportingAgent] = _BorrowingAgent) -> _ExportingAgent:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    agent = agent_type(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent"), server_client=client
    )
    agent.rebinds, agent.rebound = rebinds, []
    return agent


def _seed(attempt: int, sandbox_id: Optional[str] = "s1") -> dict:
    seed: dict[str, Any] = {
        "agent_session_id": "session",
        "episode_id": {"rollout_id": "r", "attempt": attempt},
        "task_id": {"taskset": "t", "task_id": "0"},
    }
    if sandbox_id is not None:
        seed["sandbox_access"] = {
            "connection": {"kind": "direct", "provider_config_ref": "p", "descriptor": {"sandbox_id": sandbox_id}},
            "workdir": "/testbed",
        }
    return seed


async def _restored_agent(
    tmp_path: Path,
    *,
    rebinds: bool = True,
    agent_type: type[_ExportingAgent] = _BorrowingAgent,
    sandbox_id: Optional[str] = "s1",
) -> tuple[_ExportingAgent, httpx.AsyncClient]:
    """An agent restored from a checkpoint of ``session`` at attempt 0, as a fresh process after a crash."""
    first = _borrowing_agent(agent_type=agent_type)
    async with _http(first) as http:
        seeded = await http.post("/v1/agent_sessions", json=_seed(0, sandbox_id))
        assert seeded.status_code == 200, seeded.text
        for operation, body in (
            ("prepare", control("c1")),
            ("commit", control("c1", checkpoint_dir=str(tmp_path))),
            ("resume", control("c1")),
        ):
            assert (await http.post(f"/ng-control/v1/checkpoint/{operation}", json=body)).status_code == 200
    restored = _borrowing_agent(rebinds=rebinds, agent_type=agent_type)
    app = restored.setup_webserver()
    scope = [{"rollout_id": "r"}]
    async with _http(restored, app) as http:
        for operation, body in (
            ("restore", control("r1", checkpoint_dir=str(tmp_path), episode_ids=scope)),
            ("resume", control("r1")),
        ):
            reply = await http.post(f"/ng-control/v1/checkpoint/{operation}", json=body)
            assert reply.status_code == 200, reply.text
    return restored, _http(restored, app)


def _http(agent: _ExportingAgent, app: Any = None) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app if app is not None else agent.setup_webserver()),
        base_url="http://agent",
        headers={"Authorization": "Bearer t"},
    )


async def test_the_replacement_attempt_s_seed_re_binds_a_restored_session_to_its_new_sandbox(tmp_path: Path) -> None:
    agent, http = await _restored_agent(tmp_path)
    async with http:
        reseeded = await http.post("/v1/agent_sessions", json=_seed(1, "s2"))
        repeated = await http.post("/v1/agent_sessions", json=_seed(1, "s2"))
        conflicting = await http.post("/v1/agent_sessions", json=_seed(1, "s3"))

    assert reseeded.status_code == 200, reseeded.text
    assert reseeded.headers.get(CHECKPOINT_REBINDS_TOOL_ACCESSES_HEADER) == "1"
    assert CHECKPOINT_RESTART_HEADER not in reseeded.headers
    [(before, after)] = agent.rebound
    assert before.connection.descriptor == {"sandbox_id": "s1"} and after.connection.descriptor == {"sandbox_id": "s2"}
    assert agent._session_state("session").request.sandbox_access.connection.descriptor == {"sandbox_id": "s2"}
    # An identical retry reuses the re-bound session; once re-bound, another seed conflicts as before.
    assert repeated.status_code == 200
    assert conflicting.status_code == 409


async def test_an_unchanged_seed_of_the_replacement_attempt_still_re_binds_the_restored_session(
    tmp_path: Path,
) -> None:
    # The restored request names the same sandbox, as when the resources server lends the same one again.
    agent, http = await _restored_agent(tmp_path)
    async with http:
        reseeded = await http.post("/v1/agent_sessions", json=_seed(1, "s1"))

    assert reseeded.status_code == 200, reseeded.text
    assert len(agent.rebound) == 1


async def test_a_re_bind_is_refused_for_another_attempt_and_by_an_agent_that_does_not_re_bind(tmp_path: Path) -> None:
    agent, http = await _restored_agent(tmp_path)
    async with http:
        later_attempt = await http.post("/v1/agent_sessions", json=_seed(2, "s2"))
    refusing, refusing_http = await _restored_agent(tmp_path / "other", rebinds=False)
    async with refusing_http:
        refused = await refusing_http.post("/v1/agent_sessions", json=_seed(1, "s2"))

    assert later_attempt.status_code == 409 and agent.rebound == []
    assert agent.checkpoint_participant.restored_episode("session") == EpisodeId(rollout_id="r", attempt=1)
    assert refused.status_code == 409 and refusing.rebound == []


async def test_an_agent_that_does_not_re_bind_accepts_only_an_identical_seed_of_a_restored_session(
    tmp_path: Path,
) -> None:
    # The agent leaves the re-bind hook alone, as Simple Agent does: the default gate decides.
    # Its session is lent no sandbox: one that is would be a restart, and no checkpoint would hold it.
    agent, http = await _restored_agent(tmp_path, agent_type=_ExportingAgent, sandbox_id=None)
    async with http:
        identical = await http.post("/v1/agent_sessions", json=_seed(1, None))
    other, other_http = await _restored_agent(tmp_path / "other", agent_type=_ExportingAgent, sandbox_id=None)
    async with other_http:
        different = await other_http.post("/v1/agent_sessions", json=_seed(1, "s2"))

    assert identical.status_code == 200, identical.text
    assert CHECKPOINT_REBINDS_TOOL_ACCESSES_HEADER not in identical.headers
    assert CHECKPOINT_RESTART_HEADER not in identical.headers
    assert agent._session_state("session").request.sandbox_access is None
    # The restored session is there, so a seed that differs from it conflicts instead of starting a session.
    assert different.status_code == 409


def _mcp_seed(attempt: int, sandbox_id: Optional[str] = "s1") -> dict:
    grant = {
        "kind": "mcp",
        "name": "resources",
        "required": True,
        "connection": {"url": "http://resources:8000/mcp", "headers": {"X-NeMo-Gym-Session-Token": "token"}},
    }
    return _seed(attempt, sandbox_id) | {"tool_accesses": [grant]}


@pytest.mark.parametrize(
    ("agent_type", "seed", "restarts"),
    [
        # Without a lent sandbox, so only the grant decides: an agent that does not re-bind restarts a lent one too.
        (_ExportingAgent, _mcp_seed(0, sandbox_id=None), True),
        (_ExportingAgent, _seed(0, sandbox_id=None), False),
        (_BorrowingAgent, _mcp_seed(0), False),
    ],
    ids=["mcp_grant", "no_mcp_grant", "mcp_grant_re_bound"],
)
async def test_an_agent_that_cannot_take_new_mcp_grants_reports_the_restart_on_its_own_seed(
    agent_type: type[_ExportingAgent], seed: dict, restarts: bool
) -> None:
    agent = _borrowing_agent(agent_type=agent_type)
    async with _http(agent) as http:
        seeded = await http.post("/v1/agent_sessions", json=seed)
        # The checkpoint lands right after the seed reply, before the environment server could mark the episode.
        prepared = await http.post("/ng-control/v1/checkpoint/prepare", json=control("c1"))
        exported = await agent.checkpoint_participant.export(None, checkpoint_id="c1")
        await http.post("/ng-control/v1/checkpoint/resume", json=control("c1"))

    assert seeded.status_code == 200, seeded.text
    assert (seeded.headers.get(CHECKPOINT_RESTART_HEADER) == "1") is restarts
    assert prepared.json()["report"]["restarts"] == (["r"] if restarts else [])
    assert [record.session_key for record in exported] == ([] if restarts else ["session"])


def _agent_startup(agent_type: type[_ExportingAgent]) -> None:
    _borrowing_agent(agent_type=agent_type).setup_webserver()


def test_an_agent_whose_export_hook_cannot_take_the_checkpoint_id_fails_at_startup() -> None:
    class _ExportsWithoutTheId(_ExportingAgent):
        async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
            return {}

    class _ExportsWithKeywords(_ExportingAgent):
        async def export_agent_sessions(
            self, session_keys: list[str], **kwargs: Any
        ) -> dict[str, dict[str, JsonValue]]:
            return {}

    with pytest.raises(ValueError, match="_ExportsWithoutTheId.export_agent_sessions must accept .*checkpoint_id"):
        _agent_startup(_ExportsWithoutTheId)
    _agent_startup(_ExportsWithKeywords)
    _agent_startup(_ExportingAgent)


def test_an_agent_that_declares_re_binding_without_the_re_bind_hook_fails_at_startup() -> None:
    class _DeclaresOnly(_ExportingAgent):
        rebinds_tool_accesses = True

    with pytest.raises(ValueError, match="_DeclaresOnly declares rebinds_tool_accesses but does not override"):
        _agent_startup(_DeclaresOnly)
    _agent_startup(_BorrowingAgent)


async def test_a_session_the_agent_cannot_capture_is_a_restart_and_is_not_exported() -> None:
    agent = _borrowing_agent()
    async with _http(agent) as http:
        owned = await http.post("/v1/agent_sessions", json=_seed(0, sandbox_id=None))
        prepared = await http.post("/ng-control/v1/checkpoint/prepare", json=control("c1"))
        exported = await agent.checkpoint_participant.export(None, checkpoint_id="c1")
        await http.post("/ng-control/v1/checkpoint/resume", json=control("c1"))

    assert owned.headers.get(CHECKPOINT_RESTART_HEADER) == "1"
    assert seed_rebinds_tool_accesses(owned.headers)
    assert prepared.json()["report"]["restarts"] == ["r"]
    assert exported == []


async def test_a_session_lent_a_sandbox_is_a_restart_unless_the_agent_re_binds_restored_sessions() -> None:
    # The replacement attempt would seed it with another sandbox, which the default re-bind hook refuses.
    async with _http(_borrowing_agent(agent_type=_ExportingAgent)) as http:
        lent = await http.post("/v1/agent_sessions", json=_seed(0))
        prepared = await http.post("/ng-control/v1/checkpoint/prepare", json=control("c1"))
        await http.post("/ng-control/v1/checkpoint/resume", json=control("c1"))
    async with _http(_borrowing_agent()) as http:
        rebound = await http.post("/v1/agent_sessions", json=_seed(0))

    assert lent.status_code == 200, lent.text
    assert lent.headers.get(CHECKPOINT_RESTART_HEADER) == "1"
    assert prepared.json()["report"]["restarts"] == ["r"]
    assert rebound.status_code == 200 and CHECKPOINT_RESTART_HEADER not in rebound.headers


async def test_an_override_of_the_restart_hook_cannot_keep_an_mcp_session_from_restarting() -> None:
    class _AddsNoRestarts(_ExportingAgent):
        def _seed_restarts(self, body: AgentSeedSessionRequest) -> bool:
            return False

    async with _http(_borrowing_agent(agent_type=_AddsNoRestarts)) as http:
        granted = await http.post("/v1/agent_sessions", json=_mcp_seed(0, sandbox_id=None))
    async with _http(_borrowing_agent(agent_type=_AddsNoRestarts)) as http:
        plain = await http.post("/v1/agent_sessions", json=_seed(0, sandbox_id=None))

    assert granted.status_code == 200, granted.text
    # The hook only adds restarts: an agent that cannot take a new MCP token restarts whatever the hook says.
    assert granted.headers.get(CHECKPOINT_RESTART_HEADER) == "1"
    assert plain.status_code == 200 and CHECKPOINT_RESTART_HEADER not in plain.headers


def test_an_agent_reports_its_checkpoint_mode_in_effect() -> None:
    class _ConfiguredAgent(_BorrowingAgent):
        def checkpoint_mode_in_effect(self) -> str:
            return "restart_only"

    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    agent = _ConfiguredAgent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent"), server_client=client
    )
    agent.setup_webserver()

    assert agent.checkpoint_participant is None and agent._restart_only is not None
