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

from nemo_gym._checkpoint.agent import AgentSessionParticipant, RestoredAgentSession
from nemo_gym._checkpoint.control import (
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    CommitRequest,
    ParticipantControlPlane,
    PrepareReport,
    ReleaseRequest,
)
from nemo_gym._checkpoint.errors import InvalidPhaseError, StaleAttemptError
from nemo_gym._checkpoint.settings import CheckpointSettings, checkpoint_settings
from nemo_gym._checkpoint.steps import (
    CHECKPOINT_REBINDS_TOOLS_HEADER,
    CHECKPOINT_RESTART_HEADER,
    seed_rebinds_tools,
)
from nemo_gym.base_resources_server import BaseRunRequest, BaseVerifyResponse
from nemo_gym.base_responses_api_agent import (
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import ServerClient
from tests.unit_tests.test_checkpoint_participant_workers import (  # noqa: F401 (fixtures)
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

    async def release_external_state(self, kept: frozenset[str]) -> None:
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
        participant.hooks.release_external_state = lambda kept: _record(released, kept)
        return participant, app

    async with coordinated(socket_dir, releasing_worker) as deployment:
        await deployment.controller.release(ReleaseRequest(**control("c3", kept=["c3", "c2"])))

    assert released == [frozenset({"c2", "c3"})] * 2


async def _record(released: list[frozenset[str]], kept: frozenset[str]) -> None:
    released.append(kept)


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


async def test_a_retire_wakes_an_activation_waiting_for_a_park_request_and_refuses_it() -> None:
    participant = AgentSessionParticipant(Hooks())
    episode_id = EpisodeId(rollout_id="r")
    participant.open_session("s", episode_id)
    started = asyncio.Event()
    outcome: list[BaseException] = []

    async def waiting() -> None:
        async with participant.activation("s", episode_id) as activation:
            started.set()
            try:
                # Shielded so the retire's cancellation does not hide what the wait itself raises.
                await asyncio.shield(activation.wait_park_requested())
            except BaseException as error:
                outcome.append(error)
                raise

    task = asyncio.create_task(waiting())
    await started.wait()
    await participant.mark_retired([episode_id])
    await participant.retire(episode_id)

    assert task.done()
    assert outcome and isinstance(outcome[0], (StaleAttemptError, asyncio.CancelledError))


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


class _BorrowingAgent(SimpleResponsesAPIAgent):
    """An exported agent whose sessions borrow a sandbox, optionally re-binding a restored one."""

    checkpoint_mode = "exported"
    rebinds_tool_accesses = True
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


def _borrowing_agent(*, rebinds: bool = True) -> _BorrowingAgent:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    agent = _BorrowingAgent(
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


async def _restored_agent(tmp_path: Path, *, rebinds: bool = True) -> tuple[_BorrowingAgent, httpx.AsyncClient]:
    """An agent restored from a checkpoint of ``session`` at attempt 0, as a fresh process after a crash."""
    first = _borrowing_agent()
    async with _http(first) as http:
        seeded = await http.post("/v1/agent_sessions", json=_seed(0))
        assert seeded.status_code == 200, seeded.text
        for operation, body in (
            ("prepare", control("c1")),
            ("commit", control("c1", checkpoint_dir=str(tmp_path))),
            ("resume", control("c1")),
        ):
            assert (await http.post(f"/ng-control/v1/checkpoint/{operation}", json=body)).status_code == 200
    restored = _borrowing_agent(rebinds=rebinds)
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


def _http(agent: _BorrowingAgent, app: Any = None) -> httpx.AsyncClient:
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
    assert reseeded.headers.get(CHECKPOINT_REBINDS_TOOLS_HEADER) == "1"
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


async def test_a_session_the_agent_cannot_capture_is_a_restart_and_is_not_exported() -> None:
    agent = _borrowing_agent()
    async with _http(agent) as http:
        owned = await http.post("/v1/agent_sessions", json=_seed(0, sandbox_id=None))
        prepared = await http.post("/ng-control/v1/checkpoint/prepare", json=control("c1"))
        exported = await agent.checkpoint_participant.export(None, checkpoint_id="c1")
        await http.post("/ng-control/v1/checkpoint/resume", json=control("c1"))

    assert owned.headers.get(CHECKPOINT_RESTART_HEADER) == "1"
    assert seed_rebinds_tools(owned.headers)
    assert prepared.json()["report"]["restarts"] == ["r"]
    assert exported == []


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
