# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment, agent, and resources participants across several workers, over a real coordinator socket."""

import asyncio
import contextlib
import glob
import json
import os
import shutil
import socket
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI, Request
from pydantic import Field, JsonValue

import nemo_gym._checkpoint.participant_workers as participant_workers_module
import nemo_gym._checkpoint.workers as workers_module
from nemo_gym._checkpoint.agent import AgentSessionParticipant, AgentSessionRecord, RestoredAgentSession
from nemo_gym._checkpoint.control import (
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    CommitRequest,
    ForgetRequest,
    RestoreRequest,
    RetireRequest,
    dispatch_control,
)
from nemo_gym._checkpoint.environment import EnvironmentParticipant, task_digest
from nemo_gym._checkpoint.errors import (
    AdmissionClosedError,
    ControlError,
    ExportFailedError,
    RestartInScopeError,
    StaleAttemptError,
)
from nemo_gym._checkpoint.participant_workers import (
    COORDINATOR_SOCKET_ENV,
    CoordinatedServerParticipant,
    ParticipantWorkerLink,
)
from nemo_gym._checkpoint.resources import ResourcesParticipant, ResourcesSessionRecord
from nemo_gym._checkpoint.workers import WorkerCoordinator
from nemo_gym.base_responses_api_agent import (
    AGENT_SESSION_COOKIE_KEY,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSessionState,
    BaseResponsesAPIAgentConfig,
    SimpleResponsesAPIAgent,
)
from nemo_gym.episode_types import EpisodeId
from nemo_gym.mcp_auto_exposure import TOKEN_HEADER, maybe_auto_expose, session_token_serializer
from nemo_gym.server_utils import IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, ServerClient
from nemo_gym.session_routing import (
    ResourcesSessionLifetimes,
    SessionClaims,
    install_session_routing,
)
from resources_servers.example_session_state_mgmt.app import (
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
)


TASK = {"task": "t"}


def control(checkpoint_id: str = "c1", *, timeout: float = 5, **extra: Any) -> dict:
    return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + timeout, **extra}


class AgentHooks:
    def __init__(self) -> None:
        self.restored: list[RestoredAgentSession] = []

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        return {session_key: {"key": session_key} for session_key in session_keys}

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        self.restored.extend(sessions)

    async def retire_agent_session(self, session_key: str) -> None:
        pass


@pytest.fixture(autouse=True)
def lost_coordinators(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    # The real callback terminates the worker process, which here would be the test runner.
    lost: list[str] = []
    monkeypatch.setattr(workers_module, "_terminate_this_worker", lambda: lost.append("lost"))
    return lost


@pytest.fixture
def socket_dir() -> AsyncIterator[str]:
    # AF_UNIX paths must be short, so sockets live in their own directory under /tmp.
    directory = tempfile.mkdtemp(prefix="ngw-", dir="/tmp")
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


class Workers:
    """One coordinator and the links of the workers that joined it."""

    def __init__(self, coordinator: WorkerCoordinator, stack: AsyncExitStack, make: Callable[[], Any]) -> None:
        self.coordinator = coordinator
        self.controller = coordinator.controller
        self.participant: CoordinatedServerParticipant = coordinator.participant
        self.links: list[ParticipantWorkerLink] = []
        self._stack = stack
        self._make = make

    async def join(self) -> ParticipantWorkerLink:
        participant, app = self._make()
        link = ParticipantWorkerLink(participant, socket_path=self.coordinator.socket_path, app=app)
        # Registered first, so a worker whose join fails half way still closes its connection.
        self._stack.push_async_callback(link.disconnect)
        await link.connect()
        self.links.append(link)
        return link


@asynccontextmanager
async def coordinated(
    socket_dir: str, make: Callable[[], tuple[CheckpointParticipant, FastAPI]], *, workers: int = 2, expected: int = 2
) -> AsyncIterator[Workers]:
    """Workers whose participants ``make`` builds, joined to a fresh coordinator."""
    participant, _ = make()
    coordinator = WorkerCoordinator(
        CoordinatedServerParticipant(participant, expected_workers=expected),
        instance_name=participant.kind,
        lease_grace_seconds=60,
        socket_path=os.path.join(socket_dir, f"{uuid4().hex[:8]}.sock"),
    )
    server = await coordinator.serve()
    async with AsyncExitStack() as stack:
        deployment = Workers(coordinator, stack, make)
        for _ in range(workers):
            await deployment.join()
        try:
            yield deployment
        finally:
            await stack.aclose()
            server.close()
            await server.wait_closed()


def environment_worker() -> tuple[EnvironmentParticipant, FastAPI]:
    return EnvironmentParticipant(), FastAPI()


def agent_worker() -> tuple[AgentSessionParticipant, FastAPI]:
    app = FastAPI()
    # Session routing sets this to the worker's routing ID.
    app.state.nemo_gym_routing_id = uuid4().hex
    return AgentSessionParticipant(AgentHooks()), app


def routed(
    make: Callable[[], tuple[CheckpointParticipant, FastAPI]], claims_dir: str
) -> Callable[[], tuple[CheckpointParticipant, FastAPI]]:
    """Workers ``make`` builds, with the session claims and lifetimes session routing would give them."""

    def make_routed() -> tuple[CheckpointParticipant, FastAPI]:
        participant, app = make()
        routing_id = getattr(app.state, "nemo_gym_routing_id", None) or uuid4().hex
        app.state.nemo_gym_routing_id = routing_id
        if isinstance(participant, AgentSessionParticipant):
            holds = participant.has_session
        else:
            lifetimes = ResourcesSessionLifetimes(0.2, starts=("/seed_session",), ends=("/close_session",))
            app.state.nemo_gym_resources_session_lifetimes = lifetimes
            holds = lifetimes.holds
        app.state.nemo_gym_session_claims = SessionClaims(claims_dir, routing_id, holds)
        return participant, app

    return make_routed


def claimant(claims_dir: str, session_id: str) -> Optional[str]:
    """The worker the session's claim file names, as any worker of the server reads it."""
    return SessionClaims(claims_dir, "0" * 32, lambda _: False).claimant(session_id)


def records(checkpoint_dir: Path, kind: str) -> list[dict]:
    [path] = (checkpoint_dir / "gym" / kind / kind).glob("records-*.jsonl")
    return [json.loads(line) for line in path.read_text().splitlines()]


async def commit(
    deployment: Workers, checkpoint_dir: Path, capture_keys: list[str], checkpoint_id: str = "c1"
) -> None:
    """Commit with a scope of these capture keys; a restored episode is named by its replacement attempt."""
    prepared = await deployment.controller.prepare(CheckpointRequest(**control(checkpoint_id)))
    assert prepared["phase"] == "prepared", prepared
    episode_ids = [EpisodeId.from_capture_key(key).model_dump(mode="json") for key in capture_keys]
    await deployment.controller.commit(
        CommitRequest(**control(checkpoint_id, checkpoint_dir=str(checkpoint_dir), episode_ids=episode_ids))
    )


async def restore(deployment: Workers, checkpoint_dir: Path, episode_ids: list[dict], restore_id: str = "r1") -> None:
    await deployment.controller.restore(
        RestoreRequest(**control(restore_id, checkpoint_dir=str(checkpoint_dir), episode_ids=episode_ids))
    )
    await deployment.controller.resume(CheckpointRequest(**control(restore_id)))


async def parked_episode(participant: EnvironmentParticipant, rollout_id: str, boundary: dict) -> asyncio.Task:
    """An episode that records ``boundary`` and then waits in a replay step until cancelled."""
    started = asyncio.Event()

    async def run() -> None:
        episode_id = EpisodeId(rollout_id=rollout_id)
        participant.begin(episode_id, TASK, None)
        await participant.boundary(episode_id, boundary)
        async with participant.step(episode_id, "replay"):
            started.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await started.wait()
    return task


# -- prepare and commit ----------------------------------------------------------------------------


async def test_prepare_closes_every_worker_and_waits_for_the_episodes_of_all_of_them(socket_dir: str) -> None:
    async with coordinated(socket_dir, environment_worker) as deployment:
        first, second = deployment.links
        episode_id = EpisodeId(rollout_id="r")
        # Between boundaries on the second worker: a checkpoint must wait for it.
        second.participant.begin(episode_id, TASK, None)
        missed = await deployment.controller.prepare(CheckpointRequest(**control(timeout=0.3)))
        closed = [first.participant.steps.closed, second.participant.steps.closed]
        parking = asyncio.create_task(second.participant.boundary(episode_id, {"next": "verify"}))
        prepared = await deployment.controller.prepare(CheckpointRequest(**control()))
        await deployment.controller.resume(CheckpointRequest(**control()))
        await parking
        reopened = [first.participant.steps.closed, second.participant.steps.closed]

    assert missed["report"]["blockers"] == ["r"]
    assert closed == [True, True]
    assert prepared["phase"] == "prepared"
    assert reopened == [False, False]


async def test_commit_collects_the_records_of_every_worker(socket_dir: str, tmp_path: Path) -> None:
    async with coordinated(socket_dir, environment_worker) as deployment:
        first, second = deployment.links
        tasks = [
            await parked_episode(first.participant, "r1", {"next": "one"}),
            await parked_episode(second.participant, "r2", {"next": "two"}),
        ]
        await commit(deployment, tmp_path / "ckpt", ["r1", "r2"])
        await deployment.controller.resume(CheckpointRequest(**control()))
        for task in tasks:
            task.cancel()

    exported = {
        record["episode_id"]["rollout_id"]: record["boundary"] for record in records(tmp_path / "ckpt", "environment")
    }
    assert exported == {"r1": {"next": "one"}, "r2": {"next": "two"}}


class ReplyingEnvironmentParticipant(EnvironmentParticipant):
    """A participant kind that adds a field of its own to the commit reply."""

    def commit_reply(self, records: list[CheckpointRecord]) -> dict[str, Any]:
        return {"exported_episodes": sorted(record.episode_id.capture_key for record in records)}


async def test_the_commit_reply_carries_the_fields_the_participant_kind_adds(socket_dir: str, tmp_path: Path) -> None:
    """With several workers the coordinator answers the commit, so it must keep the kind's own reply fields."""

    def worker() -> tuple[EnvironmentParticipant, FastAPI]:
        return ReplyingEnvironmentParticipant(), FastAPI()

    async with coordinated(socket_dir, worker) as deployment:
        first, second = deployment.links
        tasks = [
            await parked_episode(first.participant, "r1", {"next": "one"}),
            await parked_episode(second.participant, "r2", {"next": "two"}),
        ]
        prepared = await deployment.controller.prepare(CheckpointRequest(**control()))
        assert prepared["phase"] == "prepared", prepared
        scope = [EpisodeId(rollout_id=key).model_dump(mode="json") for key in ("r1", "r2")]
        reply = await deployment.controller.commit(
            CommitRequest(**control(checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=scope))
        )
        await deployment.controller.resume(CheckpointRequest(**control()))
        for task in tasks:
            task.cancel()

    # The records come from both workers, and the reply field is built over all of them.
    assert reply["exported_episodes"] == ["r1", "r2"]


async def test_readiness_a_worker_changes_without_notifying_still_reaches_the_coordinator(socket_dir: str) -> None:
    def resources_worker() -> tuple[ResourcesParticipant, FastAPI]:
        return ResourcesParticipant(MagicMock(), "restart_only"), FastAPI()

    async with coordinated(socket_dir, resources_worker) as deployment:
        _, second = deployment.links
        second.participant.seeded("s", EpisodeId(rollout_id="r"))
        prepared = await deployment.controller.prepare(CheckpointRequest(**control()))
        # A session that ends records no notification; the worker re-reports it while the checkpoint is open.
        second.participant.ended("s")
        deadline = time.monotonic() + 5
        while deployment.participant.readiness().restarts and time.monotonic() < deadline:
            await asyncio.sleep(0.05)
        after = deployment.participant.readiness()
        await deployment.controller.resume(CheckpointRequest(**control()))

    assert prepared["phase"] == "prepared" and prepared["report"]["restarts"] == ["r"]
    assert after.restarts == []


async def test_marking_an_attempt_marks_the_sessions_each_worker_already_has(socket_dir: str) -> None:
    def resources_worker() -> tuple[ResourcesParticipant, FastAPI]:
        return ResourcesParticipant(MagicMock(), "exported"), FastAPI()

    async with coordinated(socket_dir, resources_worker) as deployment:
        _, second = deployment.links
        second.participant.seeded("s", EpisodeId(rollout_id="r"))
        # Only the mark: a retire that arrives after its deadline stops nothing on any worker.
        await deployment.participant.mark_retired([EpisodeId(rollout_id="r")])

    # The worker refuses the session, and readiness waits for a retire that frees it.
    assert second.participant._retired_session_rollouts == {"s": "r"}
    assert not second.participant.ready()


async def test_deleting_a_restored_episode_frees_its_unused_sessions_without_refusing_them_later(
    socket_dir: str,
) -> None:
    def resources_worker() -> tuple[ResourcesParticipant, FastAPI]:
        return ResourcesParticipant(AsyncMock(), "exported"), FastAPI()

    replacement = EpisodeId(rollout_id="r", attempt=1)
    async with coordinated(socket_dir, resources_worker) as deployment:
        first, second = deployment.links
        # A restore installed the session for the replacement, which never used it.
        await second.participant.install(
            [ResourcesSessionRecord(session_id="s", episode_id=EpisodeId(rollout_id="r"), state={})], [replacement]
        )
        # The session the replacement did use, on another worker, is not this deletion's to free.
        first.participant.seeded("used", replacement)
        await deployment.participant.delete_restored(replacement)

    # A commit whose scope leaves the episode out deletes it: nothing refuses the attempt or its session later.
    assert "s" not in second.participant._sessions and second.participant._retired_session_rollouts == {}
    assert "used" in first.participant._sessions and first.participant._retired_session_rollouts == {}


async def test_every_worker_s_restarts_reach_the_coordinator_which_refuses_a_scope_naming_one(
    socket_dir: str, tmp_path: Path
) -> None:
    def resources_worker() -> tuple[ResourcesParticipant, FastAPI]:
        return ResourcesParticipant(MagicMock(), "restart_only"), FastAPI()

    async with coordinated(socket_dir, resources_worker) as deployment:
        first, second = deployment.links
        first.participant.seeded("s1", EpisodeId(rollout_id="r"))
        second.participant.seeded("s2", EpisodeId(rollout_id="q", attempt=2))
        prepared = await deployment.controller.prepare(CheckpointRequest(**control()))
        with pytest.raises(RestartInScopeError):
            await deployment.controller.commit(
                CommitRequest(
                    **control(checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "q", "attempt": 2}])
                )
            )
        await deployment.controller.resume(CheckpointRequest(**control()))

    assert prepared["phase"] == "prepared" and prepared["report"]["restarts"] == ["q-a2", "r"]


async def test_a_lost_worker_blocks_the_checkpoint_until_resume(socket_dir: str, tmp_path: Path) -> None:
    async with coordinated(socket_dir, environment_worker) as deployment:
        _, second = deployment.links
        await deployment.controller.prepare(CheckpointRequest(**control()))
        # What it held is unknown now, so a commit could miss its episodes.
        await second.disconnect()
        report = deployment.participant.readiness()
        # Final for this checkpoint: retrying later could export a later cut.
        with pytest.raises(ExportFailedError, match="no longer ready"):
            await deployment.controller.commit(
                CommitRequest(**control(checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[]))
            )
        await deployment.controller.resume(CheckpointRequest(**control()))
        # uvicorn restarts the worker; the next checkpoint proceeds.
        await deployment.join()
        prepared = await deployment.controller.prepare(CheckpointRequest(**control("c2")))

    assert report.blockers == ["environment-workers-unreported:1", "environment-worker-lost"]
    assert prepared["phase"] == "prepared"


async def test_a_worker_lost_between_readiness_and_export_fails_the_commit(socket_dir: str, tmp_path: Path) -> None:
    async with coordinated(socket_dir, environment_worker) as deployment:
        first, second = deployment.links
        participant = deployment.participant
        tasks = [
            await parked_episode(first.participant, "r1", {"next": "one"}),
            await parked_episode(second.participant, "r2", {"next": "two"}),
        ]
        await deployment.controller.prepare(CheckpointRequest(**control()))
        export = participant.export

        async def export_after_the_worker_left(episode_ids: Optional[list[EpisodeId]]) -> list:
            # The commit has checked readiness; the worker holding r2 leaves before the export.
            await second.disconnect()
            async with asyncio.timeout(5):
                while len(participant.workers) > 1:
                    await asyncio.sleep(0.01)
            return await export(episode_ids)

        participant.export = export_after_the_worker_left
        request = CommitRequest(
            **control(checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": "r1"}, {"rollout_id": "r2"}])
        )
        with pytest.raises(ExportFailedError, match="worker"):
            await deployment.controller.commit(request)
        # Final for this checkpoint: retrying later could export a later cut.
        with pytest.raises(ExportFailedError):
            await deployment.controller.commit(request)
        await deployment.controller.resume(CheckpointRequest(**control()))
        for task in tasks:
            task.cancel()

    assert not (tmp_path / "ckpt").exists()


async def test_fewer_workers_than_configured_block_prepare(socket_dir: str) -> None:
    async with coordinated(socket_dir, environment_worker, workers=1) as deployment:
        missed = await deployment.controller.prepare(CheckpointRequest(**control(timeout=0.2)))
        late = await deployment.join()
        prepared = await deployment.controller.prepare(CheckpointRequest(**control()))

    assert missed["report"]["blockers"] == ["environment-workers-unreported:1"]
    # A worker that joins while a checkpoint is open closes at once.
    assert late.participant.steps.closed
    assert prepared["phase"] == "prepared"


# -- whole-episode state: claims ---------------------------------------------------------------------


@asynccontextmanager
async def restored_environment(socket_dir: str, tmp_path: Path) -> AsyncIterator[Workers]:
    """A fresh pair of workers that restored episode ``r`` at its boundary ``{"next": "verify"}``."""
    async with coordinated(socket_dir, environment_worker) as before:
        task = await parked_episode(before.links[0].participant, "r", {"next": "verify"})
        await commit(before, tmp_path / "ckpt", ["r"])
        task.cancel()
    async with coordinated(socket_dir, environment_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        yield after


async def test_a_restored_episode_is_claimed_by_exactly_one_worker(socket_dir: str, tmp_path: Path) -> None:
    async with restored_environment(socket_dir, tmp_path) as deployment:
        replacement = EpisodeId(rollout_id="r", attempt=1)
        participants = [link.participant for link in deployment.links]
        # Both workers receive a /run for the replacement at once; each claims before it begins.
        await asyncio.gather(*(participant.claim(replacement) for participant in participants))
        for participant in participants:
            participant.begin(replacement, TASK, None)
        continuations = [participant.continuation(replacement) for participant in participants]

    assert sorted(continuations, key=str) == [None, {"next": "verify"}]
    assert deployment.participant.restored == {}


async def test_an_unclaimed_restored_episode_is_exported_by_the_next_checkpoint(
    socket_dir: str, tmp_path: Path
) -> None:
    async with restored_environment(socket_dir, tmp_path) as deployment:
        await commit(deployment, tmp_path / "ckpt2", ["r-a1"], checkpoint_id="c2")
        await deployment.controller.resume(CheckpointRequest(**control("c2")))

    [record] = records(tmp_path / "ckpt2", "environment")
    assert record["episode_id"] == {"rollout_id": "r", "attempt": 1}
    assert record["boundary"] == {"next": "verify"}
    assert record["task_digest"] == task_digest(TASK)


async def test_a_claim_is_refused_while_a_checkpoint_is_open(socket_dir: str, tmp_path: Path) -> None:
    async with restored_environment(socket_dir, tmp_path) as deployment:
        replacement = EpisodeId(rollout_id="r", attempt=1)
        participant = deployment.links[0].participant
        await deployment.controller.prepare(CheckpointRequest(**control("c2")))
        with pytest.raises(ControlError) as refused:
            await participant.claim(replacement)
        held_while_open = sorted(deployment.participant.restored)
        await deployment.controller.resume(CheckpointRequest(**control("c2")))
        # The caller retries the episode after the checkpoint; the record is still there to claim.
        await participant.claim(replacement)
        participant.begin(replacement, TASK, None)

    # Refused with the code of a new episode during a checkpoint, so the caller retries it the same way.
    assert refused.value.code == AdmissionClosedError.code
    assert held_while_open == ["r-a1"]
    assert participant.continuation(replacement) == {"next": "verify"}


async def test_retiring_an_attempt_drops_its_unclaimed_restored_state(socket_dir: str, tmp_path: Path) -> None:
    async with restored_environment(socket_dir, tmp_path) as deployment:
        await deployment.controller.retire(
            RetireRequest(**control("x", episode_ids=[{"rollout_id": "r", "attempt": 1}]))
        )

    assert deployment.participant.restored == {}


async def test_a_restored_agent_run_episode_is_claimed_by_exactly_one_worker(socket_dir: str, tmp_path: Path) -> None:
    async with coordinated(socket_dir, agent_worker) as before:
        participant = before.links[0].participant
        started = asyncio.Event()

        async def run_episode() -> None:
            async with participant.legacy_run("run:r", EpisodeId(rollout_id="r")) as run:
                await run.boundary({"next": "loop"})
                async with run.step("replay"):
                    started.set()
                    await asyncio.Event().wait()

        task = asyncio.create_task(run_episode())
        await started.wait()
        await commit(before, tmp_path / "ckpt", ["r"])
        task.cancel()

    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        replacement = EpisodeId(rollout_id="r", attempt=1)
        continuations: list[Optional[dict]] = []
        release = asyncio.Event()

        async def replacement_run(participant: AgentSessionParticipant) -> None:
            async with participant.legacy_run("run:r", replacement) as run:
                continuations.append(run.continuation)
                await release.wait()

        runs = [asyncio.create_task(replacement_run(link.participant)) for link in after.links]
        await asyncio.sleep(0.1)
        release.set()
        await asyncio.gather(*runs)
        installed_by_hook = [len(link.participant.hooks.restored) for link in after.links]

    assert sorted(continuations, key=str) == [None, {"next": "loop"}]
    assert sorted(installed_by_hook) == [0, 1]


# -- multi-request sessions: placement and claims -------------------------------------------------


async def test_a_restored_agent_session_is_installed_on_one_worker_which_claims_it(
    socket_dir: str, tmp_path: Path
) -> None:
    async with coordinated(socket_dir, routed(agent_worker, socket_dir)) as before:
        first, second = before.links
        first.participant.open_session("one", EpisodeId(rollout_id="r1"))
        second.participant.open_session("two", EpisodeId(rollout_id="r2"))
        await commit(before, tmp_path / "ckpt", ["r1", "r2"])
        await before.controller.resume(CheckpointRequest(**control()))

    claims_dir = tempfile.mkdtemp(prefix="ngc-", dir="/tmp")
    try:
        async with coordinated(socket_dir, routed(agent_worker, claims_dir), expected=3) as after:
            await restore(after, tmp_path / "ckpt", [{"rollout_id": "r1"}, {"rollout_id": "r2"}])
            holders = {
                key: next(link.routing_id for link in after.links if link.participant.has_session(key))
                for key in ("one", "two")
            }
            claims = {key: claimant(claims_dir, key) for key in holders}
            # A worker that joins later reads the same claims.
            late = await after.join()
            late_reads = {key: late.app.state.nemo_gym_session_claims.claimant(key) for key in holders}
    finally:
        shutil.rmtree(claims_dir, ignore_errors=True)

    # The two sessions are spread over the two live workers, and each one's claim names its holder.
    assert len(set(holders.values())) == 2
    assert claims == holders and late_reads == holders


async def test_a_stateless_session_is_claimed_by_its_holder_after_every_restore(
    socket_dir: str, tmp_path: Path
) -> None:
    """A stateless session has no state, but its old cookie must reach a live worker after every restore."""

    def stateless_worker() -> tuple[ResourcesParticipant, FastAPI]:
        return ResourcesParticipant(SessionStates(), "stateless"), FastAPI()

    holders = []
    for index, (source, target) in enumerate([(None, "c1"), ("c1", "c2"), ("c2", None)]):
        claims_dir = tempfile.mkdtemp(prefix="ngc-", dir="/tmp")
        try:
            async with coordinated(socket_dir, routed(stateless_worker, claims_dir)) as deployment:
                if source is None:
                    deployment.links[0].participant.seeded("s", EpisodeId(rollout_id="r"))
                else:
                    await restore(deployment, tmp_path / source, [{"rollout_id": "r", "attempt": index - 1}])
                    [holder] = [link for link in deployment.links if link.participant.readiness().counts["sessions"]]
                    holders.append((holder.routing_id, claimant(claims_dir, "s")))
                if target is not None:
                    # The restored session's episode is still continued, by its replacement attempt.
                    await commit(deployment, tmp_path / target, ["r" if index == 0 else f"r-a{index}"])
                    await deployment.controller.resume(CheckpointRequest(**control()))
        finally:
            shutil.rmtree(claims_dir, ignore_errors=True)

    assert len(holders) == 2 and all(holder == claim for holder, claim in holders)


class SessionStates:
    """Resources session hooks over an in-memory map."""

    def __init__(self) -> None:
        self.states: dict[str, JsonValue] = {}

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        return {session_id: self.states[session_id] for session_id in session_ids if session_id in self.states}

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        self.states.update(states)

    async def retire_session_state(self, session_id: str) -> None:
        self.states.pop(session_id, None)


def resources_worker(*, routed: bool) -> Callable[[], tuple[ResourcesParticipant, FastAPI]]:
    def make() -> tuple[ResourcesParticipant, FastAPI]:
        app = FastAPI()
        if routed:
            app.state.nemo_gym_routing_id = uuid4().hex
        return ResourcesParticipant(SessionStates(), "exported"), app

    return make


async def test_sessions_of_a_single_process_server_are_claimed_by_their_holders_through_later_restores(
    socket_dir: str, tmp_path: Path
) -> None:
    """A single-process server claims nothing; after each restore, each session's claim names its new holder."""
    async with coordinated(socket_dir, resources_worker(routed=False)) as before:
        for index, link in enumerate(before.links):
            link.participant.hooks.states[f"s{index}"] = index
            link.participant.seeded(f"s{index}", EpisodeId(rollout_id=f"r{index}"))
        await commit(before, tmp_path / "ckpt1", ["r0", "r1"])
        await before.controller.resume(CheckpointRequest(**control()))

    rounds = []
    for source, target, attempt in (("ckpt1", "ckpt2", 0), ("ckpt2", None, 1)):
        claims_dir = tempfile.mkdtemp(prefix="ngc-", dir="/tmp")
        try:
            async with coordinated(socket_dir, routed(resources_worker(routed=False), claims_dir)) as deployment:
                episode_ids = [{"rollout_id": f"r{index}", "attempt": attempt} for index in range(2)]
                await restore(deployment, tmp_path / source, episode_ids, restore_id=f"restore-{source}")
                holders = {
                    session_id: link.routing_id
                    for link in deployment.links
                    for session_id in link.participant.hooks.states
                }
                rounds.append((holders, {session_id: claimant(claims_dir, session_id) for session_id in holders}))
                if target is not None:
                    await commit(deployment, tmp_path / target, ["r0-a1", "r1-a1"], checkpoint_id="c2")
                    await deployment.controller.resume(CheckpointRequest(**control("c2")))
        finally:
            shutil.rmtree(claims_dir, ignore_errors=True)

    for holders, claims in rounds:
        # Spread over both workers, and each session's claim names the worker that holds it.
        assert sorted(holders) == ["s0", "s1"] and len(set(holders.values())) == 2
        assert claims == holders


async def test_a_retired_attempt_is_refused_on_every_worker_and_one_that_joins_later_until_forget(
    socket_dir: str,
) -> None:
    async with coordinated(socket_dir, environment_worker) as deployment:
        await deployment.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r"}])))
        late = await deployment.join()
        refused = []
        for link in deployment.links:
            with pytest.raises(StaleAttemptError):
                link.participant.begin(EpisodeId(rollout_id="r"), TASK, None)
            refused.append(link)
        await deployment.controller.forget(ForgetRequest(**control(rollout_ids=["r"])))
        remaining = [len(link.retired) for link in deployment.links] + [len(deployment.participant.retired)]

    assert late in refused and len(refused) == len(deployment.links)
    assert remaining == [0] * (len(deployment.links) + 1)


def counter_config(
    num_workers: int, *, mcp: bool = False, window: float = 300
) -> StatefulCounterResourcesServerConfig:
    return StatefulCounterResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="resources",
        num_workers=num_workers,
        domain="agent",
        verified=False,
        expose_tools_over_mcp=mcp,
        session_close_retry_window_seconds=window,
    )


def checkpointing_client() -> ServerClient:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    return client


class CounterWorkers:
    """A coordinator and the counter server workers that joined it, with session routing."""

    def __init__(
        self, coordinator: WorkerCoordinator, stack: AsyncExitStack, routing_dir: str, *, mcp: bool, window: float
    ) -> None:
        self.coordinator = coordinator
        self.counters: list[StatefulCounterResourcesServer] = []
        self.apps: list[FastAPI] = []
        self._stack = stack
        self._routing_dir = routing_dir
        self._mcp = mcp
        self._window = window

    async def join(self) -> FastAPI:
        """Build a worker as uvicorn builds one, and start it, which links it to the coordinator."""
        counter = StatefulCounterResourcesServer(
            config=counter_config(2, mcp=self._mcp, window=self._window), server_client=checkpointing_client()
        )
        # The checkpoint link, MCP exposure as run_webserver adds it, then session routing outside the rest.
        app = counter.setup_webserver()
        routing: dict[str, Any] = {}
        if self._mcp:
            maybe_auto_expose(counter, app)
            routing = dict(mcp_token_header=TOKEN_HEADER, mcp_token_serializer=session_token_serializer(counter))
        key = counter.get_session_middleware_key()
        install_session_routing(
            app,
            socket_dir=self._routing_dir,
            session_cookie=key,
            secret_key=key,
            **routing,
            **counter.claimed_session_routing(),
        )
        await self._stack.enter_async_context(app.router.lifespan_context(app))
        self.counters.append(counter)
        self.apps.append(app)
        return app


@asynccontextmanager
async def counter_deployment(
    socket_dir: str, monkeypatch: pytest.MonkeyPatch, *, mcp: bool = False, window: float = 300
) -> AsyncIterator[CounterWorkers]:
    """A coordinator and two counter server workers, each built as uvicorn builds one."""
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, "1")
    coordinator = WorkerCoordinator(
        CoordinatedServerParticipant(ResourcesParticipant(MagicMock(), "exported"), expected_workers=2),
        instance_name="resources",
        lease_grace_seconds=60,
        socket_path=os.path.join(socket_dir, f"{uuid4().hex[:8]}.sock"),
    )
    server = await coordinator.serve()
    monkeypatch.setenv(COORDINATOR_SOCKET_ENV, coordinator.socket_path)
    routing_dir = tempfile.mkdtemp(prefix="ngr-", dir="/tmp")
    try:
        async with AsyncExitStack() as stack:
            deployment = CounterWorkers(coordinator, stack, routing_dir, mcp=mcp, window=window)
            for _ in range(2):
                await deployment.join()
            yield deployment
    finally:
        server.close()
        await server.wait_closed()
        shutil.rmtree(routing_dir, ignore_errors=True)


@asynccontextmanager
async def counter_workers(
    socket_dir: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[WorkerCoordinator, list[StatefulCounterResourcesServer], list[FastAPI]]]:
    """A coordinator and two counter server workers with session routing, each built as uvicorn builds one."""
    async with counter_deployment(socket_dir, monkeypatch) as deployment:
        yield deployment.coordinator, deployment.counters, deployment.apps


def http(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://worker")


async def test_a_restored_resources_session_serves_a_request_with_its_old_cookie_on_any_worker(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real counter servers with session routing: the old cookie reaches the worker that holds the session."""
    async with counter_workers(socket_dir, monkeypatch) as (coordinator, _, apps):
        cookies = []
        for index, app in enumerate(apps):
            async with http(app) as worker:
                seeded = await worker.post(f"/ng-rollout/r{index}/seed_session", json={"initial_count": 10 * index})
                await worker.post(f"/ng-rollout/r{index}/increment_counter", json={"count": 1})
                cookies.append(dict(seeded.cookies))
        await commit(coordinator, tmp_path / "ckpt", ["r0", "r1"])
        await coordinator.controller.resume(CheckpointRequest(**control()))
    exported = records(tmp_path / "ckpt", "resources")

    async with counter_workers(socket_dir, monkeypatch) as (coordinator, counters, apps):
        await restore(coordinator, tmp_path / "ckpt", [{"rollout_id": "r0"}, {"rollout_id": "r1"}])
        counts = []
        for index, cookie in enumerate(cookies):
            for app in apps:
                async with http(app) as worker:
                    # The restored attempt's prefix: a request with the cookie alone names no attempt.
                    value = await worker.post(f"/ng-rollout/r{index}-a1/get_counter_value", cookies=cookie)
                    counts.append(value.json()["count"])
        held = [sorted(counter.session_id_to_counter.values()) for counter in counters]

    # Records name sessions by session ID only.
    assert len(exported) == 2 and all("owner" not in record for record in exported)
    # Either worker serves either restored session: one directly, the other by forwarding to its holder.
    assert counts == [1, 1, 11, 11]
    # The two sessions went to different workers, and no request ran against a worker without the state.
    assert sorted(held) == [[1], [11]]


async def test_sessions_of_a_single_process_server_restore_onto_several_workers(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint taken with one worker: its cookies still reach their session's state on several workers.

    The MCP token, which names the session the same way, is covered in the test after this one.
    """
    single = StatefulCounterResourcesServer(config=counter_config(1), server_client=checkpointing_client())
    bearer = {"authorization": "Bearer t"}
    cookies = []
    app = single.setup_webserver()
    for index in range(4):
        # A client per rollout, so each seed starts its own session.
        async with http(app) as rollout:
            seeded = await rollout.post(f"/ng-rollout/r{index}/seed_session", json={"initial_count": 10 * index})
            cookies.append(dict(seeded.cookies))
    async with http(app) as server:
        await server.post("/ng-control/v1/checkpoint/prepare", json=control(), headers=bearer)
        committed = await server.post(
            "/ng-control/v1/checkpoint/commit",
            json=control(
                checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[{"rollout_id": f"r{i}"} for i in range(4)]
            ),
            headers=bearer,
        )
    assert committed.status_code == 200, committed.text
    exported = records(tmp_path / "ckpt", "resources")

    async with counter_workers(socket_dir, monkeypatch) as (coordinator, counters, apps):
        await restore(coordinator, tmp_path / "ckpt", [{"rollout_id": f"r{index}"} for index in range(4)])
        counts = []
        for index, cookie in enumerate(cookies):
            prefix = f"/ng-rollout/r{index}-a1"
            for app in apps:
                async with http(app) as worker:
                    await worker.post(f"{prefix}/increment_counter", json={"count": 1}, cookies=cookie)
                    counts.append((await worker.post(f"{prefix}/get_counter_value", cookies=cookie)).json()["count"])
        held = [sorted(counter.session_id_to_counter.values()) for counter in counters]
        holders = {
            session_id: app.state.nemo_gym_routing_id
            for app, counter in zip(apps, counters)
            for session_id in counter.session_id_to_counter
        }
        claims = {session_id: apps[0].state.nemo_gym_session_claims.claimant(session_id) for session_id in holders}

    # Each worker incremented each session once, whichever worker held it.
    assert counts == [1, 2, 11, 12, 21, 22, 31, 32]
    # Spread evenly, one session at a time, and no request ran against a worker without the state.
    assert sorted(len(values) for values in held) == [2, 2]
    assert sorted(value for values in held for value in values) == [2, 12, 22, 32]
    # Each session's claim names the worker that holds it.
    assert claims == holders and set(holders) == {record["session_id"] for record in exported}


async def mcp_tool_call(app: FastAPI, token: str, name: str, arguments: dict) -> httpx.Response:
    """Call a tool as an agent harness does: the MCP session token only, never the session cookie."""
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
    headers = {"accept": "application/json, text/event-stream", TOKEN_HEADER: token}
    async with http(app) as worker:
        return await worker.post("/mcp", headers=headers, json=body)


def tool_result(response: httpx.Response) -> dict:
    assert response.status_code == 200, response.text
    result = response.json()["result"]
    assert result.get("isError") is not True, result
    return json.loads(result["content"][0]["text"])


async def test_after_a_restore_onto_several_workers_every_worker_refuses_the_replaced_attempt_s_mcp_token(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every worker forwards an old token to the worker that holds its session, which refuses it.

    A worker that joins after the restore refuses it too, and a seed under the new attempt gets a token that works.
    """
    async with counter_deployment(socket_dir, monkeypatch, mcp=True) as before:
        tokens, cookies = [], []
        for index, app in enumerate(before.apps):
            async with http(app) as worker:
                seeded = await worker.post(f"/ng-rollout/r{index}/seed_session", json={"initial_count": 10 * index})
            tokens.append(seeded.json()["mcp"]["headers"][TOKEN_HEADER])
            cookies.append(dict(seeded.cookies))
            tool_result(await mcp_tool_call(app, tokens[-1], "increment_counter", {"count": 1}))
        await commit(before.coordinator, tmp_path / "ckpt", ["r0", "r1"])
        await before.coordinator.controller.resume(CheckpointRequest(**control()))

    async with counter_deployment(socket_dir, monkeypatch, mcp=True) as after:
        await restore(after.coordinator, tmp_path / "ckpt", [{"rollout_id": "r0"}, {"rollout_id": "r1"}])
        await after.join()
        stale = [
            await mcp_tool_call(app, token, "increment_counter", {"count": 100})
            for token in tokens
            for app in after.apps
        ]
        held = sorted(value for counter in after.counters for value in counter.session_id_to_counter.values())
        async with http(after.apps[2]) as late:
            reseeded = await late.post("/ng-rollout/r1-a1/seed_session", json={"initial_count": 0}, cookies=cookies[1])
        token = reseeded.json()["mcp"]["headers"][TOKEN_HEADER]
        tool_result(await mcp_tool_call(after.apps[0], token, "increment_counter", {"count": 1}))
        continued = tool_result(await mcp_tool_call(after.apps[1], token, "get_counter_value", {}))["count"]

    assert len(stale) == 6
    assert all(
        reply.status_code == 409 and reply.json()["error"]["code"] == "checkpoint_stale_attempt" for reply in stale
    )
    # No stale call changed a restored session.
    assert held == [1, 11]
    assert continued == 12


async def test_a_commit_that_no_longer_continues_a_restored_episode_retires_it(
    socket_dir: str, tmp_path: Path
) -> None:
    async with restored_environment(socket_dir, tmp_path) as deployment:
        await commit(deployment, tmp_path / "ckpt2", [], checkpoint_id="c2")
        await deployment.controller.resume(CheckpointRequest(**control("c2")))
        worker = deployment.links[0].participant
        # Retired: a later /run for attempt 1 finds nothing to claim and starts from its input.
        await worker.claim(EpisodeId(rollout_id="r", attempt=1))
        worker.begin(EpisodeId(rollout_id="r", attempt=1), TASK, None)

        assert deployment.participant.restored == {}
        assert worker.continuation(EpisodeId(rollout_id="r", attempt=1)) is None


async def test_a_commit_that_no_longer_continues_a_restored_agent_session_retires_it_on_its_worker(
    socket_dir: str, tmp_path: Path
) -> None:
    async with coordinated(socket_dir, agent_worker) as before:
        first, second = before.links
        first.participant.open_session("kept", EpisodeId(rollout_id="k"))
        second.participant.open_session("dropped", EpisodeId(rollout_id="d"))
        await commit(before, tmp_path / "ckpt", ["k", "d"])
        await before.controller.resume(CheckpointRequest(**control()))

    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "k"}, {"rollout_id": "d"}])
        await after.controller.prepare(CheckpointRequest(**control("c2")))
        # The controller continues only rollout k, as its replacement attempt.
        await after.controller.commit(
            CommitRequest(
                **control(
                    "c2", checkpoint_dir=str(tmp_path / "ckpt2"), episode_ids=[{"rollout_id": "k", "attempt": 1}]
                )
            )
        )
        await after.controller.resume(CheckpointRequest(**control("c2")))
        held = sorted(key for link in after.links for key in ("kept", "dropped") if link.participant.has_session(key))

    assert held == ["kept"]


# -- worker messages, joins, and exports -----------------------------------------------------------


async def test_worker_messages_carry_what_the_checkpoint_writer_carries() -> None:
    import math

    reader = asyncio.StreamReader()
    # A -inf logprob and an integer beyond 64 bits, which orjson would turn into null or refuse.
    reader.feed_data(workers_module._frame({"logprobs": [-math.inf, -0.5], "nan": math.nan, "hash": 2**70}))

    message = await workers_module._read_frame(reader)

    assert message["logprobs"] == [-math.inf, -0.5] and math.isnan(message["nan"]) and message["hash"] == 2**70


async def test_a_reply_that_cannot_be_framed_is_answered_with_a_typed_error() -> None:
    reader = asyncio.StreamReader()
    reader.feed_data(
        workers_module._frame_reply({"id": 7, "kind": "export"}, {"reply_to": 7, "ok": True, "body": object()})
    )

    reply = await workers_module._read_frame(reader)

    assert reply["reply_to"] == 7 and reply["ok"] is False
    assert reply["error"]["code"] == "checkpoint_invalid_state"


async def test_a_worker_that_joins_while_a_restore_installs_reads_the_claims_it_wrote(
    socket_dir: str, tmp_path: Path
) -> None:
    async with coordinated(socket_dir, agent_worker) as before:
        first, second = before.links
        first.participant.open_session("one", EpisodeId(rollout_id="r1"))
        second.participant.open_session("two", EpisodeId(rollout_id="r2"))
        await commit(before, tmp_path / "ckpt", ["r1", "r2"])
        await before.controller.resume(CheckpointRequest(**control()))

    installing, joined = asyncio.Event(), asyncio.Event()

    class SlowRestoreHooks(AgentHooks):
        async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
            installing.set()
            # Held until the late worker has joined, so it joins while the install runs.
            await joined.wait()
            await super().restore_agent_sessions(sessions)

    def slow_agent_worker() -> tuple[AgentSessionParticipant, FastAPI]:
        app = FastAPI()
        app.state.nemo_gym_routing_id = uuid4().hex
        return AgentSessionParticipant(SlowRestoreHooks()), app

    claims_dir = tempfile.mkdtemp(prefix="ngc-", dir="/tmp")
    try:
        async with coordinated(socket_dir, routed(slow_agent_worker, claims_dir), expected=3) as after:
            restoring = asyncio.create_task(
                after.controller.restore(
                    RestoreRequest(
                        **control(
                            "r1",
                            checkpoint_dir=str(tmp_path / "ckpt"),
                            episode_ids=[{"rollout_id": "r1"}, {"rollout_id": "r2"}],
                        )
                    )
                )
            )
            await installing.wait()
            # Joins while the install runs, before any claim is rewritten.
            late = await after.join()
            joined.set()
            await restoring
            await after.controller.resume(CheckpointRequest(**control("r1")))
            holders = {
                key: next(link.routing_id for link in after.links if link.participant.has_session(key))
                for key in ("one", "two")
            }
            late_reads = {key: late.app.state.nemo_gym_session_claims.claimant(key) for key in holders}
    finally:
        shutil.rmtree(claims_dir, ignore_errors=True)

    # Nothing to adopt on joining: claims are shared files, so the late worker forwards to each holder.
    assert late_reads == holders


async def test_restoring_many_sessions_rewrites_their_claims_without_holding_up_the_event_loop(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    keys = [f"s{index}" for index in range(200)]
    async with coordinated(socket_dir, agent_worker) as before:
        for key in keys:
            before.links[0].participant.open_session(key, EpisodeId(rollout_id=key))
        await commit(before, tmp_path / "ckpt", keys)
        await before.controller.resume(CheckpointRequest(**control()))

    real_replace = os.replace

    def slow_replace(source: str, target: str) -> None:
        # Each claim takes a few milliseconds to replace, as on a slow or busy filesystem.
        if "session-claims" in target:
            time.sleep(0.005)
        real_replace(source, target)

    claims_dir = tempfile.mkdtemp(prefix="ngc-", dir="/tmp")
    gaps: list[float] = []
    try:
        async with coordinated(socket_dir, routed(agent_worker, claims_dir)) as after:
            monkeypatch.setattr(os, "replace", slow_replace)
            stop = asyncio.Event()

            async def tick() -> None:
                last = time.monotonic()
                while not stop.is_set():
                    await asyncio.sleep(0.005)
                    now = time.monotonic()
                    gaps.append(now - last)
                    last = now

            ticker = asyncio.create_task(tick())
            await after.controller.restore(
                RestoreRequest(
                    **control(
                        "r1",
                        timeout=30,
                        checkpoint_dir=str(tmp_path / "ckpt"),
                        episode_ids=[{"rollout_id": key} for key in keys],
                    )
                )
            )
            stop.set()
            await ticker
            monkeypatch.setattr(os, "replace", real_replace)
            await after.controller.resume(CheckpointRequest(**control("r1")))
            claimed = {
                key: claimant(claims_dir, key)
                == next(link.routing_id for link in after.links if link.participant.has_session(key))
                for key in keys
            }
    finally:
        shutil.rmtree(claims_dir, ignore_errors=True)

    assert all(claimed.values())
    # About half a second of claim files per worker, which the event loop does not wait for.
    assert max(gaps) < 0.2, max(gaps)


async def test_a_worker_whose_readiness_regressed_refuses_to_export(socket_dir: str, tmp_path: Path) -> None:
    async with coordinated(socket_dir, environment_worker) as deployment:
        prepared = await deployment.controller.prepare(CheckpointRequest(**control()))
        # Readiness that went backwards after the worker's last report reached the coordinator.
        deployment.links[1].participant.ready = lambda: False
        with pytest.raises(ControlError) as refused:
            await deployment.controller.commit(CommitRequest(**control(checkpoint_dir=str(tmp_path))))
        await deployment.controller.resume(CheckpointRequest(**control()))

    # The worker's export failed, so this checkpoint can never be committed: the controller prepares a new one.
    assert prepared["phase"] == "prepared" and refused.value.code == "checkpoint_export_failed"
    assert "no longer ready to commit" in refused.value.detail


async def test_a_forget_one_worker_cannot_do_yet_leaves_every_refusal_in_place(socket_dir: str) -> None:
    release = asyncio.Event()

    def make() -> tuple[ResourcesParticipant, FastAPI]:
        participant, app = resources_worker(routed=False)()
        return participant, app

    async with coordinated(socket_dir, make) as deployment:
        slow, fast = deployment.links
        original = slow.participant.hooks.retire_session_state
        stopping = asyncio.Event()

        async def slow_retire(session_id: str) -> None:
            stopping.set()
            await release.wait()
            await original(session_id)

        slow.participant.hooks.retire_session_state = slow_retire
        for index, link in enumerate(deployment.links):
            link.participant.hooks.states[f"s{index}"] = index
            link.participant.seeded(f"s{index}", EpisodeId(rollout_id="r"))
        with pytest.raises(ControlError):
            # Long enough for the marks and the retire to reach both workers on a loaded machine;
            # the slow worker's retire then holds it past the deadline.
            await deployment.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r"}], timeout=2)))
        assert stopping.is_set()
        with pytest.raises(ControlError) as refused:
            await deployment.controller.forget(ForgetRequest(**control(rollout_ids=["r"])))
        kept = [len(link.retired) for link in deployment.links] + [len(deployment.participant.retired)]
        release.set()
        await deployment.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r"}])))
        await deployment.controller.forget(ForgetRequest(**control(rollout_ids=["r"])))
        forgotten = [len(link.retired) for link in deployment.links] + [len(deployment.participant.retired)]

    # The worker that had freed its session would otherwise admit the retired attempt's late requests again.
    assert refused.value.code == "checkpoint_retire_incomplete"
    assert kept == [1, 1, 1]
    assert forgotten == [0, 0, 0]


async def test_a_restored_session_seeded_again_on_another_worker_is_exported_once(
    socket_dir: str, tmp_path: Path
) -> None:
    async with coordinated(socket_dir, resources_worker(routed=False)) as before:
        before.links[0].participant.hooks.states["x"] = "before the crash"
        before.links[0].participant.seeded("x", EpisodeId(rollout_id="r"))
        await commit(before, tmp_path / "ckpt1", ["r"])
        await before.controller.resume(CheckpointRequest(**control()))

    async with coordinated(socket_dir, resources_worker(routed=True)) as after:
        await restore(after, tmp_path / "ckpt1", [{"rollout_id": "r"}])
        [holder] = [link for link in after.links if "x" in link.participant.hooks.states]
        [other] = [link for link in after.links if link is not holder]
        # The episode continues from before its seed, and the seed reaches the other worker.
        other.participant.hooks.states["x"] = "seeded again"
        other.participant.seeded("x", EpisodeId(rollout_id="r", attempt=1))
        await commit(after, tmp_path / "ckpt2", ["r-a1"], checkpoint_id="c2")
        await after.controller.resume(CheckpointRequest(**control("c2")))
        stale_left = "x" in holder.participant.hooks.states or bool(holder.participant.unclaimed_sessions())

    exported = records(tmp_path / "ckpt2", "resources")
    assert [(record["session_id"], record["state"]) for record in exported] == [("x", "seeded again")]
    assert not stale_left


async def test_after_a_restore_every_worker_refuses_the_replaced_attempt_even_one_that_joins_later(
    socket_dir: str, tmp_path: Path
) -> None:
    async with restored_environment(socket_dir, tmp_path) as deployment:
        joined = await deployment.join()
        # A harness of the crashed attempt that outlived the crash reaches any worker, including one that joined later.
        for link in deployment.links:
            with pytest.raises(StaleAttemptError):
                await link.participant.claim(EpisodeId(rollout_id="r"))
            with pytest.raises(StaleAttemptError):
                link.participant.begin(EpisodeId(rollout_id="r"), TASK, None)
        await joined.participant.claim(EpisodeId(rollout_id="r", attempt=1))
        joined.participant.begin(EpisodeId(rollout_id="r", attempt=1), TASK, None)

    assert len(deployment.links) == 3
    assert joined.participant.continuation(EpisodeId(rollout_id="r", attempt=1)) == {"next": "verify"}


async def test_a_worker_does_not_check_a_restored_payload_again(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import orjson

    checked: list[object] = []
    real_dumps = orjson.dumps

    def counting(value: Any, *args: Any, **kwargs: Any) -> bytes:
        if value == {"next": "verify"}:
            checked.append(value)
        return real_dumps(value, *args, **kwargs)

    async with restored_environment(socket_dir, tmp_path) as deployment:
        monkeypatch.setattr(orjson, "dumps", counting)
        replacement = EpisodeId(rollout_id="r", attempt=1)
        await deployment.links[0].participant.claim(replacement)
        deployment.links[0].participant.begin(replacement, TASK, None)

    # The boundary was read back from storage and parsed from the coordinator's reply: it is JSON already.
    assert deployment.links[0].participant.continuation(replacement) == {"next": "verify"} and checked == []


async def test_a_retire_past_its_deadline_still_makes_every_worker_refuse_its_attempt(socket_dir: str) -> None:
    async with coordinated(socket_dir, environment_worker) as deployment:
        late = RetireRequest(checkpoint_id="retire", deadline_ts=time.time() - 1, episode_ids=[{"rollout_id": "r"}])
        with pytest.raises(ControlError) as refused:
            await deployment.controller.retire(late)
        for link in deployment.links:
            with pytest.raises(StaleAttemptError):
                link.participant.begin(EpisodeId(rollout_id="r"), TASK, None)

    assert refused.value.code == "checkpoint_deadline_exceeded"


async def test_every_worker_leaves_out_the_restarts_an_earlier_stage_reported_until_resume(socket_dir: str) -> None:
    """r's activation runs on the second worker; the prepare request names r as a restart.

    Every worker gets the restarts, one that joins while the checkpoint is open included,
    so the coordinator is ready without waiting for r, and every worker forgets them at resume.
    """
    from nemo_gym._checkpoint.control import PrepareRequest

    async with coordinated(socket_dir, agent_worker, expected=3) as deployment:
        _, second = deployment.links
        episode_id = EpisodeId(rollout_id="r")
        second.participant.open_session("s", episode_id)
        released = asyncio.Event()
        in_tool_call = asyncio.Event()

        async def activation() -> None:
            async with second.participant.activation("s", episode_id) as running:
                in_tool_call.set()
                await released.wait()
                await running.boundary(lambda: {"step": 1})

        task = asyncio.create_task(activation())
        await asyncio.wait_for(in_tool_call.wait(), 5)
        missed = await deployment.controller.prepare(PrepareRequest(**control(timeout=0.3, restarts=["r"])))
        third = await deployment.join()
        prepared = await deployment.controller.prepare(PrepareRequest(**control(restarts=["r"])))
        carried = [set(link.participant.carried_restarts) for link in deployment.links]
        await deployment.controller.resume(CheckpointRequest(**control()))
        after_resume = [set(link.participant.carried_restarts) for link in deployment.links]
        released.set()
        await asyncio.wait_for(task, 5)

    # Only the missing third worker held it up, not r.
    assert missed["report"]["blockers"] == ["agent-workers-unreported:1"]
    assert third is deployment.links[2]
    assert prepared["phase"] == "prepared", prepared
    assert carried == [{"r"}, {"r"}, {"r"}]
    assert after_resume == [set(), set(), set()]


# -- claimed records are leased until the claiming worker settles them -----------------------------


async def committed_run_episode(socket_dir: str, checkpoint_dir: Path) -> None:
    """A checkpoint of one agent's own /run episode ``r``, parked at its loop."""
    async with coordinated(socket_dir, agent_worker) as before:
        participant = before.links[0].participant
        started = asyncio.Event()

        async def run_episode() -> None:
            async with participant.legacy_run("run:r", EpisodeId(rollout_id="r")) as run:
                await run.boundary({"next": "loop"})
                async with run.step("replay"):
                    started.set()
                    await asyncio.Event().wait()

        task = asyncio.create_task(run_episode())
        await started.wait()
        await commit(before, checkpoint_dir, ["r"])
        task.cancel()


class GatedRestoreHooks(AgentHooks):
    """Restores fail while ``fail`` is set, and wait for ``release`` while it is not set."""

    def __init__(self) -> None:
        super().__init__()
        self.fail = False
        self.installing = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        if not sessions:
            # A restore asks every worker to install, even one with nothing to install.
            return
        self.installing.set()
        await self.release.wait()
        if self.fail:
            raise RuntimeError("the sandbox could not be rebuilt")
        await super().restore_agent_sessions(sessions)


def gated_agent_worker() -> tuple[AgentSessionParticipant, FastAPI]:
    app = FastAPI()
    app.state.nemo_gym_routing_id = uuid4().hex
    return AgentSessionParticipant(GatedRestoreHooks()), app


async def replacement_run(participant: AgentSessionParticipant) -> Optional[dict]:
    """The replacement attempt's /run on this worker; returns where it continues from."""
    async with participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=1)) as run:
        return run.continuation


async def test_a_claimed_record_whose_install_fails_goes_back_and_another_worker_claims_it(
    socket_dir: str, tmp_path: Path
) -> None:
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, gated_agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first, second = after.links
        first.participant.hooks.fail = True
        with pytest.raises(RuntimeError):
            await replacement_run(first.participant)
        await wait_until(lambda: "run:r" in after.participant.restored)
        continued = await replacement_run(second.participant)
        # The second worker settles its claim in the background.
        # A worker that disconnects first takes its install with it, and its record goes back.
        await wait_until(lambda: not after.participant.leases)
        left = dict(after.participant.restored)

    assert continued == {"next": "loop"}
    assert [session.session_key for session in second.participant.hooks.restored] == ["run:r"]
    assert left == {}


async def test_a_claimed_record_whose_reply_is_lost_goes_back(socket_dir: str, tmp_path: Path) -> None:
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first, second = after.links
        real_call = first.call

        async def reply_lost(kind: str, body: dict, **kwargs: Any) -> dict:
            reply = await real_call(kind, body, **kwargs)
            if kind == "claim":
                # The coordinator answered, and the answer never reached this worker.
                raise workers_module.CoordinatorUnavailableError("the claim reply was lost")
            return reply

        first.call = reply_lost
        with pytest.raises(workers_module.CoordinatorUnavailableError):
            await replacement_run(first.participant)
        await wait_until(lambda: "run:r" in after.participant.restored)
        continued = await replacement_run(second.participant)

    assert continued == {"next": "loop"}


async def test_a_claimed_record_of_a_lost_worker_goes_back_and_the_next_checkpoint_exports_it(
    socket_dir: str, tmp_path: Path, lost_coordinators: list[str]
) -> None:
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, gated_agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first = after.links[0]
        first.participant.hooks.release.clear()
        run = asyncio.create_task(replacement_run(first.participant))
        await first.participant.hooks.installing.wait()
        # The worker dies while it installs the record; uvicorn starts a replacement.
        first._channel.close()
        await wait_until(lambda: "run:r" in after.participant.restored)
        run.cancel()
        after.links.remove(first)
        await after.join()
        await commit(after, tmp_path / "ckpt2", ["r-a1"], checkpoint_id="c2")
        await after.controller.resume(CheckpointRequest(**control("c2")))

    assert lost_coordinators == ["lost"]
    [record] = records(tmp_path / "ckpt2", "agent")
    assert record["session_key"] == "run:r" and record["episode_id"] == {"rollout_id": "r", "attempt": 1}


async def test_an_unsettled_claim_holds_up_prepare_until_the_install_finishes(socket_dir: str, tmp_path: Path) -> None:
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, gated_agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first = after.links[0]
        first.participant.hooks.release.clear()

        async def run_to_its_boundary() -> None:
            async with first.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=1)) as run:
                # Parks here while the checkpoint is open.
                await run.boundary({"next": "loop"})

        run = asyncio.create_task(run_to_its_boundary())
        await first.participant.hooks.installing.wait()
        waiting = await after.controller.prepare(CheckpointRequest(**control("c2", timeout=0.3)))
        first.participant.hooks.release.set()
        # Ready once the claim is settled.
        prepared = await after.controller.prepare(CheckpointRequest(**control("c2")))
        await after.controller.resume(CheckpointRequest(**control("c2")))
        await run

    assert waiting["phase"] == "preparing" and "r-a1" in waiting["report"]["blockers"]
    assert prepared["phase"] == "prepared"


async def wait_until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


# -- one copy per piece of state, as the participant kind identifies it ----------------------------


async def test_two_agent_run_episodes_sent_with_one_cookie_are_both_exported_and_named_in_the_reply(
    socket_dir: str, tmp_path: Path
) -> None:
    """Agent records are told apart by session key, so two /run episodes on two workers stay two records.

    The coordinator answers the commit for its workers,
    so the reply's ``run_episode_ids`` must name the /run episodes of every worker,
    and an unclaimed restored /run episode the coordinator itself holds.
    """
    await committed_run_episode(socket_dir, tmp_path / "before")
    async with coordinated(socket_dir, agent_worker) as deployment:
        await restore(deployment, tmp_path / "before", [{"rollout_id": "r"}])
        runs = []
        for link, key in zip(deployment.links, ("run:a", "run:b")):
            ready = asyncio.Event()

            # Both /run requests carry the same cookie jar, which no longer matters: nothing records a cookie.
            async def run_episode(participant=link.participant, key=key, ready=ready) -> None:
                async with participant.legacy_run(key, EpisodeId(rollout_id=key[4:])) as run:
                    await run.boundary({"next": "loop"})
                    async with run.step("replay"):
                        ready.set()
                        await asyncio.Event().wait()

            runs.append(asyncio.create_task(run_episode()))
            await ready.wait()
        prepared = await deployment.controller.prepare(CheckpointRequest(**control("c2")))
        assert prepared["phase"] == "prepared", prepared
        reply = await deployment.controller.commit(
            CommitRequest(
                **control(
                    "c2",
                    checkpoint_dir=str(tmp_path / "ckpt"),
                    episode_ids=[{"rollout_id": "a"}, {"rollout_id": "b"}, {"rollout_id": "r", "attempt": 1}],
                )
            )
        )
        for run in runs:
            run.cancel()

    exported = records(tmp_path / "ckpt", "agent")
    assert sorted(record["session_key"] for record in exported) == ["run:a", "run:b", "run:r"]
    # Built from the merged records of both workers and the coordinator's own unclaimed record.
    assert reply["run_episode_ids"] == ["a", "b", "r-a1"]


async def test_a_stale_restored_agent_session_and_its_live_copy_export_once_and_the_stale_copy_is_freed(
    socket_dir: str, tmp_path: Path
) -> None:
    async with coordinated(socket_dir, agent_worker) as before:
        before.links[0].participant.open_session("k", EpisodeId(rollout_id="r"))
        await commit(before, tmp_path / "ckpt", ["r"])
        await before.controller.resume(CheckpointRequest(**control()))

    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        [holder] = [link for link in after.links if link.participant.has_session("k")]
        [other] = [link for link in after.links if link is not holder]
        retired: list[str] = []
        holder.participant.hooks.retire_agent_session = AsyncMock(side_effect=retired.append)
        # The replacement attempt seeds the session again on the other worker.
        other.participant.open_session("k", EpisodeId(rollout_id="r", attempt=1))
        await commit(after, tmp_path / "ckpt2", ["r-a1"], checkpoint_id="c2")
        await after.controller.resume(CheckpointRequest(**control("c2")))
        stale_left = holder.participant.has_session("k")

    [record] = records(tmp_path / "ckpt2", "agent")
    assert record["session_key"] == "k" and record["episode_id"] == {"rollout_id": "r", "attempt": 1}
    assert retired == ["k"] and not stale_left


async def test_stale_copies_on_several_workers_are_deleted_at_once(socket_dir: str, tmp_path: Path) -> None:
    class SlowRetire(AgentHooks):
        async def retire_agent_session(self, session_key: str) -> None:
            await asyncio.sleep(0.3)

    def slow_agent_worker() -> tuple[AgentSessionParticipant, FastAPI]:
        app = FastAPI()
        app.state.nemo_gym_routing_id = uuid4().hex
        return AgentSessionParticipant(SlowRetire()), app

    keys = [f"k{index}" for index in range(6)]
    async with coordinated(socket_dir, agent_worker) as before:
        for index, key in enumerate(keys):
            before.links[index % 2].participant.open_session(key, EpisodeId(rollout_id=key))
        await commit(before, tmp_path / "ckpt", keys)
        await before.controller.resume(CheckpointRequest(**control()))

    async with coordinated(socket_dir, slow_agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": key} for key in keys])
        for key in keys:
            # Every session is seeded again on the worker that does not hold its restored copy.
            [other] = [link for link in after.links if not link.participant.has_session(key)]
            other.participant.open_session(key, EpisodeId(rollout_id=key, attempt=1))
        started = time.monotonic()
        await commit(after, tmp_path / "ckpt2", [f"{key}-a1" for key in keys], checkpoint_id="c2")
        elapsed = time.monotonic() - started
        await after.controller.resume(CheckpointRequest(**control("c2")))

    assert len(records(tmp_path / "ckpt2", "agent")) == len(keys)
    # Six deletions of 0.3 s each, three per worker: about one deletion's time, not their sum.
    assert elapsed < 0.9, elapsed


async def test_stale_resources_copies_on_several_workers_are_deleted_at_once(socket_dir: str, tmp_path: Path) -> None:
    """Restored resources sessions seeded again on the other worker: their stale copies go in about one teardown."""

    class SlowTeardownStates(SessionStates):
        async def retire_session_state(self, session_id: str) -> None:
            await asyncio.sleep(0.3)
            await super().retire_session_state(session_id)

    def slow_resources_worker() -> tuple[ResourcesParticipant, FastAPI]:
        app = FastAPI()
        app.state.nemo_gym_routing_id = uuid4().hex
        return ResourcesParticipant(SlowTeardownStates(), "exported"), app

    sessions = [f"s{index}" for index in range(6)]
    async with coordinated(socket_dir, resources_worker(routed=True)) as before:
        for index, session_id in enumerate(sessions):
            link = before.links[index % 2]
            link.participant.hooks.states[session_id] = index
            link.participant.seeded(session_id, EpisodeId(rollout_id=session_id))
        await commit(before, tmp_path / "ckpt", sessions)
        await before.controller.resume(CheckpointRequest(**control()))

    async with coordinated(socket_dir, slow_resources_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": session_id} for session_id in sessions])
        for session_id in sessions:
            # Each session's episode seeds it again on the worker that does not hold its restored copy.
            [other] = [link for link in after.links if session_id not in link.participant.hooks.states]
            other.participant.hooks.states[session_id] = "seeded again"
            other.participant.seeded(session_id, EpisodeId(rollout_id=session_id, attempt=1))
        started = time.monotonic()
        await commit(after, tmp_path / "ckpt2", [f"{session_id}-a1" for session_id in sessions], checkpoint_id="c2")
        elapsed = time.monotonic() - started
        await after.controller.resume(CheckpointRequest(**control("c2")))

    exported = records(tmp_path / "ckpt2", "resources")
    assert sorted(record["session_id"] for record in exported) == sessions
    assert {record["state"] for record in exported} == {"seeded again"}
    # Six teardowns of 0.3 s each, three per worker: about one teardown's time, not their sum.
    assert elapsed < 0.9, elapsed


# -- a cancelled or timed-out message stops the other side's handler -------------------------------


class SlowTeardown:
    """Resources session hooks whose teardown takes ``seconds``, recording how many run at once."""

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.running = 0
        self.peak = 0
        self.finished = 0

    async def export_session_states(self, session_ids: list[str]) -> dict[str, Any]:
        return dict.fromkeys(session_ids, {})

    async def restore_session_states(self, states: dict[str, Any]) -> None:
        pass

    async def retire_session_state(self, session_id: str) -> None:
        self.running += 1
        self.peak = max(self.peak, self.running)
        try:
            await asyncio.sleep(self.seconds)
            self.finished += 1
        finally:
            self.running -= 1


async def test_a_retire_cut_short_by_its_deadline_stops_the_worker_s_teardown_before_a_retry_starts_it(
    socket_dir: str,
) -> None:
    hooks: list[SlowTeardown] = []

    def slow_worker() -> tuple[ResourcesParticipant, FastAPI]:
        hooks.append(SlowTeardown(0.6))
        return ResourcesParticipant(hooks[-1], "exported"), FastAPI()

    async with coordinated(socket_dir, slow_worker) as deployment:
        worker_hooks = hooks[-1]
        deployment.links[1].participant.seeded("s", EpisodeId(rollout_id="r"))
        retire = {"episode_ids": [{"rollout_id": "r"}]}
        with pytest.raises(ControlError) as cut_short:
            await deployment.controller.retire(RetireRequest(**control("x", timeout=0.2, **retire)))
        still_running = worker_hooks.running
        # Retried at once, as a controller does after a deadline.
        await deployment.controller.retire(RetireRequest(**control("x", **retire)))

    assert cut_short.value.code == "checkpoint_deadline_exceeded"
    assert still_running == 0
    assert worker_hooks.peak == 1 and worker_hooks.finished == 1
    assert "s" not in deployment.links[1].participant._sessions


async def test_a_slow_export_is_stopped_on_every_worker_before_resume_returns(socket_dir: str, tmp_path: Path) -> None:
    exporting: list[str] = []

    class SlowExport(AgentHooks):
        async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
            exporting.append("start")
            try:
                await asyncio.sleep(5)
            finally:
                exporting.append("stop")
            return await super().export_agent_sessions(session_keys)

    def slow_agent_worker() -> tuple[AgentSessionParticipant, FastAPI]:
        app = FastAPI()
        app.state.nemo_gym_routing_id = uuid4().hex
        return AgentSessionParticipant(SlowExport()), app

    async with coordinated(socket_dir, slow_agent_worker) as deployment:
        deployment.links[0].participant.open_session("k", EpisodeId(rollout_id="r"))
        await deployment.controller.prepare(CheckpointRequest(**control()))
        with pytest.raises(ControlError):
            await deployment.controller.commit(
                CommitRequest(**control(timeout=0.3, checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=None))
            )
        await deployment.controller.resume(CheckpointRequest(**control()))
        after_resume = list(exporting)

    # Admission reopened only once the export it no longer uses had stopped on the worker.
    assert after_resume.count("start") == after_resume.count("stop") >= 1


async def test_a_cancel_that_arrives_after_the_handler_finished_is_answered_at_once(socket_dir: str) -> None:
    async with coordinated(socket_dir, environment_worker) as deployment:
        channel = deployment.participant.workers[1].channel
        answered = await asyncio.wait_for(channel.call("cancel", {"id": 10**9}, timeout=None), 2)

    assert answered == {}


async def test_a_teardown_slower_than_the_message_timeout_finishes_within_the_retire_deadline(
    socket_dir: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Scaled down: the fixed message timeout is shorter than the teardown, and the deadline is longer.
    monkeypatch.setattr(workers_module, "MESSAGE_TIMEOUT_SECONDS", 0.2)
    monkeypatch.setattr(participant_workers_module, "MESSAGE_TIMEOUT_SECONDS", 0.2)
    hooks: list[SlowTeardown] = []

    def slow_worker() -> tuple[ResourcesParticipant, FastAPI]:
        hooks.append(SlowTeardown(0.5))
        return ResourcesParticipant(hooks[-1], "exported"), FastAPI()

    async with coordinated(socket_dir, slow_worker) as deployment:
        deployment.links[0].participant.seeded("s", EpisodeId(rollout_id="r"))
        retired = await deployment.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r"}])))
        # A restored session the next commit no longer continues is deleted the same way.
        await deployment.links[1].participant.install(
            [ResourcesSessionRecord(session_id="t", episode_id=EpisodeId(rollout_id="q"), state={})],
            [EpisodeId(rollout_id="q", attempt=1)],
        )
        await commit(deployment, tmp_path / "ckpt", [], checkpoint_id="c2")
        await deployment.controller.resume(CheckpointRequest(**control("c2")))

    assert retired == {"retired_count": 1}
    assert sum(hook.finished for hook in hooks) == 2
    assert "t" not in deployment.links[1].participant._sessions


# -- a restore is all or nothing across workers ----------------------------------------------------


async def test_a_restore_that_fails_on_one_worker_leaves_nothing_on_the_others(
    socket_dir: str, tmp_path: Path
) -> None:
    async with coordinated(socket_dir, agent_worker) as before:
        first, second = before.links
        first.participant.open_session("one", EpisodeId(rollout_id="r1"))
        second.participant.open_session("two", EpisodeId(rollout_id="r2"))
        await commit(before, tmp_path / "ckpt", ["r1", "r2"])
        await before.controller.resume(CheckpointRequest(**control()))

    claims_dir = tempfile.mkdtemp(prefix="ngc-", dir="/tmp")
    async with coordinated(socket_dir, routed(gated_agent_worker, claims_dir)) as after:
        failing, installing = after.links
        failing.participant.hooks.fail = True
        retired: list[str] = []
        installing.participant.hooks.retire_agent_session = AsyncMock(side_effect=retired.append)
        scope = [{"rollout_id": "r1"}, {"rollout_id": "r2"}]
        with pytest.raises(ControlError):
            await after.controller.restore(
                RestoreRequest(**control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=scope))
            )
        await wait_until(lambda: all(link.participant.accepting for link in after.links))
        left = [key for link in after.links for key in ("one", "two") if link.participant.has_session(key)]
        claims_left = os.listdir(os.path.join(claims_dir, "session-claims"))
        # The controller retires the restored attempts after a failed restore, which finds nothing left.
        await after.controller.retire(
            RetireRequest(
                **control(
                    "cleanup", episode_ids=[{"rollout_id": "r1", "attempt": 1}, {"rollout_id": "r2", "attempt": 1}]
                )
            )
        )

    shutil.rmtree(claims_dir, ignore_errors=True)
    assert left == []
    assert len(retired) == 1
    # The surviving worker released the claim its install rewrote; the failing one never rewrote its own.
    assert claims_left == []
    assert after.participant.restored == {}


async def test_a_worker_lost_while_it_installs_fails_the_restore_and_leaves_nothing_on_the_others(
    socket_dir: str, tmp_path: Path, lost_coordinators: list[str]
) -> None:
    async with coordinated(socket_dir, agent_worker) as before:
        first, second = before.links
        first.participant.open_session("one", EpisodeId(rollout_id="r1"))
        second.participant.open_session("two", EpisodeId(rollout_id="r2"))
        await commit(before, tmp_path / "ckpt", ["r1", "r2"])
        await before.controller.resume(CheckpointRequest(**control()))

    async with coordinated(socket_dir, gated_agent_worker) as after:
        dying, surviving = after.links
        dying.participant.hooks.release.clear()
        scope = [{"rollout_id": "r1"}, {"rollout_id": "r2"}]
        restoring = asyncio.create_task(
            after.controller.restore(
                RestoreRequest(**control("r1", checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=scope))
            )
        )
        await dying.participant.hooks.installing.wait()
        # The worker dies in the middle of its install.
        dying._channel.close()
        with pytest.raises(ControlError):
            await restoring
        left = [key for key in ("one", "two") if surviving.participant.has_session(key)]

    assert left == [] and lost_coordinators == ["lost"]
    assert after.participant.restored == {}


async def failed_restore_whose_rollback_fails(
    socket_dir: str, tmp_path: Path, claims_dir: str, stack: AsyncExitStack
) -> tuple[Workers, ParticipantWorkerLink, ParticipantWorkerLink, list[bool]]:
    """A restore that fails on one worker, whose rollback fails on the other while the returned flag is set."""
    await committed_sessions_on_two_workers(socket_dir, tmp_path / "ckpt")
    after = await stack.enter_async_context(coordinated(socket_dir, routed(gated_agent_worker, claims_dir)))
    failing, installing = after.links
    failing.participant.hooks.fail = True
    rollback_fails = [True]

    async def retire_agent_session(session_key: str) -> None:
        if rollback_fails[0]:
            raise RuntimeError("the sandbox did not stop")

    installing.participant.hooks.retire_agent_session = retire_agent_session
    with pytest.raises(ControlError):
        await after.controller.restore(restore_both(tmp_path / "ckpt"))
    return after, failing, installing, rollback_fails


async def test_a_rollback_that_fails_on_a_worker_keeps_every_worker_closed_until_it_is_repeated(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(participant_workers_module, "_ROLLBACK_RETRY_SECONDS", 0.02, raising=False)
    monkeypatch.setattr(participant_workers_module, "_ROLLBACK_RETRY_BUDGET_SECONDS", 0.3, raising=False)
    claims_dir = tempfile.mkdtemp(prefix="ngc-", dir="/tmp")
    try:
        async with AsyncExitStack() as stack:
            after, failing, installing, rollback_fails = await failed_restore_whose_rollback_fails(
                socket_dir, tmp_path, claims_dir, stack
            )
            await asyncio.sleep(0.2)
            serving_after_the_restore = [link.accepting or link.participant.accepting for link in after.links]
            held_after_the_restore = installing.participant.has_session("two")
            claim_after_the_restore = claimant(claims_dir, "two")
            with pytest.raises(ControlError) as incomplete:
                await after.controller.resume(CheckpointRequest(**control("r1")))
            serving_after_the_resume = [link.accepting or link.participant.accepting for link in after.links]
            # The rollback succeeds once it is repeated, and the next resume reopens every worker.
            rollback_fails[0] = False
            resumed = await after.controller.resume(CheckpointRequest(**control("r1")))
            serving = [link.accepting and link.participant.accepting for link in after.links]
            left = [key for link in after.links for key in ("one", "two") if link.participant.has_session(key)]
            claims_left = os.listdir(os.path.join(claims_dir, "session-claims"))
    finally:
        shutil.rmtree(claims_dir, ignore_errors=True)

    # The worker still holding what the failed restore installed never served it, nor did any other worker.
    assert serving_after_the_restore == [False, False] and serving_after_the_resume == [False, False]
    assert held_after_the_restore and claim_after_the_restore == installing.routing_id
    assert incomplete.value.code == "checkpoint_rollback_incomplete" and incomplete.value.status_code == 503
    assert resumed["phase"] == "idle" and serving == [True, True]
    assert left == [] and claims_left == []


async def test_a_worker_lost_while_its_rollback_is_retried_lets_admission_reopen(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost_coordinators: list[str]
) -> None:
    monkeypatch.setattr(participant_workers_module, "_ROLLBACK_RETRY_SECONDS", 0.02, raising=False)
    monkeypatch.setattr(participant_workers_module, "_ROLLBACK_RETRY_BUDGET_SECONDS", 0.3, raising=False)
    claims_dir = tempfile.mkdtemp(prefix="ngc-", dir="/tmp")
    try:
        async with AsyncExitStack() as stack:
            after, failing, installing, _ = await failed_restore_whose_rollback_fails(
                socket_dir, tmp_path, claims_dir, stack
            )
            serving_while_it_holds = failing.accepting or failing.participant.accepting
            # The worker that cannot roll back is lost, and what it held goes with it.
            installing._channel.close()
            await wait_until(lambda: len(after.participant.workers) == 1)
            resumed = await after.controller.resume(CheckpointRequest(**control("r1")))
            serving = failing.accepting and failing.participant.accepting
    finally:
        shutil.rmtree(claims_dir, ignore_errors=True)

    assert lost_coordinators == ["lost"] and not serving_while_it_holds
    assert resumed["phase"] == "idle" and serving


# -- status, chunked exports, and owners -----------------------------------------------------------


async def test_status_adds_up_what_every_worker_reports(socket_dir: str) -> None:
    def resources_worker() -> tuple[ResourcesParticipant, FastAPI]:
        return ResourcesParticipant(AsyncMock(), "exported"), FastAPI()

    async with coordinated(socket_dir, resources_worker) as deployment:
        first, second = deployment.links
        first.participant.seeded("s1", EpisodeId(rollout_id="r"))
        second.participant.seeded("s2", EpisodeId(rollout_id="r"))
        before = await dispatch_control(deployment.controller, "status", None)
        # A retire past its deadline only refuses the attempt.
        # That marks each worker's session until a retire frees it.
        with pytest.raises(ControlError):
            await deployment.controller.retire(
                RetireRequest(checkpoint_id="x", deadline_ts=time.time() - 1, episode_ids=[{"rollout_id": "r"}])
            )
        marked = await dispatch_control(deployment.controller, "status", None)

    assert before["report"]["counts"]["sessions"] == 2 and before["report"]["counts"]["workers"] == 2
    assert marked["retired_sessions"] == 2 and marked["mode"] == "exported"
    assert marked["status_workers"] == 2


async def test_status_answers_without_a_worker_that_does_not(socket_dir: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(participant_workers_module, "_STATUS_TIMEOUT_SECONDS", 0.2, raising=False)

    def resources_worker() -> tuple[ResourcesParticipant, FastAPI]:
        return ResourcesParticipant(AsyncMock(), "exported"), FastAPI()

    async with coordinated(socket_dir, resources_worker) as deployment:
        stuck = deployment.links[1]
        real_handle = stuck.handle

        async def handle(kind: str, body: dict) -> dict:
            if kind == "status":
                await asyncio.Event().wait()
            return await real_handle(kind, body)

        stuck.handle = handle
        started = time.monotonic()
        status = await dispatch_control(deployment.controller, "status", None)
        elapsed = time.monotonic() - started

    assert status["status_workers"] == 1 and status["workers"] == 2
    assert elapsed < 1.0


async def test_a_large_export_arrives_in_chunks_with_every_record(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(participant_workers_module, "EXPORT_CHUNK_BYTES", 300, raising=False)
    sent: list[int] = []
    async with coordinated(socket_dir, agent_worker) as deployment:
        for link in deployment.links:
            real_call = link.call

            async def counting(kind: str, body: Any, real_call=real_call, **kwargs: Any) -> dict:
                if kind == "export_chunk":
                    sent.append(len(json.loads(bytes(body))["records"]))
                return await real_call(kind, body, **kwargs)

            link.call = counting
        keys = [f"k{index}" for index in range(10)]
        for index, key in enumerate(keys):
            deployment.links[index % 2].participant.open_session(key, EpisodeId(rollout_id=key))
        await commit(deployment, tmp_path / "ckpt", keys)
        await deployment.controller.resume(CheckpointRequest(**control()))

    assert sorted(record["session_key"] for record in records(tmp_path / "ckpt", "agent")) == sorted(keys)
    assert len(sent) > 2 and sum(sent) == len(keys)


async def test_a_worker_lost_between_export_chunks_fails_the_commit_and_keeps_nothing(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost_coordinators: list[str]
) -> None:
    monkeypatch.setattr(participant_workers_module, "EXPORT_CHUNK_BYTES", 100, raising=False)
    async with coordinated(socket_dir, agent_worker) as deployment:
        dying = deployment.links[1]
        real_call = dying.call
        chunks = 0

        async def die_after_one_chunk(kind: str, body: Any, **kwargs: Any) -> dict:
            nonlocal chunks
            if kind == "export_chunk":
                chunks += 1
                if chunks == 2:
                    dying._channel.close()
            return await real_call(kind, body, **kwargs)

        dying.call = die_after_one_chunk
        for index in range(6):
            dying.participant.open_session(f"k{index}", EpisodeId(rollout_id=f"k{index}"))
        await deployment.controller.prepare(CheckpointRequest(**control()))
        with pytest.raises(ControlError):
            await deployment.controller.commit(
                CommitRequest(**control(checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=None))
            )
        buffers = dict(deployment.participant._exports)

    assert buffers == {} and lost_coordinators == ["lost"]


# -- restored agent sessions on agent servers with several workers ---------------------------------


class CheckpointedSessionAgent(SimpleResponsesAPIAgent):
    """An agent whose sessions checkpoint: the state is the seed request."""

    checkpoint_mode = "exported"
    closes: list[str] = Field(default_factory=list)

    label: str = ""

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.get("/whoami")(self.whoami)
        return app

    async def whoami(self, request: Request) -> dict:
        """Which worker served the request, and whether it holds the cookie's agent session."""
        agent_session_id = request.session.get(AGENT_SESSION_COOKIE_KEY)
        return {
            "worker": self.label,
            "holds": agent_session_id is not None and self.holds_agent_session(agent_session_id),
        }

    async def responses(self, body):
        raise NotImplementedError

    async def run(self, body):
        raise NotImplementedError

    async def _seed_agent_session_state(self, body: AgentSeedSessionRequest) -> AgentSessionState:
        return AgentSessionState(request=body)

    async def _close_agent_session_state(self, state: AgentSessionState) -> AgentCloseSessionResponse:
        self.closes.append(state.request.agent_session_id)
        return AgentCloseSessionResponse(agent_session_id=state.request.agent_session_id)

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        return {key: {"request": self._session_state(key).request.model_dump(mode="json")} for key in session_keys}

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        for session in sessions:
            request = AgentSeedSessionRequest.model_validate(
                {**session.session["request"], "episode_id": session.episode_id.model_dump(mode="json")}
            )
            self._install_restored_session(session.session_key, AgentSessionState(request=request))


class AgentServerWorkers:
    """A coordinator and the agent server workers that joined it, with session routing."""

    def __init__(self, coordinator: WorkerCoordinator, stack: AsyncExitStack, routing_dir: str) -> None:
        self.coordinator = coordinator
        self.agents: list[CheckpointedSessionAgent] = []
        self.apps: list[FastAPI] = []
        self._stack = stack
        self._routing_dir = routing_dir

    async def join(self) -> FastAPI:
        config = BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent", num_workers=2)
        agent = CheckpointedSessionAgent(
            config=config, server_client=checkpointing_client(), label=f"worker-{len(self.agents)}"
        )
        app = agent.setup_webserver()
        key = agent.get_session_middleware_key()
        install_session_routing(
            app, socket_dir=self._routing_dir, session_cookie=key, secret_key=key, **agent.claimed_session_routing()
        )
        await self._stack.enter_async_context(app.router.lifespan_context(app))
        self.agents.append(agent)
        self.apps.append(app)
        return app


@asynccontextmanager
async def agent_server_deployment(
    socket_dir: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[AgentServerWorkers]:
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, "1")
    coordinator = WorkerCoordinator(
        CoordinatedServerParticipant(AgentSessionParticipant(MagicMock()), expected_workers=2),
        instance_name="agent",
        lease_grace_seconds=60,
        socket_path=os.path.join(socket_dir, f"{uuid4().hex[:8]}.sock"),
    )
    server = await coordinator.serve()
    monkeypatch.setenv(COORDINATOR_SOCKET_ENV, coordinator.socket_path)
    routing_dir = tempfile.mkdtemp(prefix="ngr-", dir="/tmp")
    try:
        async with AsyncExitStack() as stack:
            deployment = AgentServerWorkers(coordinator, stack, routing_dir)
            for _ in range(2):
                await deployment.join()
            yield deployment
    finally:
        server.close()
        await server.wait_closed()
        shutil.rmtree(routing_dir, ignore_errors=True)


def seed_body(agent_session_id: str, rollout_id: str, attempt: int = 0) -> dict:
    return {
        "agent_session_id": agent_session_id,
        "episode_id": {"rollout_id": rollout_id, "attempt": attempt},
        "task_id": {"taskset": "t", "task_id": "0"},
    }


async def test_a_restored_agent_session_is_seeded_and_closed_on_its_worker_without_a_cookie(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with agent_server_deployment(socket_dir, monkeypatch) as before:
        async with http(before.apps[0]) as worker:
            seeded = await worker.post("/v1/agent_sessions", json=seed_body("k", "r"))
            assert seeded.status_code == 200, seeded.text
        await commit(before.coordinator, tmp_path / "ckpt", ["r"])
        await before.coordinator.controller.resume(CheckpointRequest(**control()))

    async with agent_server_deployment(socket_dir, monkeypatch) as after:
        await restore(after.coordinator, tmp_path / "ckpt", [{"rollout_id": "r"}])
        [holder] = [agent for agent in after.agents if agent.holds_agent_session("k")]
        [other_app] = [app for app, agent in zip(after.apps, after.agents) if agent is not holder]
        async with http(other_app) as other:
            # The replacement attempt's seed, retried without a cookie, as after a lost reply.
            reseeded = await other.post("/v1/agent_sessions", json=seed_body("k", "r", attempt=1))
            closed = await other.post(
                "/v1/agent_sessions/close",
                json={"agent_session_id": "k", "episode_id": {"rollout_id": "r", "attempt": 1}},
            )
        held_elsewhere = [agent.holds_agent_session("k") for agent in after.agents if agent is not holder]

    assert reseeded.status_code == 200, reseeded.text
    assert closed.status_code == 200, closed.text
    # Both reached the worker the restore installed the session on, which closed it.
    assert holder.closes == ["k"] and held_elsewhere == [False]


async def test_an_agent_session_cookie_from_before_a_restore_reaches_the_worker_that_holds_the_session_now(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cookie names only the agent session ID, so its claim, which the restore rewrote, routes it."""
    async with agent_server_deployment(socket_dir, monkeypatch) as before:
        async with http(before.apps[0]) as worker:
            seeded = await worker.post("/v1/agent_sessions", json=seed_body("k", "r"))
            cookies = dict(seeded.cookies)
        await commit(before.coordinator, tmp_path / "ckpt", ["r"])
        await before.coordinator.controller.resume(CheckpointRequest(**control()))

    async with agent_server_deployment(socket_dir, monkeypatch) as after:
        await restore(after.coordinator, tmp_path / "ckpt", [{"rollout_id": "r"}])
        [holder] = [agent for agent in after.agents if agent.holds_agent_session("k")]
        answers = []
        for app in after.apps:
            async with http(app) as worker:
                answers.append((await worker.get("/whoami", cookies=cookies)).json())

    assert seeded.status_code == 200 and cookies
    # Whichever worker the old cookie reaches, the request is served where the restored session is.
    assert answers == [{"worker": holder.label, "holds": True}] * 2


# -- restored sessions' claims, worker loss, and the worker protocol's repairs ---------------------


async def test_a_restored_resources_session_keeps_its_claim_until_the_participant_frees_it(
    socket_dir: str, tmp_path: Path
) -> None:
    """A restored session is open on its holder, as after a seed, until a commit that drops its episode frees it."""
    async with coordinated(socket_dir, resources_worker(routed=False)) as before:
        for index, link in enumerate(before.links):
            link.participant.hooks.states[f"s{index}"] = index
            link.participant.seeded(f"s{index}", EpisodeId(rollout_id=f"r{index}"))
        await commit(before, tmp_path / "ckpt", ["r0", "r1"])
        await before.controller.resume(CheckpointRequest(**control()))

    def sweep(link: ParticipantWorkerLink) -> None:
        # What session routing's background sweeper does on a running worker.
        for session_id in link.app.state.nemo_gym_resources_session_lifetimes.sweep():
            link.app.state.nemo_gym_session_claims.release(session_id)

    claims_dir = tempfile.mkdtemp(prefix="ngc-", dir="/tmp")
    try:
        async with coordinated(socket_dir, routed(resources_worker(routed=False), claims_dir)) as after:
            await restore(after, tmp_path / "ckpt", [{"rollout_id": "r0"}, {"rollout_id": "r1"}])
            [holder] = [link for link in after.links if "s0" in link.participant.hooks.states]
            # Longer than the window: a session that is open keeps its claim however long it is idle.
            await asyncio.sleep(0.3)
            sweep(holder)
            kept = claimant(claims_dir, "s0")
            # The controller continues only rollout r1, so the commit frees r0's unused restored session.
            await commit(after, tmp_path / "ckpt2", ["r1-a1"], checkpoint_id="c2")
            await after.controller.resume(CheckpointRequest(**control("c2")))
            await asyncio.sleep(0.3)
            sweep(holder)
            freed = claimant(claims_dir, "s0")
    finally:
        shutil.rmtree(claims_dir, ignore_errors=True)

    assert kept == holder.routing_id
    assert "s0" not in holder.participant.hooks.states and freed is None


async def test_a_worker_that_claimed_a_session_before_its_restore_drops_the_claim(
    socket_dir: str, tmp_path: Path
) -> None:
    """A request for a session can reach the server before its restore, for example a cleanup close."""
    async with coordinated(socket_dir, resources_worker(routed=False)) as before:
        before.links[0].participant.hooks.states["s"] = 1
        before.links[0].participant.seeded("s", EpisodeId(rollout_id="r"))
        await commit(before, tmp_path / "ckpt", ["r"])
        await before.controller.resume(CheckpointRequest(**control()))

    claims_dir = tempfile.mkdtemp(prefix="ngc-", dir="/tmp")
    try:
        async with coordinated(socket_dir, routed(resources_worker(routed=False), claims_dir)) as after:
            # The restore places the only session on the first worker; the second claimed it first.
            early = after.links[1].app.state.nemo_gym_session_claims
            assert early.claim("s") == after.links[1].routing_id
            await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
            [holder] = [link for link in after.links if "s" in link.participant.hooks.states]
            still_mine = early.mine("s")
            claimed_by = claimant(claims_dir, "s")
    finally:
        shutil.rmtree(claims_dir, ignore_errors=True)

    assert holder is after.links[0]
    assert claimed_by == holder.routing_id and not still_mine


async def test_a_restored_run_whose_session_cannot_open_goes_back_and_leaves_nothing_on_the_worker(
    socket_dir: str, tmp_path: Path
) -> None:
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first, second = after.links
        retired: list[str] = []
        first.participant.hooks.retire_agent_session = AsyncMock(side_effect=retired.append)
        # A /run of a later attempt than the one the record continues: the restored session belongs to attempt 1.
        with pytest.raises(StaleAttemptError):
            async with first.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=2)):
                pass
        await wait_until(lambda: "run:r" in after.participant.restored)
        left = first.participant.has_session("run:r")
        continued = await replacement_run(second.participant)

    # The installed state was released, the record went back, and the attempt it continues claims it.
    assert not left and retired == ["run:r"]
    assert continued == {"next": "loop"}


async def test_a_worker_lost_with_restored_sessions_before_the_resume_fails_the_resume(
    socket_dir: str, tmp_path: Path, lost_coordinators: list[str]
) -> None:
    async with coordinated(socket_dir, agent_worker) as before:
        first, second = before.links
        first.participant.open_session("one", EpisodeId(rollout_id="r1"))
        second.participant.open_session("two", EpisodeId(rollout_id="r2"))
        await commit(before, tmp_path / "ckpt", ["r1", "r2"])
        await before.controller.resume(CheckpointRequest(**control()))

    async with coordinated(socket_dir, agent_worker) as after:
        await after.controller.restore(
            RestoreRequest(
                **control(
                    "r1",
                    checkpoint_dir=str(tmp_path / "ckpt"),
                    episode_ids=[{"rollout_id": "r1"}, {"rollout_id": "r2"}],
                )
            )
        )
        [holder] = [link for link in after.links if link.participant.has_session("one")]
        # The worker holding a restored session dies after the install and before the resume.
        holder._channel.close()
        await wait_until(lambda: len(after.participant.workers) == 1)
        failures = []
        for _ in range(2):
            with pytest.raises(ControlError) as failed:
                await after.controller.resume(CheckpointRequest(**control("r1")))
            failures.append(failed.value)

    # Every resume fails, rather than continuing the lost session's episode without its state.
    assert [failure.code for failure in failures] == ["checkpoint_worker_lost"] * 2
    assert lost_coordinators == ["lost"]


async def test_a_worker_joins_its_coordinator_only_once_its_private_socket_accepts(
    socket_dir: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    accepting: list[bool] = []
    real_join = CoordinatedServerParticipant.join

    def join(self: CoordinatedServerParticipant, worker_index: int, channel: Any, registration: dict) -> dict:
        # A restore may place sessions on the worker from now on, and other workers forward their requests there.
        paths = glob.glob(os.path.join("/tmp", "ngr-*", f"{registration['routing_id']}.sock"))
        with socket.socket(socket.AF_UNIX) as probe:
            accepting.append(len(paths) == 1 and probe.connect_ex(paths[0]) == 0)
        return real_join(self, worker_index, channel, registration)

    monkeypatch.setattr(CoordinatedServerParticipant, "join", join)
    async with counter_deployment(socket_dir, monkeypatch):
        pass

    assert accepting == [True, True]


async def test_a_readiness_report_that_did_not_arrive_is_sent_again(socket_dir: str) -> None:
    def resources_worker() -> tuple[ResourcesParticipant, FastAPI]:
        return ResourcesParticipant(AsyncMock(), "exported"), FastAPI()

    async with coordinated(socket_dir, resources_worker) as deployment:
        busy = deployment.links[1]
        busy.participant.request_started("s", counted=True)
        real_call = busy.call
        lost: list[str] = []

        async def first_report_lost(kind: str, body: Any, **kwargs: Any) -> dict:
            if kind == "report" and not lost:
                lost.append(kind)
                raise workers_module.CoordinatorUnavailableError("the report was lost")
            return await real_call(kind, body, **kwargs)

        busy.call = first_report_lost
        waiting = await deployment.controller.prepare(CheckpointRequest(**control(timeout=0.3)))
        # The request ends: the worker is ready, and its report of that is lost once.
        busy.participant.request_ended("s", counted=True)
        await busy.participant.notify()
        prepared = await deployment.controller.prepare(CheckpointRequest(**control(timeout=3)))
        await deployment.controller.resume(CheckpointRequest(**control()))

    assert waiting["phase"] == "preparing" and lost == ["report"]
    assert prepared["phase"] == "prepared"


async def test_a_forget_cut_short_leaves_every_worker_refusing_the_attempts(socket_dir: str) -> None:
    async with coordinated(socket_dir, resources_worker(routed=False)) as deployment:
        await deployment.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r"}])))
        fast, slow = deployment.links
        release = asyncio.Event()
        real_forget = slow.forget_local

        async def slow_forget(rollout_ids: list[str]) -> None:
            await release.wait()
            await real_forget(rollout_ids)

        slow.forget_local = slow_forget
        # The first worker forgets; the deadline cuts the forget short while the second still waits.
        with pytest.raises(ControlError):
            await deployment.controller.forget(ForgetRequest(**control(rollout_ids=["r"], timeout=0.5)))
        refusing = [len(link.retired) for link in deployment.links] + [len(deployment.participant.retired)]
        release.set()
        await deployment.controller.forget(ForgetRequest(**control(rollout_ids=["r"])))
        forgotten = [len(link.retired) for link in deployment.links] + [len(deployment.participant.retired)]

    # The worker that had forgotten refuses the retired attempt again, until a forget reaches every worker.
    assert refusing == [1, 1, 1]
    assert forgotten == [0, 0, 0]


async def test_a_worker_that_joins_while_a_forget_runs_forgets_too(socket_dir: str) -> None:
    async with coordinated(socket_dir, resources_worker(routed=False), expected=3) as deployment:
        await deployment.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r"}])))
        slow = deployment.links[1]
        forgetting, release = asyncio.Event(), asyncio.Event()
        real_forget = slow.forget_local

        async def slow_forget(rollout_ids: list[str]) -> None:
            forgetting.set()
            await release.wait()
            await real_forget(rollout_ids)

        slow.forget_local = slow_forget
        forget = asyncio.create_task(deployment.controller.forget(ForgetRequest(**control(rollout_ids=["r"]))))
        await forgetting.wait()
        # A worker starts while the forget is under way, and adopts the refusal the coordinator still keeps.
        late = await deployment.join()
        refusing_on_joining = len(late.retired)
        release.set()
        await forget
        forgotten = [len(link.retired) for link in deployment.links] + [len(deployment.participant.retired)]

    assert refusing_on_joining == 1
    # No worker keeps refusing a rollout the coordinator has forgotten.
    assert forgotten == [0, 0, 0, 0]


async def test_a_forget_retried_while_the_repair_of_one_cut_short_runs_waits_for_that_repair(socket_dir: str) -> None:
    async with coordinated(socket_dir, resources_worker(routed=False)) as deployment:
        await deployment.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r"}])))
        fast, slow = deployment.links
        slow_forgetting, never = asyncio.Event(), asyncio.Event()
        real_forget = slow.forget_local

        async def stuck_forget(rollout_ids: list[str]) -> None:
            slow_forgetting.set()
            await never.wait()

        slow.forget_local = stuck_forget
        marking, mark_may_finish = asyncio.Event(), asyncio.Event()
        fast_handler = fast._channel._handler

        async def held_mark(kind: str, body: dict) -> dict:
            if kind == "mark":
                # The repair reaches the worker that had already forgotten, and is held there.
                marking.set()
                await mark_may_finish.wait()
            return await fast_handler(kind, body)

        fast._channel._handler = held_mark
        forget = asyncio.create_task(deployment.controller.forget(ForgetRequest(**control(rollout_ids=["r"]))))
        await slow_forgetting.wait()
        await wait_until(lambda: len(fast.retired) == 0)
        # Cut short, then cancelled again while its repair runs.
        forget.cancel()
        await marking.wait()
        forget.cancel()
        with pytest.raises(asyncio.CancelledError):
            await forget
        slow.forget_local = real_forget
        retry = asyncio.create_task(deployment.controller.forget(ForgetRequest(**control(rollout_ids=["r"]))))
        await asyncio.sleep(0.2)
        mark_may_finish.set()
        await retry
        await wait_until(lambda: not deployment.participant._remarks)
        forgotten = [len(link.retired) for link in deployment.links] + [len(deployment.participant.retired)]

    # The older repair cannot refuse the rollout again on one worker after the retry forgot it everywhere.
    assert forgotten == [0, 0, 0]


async def test_a_worker_lost_in_the_middle_of_a_message_is_reported_as_a_lost_worker(
    socket_dir: str, lost_coordinators: list[str]
) -> None:
    async with coordinated(socket_dir, environment_worker) as deployment:
        dying = deployment.links[1]
        real_handle = dying.handle

        async def handle(kind: str, body: dict) -> dict:
            if kind == "delete_restored":
                # The worker process dies while it handles the message.
                dying._channel.close()
                await asyncio.Event().wait()
            return await real_handle(kind, body)

        dying.handle = handle
        with pytest.raises(ControlError) as lost:
            await deployment.participant.delete_restored(EpisodeId(rollout_id="r"))

    # The coordinator is running: the error names the lost worker, not an unavailable coordinator.
    assert lost.value.code == "checkpoint_worker_lost" and lost.value.status_code == 409
    assert "worker 2 was lost" in lost.value.detail


def test_every_participant_takes_the_hooks_a_worker_link_sets() -> None:
    participants = [
        ResourcesParticipant(SessionStates(), "exported"),
        EnvironmentParticipant(),
        AgentSessionParticipant(AgentHooks()),
    ]
    unset = [(participant.claim_restored, participant.on_session_freed) for participant in participants]
    for participant in participants:
        ParticipantWorkerLink(participant, socket_path="/nonexistent", app=FastAPI())

    # Part of every participant's contract, rather than set only where an attribute happens to exist.
    assert unset == [(None, None)] * 3
    assert all(participant.claim_restored is not None for participant in participants)
    assert all(participant.on_session_freed is not None for participant in participants)


# -- restores that lose a worker, claims that settle late, and clean disconnects ---------------------


async def committed_sessions_on_two_workers(socket_dir: str, checkpoint_dir: Path) -> None:
    """A checkpoint of agent sessions ``one`` (rollout r1) and ``two`` (rollout r2), one on each of two workers."""
    async with coordinated(socket_dir, agent_worker) as before:
        first, second = before.links
        first.participant.open_session("one", EpisodeId(rollout_id="r1"))
        second.participant.open_session("two", EpisodeId(rollout_id="r2"))
        await commit(before, checkpoint_dir, ["r1", "r2"])
        await before.controller.resume(CheckpointRequest(**control()))


def restore_both(checkpoint_dir: Path) -> RestoreRequest:
    return RestoreRequest(
        **control("r1", checkpoint_dir=str(checkpoint_dir), episode_ids=[{"rollout_id": "r1"}, {"rollout_id": "r2"}])
    )


def acknowledged_install(link: ParticipantWorkerLink) -> asyncio.Event:
    """Set once ``link``'s worker has sent the reply to its install."""
    acknowledged = asyncio.Event()
    channel = link._channel
    send_reply = channel._send_reply

    async def reply_then_note(message: dict, reply: dict) -> None:
        await send_reply(message, reply)
        if message["kind"] == "install":
            acknowledged.set()

    channel._send_reply = reply_then_note
    return acknowledged


async def test_a_holder_lost_after_its_install_while_another_worker_still_installs_fails_the_restore(
    socket_dir: str, tmp_path: Path, lost_coordinators: list[str]
) -> None:
    await committed_sessions_on_two_workers(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, gated_agent_worker) as after:
        quick, slow = after.links
        slow.participant.hooks.release.clear()
        acknowledged = acknowledged_install(quick)
        restoring = asyncio.create_task(after.controller.restore(restore_both(tmp_path / "ckpt")))
        await acknowledged.wait()
        # The worker acknowledged its install, then dies while the other worker is still installing.
        quick._channel.close()
        await wait_until(lambda: len(after.participant.workers) == 1)
        slow.participant.hooks.release.set()
        with pytest.raises(ControlError) as failed:
            await restoring
        await wait_until(lambda: slow.accepting and slow.participant.accepting)
        left = [key for key in ("one", "two") if slow.participant.has_session(key)]
        resumed = await after.controller.resume(CheckpointRequest(**control("r2")))

    assert quick.participant.has_session("one") and lost_coordinators == ["lost"]
    # Every install reported success, but the lost worker's session is gone: the restore fails as a whole,
    # and the surviving worker deletes what it installed before admission reopens.
    assert failed.value.code == "checkpoint_worker_lost"
    assert left == [] and resumed["phase"] == "idle"


async def test_a_restore_past_its_deadline_whose_holder_was_lost_while_installing_reopens_admission(
    socket_dir: str, tmp_path: Path, lost_coordinators: list[str]
) -> None:
    await committed_sessions_on_two_workers(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, gated_agent_worker) as after:
        quick, slow = after.links
        slow.participant.hooks.release.clear()
        acknowledged = acknowledged_install(quick)
        scope = [{"rollout_id": "r1"}, {"rollout_id": "r2"}]
        request = RestoreRequest(
            **control("r1", timeout=0.5, checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=scope)
        )
        restoring = asyncio.create_task(after.controller.restore(request))
        await acknowledged.wait()
        quick._channel.close()
        await wait_until(lambda: len(after.participant.workers) == 1)
        with pytest.raises(ControlError):
            await restoring
        # The install outlives its restore, and every worker still connected completes it.
        slow.participant.hooks.release.set()
        await wait_until(lambda: slow.accepting and slow.participant.accepting)
        left = [key for key in ("one", "two") if slow.participant.has_session(key)]
        resumed = await after.controller.resume(CheckpointRequest(**control("r2")))

    assert lost_coordinators == ["lost"]
    # Admission reopens after the failed restore, without anything the restore installed.
    assert left == [] and resumed["phase"] == "idle"


async def test_a_holder_lost_after_an_install_that_outlived_its_restore_does_not_keep_admission_closed(
    socket_dir: str, tmp_path: Path, lost_coordinators: list[str]
) -> None:
    await committed_sessions_on_two_workers(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, gated_agent_worker) as after:
        for link in after.links:
            link.participant.hooks.release.clear()
        scope = [{"rollout_id": "r1"}, {"rollout_id": "r2"}]
        request = RestoreRequest(
            **control("r1", timeout=0.5, checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=scope)
        )
        with pytest.raises(ControlError):
            await after.controller.restore(request)
        # Admission reopens once the install that outlived the restore has finished; it waits for this lock.
        async with after.controller._lock:
            for link in after.links:
                link.participant.hooks.release.set()
            await wait_until(lambda: after.controller._install.done())
            [holder] = [link for link in after.links if link.participant.has_session("one")]
            holder._channel.close()
            await wait_until(lambda: len(after.participant.workers) == 1)
        [other] = [link for link in after.links if link is not holder]
        await wait_until(lambda: other.accepting and other.participant.accepting)
        left = [key for key in ("one", "two") if other.participant.has_session(key)]
        resumed = await after.controller.resume(CheckpointRequest(**control("r2")))

    assert lost_coordinators == ["lost"]
    # The failed restore is rolled back on the worker that is left, rather than served in part.
    assert left == [] and resumed["phase"] == "idle"


async def test_no_worker_reopens_until_every_worker_holding_restored_sessions_can(
    socket_dir: str, tmp_path: Path, lost_coordinators: list[str]
) -> None:
    await committed_sessions_on_two_workers(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await after.controller.restore(restore_both(tmp_path / "ckpt"))
        [holder] = [link for link in after.links if link.participant.has_session("one")]
        [other] = [link for link in after.links if link is not holder]
        other_serving_before_the_loss: list[bool] = []
        handler = holder._channel._handler

        def serving(link: ParticipantWorkerLink) -> bool:
            return link.accepting or link.participant.accepting

        async def lost_while_reopening(kind: str, body: dict) -> dict:
            if kind in ("prepare_open", "open"):
                # Time for a worker that reopens before every worker is ready to start serving.
                with contextlib.suppress(AssertionError):
                    await wait_until(lambda: serving(other), timeout=0.3)
                other_serving_before_the_loss.append(serving(other))
                # The worker process dies while the workers reopen.
                holder._channel.close()
                raise RuntimeError("the worker died")
            return await handler(kind, body)

        holder._channel._handler = lost_while_reopening
        with pytest.raises(ControlError) as failed:
            await after.controller.resume(CheckpointRequest(**control("r1")))
        other_serving = serving(other)
        # uvicorn starts a replacement worker, which joins the coordinator that stays closed.
        after.links.remove(holder)
        replacement = await after.join()
        replacement_serving = serving(replacement)

    assert failed.value.code == "checkpoint_worker_lost" and lost_coordinators == ["lost"]
    # No worker served while the holder could still fail the resume, nor after it did.
    assert other_serving_before_the_loss == [False]
    assert not other_serving and not replacement_serving


async def test_a_holder_lost_after_every_worker_is_ready_to_reopen_is_lost_as_after_the_resume(
    socket_dir: str, tmp_path: Path, lost_coordinators: list[str]
) -> None:
    await committed_sessions_on_two_workers(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await after.controller.restore(restore_both(tmp_path / "ckpt"))
        [holder] = [link for link in after.links if link.participant.has_session("one")]
        [other] = [link for link in after.links if link is not holder]
        handler = holder._channel._handler

        async def lost_while_opening(kind: str, body: dict) -> dict:
            if kind == "open":
                holder._channel.close()
                raise RuntimeError("the worker died")
            return await handler(kind, body)

        holder._channel._handler = lost_while_opening
        resumed = await after.controller.resume(CheckpointRequest(**control("r1")))
        other_serving = other.accepting and other.participant.accepting
        coordinator_open = after.participant.accepting
        await wait_until(lambda: len(after.participant.workers) == 1)
        after.links.remove(holder)
        replacement = await after.join()

    assert lost_coordinators == ["lost"] and resumed["phase"] == "idle"
    # Every worker was ready, so the workers reopen; the lost worker's sessions are gone, as after any resume.
    assert other_serving and coordinator_open and replacement.accepting


async def restore_past_its_deadline_with_installs_held(deployment: Workers, checkpoint_dir: Path) -> None:
    """A restore of both sessions that fails at its deadline while every worker's install is held."""
    for link in deployment.links:
        link.participant.hooks.release.clear()
    scope = [{"rollout_id": "r1"}, {"rollout_id": "r2"}]
    request = RestoreRequest(**control("r1", timeout=0.5, checkpoint_dir=str(checkpoint_dir), episode_ids=scope))
    with pytest.raises(ControlError):
        await deployment.controller.restore(request)


async def test_a_prepare_cannot_overtake_a_failed_restore_s_recovery_whose_holder_is_then_lost(
    socket_dir: str, tmp_path: Path, lost_coordinators: list[str]
) -> None:
    await committed_sessions_on_two_workers(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, gated_agent_worker) as after:
        await restore_past_its_deadline_with_installs_held(after, tmp_path / "ckpt")
        with pytest.raises(ControlError) as refused:
            await after.controller.prepare(CheckpointRequest(**control("c2", timeout=0.3)))
        async with after.controller._lock:
            for link in after.links:
                link.participant.hooks.release.set()
            await wait_until(lambda: after.controller._install.done())
            [holder] = [link for link in after.links if link.participant.has_session("one")]
            holder._channel.close()
            await wait_until(lambda: len(after.participant.workers) == 1)
        [other] = [link for link in after.links if link is not holder]
        # The recovery the prepare could not overtake rolls the failed restore back and reopens admission.
        await wait_until(lambda: other.accepting and other.participant.accepting)
        left = [key for key in ("one", "two") if other.participant.has_session(key)]
        after.links.remove(holder)
        await after.join()
        prepared = await after.controller.prepare(CheckpointRequest(**control("c3", timeout=2)))
        resumed = await after.controller.resume(CheckpointRequest(**control("c3")))

    assert lost_coordinators == ["lost"]
    assert refused.value.code == "checkpoint_restore_recovery_pending" and refused.value.status_code == 503
    assert left == [] and prepared["phase"] == "prepared" and resumed["phase"] == "idle"


async def test_a_resume_after_a_checkpoint_that_overtook_a_failed_restore_still_rolls_that_restore_back(
    socket_dir: str, tmp_path: Path, lost_coordinators: list[str]
) -> None:
    await committed_sessions_on_two_workers(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, gated_agent_worker) as after:
        await restore_past_its_deadline_with_installs_held(after, tmp_path / "ckpt")
        # A checkpoint begins before the failed restore's recovery, as a prepare could before that was refused.
        after.controller._reopen_pending = False
        await after.controller.prepare(CheckpointRequest(**control("c2", timeout=0.3)))
        after.controller._reopen_pending = True
        for link in after.links:
            link.participant.hooks.release.set()
        await wait_until(lambda: after.controller._install.done())
        [holder] = [link for link in after.links if link.participant.has_session("one")]
        holder._channel.close()
        await wait_until(lambda: len(after.participant.workers) == 1)
        [other] = [link for link in after.links if link is not holder]
        resumed = await after.controller.resume(CheckpointRequest(**control("c2")))
        serving = other.accepting and other.participant.accepting
        left = [key for key in ("one", "two") if other.participant.has_session(key)]

    # The resume rolls the failed restore back rather than fail for good on the restored state the lost worker held.
    assert lost_coordinators == ["lost"]
    assert resumed["phase"] == "idle" and serving and left == []


async def test_a_holder_lost_after_a_retried_restore_succeeded_still_fails_its_resume(
    socket_dir: str, tmp_path: Path, lost_coordinators: list[str]
) -> None:
    await committed_sessions_on_two_workers(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        first = after.links[0]
        real_install, real_reopen = first.participant.install, after.participant.reopen_workers
        first.participant.install = AsyncMock(side_effect=RuntimeError("the sandbox could not be rebuilt"))
        # Admission stays closed after the failed restore, and the controller retries the restore, which succeeds.
        after.participant.reopen_workers = AsyncMock(side_effect=RuntimeError("the reopen failed"))
        with pytest.raises(ControlError):
            await after.controller.restore(restore_both(tmp_path / "ckpt"))
        first.participant.install, after.participant.reopen_workers = real_install, real_reopen
        await after.controller.restore(
            RestoreRequest(
                **control(
                    "r2",
                    checkpoint_dir=str(tmp_path / "ckpt"),
                    episode_ids=[{"rollout_id": "r1"}, {"rollout_id": "r2"}],
                )
            )
        )
        [holder] = [link for link in after.links if link.participant.has_session("one")]
        holder._channel.close()
        await wait_until(lambda: len(after.participant.workers) == 1)
        with pytest.raises(ControlError) as failed:
            await after.controller.resume(CheckpointRequest(**control("r2")))

    # The earlier failed restore's recovery does not apply to this one: its lost state fails the resume, as always.
    assert lost_coordinators == ["lost"] and failed.value.code == "checkpoint_worker_lost"


@pytest.mark.parametrize("failure", ["raises", "raises_cancelled_error", "hangs"])
async def test_a_restored_run_whose_teardown_fails_keeps_its_record_until_a_teardown_succeeds(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(participant_workers_module, "_SETTLE_RETRY_SECONDS", 0.05)
    monkeypatch.setattr(participant_workers_module, "_TEARDOWN_ATTEMPT_SECONDS", 0.2, raising=False)
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first, second = after.links
        attempts: list[str] = []
        second_attempt_may_finish = asyncio.Event()

        async def retire_agent_session(session_key: str) -> None:
            attempts.append("started")
            if len(attempts) == 1:
                if failure == "raises":
                    raise RuntimeError("the sandbox did not stop")
                if failure == "raises_cancelled_error":
                    # The hook's own call was cancelled, for example by a client library's timeout.
                    raise asyncio.CancelledError()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    attempts.append("cancelled")
                    raise
            await second_attempt_may_finish.wait()

        first.participant.hooks.retire_agent_session = retire_agent_session
        # The restored session belongs to attempt 1, so this /run cannot open it and rolls the install back.
        with pytest.raises(StaleAttemptError):
            async with first.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=2)):
                pass
        await wait_until(lambda: attempts.count("started") == 2)
        # The installed copy may still be on this worker, so the record cannot be claimed anywhere else yet.
        left = first.participant.has_session("run:r")
        claimable_before = "run:r" in after.participant.restored
        second_attempt_may_finish.set()
        await wait_until(lambda: "run:r" in after.participant.restored)
        continued = await replacement_run(second.participant)

    assert not left and not claimable_before
    assert continued == {"next": "loop"}
    # A teardown that does not return in time is cancelled, and tried again like one that failed.
    assert attempts == (["started", "cancelled", "started"] if failure == "hangs" else ["started", "started"])


async def test_a_stuck_teardown_keeps_its_record_blocks_prepare_by_name_and_a_run_of_its_key_gives_up(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(participant_workers_module, "_SETTLE_RETRY_SECONDS", 0.05)
    monkeypatch.setattr(participant_workers_module, "_TEARDOWN_ATTEMPT_SECONDS", 0.1, raising=False)
    monkeypatch.setattr(participant_workers_module, "MESSAGE_TIMEOUT_SECONDS", 0.5)
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first, second = after.links
        attempts = 0
        stuck = asyncio.Event()
        unstuck = asyncio.Event()

        async def retire_agent_session(session_key: str) -> None:
            nonlocal attempts
            attempts += 1
            if attempts >= 3:
                stuck.set()
            await unstuck.wait()

        first.participant.hooks.retire_agent_session = retire_agent_session
        with pytest.raises(StaleAttemptError):
            async with first.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=2)):
                pass
        async with asyncio.timeout(5):
            await stuck.wait()
        waiting = await after.controller.prepare(CheckpointRequest(**control("c2", timeout=0.3)))
        await after.controller.resume(CheckpointRequest(**control("c2")))
        # The replacement's /run lands on the worker still freeing the key, and is told to come back.
        with pytest.raises(participant_workers_module.ClaimSettlingError):
            await replacement_run(first.participant)
        claimable_while_stuck = "run:r" in after.participant.restored
        unstuck.set()
        await wait_until(lambda: "run:r" in after.participant.restored)
        continued = await replacement_run(second.participant)

    assert waiting["phase"] == "preparing" and "r-a1" in waiting["report"]["blockers"]
    assert not claimable_while_stuck
    assert continued == {"next": "loop"}


@pytest.mark.parametrize("teardown", ["always_fails", "succeeds_after_the_lease_ends"])
async def test_a_teardown_whose_lease_a_retire_ended_stops_within_its_cap(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, teardown: str
) -> None:
    monkeypatch.setattr(participant_workers_module, "_SETTLE_RETRY_SECONDS", 0.05)
    monkeypatch.setattr(participant_workers_module, "_TEARDOWN_ATTEMPTS_WITHOUT_LEASE", 2, raising=False)
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first = after.links[0]
        attempts = 0
        attempts_at_retire: Optional[int] = None
        settlements: list[str] = []
        real_call = first.call

        async def recorded(kind: str, body: Any, **kwargs: Any) -> dict:
            if kind == "settle_claim":
                settlements.append(body["key"])
            return await real_call(kind, body, **kwargs)

        async def retire_agent_session(session_key: str) -> None:
            nonlocal attempts
            attempts += 1
            if teardown == "succeeds_after_the_lease_ends" and attempts_at_retire is not None:
                return
            raise RuntimeError("the sandbox did not stop")

        first.call = recorded
        first.participant.hooks.retire_agent_session = retire_agent_session
        with pytest.raises(StaleAttemptError):
            async with first.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=2)):
                pass
        await wait_until(lambda: attempts >= 2)
        # The controller gives up on the restored attempt, which ends the record's lease.
        await after.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r", "attempt": 1}])))
        attempts_at_retire = attempts
        await wait_until(lambda: not first._settling and not first._tearing_down)
        attempts_at_the_end = attempts
        await asyncio.sleep(0.3)

    # Nothing keeps running for the rolled-back install: no task, and no entry naming it.
    assert attempts == attempts_at_the_end
    given_up = [record for record in caplog.records if "may be orphaned" in record.getMessage()]
    if teardown == "always_fails":
        # The attempt under way when the lease ended, and at most two more.
        assert attempts - attempts_at_retire <= 3 and settlements == []
        assert len(given_up) == 1 and "run:r" in given_up[0].getMessage() and "r-a1" in given_up[0].getMessage()
    else:
        # A teardown that succeeds within the budget settles its claim as before.
        assert settlements == ["run:r"] and given_up == []


async def test_a_teardown_that_ignores_cancellation_is_no_longer_awaited_once_its_lease_is_gone(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(participant_workers_module, "_SETTLE_RETRY_SECONDS", 0.05)
    monkeypatch.setattr(participant_workers_module, "_TEARDOWN_ATTEMPT_SECONDS", 0.1, raising=False)
    monkeypatch.setattr(participant_workers_module, "_TEARDOWN_CANCEL_GRACE_SECONDS", 0.1, raising=False)
    monkeypatch.setattr(participant_workers_module, "_TEARDOWN_ATTEMPTS_WITHOUT_LEASE", 2, raising=False)
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first = after.links[0]
        started = 0
        cancellations = 0
        release = asyncio.Event()

        async def retire_agent_session(session_key: str) -> None:
            nonlocal started, cancellations
            started += 1
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    # A hook that swallows its cancellation.
                    cancellations += 1

        first.participant.hooks.retire_agent_session = retire_agent_session
        try:
            with pytest.raises(StaleAttemptError):
                async with first.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=2)):
                    pass
            await wait_until(lambda: cancellations >= 1)
            await after.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r", "attempt": 1}])))
            await wait_until(lambda: not first._settling and not first._tearing_down)
        finally:
            # Lets the stuck call end, so the deployment can stop.
            release.set()
            await asyncio.sleep(0.05)

    # The stuck call was never started again alongside itself, and nothing waits for it any more.
    assert started == 1


async def test_a_key_whose_given_up_teardown_still_runs_cannot_be_claimed_on_its_worker_until_that_call_ends(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(participant_workers_module, "_SETTLE_RETRY_SECONDS", 0.05)
    monkeypatch.setattr(participant_workers_module, "_TEARDOWN_ATTEMPT_SECONDS", 0.1, raising=False)
    monkeypatch.setattr(participant_workers_module, "_TEARDOWN_CANCEL_GRACE_SECONDS", 0.1, raising=False)
    monkeypatch.setattr(participant_workers_module, "_TEARDOWN_ATTEMPTS_WITHOUT_LEASE", 1, raising=False)
    monkeypatch.setattr(participant_workers_module, "MESSAGE_TIMEOUT_SECONDS", 0.3)
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first = after.links[0]
        events: list[str] = []
        release = asyncio.Event()

        async def retire_agent_session(session_key: str) -> None:
            if "new run opened" in events:
                events.append("teardown freed the new run's session")
                return
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    # A hook that swallows its cancellation, and ends only later.
                    pass
            events.append("teardown ended")

        first.participant.hooks.retire_agent_session = retire_agent_session
        try:
            with pytest.raises(StaleAttemptError):
                async with first.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=2)):
                    pass
            await after.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r", "attempt": 1}])))
            # The teardown is given up while its call still runs.
            await wait_until(lambda: not first._settling and not first._tearing_down)
            with pytest.raises(participant_workers_module.ClaimSettlingError):
                async with first.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=2)):
                    events.append("new run opened")
        finally:
            release.set()
        # Once the call ends, the key can be claimed here again.
        await wait_until(lambda: "teardown ended" in events)
        async with first.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=2)):
            events.append("new run opened")
        blocked = dict(first._blocked)

    assert events == ["teardown ended", "new run opened"] and blocked == {}
    assert any("cannot be claimed on this worker until that call ends" in r.getMessage() for r in caplog.records)


async def test_a_run_that_reuses_a_key_whose_rolled_back_install_is_still_being_freed_waits_for_it(
    socket_dir: str, tmp_path: Path
) -> None:
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first = after.links[0]
        events: list[str] = []
        teardown_may_finish = asyncio.Event()

        async def retire_agent_session(session_key: str) -> None:
            events.append("teardown started")
            await teardown_may_finish.wait()
            events.append("teardown finished")

        first.participant.hooks.retire_agent_session = retire_agent_session
        async with asyncio.timeout(5):
            with pytest.raises(StaleAttemptError):
                async with first.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=2)):
                    pass
        await wait_until(lambda: events == ["teardown started"])
        # The controller gives up on the restored attempt, which ends the record's lease.
        await after.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r", "attempt": 1}])))

        async def later_run() -> None:
            async with first.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=2)):
                events.append("new run opened")

        run = asyncio.create_task(later_run())
        await asyncio.sleep(0.3)
        teardown_may_finish.set()
        await run

    # The teardown of the rolled-back install never frees the session the later /run opened under the same key.
    assert events == ["teardown started", "teardown finished", "new run opened"]


@pytest.mark.parametrize("next_claim", ["another_worker", "the_same_worker"])
async def test_a_settlement_retried_after_another_claim_leased_the_record_changes_nothing(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, next_claim: str
) -> None:
    monkeypatch.setattr(participant_workers_module, "_SETTLE_RETRY_SECONDS", 0.05)
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first, second = after.links
        settlements = 0
        retry_may_arrive = asyncio.Event()
        retried = asyncio.Event()
        real_call = first.call

        async def first_reply_lost(kind: str, body: Any, **kwargs: Any) -> dict:
            nonlocal settlements
            if kind != "settle_claim":
                return await real_call(kind, body, **kwargs)
            settlements += 1
            if settlements == 1:
                await real_call(kind, body, **kwargs)
                # The coordinator settled the claim, and its reply never reached this worker.
                raise workers_module.CoordinatorUnavailableError("the settlement reply was lost")
            if settlements == 2:
                await retry_may_arrive.wait()
                reply = await real_call(kind, body, **kwargs)
                retried.set()
                return reply
            return await real_call(kind, body, **kwargs)

        first.call = first_reply_lost
        with pytest.raises(RuntimeError):
            async with first.claim("run:r") as record:
                assert record is not None
                raise RuntimeError("the install failed")
        await wait_until(lambda: "run:r" in after.participant.restored)
        claimant = second if next_claim == "another_worker" else first
        async with claimant.claim("run:r") as record:
            later_lease = after.participant.leases["run:r"]
            # The first claim's settlement is sent again, and arrives while the later claim holds the record.
            retry_may_arrive.set()
            async with asyncio.timeout(5):
                await retried.wait()
            lease_after_the_retry = after.participant.leases.get("run:r")
            returned_by_the_retry = "run:r" in after.participant.restored
        await wait_until(lambda: not after.participant.leases)

    assert record is not None
    assert lease_after_the_retry == later_lease and not returned_by_the_retry
    # The later claim settled its own lease: the record was installed, so it does not go back.
    assert "run:r" not in after.participant.restored


@pytest.mark.parametrize("teardown", ["finishes_within_the_grace", "outlasts_the_grace"])
async def test_a_clean_disconnect_waits_a_bounded_grace_for_a_teardown_in_flight(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, teardown: str
) -> None:
    monkeypatch.setattr(participant_workers_module, "_TEARDOWN_GRACE_SECONDS", 0.5, raising=False)
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first = after.links[0]
        events: list[str] = []
        started = asyncio.Event()

        async def retire_agent_session(session_key: str) -> None:
            events.append("teardown started")
            started.set()
            try:
                await asyncio.sleep(0.2 if teardown == "finishes_within_the_grace" else 60)
            except asyncio.CancelledError:
                events.append("teardown cancelled")
                raise
            events.append("teardown finished")

        first.participant.hooks.retire_agent_session = retire_agent_session
        leases_when_the_worker_left: list[list[str]] = []
        real_leave = after.participant.leave

        async def leave(worker_index: int) -> None:
            leases_when_the_worker_left.append(sorted(after.participant.leases))
            await real_leave(worker_index)

        after.participant.leave = leave
        with pytest.raises(StaleAttemptError):
            async with first.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=2)):
                pass
        await started.wait()
        began = time.monotonic()
        await first.disconnect()
        took = time.monotonic() - began
        await wait_until(lambda: "run:r" in after.participant.restored)

    if teardown == "finishes_within_the_grace":
        # The teardown finished and settled the claim before the worker left.
        assert events == ["teardown started", "teardown finished"] and took < 0.5
        assert leases_when_the_worker_left[0] == []
    else:
        # Past the grace the teardown is cancelled, and the coordinator takes the record back as the worker leaves.
        assert events == ["teardown started", "teardown cancelled"] and 0.5 <= took < 2.0
        assert leases_when_the_worker_left[0] == ["run:r"]


async def test_a_claim_that_arrives_while_a_failed_claim_settles_waits_and_continues_from_the_record(
    socket_dir: str, tmp_path: Path
) -> None:
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, gated_agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        first, second = after.links
        first.participant.hooks.fail = True
        settle_may_arrive = asyncio.Event()
        real_call = first.call

        async def slow_settlement(kind: str, body: Any, **kwargs: Any) -> dict:
            if kind == "settle_claim":
                await settle_may_arrive.wait()
            return await real_call(kind, body, **kwargs)

        first.call = slow_settlement
        with pytest.raises(RuntimeError):
            await replacement_run(first.participant)
        # The failed claim's record is on its way back; the retried /run reaches the other worker meanwhile.
        retried = asyncio.create_task(replacement_run(second.participant))
        await asyncio.sleep(0.3)
        waited = not retried.done()
        settle_may_arrive.set()
        continued = await retried

    assert waited and continued == {"next": "loop"}


async def test_a_retired_session_s_cookie_is_refused_on_every_worker_until_forget(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with counter_deployment(socket_dir, monkeypatch, window=0.2) as deployment:
        holder, other = deployment.apps
        async with http(holder) as worker:
            seeded = await worker.post("/ng-rollout/r/seed_session", json={"initial_count": 3})
        cookie = dict(seeded.cookies)
        [session_id] = deployment.counters[0].session_id_to_counter
        await deployment.coordinator.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r"}])))
        # Well past the close window and the sweeps after it.
        await asyncio.sleep(0.6)
        claimed_after_retire = claimant(deployment._routing_dir, session_id)
        async with http(other) as worker:
            # A late request of the retired attempt that carries only the cookie lands on the other worker.
            refused = await worker.post("/get_counter_value", cookies=cookie)
        await deployment.coordinator.controller.forget(ForgetRequest(**control(rollout_ids=["r"])))
        await asyncio.sleep(0.6)
        claimed_after_forget = claimant(deployment._routing_dir, session_id)
        left = [dict(counter.session_id_to_counter) for counter in deployment.counters]

    assert claimed_after_retire == holder.state.nemo_gym_routing_id
    assert refused.status_code == 409 and refused.json()["error"]["code"] == "checkpoint_stale_attempt"
    # The forget ends the claim's life, one close window later; nothing of the session is left on either worker.
    assert claimed_after_forget is None and left == [{}, {}]


async def test_a_clean_disconnect_stops_the_worker_link_s_background_tasks(socket_dir: str, tmp_path: Path) -> None:
    await committed_run_episode(socket_dir, tmp_path / "ckpt")
    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r"}])
        link = after.links[0]
        disconnected = False
        after_disconnect: list[str] = []
        real_call = link.call

        async def unanswered_reports_and_settlements(kind: str, body: Any, **kwargs: Any) -> dict:
            if kind in ("report", "settle_claim"):
                if disconnected:
                    after_disconnect.append(kind)
                raise workers_module.CoordinatorUnavailableError("the coordinator did not answer")
            return await real_call(kind, body, **kwargs)

        link.call = unanswered_reports_and_settlements
        # A claim that fails: its settlement keeps retrying.
        with pytest.raises(StaleAttemptError):
            async with link.participant.legacy_run("run:r", EpisodeId(rollout_id="r", attempt=2)):
                pass
        # A checkpoint opens: the worker polls its readiness, and its report keeps retrying.
        await after.controller.prepare(CheckpointRequest(**control(timeout=0.2)))
        await link.participant.notify()
        await asyncio.sleep(0.2)
        tasks = [link._poll_task, link._report_task, *link._settling]
        await link.disconnect()
        disconnected = True
        await asyncio.sleep(0.3)
        await after.controller.resume(CheckpointRequest(**control()))

    assert len(tasks) == 3 and all(task is not None and task.done() for task in tasks)
    assert link._channel is None and not link._settling
    assert after_disconnect == []


async def test_an_export_sends_its_first_chunk_before_it_has_encoded_every_record(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(participant_workers_module, "EXPORT_CHUNK_BYTES", 100, raising=False)
    encoded = 0
    to_json_record = AgentSessionRecord.to_json_record

    def counted(record: AgentSessionRecord) -> dict:
        nonlocal encoded
        encoded += 1
        return to_json_record(record)

    monkeypatch.setattr(AgentSessionRecord, "to_json_record", counted)
    encoded_at_first_chunk: list[int] = []
    keys = [f"k{index}" for index in range(20)]
    async with coordinated(socket_dir, agent_worker, workers=1, expected=1) as deployment:
        link = deployment.links[0]
        real_call = link.call

        async def watching(kind: str, body: Any, **kwargs: Any) -> dict:
            if kind == "export_chunk" and not encoded_at_first_chunk:
                encoded_at_first_chunk.append(encoded)
            return await real_call(kind, body, **kwargs)

        link.call = watching
        for key in keys:
            link.participant.open_session(key, EpisodeId(rollout_id=key))
        await commit(deployment, tmp_path / "ckpt", keys)
        await deployment.controller.resume(CheckpointRequest(**control()))

    assert sorted(record["session_key"] for record in records(tmp_path / "ckpt", "agent")) == sorted(keys)
    # Only the first few chunks were encoded when the first one went out, not the whole export.
    assert encoded_at_first_chunk and encoded_at_first_chunk[0] <= 6
