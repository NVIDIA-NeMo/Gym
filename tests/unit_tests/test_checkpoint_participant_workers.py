# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Environment, agent, and resources participants across several workers, over a real coordinator socket."""

import asyncio
import json
import os
import shutil
import tempfile
import time
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock
from uuid import uuid4

import httpx
import pytest
from fastapi import FastAPI
from pydantic import JsonValue

import nemo_gym._checkpoint.workers as workers_module
from nemo_gym._checkpoint.agent import AgentSessionParticipant, RestoredAgentSession
from nemo_gym._checkpoint.control import (
    CheckpointParticipant,
    CheckpointRequest,
    CommitRequest,
    ForgetRequest,
    RestoreRequest,
    RetireRequest,
)
from nemo_gym._checkpoint.environment import EnvironmentParticipant, task_digest
from nemo_gym._checkpoint.errors import (
    AdmissionClosedError,
    ControlError,
    InvalidPhaseError,
    RestartInScopeError,
    StaleAttemptError,
)
from nemo_gym._checkpoint.participant_workers import (
    COORDINATOR_SOCKET_ENV,
    CoordinatedServerParticipant,
    ParticipantWorkerLink,
)
from nemo_gym._checkpoint.resources import ResourcesParticipant
from nemo_gym._checkpoint.workers import WorkerCoordinator
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, ServerClient
from nemo_gym.session_routing import install_session_routing, session_aliases, session_placements
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
        await link.connect()
        self._stack.push_async_callback(link.disconnect)
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
        with pytest.raises(InvalidPhaseError, match="no longer ready"):
            await deployment.controller.commit(
                CommitRequest(**control(checkpoint_dir=str(tmp_path / "ckpt"), episode_ids=[]))
            )
        await deployment.controller.resume(CheckpointRequest(**control()))
        # uvicorn restarts the worker; the next checkpoint proceeds.
        await deployment.join()
        prepared = await deployment.controller.prepare(CheckpointRequest(**control("c2")))

    assert report.blockers == ["environment-workers-unreported:1", "environment-worker-lost"]
    assert prepared["phase"] == "prepared"


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


async def test_a_restored_legacy_agent_run_is_claimed_by_exactly_one_worker(socket_dir: str, tmp_path: Path) -> None:
    async with coordinated(socket_dir, agent_worker) as before:
        participant = before.links[0].participant
        started = asyncio.Event()

        async def legacy() -> None:
            async with participant.legacy_run("run:r", EpisodeId(rollout_id="r")) as run:
                await run.boundary({"next": "loop"})
                async with run.step("replay"):
                    started.set()
                    await asyncio.Event().wait()

        task = asyncio.create_task(legacy())
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


# -- multi-request sessions: placement and aliases -------------------------------------------------


async def test_a_restored_agent_session_is_installed_on_one_worker_and_every_router_aliases_it(
    socket_dir: str, tmp_path: Path
) -> None:
    async with coordinated(socket_dir, agent_worker) as before:
        first, second = before.links
        first.participant.open_session("one", EpisodeId(rollout_id="r1"))
        second.participant.open_session("two", EpisodeId(rollout_id="r2"))
        old_owners = {"one": first.routing_id, "two": second.routing_id}
        await commit(before, tmp_path / "ckpt", ["r1", "r2"])
        await before.controller.resume(CheckpointRequest(**control()))

    async with coordinated(socket_dir, agent_worker, expected=3) as after:
        await restore(after, tmp_path / "ckpt", [{"rollout_id": "r1"}, {"rollout_id": "r2"}])
        holders = {
            key: next(link.routing_id for link in after.links if link.participant.has_session(key))
            for key in ("one", "two")
        }
        tables = [dict(session_aliases(link.app)) for link in after.links]
        # A worker that joins later, or restarts, gets the current table.
        late = await after.join()
        late_table = dict(session_aliases(late.app))

    # The two pre-crash owners are spread over the two live workers.
    assert len(set(holders.values())) == 2
    expected_table = {old_owners[key]: holders[key] for key in holders}
    assert tables == [expected_table, expected_table]
    assert late_table == expected_table


async def test_a_stateless_session_keeps_its_owner_through_restores_and_no_other_owner_is_carried(
    socket_dir: str, tmp_path: Path
) -> None:
    """A stateless session has no state, but its old cookie must reach a live worker after every restore."""

    def stateless_worker() -> tuple[ResourcesParticipant, FastAPI]:
        app = FastAPI()
        app.state.nemo_gym_routing_id = uuid4().hex
        return ResourcesParticipant(SessionStates(), "stateless"), app

    async with coordinated(socket_dir, stateless_worker) as first:
        first.links[0].participant.seeded("s", EpisodeId(rollout_id="r"))
        owner = first.links[0].routing_id
        await commit(first, tmp_path / "c1", ["r"])
        await first.controller.resume(CheckpointRequest(**control()))

    async with coordinated(socket_dir, stateless_worker) as second:
        await restore(second, tmp_path / "c1", [{"rollout_id": "r"}])
        # The restored session's episode is still continued, by its replacement attempt.
        await commit(second, tmp_path / "c2", ["r-a1"])
        await second.controller.resume(CheckpointRequest(**control()))

    async with coordinated(socket_dir, stateless_worker) as third:
        await restore(third, tmp_path / "c2", [{"rollout_id": "r", "attempt": 1}])
        tables = [dict(session_aliases(link.app)) for link in third.links]
        holder = next(link for link in third.links if link.participant.readiness().counts["sessions"])

    # Only the owner the session's cookie names is carried, not every worker of every earlier generation.
    assert all(list(table) == [owner] for table in tables)
    assert tables[0][owner] == holder.routing_id


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


async def test_a_session_placed_without_an_owner_stays_routable_through_later_restores(
    socket_dir: str, tmp_path: Path
) -> None:
    """Placed once by session ID, a session follows its new owner, and keeps a table entry for its old cookie."""
    async with coordinated(socket_dir, resources_worker(routed=False)) as before:
        for index, link in enumerate(before.links):
            link.participant.hooks.states[f"s{index}"] = index
            link.participant.seeded(f"s{index}", EpisodeId(rollout_id=f"r{index}"))
        await commit(before, tmp_path / "ckpt1", ["r0", "r1"])
        await before.controller.resume(CheckpointRequest(**control()))

    episode_ids = [{"rollout_id": "r0", "attempt": 1}, {"rollout_id": "r1", "attempt": 1}]
    async with coordinated(socket_dir, resources_worker(routed=True)) as first:
        await restore(first, tmp_path / "ckpt1", [{"rollout_id": "r0"}, {"rollout_id": "r1"}])
        first_table = dict(session_placements(first.links[0].app))
        await commit(first, tmp_path / "ckpt2", ["r0-a1", "r1-a1"], checkpoint_id="c2")
        await first.controller.resume(CheckpointRequest(**control("c2")))
    second_owners = {record["session_id"]: record["owner"] for record in records(tmp_path / "ckpt2", "resources")}

    async with coordinated(socket_dir, resources_worker(routed=True)) as second:
        await restore(second, tmp_path / "ckpt2", episode_ids, restore_id="r2")
        table = dict(session_placements(second.links[0].app))
        aliases = dict(session_aliases(second.links[0].app))
        holders = {
            session_id: link.routing_id for link in second.links for session_id in link.participant.hooks.states
        }

    # The first restore spread the two sessions over both workers, which then owned them.
    assert len(set(first_table.values())) == 2
    assert second_owners == first_table
    # After the next restore,
    # an old cookie without an owner and a newer one naming its owner both arrive where the session is.
    assert table == holders
    assert {session_id: aliases[owner] for session_id, owner in second_owners.items()} == holders


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


def counter_config(num_workers: int) -> StatefulCounterResourcesServerConfig:
    return StatefulCounterResourcesServerConfig(
        host="",
        port=0,
        entrypoint="",
        name="resources",
        num_workers=num_workers,
        domain="agent",
        verified=False,
    )


def checkpointing_client() -> ServerClient:
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    return client


@asynccontextmanager
async def counter_workers(
    socket_dir: str, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[WorkerCoordinator, list[StatefulCounterResourcesServer], list[FastAPI]]]:
    """A coordinator and two counter server workers with session routing, each built as uvicorn builds one."""
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
    counters, apps = [], []
    try:
        async with AsyncExitStack() as stack:
            for _ in range(2):
                counter = StatefulCounterResourcesServer(
                    config=counter_config(2), server_client=checkpointing_client()
                )
                # The checkpoint link, then session routing outside every other middleware.
                app = counter.setup_webserver()
                key = counter.get_session_middleware_key()
                install_session_routing(app, socket_dir=routing_dir, session_cookie=key, secret_key=key)
                await stack.enter_async_context(app.router.lifespan_context(app))
                counters.append(counter)
                apps.append(app)
            yield coordinator, counters, apps
    finally:
        server.close()
        await server.wait_closed()
        shutil.rmtree(routing_dir, ignore_errors=True)


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
        for cookie in cookies:
            for app in apps:
                async with http(app) as worker:
                    counts.append((await worker.post("/get_counter_value", cookies=cookie)).json()["count"])
        held = [sorted(counter.session_id_to_counter.values()) for counter in counters]

    # Each session was created on its own worker and is exported with that worker as its owner.
    assert len({record["owner"] for record in exported}) == 2
    # Either worker serves either restored session: one directly, the other by forwarding to its holder.
    assert counts == [1, 1, 11, 11]
    # The two pre-crash owners went to different workers, and no request ran against a worker without the state.
    assert sorted(held) == [[1], [11]]


async def test_sessions_of_a_single_process_server_restore_onto_several_workers(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checkpoint taken with one worker: its cookies name no worker, yet still reach their session's state.

    The MCP token path through the same table is covered in test_session_routing.py:
    checkpointing and MCP exposure are not combined on one server.
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
        for cookie in cookies:
            for app in apps:
                async with http(app) as worker:
                    await worker.post("/increment_counter", json={"count": 1}, cookies=cookie)
                    counts.append((await worker.post("/get_counter_value", cookies=cookie)).json()["count"])
        held = [sorted(counter.session_id_to_counter.values()) for counter in counters]
        tables = [dict(session_placements(app)) for app in apps]

    assert {record["owner"] for record in exported} == {None}
    # Each worker incremented each session once, whichever worker held it.
    assert counts == [1, 2, 11, 12, 21, 22, 31, 32]
    # Spread evenly, one session at a time, and no request ran against a worker without the state.
    assert sorted(len(values) for values in held) == [2, 2]
    assert sorted(value for values in held for value in values) == [2, 12, 22, 32]
    # Every worker's router knows where each of them is.
    assert tables[0] == tables[1]
    assert set(tables[0]) == {record["session_id"] for record in exported}


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
    assert reply["error"]["code"] == "invalid_checkpoint_state"


async def test_a_worker_that_joins_while_a_restore_installs_gets_the_routing_tables_on_reopen(
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

    async with coordinated(socket_dir, slow_agent_worker, expected=3) as after:
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
        # Joins while the install runs, so its join state still has the tables from before the restore.
        late = await after.join()
        joined.set()
        await restoring
        await after.controller.resume(CheckpointRequest(**control("r1")))
        expected = dict(session_aliases(after.links[0].app))

    assert expected and dict(session_aliases(late.app)) == expected


async def test_a_worker_whose_readiness_regressed_refuses_to_export(socket_dir: str, tmp_path: Path) -> None:
    async with coordinated(socket_dir, environment_worker) as deployment:
        prepared = await deployment.controller.prepare(CheckpointRequest(**control()))
        # Readiness that went backwards after the worker's last report reached the coordinator.
        deployment.links[1].participant.ready = lambda: False
        with pytest.raises(ControlError) as refused:
            await deployment.controller.commit(CommitRequest(**control(checkpoint_dir=str(tmp_path))))
        await deployment.controller.resume(CheckpointRequest(**control()))

    assert prepared["phase"] == "prepared" and refused.value.code == "invalid_phase"


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
    assert refused.value.code == "retire_incomplete"
    assert kept == [1, 1, 1]
    assert forgotten == [0, 0, 0]


def test_a_seed_s_owner_replaces_the_restored_one_and_a_protocol_start_keeps_it() -> None:
    participant = ResourcesParticipant(SessionStates(), "exported")
    participant.owner = "this-worker"
    participant._owners["x"] = "old-owner"
    participant.seeded("x", EpisodeId(rollout_id="r", attempt=1))
    kept = participant._owners["x"]
    participant.seeded("x", EpisodeId(rollout_id="r", attempt=1), "this-worker")

    assert kept == "old-owner"
    # The seed's reply cookie names the worker that served it.
    assert participant._owners["x"] == "this-worker"


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
        other.participant.seeded("x", EpisodeId(rollout_id="r", attempt=1), other.routing_id)
        await commit(after, tmp_path / "ckpt2", ["r-a1"], checkpoint_id="c2")
        await after.controller.resume(CheckpointRequest(**control("c2")))
        stale_left = "x" in holder.participant.hooks.states or bool(holder.participant.unclaimed_sessions())

    exported = records(tmp_path / "ckpt2", "resources")
    assert [(record["session_id"], record["state"]) for record in exported] == [("x", "seeded again")]
    assert not stale_left
