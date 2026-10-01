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
    RestoreRequest,
    RetireRequest,
)
from nemo_gym._checkpoint.environment import EnvironmentParticipant, task_digest
from nemo_gym._checkpoint.errors import AdmissionClosedError, ControlError, InvalidPhaseError
from nemo_gym._checkpoint.participant_workers import (
    COORDINATOR_SOCKET_ENV,
    CoordinatedServerParticipant,
    ParticipantWorkerLink,
)
from nemo_gym._checkpoint.resources import ResourcesParticipant
from nemo_gym._checkpoint.workers import WorkerCoordinator
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, ServerClient
from nemo_gym.session_routing import install_session_routing, session_aliases
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
    app.state.nemo_gym_session_owner = uuid4().hex
    return AgentSessionParticipant(AgentHooks()), app


def records(checkpoint_dir: Path, kind: str) -> list[dict]:
    path = checkpoint_dir / "gym" / kind / kind / "records.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


async def commit(deployment: Workers, checkpoint_dir: Path, rollout_ids: list[str], checkpoint_id: str = "c1") -> None:
    prepared = await deployment.controller.prepare(CheckpointRequest(**control(checkpoint_id)))
    assert prepared["phase"] == "prepared", prepared
    episode_ids = [{"rollout_id": rollout_id} for rollout_id in rollout_ids]
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
        missed = await deployment.controller.prepare(CheckpointRequest(**control(timeout=0.3)))
        # A session that ends records no notification; the worker re-reports it while the checkpoint is open.
        second.participant.ended("s")
        prepared = await deployment.controller.prepare(CheckpointRequest(**control()))
        await deployment.controller.resume(CheckpointRequest(**control()))

    assert missed["report"]["blockers"] == ["r"]
    assert prepared["phase"] == "prepared"


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
        await commit(deployment, tmp_path / "ckpt2", ["r"], checkpoint_id="c2")
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


async def test_owners_whose_sessions_exported_nothing_are_aliased_too(socket_dir: str, tmp_path: Path) -> None:
    """A stateless server exports no session, but its old cookies must still reach a live worker."""
    async with coordinated(socket_dir, agent_worker) as before:
        old_owners = sorted(link.routing_id for link in before.links)
        await commit(before, tmp_path / "ckpt", [])
        await before.controller.resume(CheckpointRequest(**control()))

    async with coordinated(socket_dir, agent_worker) as after:
        await restore(after, tmp_path / "ckpt", [])
        table = dict(session_aliases(after.links[0].app))
        live = {link.routing_id for link in after.links}

    assert sorted(table) == old_owners
    # Spread over the live workers, like owners with records.
    assert set(table.values()) == live


async def test_a_retire_leaves_no_fence_on_any_worker_or_on_one_that_joins_later(socket_dir: str) -> None:
    async with coordinated(socket_dir, environment_worker) as deployment:
        await deployment.controller.retire(RetireRequest(**control(episode_ids=[{"rollout_id": "r"}])))
        late = await deployment.join()
        fences = [len(link.attempts) for link in deployment.links] + [len(deployment.participant.attempts)]

    assert late in deployment.links
    assert fences == [0] * (len(deployment.links) + 1)


async def test_a_restored_resources_session_serves_a_request_with_its_old_cookie_on_any_worker(
    socket_dir: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real counter servers with session routing: the old cookie reaches the worker that holds the session."""
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, "1")
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {"checkpoint": {"enabled": True, "control_auth_token": "t"}}
    config = StatefulCounterResourcesServerConfig(
        host="", port=0, entrypoint="", name="resources", num_workers=2, domain="agent", verified=False
    )

    @asynccontextmanager
    async def counter_workers() -> AsyncIterator[tuple[WorkerCoordinator, list[StatefulCounterResourcesServer], list]]:
        """A coordinator and two counter server workers, each built as uvicorn builds a worker."""
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
                    counter = StatefulCounterResourcesServer(config=config, server_client=client)
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

    async with counter_workers() as (coordinator, _, apps):
        cookies = []
        for index, app in enumerate(apps):
            async with http(app) as worker:
                seeded = await worker.post(f"/ng-rollout/r{index}/seed_session", json={"initial_count": 10 * index})
                await worker.post(f"/ng-rollout/r{index}/increment_counter", json={"count": 1})
                cookies.append(dict(seeded.cookies))
        await commit(coordinator, tmp_path / "ckpt", ["r0", "r1"])
        await coordinator.controller.resume(CheckpointRequest(**control()))
    exported = records(tmp_path / "ckpt", "resources")

    async with counter_workers() as (coordinator, counters, apps):
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
