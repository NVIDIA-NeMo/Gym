# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Random sequences of checkpoint operations against a whole deployment, driven through ``coordination``.

A deployment here is an environment server, an agent, a resources server, and a policy model,
each a real participant behind its own control routes.
Each episode runs as it does in production: the environment server steps it, each step is an agent activation,
and each activation makes policy model calls and resources tool calls on the episode's sessions.
Each seed interleaves those steps with prepare, commit, resume, retire, forget,
and restore through the coordination functions a controller calls, at short or long deadlines,
while writes are held or fail and one participant's close or install fails.

After every step:

- every participant is open whenever every participant is idle;
- prepare closes participants in Gym's order and resume reopens them in reverse; retire stops callers first;
- after a retire that succeeded, every participant refuses the retired attempt;
- a commit returns, for every participant, only episodes the controller continues;
- every manifest on disk verifies and stores that participant's first export of the checkpoint;
- a checkpoint resumed without a manifest never gets one.

Every committed checkpoint must restore into a fresh deployment;
a restore that fails anywhere must leave every participant open with nothing restored,
and a restore that succeeds must restore agent
and resources sessions only for episodes the environment server restored.

Set ``NEMO_GYM_CHECKPOINT_MIXED_SEQUENCE_SEEDS`` to run more seeds.
"""

import asyncio
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import JsonValue

from nemo_gym._checkpoint import coordination
from nemo_gym._checkpoint.agent import AgentSessionParticipant, RestoredAgentSession
from nemo_gym._checkpoint.control import CheckpointParticipant, CheckpointRecord, install_participant
from nemo_gym._checkpoint.coordination import PREPARE_ORDER, RETIRE_ORDER, CoordinationError, Participants
from nemo_gym._checkpoint.environment import EnvironmentParticipant
from nemo_gym._checkpoint.errors import AdmissionClosedError, ControlError, StaleAttemptError
from nemo_gym._checkpoint.model import PolicyModelParticipant
from nemo_gym._checkpoint.resources import ResourcesParticipant
from nemo_gym._checkpoint.store import participant_dir, read_participant_state
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import BaseServerConfig
from tests.unit_tests.test_checkpoint_coordination import InProcessClient
from tests.unit_tests.test_checkpoint_sequences import GatedWrites


SEEDS = int(os.environ.get("NEMO_GYM_CHECKPOINT_MIXED_SEQUENCE_SEEDS", "12"))
STEPS = 40
SHORT = 0.02
LONG = 2.0
TOKEN = "t"
TASK = {"task": "t"}
INSTALL_FAILURE = "install failed"
CLOSE_FAILURE = "admission failed to close part way"
KINDS = ("environment", "model", "agent", "resources")


class AgentHooks:
    def __init__(self) -> None:
        self.sessions: dict[str, dict[str, JsonValue]] = {}

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        return {key: self.sessions.get(key, {}) for key in session_keys}

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        self.sessions.update({session.session_key: dict(session.session) for session in sessions})

    async def retire_agent_session(self, session_key: str) -> None:
        self.sessions.pop(session_key, None)


class SessionStates:
    def __init__(self) -> None:
        self.states: dict[str, JsonValue] = {}

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        return {session_id: self.states[session_id] for session_id in session_ids if session_id in self.states}

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        self.states.update(states)

    async def retire_session_state(self, session_id: str) -> None:
        self.states.pop(session_id, None)


class Faults:
    """Wraps a participant so its next install can fail, and records the order of its admission changes."""

    def __init__(self, participant: CheckpointParticipant, name: str, events: list[tuple[str, str]]) -> None:
        self.fail_next_install = False
        self.fail_next_close = False
        install, close, open_, retire = (
            participant.install,
            participant.close_admission,
            participant.open_admission,
            participant.retire,
        )

        async def failing_install(records: list[CheckpointRecord], scope: list[EpisodeId]) -> None:
            if self.fail_next_install:
                self.fail_next_install = False
                raise ControlError(INSTALL_FAILURE)
            await install(records, scope)

        async def recording_close(request: Any) -> None:
            events.append(("close", name))
            await close(request)
            if self.fail_next_close:
                # Closed, then failed: the caller must reopen it.
                self.fail_next_close = False
                raise ControlError(CLOSE_FAILURE)

        async def recording_open() -> None:
            events.append(("open", name))
            await open_()

        async def recording_retire(episode_id: EpisodeId) -> None:
            events.append(("retire", name))
            await retire(episode_id)

        participant.install = failing_install  # type: ignore[method-assign]
        participant.close_admission = recording_close  # type: ignore[method-assign]
        participant.open_admission = recording_open  # type: ignore[method-assign]
        participant.retire = recording_retire  # type: ignore[method-assign]


class Deployment:
    """One participant of each kind, each behind its own control routes, reached by name like real servers."""

    def __init__(self) -> None:
        self.environment = EnvironmentParticipant()
        self.agent_hooks = AgentHooks()
        self.agent = AgentSessionParticipant(self.agent_hooks)
        self.resources_states = SessionStates()
        self.resources = ResourcesParticipant(self.resources_states, "exported")
        self.model = PolicyModelParticipant()
        self.by_kind: dict[str, CheckpointParticipant] = {
            "environment": self.environment,
            "model": self.model,
            "agent": self.agent,
            "resources": self.resources,
        }
        self.events: list[tuple[str, str]] = []
        self.faults = {kind: Faults(participant, kind, self.events) for kind, participant in self.by_kind.items()}
        self.first_exports: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.exporting = ""
        apps: dict[str, FastAPI] = {}
        global_config: dict[str, Any] = {}
        server_types = {
            "environment": "environment_servers",
            "model": "responses_api_models",
            "agent": "responses_api_agents",
            "resources": "resources_servers",
        }
        for kind, participant in self.by_kind.items():
            self._remember_exports(kind, participant)
            app = FastAPI()
            install_participant(app, participant, auth_token=TOKEN, instance_name=kind, lease_grace_seconds=60)
            apps[kind] = app
            global_config[kind] = {server_types[kind]: {kind: {}}}
        self.client = InProcessClient(
            head_server_config=BaseServerConfig(host="head", port=1),
            global_config_dict=OmegaConf.create(global_config),
            apps=apps,
        )
        self.participants: Optional[Participants] = None

    def _remember_exports(self, kind: str, participant: CheckpointParticipant) -> None:
        export = participant.export

        async def remembering(episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
            records = await export(episode_ids)
            self.first_exports.setdefault((self.exporting, kind), [record.to_json_record() for record in records])
            return records

        participant.export = remembering  # type: ignore[method-assign]

    async def discover(self) -> Participants:
        if self.participants is None:
            self.participants = await coordination.discover(self.client, auth_token=TOKEN)
        return self.participants

    def open_everywhere(self) -> bool:
        return (
            not self.environment.steps.closed
            and self.agent.accepting
            and self.resources.accepting
            and self.model.gate.accepting
        )


class Episode:
    """An episode the environment server steps; each step is an agent activation with model and tool calls."""

    def __init__(self, deployment: Deployment, rollout_id: str, rng: random.Random) -> None:
        self.deployment = deployment
        self.episode_id = EpisodeId(rollout_id=rollout_id)
        self.key = self.episode_id.capture_key
        self.session_id = f"s-{rollout_id}"
        self.rng = rng
        self.advance = asyncio.Event()
        # Released at once, or held so a retire's wait for the episode is cut short by its deadline.
        self.cleanup = asyncio.Event()
        if rng.random() < 0.6:
            self.cleanup.set()
        self.finish = False
        self.task = asyncio.create_task(self.run())

    async def _next(self) -> None:
        await self.advance.wait()
        self.advance.clear()

    async def model_call(self) -> None:
        gate = self.deployment.model.gate
        # The policy model waits out an open checkpoint before it admits a call.
        await gate.admit(self.key)
        ticket = gate.enter(self.key)
        try:
            await self._next()
            await gate.deliver_response(ticket)
        finally:
            await gate.exit(ticket)

    async def tool_call(self) -> None:
        resources = self.deployment.resources
        while True:
            try:
                resources.admit(self.session_id, "/increment")
                break
            except AdmissionClosedError:
                await resources.wait_open()
        resources.request_started(self.session_id, counted=True)
        try:
            states = self.deployment.resources_states.states
            states[self.session_id] = {"count": int((states.get(self.session_id) or {}).get("count", 0)) + 1}
        finally:
            resources.request_ended(self.session_id, counted=True)
            await resources.notify()

    async def seed_resources(self) -> None:
        """A ``/seed_session`` as the resources middleware admits it:
        it counts as a seed of its attempt before it is admitted, waits out an open checkpoint,
        and counts as a request in flight until its session is registered."""
        resources = self.deployment.resources
        resources.seed_started(self.episode_id)
        try:
            while not resources.accepting:
                await resources.wait_open()
            resources.request_started(None, counted=True)
            try:
                resources.seeded(self.session_id, self.episode_id)
                self.deployment.resources_states.states.setdefault(self.session_id, {"count": 0})
            finally:
                resources.request_ended(None, counted=True)
        finally:
            resources.seed_ended(self.episode_id)
            await resources.notify()

    async def run(self) -> None:
        environment, agent, resources = self.deployment.environment, self.deployment.agent, self.deployment.resources
        environment.begin(self.episode_id, TASK, None)
        try:
            # Seeding, as single_agent_turn does it: a replay step a checkpoint does not wait for.
            async with environment.step(self.episode_id, "replay"):
                await self.seed_resources()
                await self._next()
                async with agent.seeding(self.key, self.episode_id):
                    agent.open_session(self.key, self.episode_id, seed=True)
            await environment.boundary(self.episode_id, {"step": 0})
            step = 0
            while not self.finish:
                await self._next()
                # An agent activation continues from the agent's own boundaries, so the step may be replayed.
                async with environment.step(self.episode_id, "replay"):
                    async with agent.activation(self.key, self.episode_id) as activation:
                        for turn in range(self.rng.randint(1, 2)):
                            async with activation.awaiting_model():
                                await self.model_call()
                            await self.tool_call()
                            await activation.boundary(lambda step=step, turn=turn: {"step": step, "turn": turn})
                            await self._next()
                step += 1
                await environment.boundary(self.episode_id, {"step": step})
        except asyncio.CancelledError:
            await self.cleanup.wait()
            raise
        finally:
            # Final cleanup closes the episode's sessions, as the environment server's cleanup does.
            await agent.close_session(self.key)
            resources.ended(self.session_id)
            await environment.end(self.episode_id)


class MixedSimulation:
    def __init__(self, seed: int, tmp_path: Path, writes: GatedWrites) -> None:
        self.rng = random.Random(seed)
        self.tmp_path = tmp_path
        self.writes = writes
        self.deployment = Deployment()
        self.episodes: dict[str, Episode] = {}
        self.checkpoints = 0
        self.open_checkpoint: Optional[str] = None
        self.retired: set[str] = set()
        self.committed: dict[str, tuple[dict[str, dict[str, Any]], list[str]]] = {}
        self.resumed_unpublished: set[str] = set()
        self.log: list[str] = []

    # -- helpers ------------------------------------------------------------------------------------

    def directory(self, checkpoint_id: str) -> Path:
        return self.tmp_path / checkpoint_id

    def manifest_path(self, checkpoint_id: str, kind: str) -> Path:
        return participant_dir(self.directory(checkpoint_id), kind=kind, instance=kind) / "manifest.json"

    def live(self) -> list[Episode]:
        return [episode for episode in self.episodes.values() if not episode.task.done()]

    def scope(self) -> list[EpisodeId]:
        """Every episode the controller continues: those in flight, and restored ones never re-dispatched here."""
        return [episode.episode_id for episode in self.live() if episode.key not in self.retired]

    async def attempt(self, operation: Any) -> Any:
        try:
            return await operation
        except CoordinationError as error:
            return error

    async def settle(self) -> None:
        for _ in range(10):
            await asyncio.sleep(0)

    # -- operations ---------------------------------------------------------------------------------

    async def start(self) -> None:
        rollout_id = f"r{len(self.episodes)}"
        self.episodes[rollout_id] = Episode(self.deployment, rollout_id, random.Random(self.rng.random()))

    async def advance(self) -> None:
        if live := self.live():
            self.rng.choice(live).advance.set()

    async def finish(self) -> None:
        if live := self.live():
            episode = self.rng.choice(live)
            episode.finish = True
            episode.advance.set()

    async def prepare(self) -> None:
        """Prepare through coordination, usually while episodes keep stepping and park.

        A prepare that is not ready leaves stragglers; the controller then does what it must: resume,
        retire the blockers, and leave the next prepare to a later step.
        """
        participants = await self.deployment.discover()
        if self.open_checkpoint is None:
            self.checkpoints += 1
            self.open_checkpoint = f"c{self.checkpoints}"
        before = len(self.deployment.events)
        progress = self.rng.random() < 0.8
        seconds = 0.3 if progress else SHORT
        preparing = asyncio.ensure_future(
            self.attempt(coordination.prepare(participants, self.open_checkpoint, deadline_ts=time.time() + seconds))
        )
        for _ in range(6 if progress else 0):
            if preparing.done():
                break
            for episode in self.live():
                episode.advance.set()
            await self.settle()
        result = await preparing
        closes = [name for kind, name in self.deployment.events[before:] if kind == "close"]
        stages = [PREPARE_ORDER.index(name) for name in closes]
        assert stages == sorted(stages), f"prepare closed participants out of order: {closes}"
        if isinstance(result, coordination.PrepareResult) and not result.prepared and self.rng.random() < 0.7:
            stragglers = {key for keys in result.blockers().values() for key in keys if key in self.episodes}
            await self.resume()
            for key in sorted(stragglers):
                await self.retire(self.episodes[key])

    async def commit(self) -> None:
        if self.open_checkpoint is None or self.rng.random() < 0.7:
            await self.prepare()
        if self.open_checkpoint is None:
            # The prepare left stragglers, and the controller resumed to retire them.
            return
        participants = await self.deployment.discover()
        checkpoint_id = self.open_checkpoint
        self.deployment.exporting = checkpoint_id
        self.writes.hold_publishing = self.rng.random() < 0.3
        scope = self.scope()
        held = self.rng.random() < 0.5
        if not held:
            self.writes.gate.set()
        seconds = SHORT if held else LONG
        reply = await self.attempt(
            coordination.commit(
                participants,
                checkpoint_id,
                str(self.directory(checkpoint_id)),
                scope,
                deadline_ts=time.time() + seconds,
            )
        )
        if not held:
            await self.writes.drain()
        if isinstance(reply, dict):
            continued = {episode_id.capture_key for episode_id in scope}
            for kind, result in reply.items():
                assert set(result["episode_ids"]) <= continued, f"{kind} committed an episode not in scope"
                assert not set(result["episode_ids"]) & self.retired, f"{kind} committed a retired attempt"
            self.check_sessions_match_boundaries(checkpoint_id)
            self.committed[checkpoint_id] = (
                {kind: result["manifest"] for kind, result in reply.items()},
                sorted(continued),
            )

    async def resume(self) -> None:
        if self.open_checkpoint is None:
            return
        participants = await self.deployment.discover()
        checkpoint_id, self.open_checkpoint = self.open_checkpoint, None
        before = len(self.deployment.events)
        resuming = asyncio.ensure_future(
            self.attempt(coordination.resume(participants, checkpoint_id, deadline_ts=time.time() + LONG))
        )
        await self.settle()
        if not resuming.done():
            await self.writes.drain()
        await resuming
        opens = [name for kind, name in self.deployment.events[before:] if kind == "open"]
        stages = [PREPARE_ORDER.index(name) for name in opens]
        assert stages == sorted(stages, reverse=True), f"resume reopened participants out of order: {opens}"
        for kind in KINDS:
            if not self.manifest_path(checkpoint_id, kind).exists():
                self.resumed_unpublished.add(f"{checkpoint_id}/{kind}")

    async def retire(self, episode: Optional[Episode] = None) -> None:
        if not self.episodes or self.open_checkpoint is not None:
            return
        participants = await self.deployment.discover()
        episode = episode or self.rng.choice(list(self.episodes.values()))
        before = len(self.deployment.events)
        # A retire waits for the episode to end; one whose cleanup is held is retired with a short deadline.
        seconds = LONG if episode.cleanup.is_set() and self.rng.random() < 0.5 else SHORT
        reply = await self.attempt(
            coordination.retire(
                participants, f"retire-{len(self.log)}", [episode.episode_id], deadline_ts=time.time() + seconds
            )
        )
        retires = [name for kind, name in self.deployment.events[before:] if kind == "retire"]
        stages = [next(index for index, group in enumerate(RETIRE_ORDER) if name in group) for name in retires]
        assert stages == sorted(stages), f"retire stopped callees before callers: {retires}"
        if not isinstance(reply, CoordinationError):
            for kind, participant in self.deployment.by_kind.items():
                with pytest.raises(StaleAttemptError):
                    participant.retired.check(episode.episode_id)
            self.retired.add(episode.key)
        else:
            try:
                self.deployment.environment.retired.check(episode.episode_id)
            except StaleAttemptError:
                self.retired.add(episode.key)

    async def forget(self) -> None:
        if self.open_checkpoint is not None:
            return
        finished = [key for key in self.retired if self.episodes[key].task.done()]
        if finished:
            participants = await self.deployment.discover()
            rollout_id = self.rng.choice(sorted(finished))
            await self.attempt(
                coordination.forget(
                    participants, f"forget-{len(self.log)}", [rollout_id], deadline_ts=time.time() + LONG
                )
            )

    async def fail_next_close(self) -> None:
        self.deployment.faults[self.rng.choice(KINDS)].fail_next_close = True

    async def resume_then_retire(self) -> None:
        """Resume and retire at once, before parked work wakes: a controller retiring a straggler."""
        if self.open_checkpoint is None or self.writes.running:
            return
        participants = await self.deployment.discover()
        checkpoint_id, self.open_checkpoint = self.open_checkpoint, None
        await self.attempt(coordination.resume(participants, checkpoint_id, deadline_ts=time.time() + LONG))
        for kind in KINDS:
            if not self.manifest_path(checkpoint_id, kind).exists():
                self.resumed_unpublished.add(f"{checkpoint_id}/{kind}")
        await self.retire()

    async def fail_next_write(self) -> None:
        self.writes.fail_next = True

    async def drain_writes(self) -> None:
        await self.writes.drain()

    async def restore(self) -> None:
        if self.committed:
            await self.restore_into_fresh(self.rng.choice(sorted(self.committed)), fail=self.rng.random() < 0.3)

    async def restore_into_fresh(self, checkpoint_id: str, *, fail: bool) -> None:
        _, continued = self.committed[checkpoint_id]
        fresh = Deployment()
        failing = self.rng.choice(KINDS)
        if fail:
            fault = fresh.faults[failing]
            if self.rng.random() < 0.5:
                fault.fail_next_install = True
            else:
                fault.fail_next_close = True
        participants = await fresh.discover()
        scope = [EpisodeId.from_capture_key(key) for key in continued]
        restore_id = f"restore-{len(self.log)}"
        reply = await self.attempt(
            coordination.restore(
                participants, restore_id, str(self.directory(checkpoint_id)), scope, deadline_ts=time.time() + LONG
            )
        )
        if isinstance(reply, CoordinationError):
            assert fail and (INSTALL_FAILURE in str(reply) or CLOSE_FAILURE in str(reply)), f"did not restore: {reply}"
            assert fresh.open_everywhere(), "a failed restore left a participant closed"
            assert not fresh.environment._restored, "a failed restore left environment state restored"
            assert not fresh.agent._sessions, "a failed restore left agent sessions restored"
            assert not fresh.resources._sessions, "a failed restore left resources sessions restored"
            return
        restored = {key for result in reply.values() for key in result.get("restored", [])}
        environment = set(reply["environment"]["restored"])
        for kind in ("agent", "resources"):
            assert set(reply[kind]["restored"]) <= environment, f"{kind} restored an episode the environment did not"
        assert restored <= {f"{EpisodeId.from_capture_key(key).rollout_id}-a1" for key in continued}
        await coordination.resume(participants, restore_id, deadline_ts=time.time() + LONG)
        assert fresh.open_everywhere(), "a participant stayed closed after the restore resumed"

    # -- invariants ---------------------------------------------------------------------------------

    def check_sessions_match_boundaries(self, checkpoint_id: str) -> None:
        """An episode the environment exported past seeding must have its agent and resources sessions exported."""
        exports = self.deployment.first_exports
        agent_keys = {record["session_key"] for record in exports.get((checkpoint_id, "agent"), [])}
        resources_ids = {record["session_id"] for record in exports.get((checkpoint_id, "resources"), [])}
        for record in exports.get((checkpoint_id, "environment"), []):
            if record["boundary"] is None:
                continue
            key = EpisodeId.model_validate(record["episode_id"]).capture_key
            rollout_id = EpisodeId.model_validate(record["episode_id"]).rollout_id
            assert key in agent_keys, f"{key} is past seeding, but its agent session was not exported"
            assert f"s-{rollout_id}" in resources_ids, f"{key} is past seeding, but its resources session was not"

    def check(self) -> None:
        if self.open_checkpoint is None and self.writes.running == 0:
            assert self.deployment.open_everywhere(), "a participant is closed while no checkpoint is open"
        for checkpoint_id in [f"c{index}" for index in range(1, self.checkpoints + 1)]:
            for kind in KINDS:
                if not self.manifest_path(checkpoint_id, kind).exists():
                    continue
                assert f"{checkpoint_id}/{kind}" not in self.resumed_unpublished, "a resumed checkpoint published"
                manifest, records = read_participant_state(self.directory(checkpoint_id), kind=kind, instance=kind)
                if checkpoint_id in self.committed:
                    assert manifest == self.committed[checkpoint_id][0][kind], f"{kind}'s manifest is not committed"
                assert records == self.deployment.first_exports[checkpoint_id, kind], f"{kind} stored a later export"

    async def run(self) -> None:
        operations = [
            (self.start, 3),
            (self.advance, 8),
            (self.finish, 1),
            (self.prepare, 2),
            (self.commit, 3),
            (self.resume, 3),
            (self.retire, 2),
            (self.forget, 1),
            (self.release_cleanup, 1),
            (self.fail_next_close, 1),
            (self.resume_then_retire, 2),
            (self.fail_next_write, 1),
            (self.drain_writes, 1),
            (self.restore, 1),
        ]
        for _ in range(STEPS):
            operation = self.rng.choices([op for op, _ in operations], weights=[w for _, w in operations])[0]
            self.log.append(operation.__name__)
            await operation()
            await self.settle()
            self.check()
        await self.wind_down()

    async def release_cleanup(self) -> None:
        if held := [episode for episode in self.episodes.values() if not episode.cleanup.is_set()]:
            self.rng.choice(held).cleanup.set()

    async def stop_episodes(self) -> None:
        for episode in self.episodes.values():
            episode.cleanup.set()
            episode.task.cancel()
        if self.episodes:
            await asyncio.wait([episode.task for episode in self.episodes.values()], timeout=5)

    async def wind_down(self) -> None:
        await self.resume()
        self.writes.gate.set()
        await self.stop_episodes()
        await self.writes.drain()
        await self.settle()
        self.check()
        for checkpoint_id in sorted(self.committed):
            await self.restore_into_fresh(checkpoint_id, fail=False)


@pytest.mark.parametrize("seed", range(SEEDS))
async def test_random_sequences_keep_a_whole_deployment_consistent(
    seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    simulation = MixedSimulation(seed, tmp_path, GatedWrites(monkeypatch))
    try:
        async with asyncio.timeout(60):
            await simulation.run()
    except TimeoutError as error:
        raise AssertionError(f"seed {seed} hung after {simulation.log}") from error
    except AssertionError as error:
        raise AssertionError(f"seed {seed} after {simulation.log}: {error}") from error
    finally:
        simulation.writes.gate.set()
        await simulation.stop_episodes()
