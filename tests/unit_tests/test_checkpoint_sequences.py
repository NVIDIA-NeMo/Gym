# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Random sequences of checkpoint operations against environment participants, with invariants after every step.

Each seed interleaves episode steps with prepare, commit, resume, retire, forget, and restore,
at short or long deadlines, while writes wait at a gate, episodes clean up slowly,
and admission can fail to close part way.
It runs against one participant in one process,
and against a coordinator with several workers over its real Unix socket,
where workers also join while a checkpoint is open or a restore installs,
one worker's close fails while the others succeed, and checkpoints restore onto a different number of workers.
Example tests check what each operation does on its own; these check what must hold however operations overlap:

- admission is open on every worker whenever the participant is idle;
- every worker refuses the same retired attempts as its coordinator;
- each worker's running blocker count equals a fresh count;
- every manifest on disk verifies against its records, and is the one a successful commit returned;
- what a checkpoint stores is its first export, even when a commit is retried after a failed write;
- once resume returns, no manifest appears for a checkpoint that had none, even if a write was publishing;
- a commit never includes an attempt that was retired before it;
- every committed checkpoint restores into a fresh participant, with any number of workers.

A failing seed reproduces on its own, for example ``pytest -k "test_random_sequences_keep_checkpoints_safe and 17"``.
Set ``NEMO_GYM_CHECKPOINT_SEQUENCE_SEEDS`` and ``NEMO_GYM_CHECKPOINT_WORKER_SEQUENCE_SEEDS`` to run more seeds.
"""

import asyncio
import logging
import os
import random
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import pytest
from fastapi import FastAPI

import nemo_gym._checkpoint.control as control
import nemo_gym._checkpoint.store as store
from nemo_gym._checkpoint.control import (
    CheckpointPhase,
    CheckpointRecord,
    CheckpointRequest,
    CommitRequest,
    ForgetRequest,
    ParticipantControlPlane,
    RestoreRequest,
    RetireRequest,
)
from nemo_gym._checkpoint.environment import EnvironmentParticipant
from nemo_gym._checkpoint.errors import CheckpointStateError, ControlError, StaleAttemptError
from nemo_gym._checkpoint.participant_workers import CoordinatedServerParticipant, ParticipantWorkerLink
from nemo_gym._checkpoint.store import participant_dir, read_participant_state
from nemo_gym._checkpoint.workers import WorkerCoordinator
from nemo_gym.episode_types import EpisodeId


SEEDS = int(os.environ.get("NEMO_GYM_CHECKPOINT_SEQUENCE_SEEDS", "150"))
WORKER_SEEDS = int(os.environ.get("NEMO_GYM_CHECKPOINT_WORKER_SEQUENCE_SEEDS", "30"))
STEPS = 40
SHORT = 0.01
LONG = 10.0
TASK = {"task": "t"}
CLOSE_FAILURE = "admission failed to close part way"


class FaultyEnvironment(EnvironmentParticipant):
    """An environment participant whose next close of admission can be made to fail after it closed."""

    def __init__(self) -> None:
        super().__init__()
        self.fail_next_close = False

    async def close_admission(self, request: CheckpointRequest) -> None:
        await super().close_admission(request)
        if self.fail_next_close:
            self.fail_next_close = False
            raise RuntimeError(CLOSE_FAILURE)


class GatedWrites:
    """Hold every checkpoint write at a gate, so a write can still be running while other operations happen.

    A write is held before it starts, or, if ``hold_publishing`` was set when it started,
    inside the creation of its manifest, after its last check for a resume.
    ``fail_next`` makes the next write fail, as a full disk would.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.gate = threading.Event()
        self.running = 0
        self.hold_publishing = False
        self.fail_next = False
        self._lock = threading.Lock()
        self._local = threading.local()
        real_write = control.write_participant_state
        real_create = store._create

        def gated(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                self.running += 1
            try:
                self._local.hold_publishing = self.hold_publishing
                if not self.hold_publishing:
                    self.gate.wait(10)
                if self.fail_next:
                    self.fail_next = False
                    raise CheckpointStateError("disk full")
                return real_write(*args, **kwargs)
            finally:
                with self._lock:
                    self.running -= 1

        def held_create(*args: Any, **kwargs: Any) -> bool:
            if getattr(self._local, "hold_publishing", False):
                self.gate.wait(10)
            return real_create(*args, **kwargs)

        monkeypatch.setattr(control, "write_participant_state", gated)
        monkeypatch.setattr(store, "_create", held_create)

    async def drain(self) -> None:
        """Let every held write finish, then hold new ones again."""
        self.gate.set()
        give_up = time.monotonic() + 10
        while self.running and time.monotonic() < give_up:
            await asyncio.sleep(0.001)
        assert not self.running, "a checkpoint write never finished"
        self.gate.clear()


class Deployment:
    """One participant's control plane and the environment participants episodes run on.

    It remembers each checkpoint's first export, which every later write of that checkpoint must store.
    """

    controller: ParticipantControlPlane
    environments: list[FaultyEnvironment]
    # How long to let in-flight work and messages settle between steps.
    settle_seconds = 0.0

    def remember_exports(self) -> None:
        self.exporting = ""
        self.first_exports: dict[str, list[dict[str, Any]]] = {}
        participant = self.controller.participant
        export = participant.export

        async def remembering(episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
            records = await export(episode_ids)
            self.first_exports.setdefault(self.exporting, [record.to_json_record() for record in records])
            return records

        participant.export = remembering  # type: ignore[method-assign]

    def open_everywhere(self) -> bool:
        return all(not environment.steps.closed for environment in self.environments)

    async def join(self) -> None:
        """Add a worker; one process has none to add."""

    async def close(self) -> None:
        pass


class OneProcess(Deployment):
    def __init__(self) -> None:
        self.environments = [FaultyEnvironment()]
        self.controller = ParticipantControlPlane(self.environments[0], instance_name="env", lease_grace_seconds=60)
        self.remember_exports()


class Workers(Deployment):
    """A coordinator and its workers over the coordinator's real Unix socket, all in this event loop."""

    settle_seconds = 0.005

    def __init__(self, socket_dir: str, expected: int) -> None:
        self.socket_path = os.path.join(socket_dir, f"{uuid.uuid4().hex[:8]}.sock")
        self.coordinator = WorkerCoordinator(
            CoordinatedServerParticipant(FaultyEnvironment(), expected_workers=expected),
            instance_name="env",
            lease_grace_seconds=60,
            socket_path=self.socket_path,
        )
        self.controller = self.coordinator.controller
        self.environments = []
        self.links: list[ParticipantWorkerLink] = []
        self.remember_exports()

    @classmethod
    async def start(cls, socket_dir: str, count: int) -> "Workers":
        deployment = cls(socket_dir, count)
        deployment.server = await deployment.coordinator.serve()
        for _ in range(count):
            await deployment.join()
        return deployment

    async def join(self) -> None:
        environment = FaultyEnvironment()
        # A worker that lost its coordinator stops its own process; here that would be the test runner.
        link = ParticipantWorkerLink(
            environment, socket_path=self.socket_path, app=FastAPI(), on_coordinator_lost=lambda: None
        )
        await link.connect()
        self.environments.append(environment)
        self.links.append(link)

    def open_everywhere(self) -> bool:
        return super().open_everywhere() and self.coordinator.participant.accepting

    async def close(self) -> None:
        for link in self.links:
            await link.disconnect()
        self.server.close()
        await self.server.wait_closed()


class Episode:
    """An environment episode that takes one step per ``advance``, in a mode it draws from its own generator."""

    def __init__(self, participant: EnvironmentParticipant, rollout_id: str, rng: random.Random) -> None:
        self.participant = participant
        self.episode_id = EpisodeId(rollout_id=rollout_id)
        self.key = self.episode_id.capture_key
        self.rng = rng
        self.advance = asyncio.Event()
        # Released at once, or held so a retire's wait for the episode is cut short by its deadline.
        self.cleanup = asyncio.Event()
        if rng.random() < 0.5:
            self.cleanup.set()
        self.finish = False
        self.task = asyncio.create_task(self.run())

    async def _next(self) -> None:
        await self.advance.wait()
        self.advance.clear()

    async def run(self) -> None:
        self.participant.begin(self.episode_id, TASK, None)
        try:
            step = 0
            while not self.finish:
                await self._next()
                async with self.participant.step(self.episode_id, self.rng.choice(["wait", "replay"])):
                    await self._next()
                step += 1
                await self.participant.boundary(self.episode_id, {"step": step})
        except asyncio.CancelledError:
            await self.cleanup.wait()
            raise
        finally:
            await self.participant.end(self.episode_id)


class Simulation:
    def __init__(
        self, seed: int, tmp_path: Path, writes: GatedWrites, deployment: Deployment, socket_dir: Optional[str]
    ) -> None:
        self.rng = random.Random(seed)
        self.tmp_path = tmp_path
        self.writes = writes
        self.deployment = deployment
        self.controller = deployment.controller
        # Set for workers: restores then go to a fresh coordinator with one to three workers.
        self.socket_dir = socket_dir
        self.episodes: dict[str, Episode] = {}
        self.checkpoints = 0
        # Capture keys whose attempt was marked retired: no later commit may include them.
        self.retired: set[str] = set()
        # Capture keys of episodes a server could not capture: no commit may include them.
        self.restarts: set[str] = set()
        # Restart markings waiting for an open checkpoint to end.
        self.markings: set[asyncio.Future] = set()
        # Checkpoint ID -> the manifest and episodes its successful commit returned.
        self.committed: dict[str, tuple[dict[str, Any], list[str]]] = {}
        # Checkpoints resumed while they had no manifest: none may ever appear.
        self.resumed_unpublished: set[str] = set()
        self.log: list[str] = []

    # -- helpers ------------------------------------------------------------------------------------

    def directory(self, checkpoint_id: str) -> Path:
        return self.tmp_path / checkpoint_id

    def manifest_path(self, checkpoint_id: str) -> Path:
        return participant_dir(self.directory(checkpoint_id), kind="environment", instance="env") / "manifest.json"

    def current_checkpoint(self) -> str:
        if self.controller.checkpoint_id is not None:
            return self.controller.checkpoint_id
        self.checkpoints += 1
        return f"c{self.checkpoints}"

    def request(self, checkpoint_id: str, seconds: float, **extra: Any) -> dict[str, Any]:
        return {"checkpoint_id": checkpoint_id, "deadline_ts": time.time() + seconds, **extra}

    def live(self) -> list[Episode]:
        return [episode for episode in self.episodes.values() if not episode.task.done()]

    async def attempt(self, operation: Any) -> Any:
        """Run one control operation; a typed refusal or the injected close failure is an outcome, not a bug."""
        try:
            return await operation
        except ControlError as error:
            return error
        except RuntimeError as error:
            if str(error) != CLOSE_FAILURE:
                raise
            return error

    async def settle(self) -> None:
        for _ in range(10):
            await asyncio.sleep(0)
        if self.deployment.settle_seconds:
            await asyncio.sleep(self.deployment.settle_seconds)

    # -- operations ---------------------------------------------------------------------------------

    async def start(self) -> None:
        rollout_id = f"r{len(self.episodes)}"
        environment = self.rng.choice(self.deployment.environments)
        self.episodes[rollout_id] = Episode(environment, rollout_id, random.Random(self.rng.random()))

    async def advance(self) -> None:
        if live := self.live():
            self.rng.choice(live).advance.set()

    async def finish(self) -> None:
        if live := self.live():
            episode = self.rng.choice(live)
            episode.finish = True
            episode.advance.set()

    async def mark_restart(self) -> None:
        """A seed reply says a server cannot capture this episode, so it must start over after a crash.

        The reply may arrive while a checkpoint is open; the episode becomes a restart once it is not.
        """
        if live := self.live():
            episode = self.rng.choice(live)
            marking = asyncio.ensure_future(episode.participant.mark_restart(episode.episode_id))
            marking.add_done_callback(
                lambda done: done.cancelled() or done.exception() or self.restarts.add(episode.key)
            )
            self.markings.add(marking)
            marking.add_done_callback(self.markings.discard)

    async def prepare(self) -> None:
        checkpoint_id = self.current_checkpoint()
        seconds = 3 * SHORT + 2 * self.deployment.settle_seconds * 10
        await self.attempt(self.controller.prepare(CheckpointRequest(**self.request(checkpoint_id, seconds))))

    async def prepare_while_episodes_park(self) -> None:
        """Prepare while episodes keep stepping, so they reach boundaries and park, as they do in a real run."""
        checkpoint_id = self.current_checkpoint()
        seconds = 10 * SHORT + 20 * self.deployment.settle_seconds
        preparing = asyncio.ensure_future(
            self.attempt(self.controller.prepare(CheckpointRequest(**self.request(checkpoint_id, seconds))))
        )
        for _ in range(4):
            if preparing.done():
                break
            for episode in self.live():
                episode.advance.set()
            await self.settle()
        await preparing

    async def commit(self) -> None:
        if self.controller.phase != CheckpointPhase.PREPARED and self.rng.random() < 0.8:
            # Usually as a controller does, after a prepare; sometimes without one, which must be refused.
            await self.prepare_while_episodes_park()
        checkpoint_id = self.current_checkpoint()
        self.deployment.exporting = checkpoint_id
        self.writes.hold_publishing = self.rng.random() < 0.3
        retired_before = set(self.retired)
        # A restart that has ended left nothing to continue, so naming it is harmless.
        restarts_before = {key for key in self.restarts if not self.episodes[key].task.done()}
        body = self.request(checkpoint_id, LONG, checkpoint_dir=str(self.directory(checkpoint_id)))
        scope: Optional[set[str]] = None
        if self.episodes and self.rng.random() < 0.5:
            # A controller's scope, which may name a restart, as one that ignores the reported restarts would.
            scope = set(self.rng.sample(sorted(self.episodes), self.rng.randint(1, len(self.episodes))))
            body["episode_ids"] = [EpisodeId.from_capture_key(key).model_dump() for key in sorted(scope)]
        if self.rng.random() < 0.5:
            # The write waits at its gate past the commit's short deadline and keeps running.
            reply = await self.attempt(control_commit(self.controller, {**body, "deadline_ts": time.time() + SHORT}))
        else:
            self.writes.gate.set()
            reply = await self.attempt(control_commit(self.controller, body))
            await self.writes.drain()
        if isinstance(reply, dict):
            assert not set(reply["episode_ids"]) & retired_before, "a commit included a retired attempt"
            assert not set(reply["episode_ids"]) & restarts_before, "a commit included a restart"
            assert not (scope or set()) & restarts_before, "a commit whose scope names a restart was accepted"
            if checkpoint_id in self.committed:
                assert self.committed[checkpoint_id][0] == reply["manifest"], "a retried commit changed its manifest"
            self.committed[checkpoint_id] = (reply["manifest"], reply["episode_ids"])

    async def resume(self) -> None:
        checkpoint_id = self.controller.checkpoint_id or f"c{self.checkpoints}"
        resume = asyncio.ensure_future(
            self.attempt(self.controller.resume(CheckpointRequest(**self.request(checkpoint_id, LONG))))
        )
        await self.settle()
        if not resume.done():
            # Resume is waiting for a write that is publishing; the write finishes, as it would on its own.
            await self.writes.drain()
        reply = await resume
        if isinstance(reply, dict) and not reply.get("idempotent") and not self.manifest_path(checkpoint_id).exists():
            self.resumed_unpublished.add(checkpoint_id)

    async def resume_then_retire(self) -> None:
        """Resume and retire at once, before parked episodes wake: a controller retiring a straggler."""
        checkpoint_id = self.controller.checkpoint_id
        # A held write may be publishing, and resume would wait for it; the plain resume operation covers that.
        if checkpoint_id is None or self.writes.running:
            return
        reply = await self.attempt(self.controller.resume(CheckpointRequest(**self.request(checkpoint_id, LONG))))
        if isinstance(reply, dict) and not self.manifest_path(checkpoint_id).exists():
            self.resumed_unpublished.add(checkpoint_id)
        await self.retire()

    async def retire(self) -> None:
        if not self.episodes:
            return
        episode = self.rng.choice(list(self.episodes.values()))
        # A retire waits for the episode to end; one whose cleanup is held is retired with a short deadline.
        seconds = LONG if episode.cleanup.is_set() and self.rng.random() < 0.5 else SHORT
        body = self.request(f"retire-{len(self.log)}", seconds, episode_ids=[episode.episode_id.model_dump()])
        await self.attempt(self.controller.retire(RetireRequest(**body)))
        try:
            self.controller.participant.retired.check(episode.episode_id)
        except StaleAttemptError:
            self.retired.add(episode.key)

    async def forget(self) -> None:
        finished = [key for key in self.retired if self.episodes[key].task.done()]
        if finished:
            rollout_id = self.rng.choice(sorted(finished))
            body = self.request(f"forget-{len(self.log)}", LONG, rollout_ids=[rollout_id])
            await self.attempt(self.controller.forget(ForgetRequest(**body)))

    async def release_cleanup(self) -> None:
        if held := [episode for episode in self.episodes.values() if not episode.cleanup.is_set()]:
            self.rng.choice(held).cleanup.set()

    async def fail_next_close(self) -> None:
        # With several workers, one worker fails to close while the others close.
        self.rng.choice(self.deployment.environments).fail_next_close = True

    async def join_worker(self) -> None:
        await self.deployment.join()

    async def fail_next_write(self) -> None:
        self.writes.fail_next = True

    async def drain_writes(self) -> None:
        await self.writes.drain()

    async def restore(self) -> None:
        if self.committed:
            await self.restore_into_fresh(self.rng.choice(sorted(self.committed)), fail_close=self.rng.random() < 0.3)

    async def restore_into_fresh(self, checkpoint_id: str, *, fail_close: bool) -> None:
        """Restore a committed checkpoint into a fresh deployment, which must end idle and open, or restored."""
        _, episode_ids = self.committed[checkpoint_id]
        if self.socket_dir is None:
            fresh: Deployment = OneProcess()
        else:
            fresh = await Workers.start(self.socket_dir, self.rng.randint(1, 3))
        try:
            if fail_close:
                self.rng.choice(fresh.environments).fail_next_close = True
            scope = [EpisodeId.from_capture_key(key).model_dump() for key in episode_ids]
            restore_id = f"restore-{len(self.log)}"
            body = self.request(restore_id, LONG, checkpoint_dir=str(self.directory(checkpoint_id)))
            restoring = asyncio.ensure_future(
                self.attempt(fresh.controller.restore(RestoreRequest(**body, episode_ids=scope)))
            )
            if self.socket_dir is not None and self.rng.random() < 0.3:
                # A worker that starts while the restore installs.
                await fresh.join()
            reply = await restoring
            if isinstance(reply, dict):
                assert fresh.controller.phase == CheckpointPhase.RESTORED
                expected = sorted(f"{EpisodeId.from_capture_key(key).rollout_id}-a1" for key in episode_ids)
                assert reply["restored"] == expected, "a restore installed other episodes than were committed"
                await fresh.controller.resume(CheckpointRequest(**self.request(restore_id, LONG)))
                await self.settle()
                assert fresh.open_everywhere(), "a worker stayed closed after the restore resumed"
            else:
                assert fail_close, f"a committed checkpoint did not restore: {reply}"
                await self.settle()
                assert fresh.controller.phase == CheckpointPhase.IDLE, "a failed restore did not end idle"
                assert fresh.open_everywhere(), "a failed restore left a worker closed"
        finally:
            await fresh.close()

    # -- invariants ---------------------------------------------------------------------------------

    def check(self) -> None:
        if self.controller.phase == CheckpointPhase.IDLE:
            assert self.deployment.open_everywhere(), "admission is closed while the participant is idle"
        marks = self.controller.participant.retired.marks()
        for environment in self.deployment.environments:
            assert environment.retired.marks() == marks, "a worker refuses other attempts than its coordinator"
            steps = environment.steps
            assert steps.blocker_count() == len(steps.blockers(10**6)), "the running blocker count drifted"
        for checkpoint_id in [f"c{index}" for index in range(1, self.checkpoints + 1)]:
            if not self.manifest_path(checkpoint_id).exists():
                continue
            assert checkpoint_id not in self.resumed_unpublished, "a resumed checkpoint published a manifest"
            manifest, records = read_participant_state(
                self.directory(checkpoint_id), kind="environment", instance="env"
            )
            if checkpoint_id in self.committed:
                assert manifest == self.committed[checkpoint_id][0], "the manifest on disk is not the one committed"
            assert records == self.deployment.first_exports[checkpoint_id], "a checkpoint stored a later export"

    async def run(self) -> None:
        operations = [
            (self.start, 3),
            (self.mark_restart, 1),
            (self.advance, 8),
            (self.finish, 1),
            (self.prepare, 2),
            (self.prepare_while_episodes_park, 3),
            (self.commit, 3),
            (self.resume, 3),
            (self.retire, 2),
            (self.resume_then_retire, 2),
            (self.forget, 1),
            (self.release_cleanup, 1),
            (self.fail_next_close, 1),
            (self.fail_next_write, 2),
            (self.drain_writes, 1),
            (self.restore, 1),
        ]
        if self.socket_dir is not None:
            operations.append((self.join_worker, 1))
        for _ in range(STEPS):
            operation = self.rng.choices([op for op, _ in operations], weights=[w for _, w in operations])[0]
            self.log.append(operation.__name__)
            await operation()
            await self.settle()
            self.check()
        await self.wind_down()

    async def stop_episodes(self) -> None:
        for marking in list(self.markings):
            marking.cancel()
        for episode in self.episodes.values():
            episode.cleanup.set()
            episode.task.cancel()
        if self.episodes:
            await asyncio.wait([episode.task for episode in self.episodes.values()], timeout=5)

    async def wind_down(self) -> None:
        """Abort any open checkpoint, let everything finish, then check that every commit still restores."""
        if self.controller.checkpoint_id is not None:
            await self.resume()
        self.writes.gate.set()
        await self.stop_episodes()
        await self.writes.drain()
        await self.settle()
        self.check()
        for checkpoint_id in sorted(self.committed):
            await self.restore_into_fresh(checkpoint_id, fail_close=False)


async def control_commit(controller: ParticipantControlPlane, body: dict[str, Any]) -> Optional[dict[str, Any]]:
    return await controller.commit(CommitRequest(**body))


async def simulate(seed: int, simulation: Simulation) -> None:
    try:
        async with asyncio.timeout(60):
            await simulation.run()
    except TimeoutError as error:
        raise AssertionError(f"seed {seed} hung after {simulation.log}") from error
    except AssertionError as error:
        raise AssertionError(f"seed {seed} after {simulation.log}: {error}") from error
    finally:
        # A failing seed skips its wind-down: release everything it holds,
        # or closing the event loop would wait forever for episodes held in cleanup,
        # and writer threads would wait at the gate.
        simulation.writes.gate.set()
        await simulation.stop_episodes()


@pytest.mark.parametrize("seed", range(SEEDS))
async def test_random_sequences_keep_checkpoints_safe(
    seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await simulate(seed, Simulation(seed, tmp_path, GatedWrites(monkeypatch), OneProcess(), None))


class _HideInjectedCloseFailures(logging.Filter):
    """Drop the coordinator's log of the injected close failure; any other failure is still logged."""

    def filter(self, record: logging.LogRecord) -> bool:
        return not (record.exc_info and str(record.exc_info[1]) == CLOSE_FAILURE)


@pytest.fixture
def socket_dir() -> Any:
    # AF_UNIX paths must be short, so sockets live in their own directory under /tmp.
    directory = tempfile.mkdtemp(prefix="ngs-", dir="/tmp")
    yield directory
    shutil.rmtree(directory, ignore_errors=True)


@pytest.mark.parametrize("seed", range(WORKER_SEEDS))
async def test_random_sequences_keep_checkpoints_safe_with_workers(
    seed: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, socket_dir: str
) -> None:
    logger = logging.getLogger("nemo_gym._checkpoint.workers")
    hide = _HideInjectedCloseFailures()
    logger.addFilter(hide)
    deployment = await Workers.start(socket_dir, random.Random(seed).randint(2, 3))
    try:
        await simulate(seed, Simulation(seed, tmp_path, GatedWrites(monkeypatch), deployment, socket_dir))
    finally:
        await deployment.close()
        logger.removeFilter(hide)
