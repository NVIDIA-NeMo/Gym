# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Random checkpoint sequences against a real ``single_agent_turn`` deployment, as a training controller drives it.

Each seed starts real server processes: ``single_agent_turn`` over Simple Agent, the weather resources server,
and the policy model server, against the fake inference backend,
which holds every model call until the sequence releases it.
It then dispatches ``/run`` episodes and, in a random order, releases model calls, checkpoints, resumes,
crashes and restores (sometimes with a different number of workers), and retires stragglers,
doing what the controller must: re-dispatch every rollout it continues as its next attempt,
wait for replies already in flight, and forget the rollouts it retired once they are done.

At the end every rollout must have finished with the right reward, failing only in ways the controller caused;
no attempt may make more model calls than its own episode needs;
every server must be idle with nothing left over (no episodes, sessions, requests in flight, or refusals);
and every committed checkpoint must verify.

Each seed takes tens of seconds; set ``NEMO_GYM_CHECKPOINT_E2E_SEQUENCE_SEEDS`` to run more.
"""

import asyncio
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import httpx
import pytest
from checkpoint_deployment import TOKEN, Deployment, weather_episode
from test_checkpoint_e2e import deploy, gym_http_client, wait_until  # noqa: F401

from nemo_gym._checkpoint import coordination
from nemo_gym._checkpoint.coordination import CoordinationError
from nemo_gym._checkpoint.store import read_participant_state
from nemo_gym.episode_types import EpisodeId


SEEDS = int(os.environ.get("NEMO_GYM_CHECKPOINT_E2E_SEQUENCE_SEEDS", "4"))
STEPS = 30
ROLLOUTS = 10
# Calls an uninterrupted weather episode makes: the tool call, then the final answer.
CALLS_PER_ATTEMPT = 2
SERVERS = {"environment": "environment", "agent": "agent", "resources": "resources", "policy_model": "model"}

pytestmark = pytest.mark.skipif(
    os.environ.get("NEMO_GYM_CHECKPOINT_E2E") != "1",
    reason="starts real server processes; set NEMO_GYM_CHECKPOINT_E2E=1 to run",
)


class Rollout:
    def __init__(self, rollout_id: str) -> None:
        self.rollout_id = rollout_id
        self.attempt = 0
        self.task: Optional[asyncio.Task] = None
        self.reward: Optional[float] = None
        # Every attempt this rollout was dispatched as.
        self.dispatched: list[int] = []
        # Attempts the controller retired: their replies no longer matter.
        self.retired_attempts: set[int] = set()
        # Set while the current attempt's /run was refused because a checkpoint is open, until it is sent again.
        self.refused = False

    @property
    def episode_id(self) -> EpisodeId:
        return EpisodeId(rollout_id=self.rollout_id, attempt=self.attempt)

    @property
    def done(self) -> bool:
        return self.reward is not None


class Sequence:
    def __init__(self, seed: int, deployment: Deployment, tmp_path: Path, http: httpx.AsyncClient) -> None:
        self.rng = random.Random(seed)
        self.deployment = deployment
        self.tmp_path = tmp_path
        self.http = http
        self.rollouts = {f"r{index}": Rollout(f"r{index}") for index in range(ROLLOUTS)}
        self.checkpoints = 0
        self.open_checkpoint: Optional[str] = None
        self.committed: list[str] = []
        self.log: list[str] = []
        # Set while no checkpoint is open: a refused /run waits for it before it is sent again.
        self.reopened = asyncio.Event()
        self.reopened.set()
        # Failures of attempts the controller still cared about, including ones a later attempt replaced since.
        self.failures: list[str] = []

    # -- the controller's side ----------------------------------------------------------------------

    async def run_attempt(self, rollout: Rollout, attempt: int) -> None:
        """One ``/run`` of one attempt; a refusal while a checkpoint is open is retried, as the controller must."""
        while True:
            try:
                reply = await self.http.post("/run", json=weather_episode(rollout.rollout_id, attempt=attempt))
            except httpx.TransportError:
                # The servers crashed or were restarted; the controller re-dispatches after the restore.
                return
            if attempt in rollout.retired_attempts or attempt != rollout.attempt:
                # Retired, or superseded by a later attempt: its outcome no longer matters.
                return
            if reply.status_code == 409 and reply.json()["error"]["code"] == "admission_closed":
                rollout.refused = True
                await self.reopened.wait()
                rollout.refused = False
                continue
            body = reply.json() if reply.status_code == 200 else None
            if body is None or body.get("result") is None:
                failure = body.get("failure") if body else reply.text[:300]
                self.failures.append(f"{rollout.rollout_id} attempt {attempt} failed after {self.log}: {failure}")
                return
            rollout.reward = body["result"]["reward"]
            return

    def dispatch(self, rollout: Rollout, attempt: int) -> None:
        rollout.attempt = attempt
        rollout.dispatched.append(attempt)
        rollout.task = asyncio.create_task(self.run_attempt(rollout, attempt))

    def unfinished(self) -> list[Rollout]:
        return [rollout for rollout in self.rollouts.values() if not rollout.done]

    async def participants(self) -> coordination.Participants:
        return await self.deployment.participants()

    # -- operations ---------------------------------------------------------------------------------

    async def start(self) -> None:
        if fresh := [rollout for rollout in self.rollouts.values() if not rollout.dispatched]:
            self.dispatch(fresh[0], 0)

    async def start_then_checkpoint(self) -> None:
        """Dispatch rollouts and checkpoint while they are still seeding their sessions, as a busy controller does."""
        for _ in range(3):
            await self.start()
        await asyncio.sleep(self.rng.uniform(0, 0.05))
        await self.checkpoint()

    async def release_calls(self) -> None:
        self.deployment.backend("/_ctl/step", {"calls": self.rng.randint(1, 3)})
        await asyncio.sleep(0.2)

    async def checkpoint(self) -> None:
        participants = await self.participants()
        self.checkpoints += 1
        checkpoint_id = self.open_checkpoint = f"c{self.checkpoints}"
        self.reopened.clear()
        prepared = await coordination.prepare(participants, checkpoint_id, deadline_ts=time.time() + 5)
        if not prepared.prepared:
            self.log.append("stragglers")
            # Stragglers: abort, retire them, and re-dispatch them as their next attempt.
            await self.resume()
            stragglers = {key for keys in prepared.blockers().values() for key in keys}
            for rollout in self.unfinished():
                if rollout.episode_id.capture_key in stragglers:
                    await self.retire(rollout)
            return
        scope = [rollout.episode_id for rollout in self.unfinished() if rollout.dispatched]
        directory = self.tmp_path / checkpoint_id
        try:
            replies = await coordination.commit(
                participants, checkpoint_id, str(directory), scope, deadline_ts=time.time() + 10
            )
        except CoordinationError:
            self.log.append("commit failed")
            await self.resume()
            return
        self.committed.append(checkpoint_id)
        exported = set(replies["environment"]["episode_ids"])
        # Replies already in flight: an episode no participant exported finished before prepare was ready,
        # unless its /run was refused and waits for the controller to reopen.
        waiting = [
            rollout
            for rollout in self.unfinished()
            if rollout.dispatched and rollout.episode_id.capture_key not in exported and rollout.task is not None
        ]
        await wait_until(lambda: all(rollout.task.done() or rollout.refused for rollout in waiting), timeout=30)
        if self.rng.random() < 0.5:
            self.log.append("resumed")
            await self.resume()
        else:
            self.log.append("crashed")
            await self.crash_and_restore(
                checkpoint_id, [rollout for rollout in self.unfinished() if rollout.dispatched]
            )

    async def resume(self) -> None:
        if self.open_checkpoint is None:
            return
        checkpoint_id, self.open_checkpoint = self.open_checkpoint, None
        await coordination.resume(await self.participants(), checkpoint_id, deadline_ts=time.time() + 10)
        self.reopened.set()

    async def crash_and_restore(self, checkpoint_id: str, continued: list[Rollout]) -> None:
        self.open_checkpoint = None
        self.deployment.crash_gym()
        for rollout in self.rollouts.values():
            if rollout.task is not None and not rollout.task.done():
                rollout.task.cancel()
        if self.rng.random() < 0.5:
            self.deployment.set_server_workers(self.rng.choice([1, 2]))
        self.deployment.start_gym()
        participants = await self.participants()
        restore_id = f"restore-{len(self.log)}"
        scope = [rollout.episode_id for rollout in continued]
        await coordination.restore(
            participants, restore_id, str(self.tmp_path / checkpoint_id), scope, deadline_ts=time.time() + 30
        )
        await coordination.resume(participants, restore_id, deadline_ts=time.time() + 10)
        self.reopened.set()
        # Every rollout still unfinished continues as its next attempt; the restored ones from their boundary.
        for rollout in self.unfinished():
            if rollout.dispatched:
                self.dispatch(rollout, rollout.attempt + 1)

    async def retire(self, rollout: Optional[Rollout] = None) -> None:
        if self.open_checkpoint is not None:
            return
        candidates = [rollout for rollout in self.unfinished() if rollout.dispatched]
        if rollout is None:
            if not candidates:
                return
            rollout = self.rng.choice(candidates)
        rollout.retired_attempts.add(rollout.attempt)
        await coordination.retire(
            await self.participants(), f"retire-{len(self.log)}", [rollout.episode_id], deadline_ts=time.time() + 30
        )
        self.dispatch(rollout, rollout.attempt + 1)

    # -- the run --------------------------------------------------------------------------------

    async def run(self) -> None:
        self.deployment.backend("/_ctl/gate")
        for _ in range(2):
            await self.start()
        operations = [
            (self.start, 3),
            (self.release_calls, 6),
            (self.checkpoint, 2),
            (self.start_then_checkpoint, 2),
            (self.retire, 1),
        ]
        for _ in range(STEPS):
            operation = self.rng.choices([op for op, _ in operations], weights=[w for _, w in operations])[0]
            self.log.append(operation.__name__)
            await operation()
        await self.resume()
        # Let every rollout finish: dispatch the rest and release every model call.
        for rollout in self.rollouts.values():
            if not rollout.dispatched:
                self.dispatch(rollout, 0)
        self.deployment.backend("/_ctl/release")
        await asyncio.wait_for(
            asyncio.gather(*(rollout.task for rollout in self.rollouts.values() if rollout.task is not None)),
            timeout=120,
        )
        retired = sorted(rollout.rollout_id for rollout in self.rollouts.values() if rollout.retired_attempts)
        if retired:
            await coordination.forget(await self.participants(), "forget", retired, deadline_ts=time.time() + 10)

    async def check(self) -> None:
        assert not self.failures, f"rollouts failed for reasons the controller did not cause: {self.failures}"
        for rollout in self.rollouts.values():
            assert rollout.reward == 1.0, f"{rollout.rollout_id} ended with reward {rollout.reward}"
        attempts = sum(len(rollout.dispatched) for rollout in self.rollouts.values())
        calls = len(self.deployment.backend_calls())
        assert calls <= CALLS_PER_ATTEMPT * attempts, f"{calls} model calls for {attempts} attempts"
        for name in SERVERS:
            status = await self.status(name)
            assert status["phase"] == "idle", f"{name} is in phase {status['phase']}"
            assert status["retired_rollouts"] == 0, f"{name} still refuses {status['retired_rollouts']} rollouts"
            counts = status["report"]["counts"]
            left = {
                key: value for key, value in counts.items() if key in ("episodes", "sessions", "inflight") and value
            }
            assert not left, f"{name} has state left over: {left}"
        for checkpoint_id in self.committed:
            for name, kind in SERVERS.items():
                read_participant_state(self.tmp_path / checkpoint_id, kind=kind, instance=name)

    async def status(self, name: str) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.deployment.url(name), timeout=10) as client:
            response = await client.get(
                "/ng-control/v1/checkpoint/status", headers={"Authorization": f"Bearer {TOKEN}"}
            )
        return response.json()


@pytest.mark.parametrize("seed", range(SEEDS))
async def test_random_checkpoint_sequences_on_a_single_agent_turn_deployment(
    deploy, tmp_path: Path, seed: int
) -> None:
    rng = random.Random(seed)
    deployment = deploy("native", policy_workers=rng.choice([1, 2]), server_workers=rng.choice([1, 2]))
    async with httpx.AsyncClient(base_url=deployment.url("environment"), timeout=120) as http:
        sequence = Sequence(seed, deployment, tmp_path, http)
        try:
            await sequence.run()
            await sequence.check()
            print(f"seed {seed}: {sequence.log}")
        except (AssertionError, TimeoutError, CoordinationError) as error:
            calls = json.dumps([call["n_messages"] for call in deployment.backend_calls()])
            raise AssertionError(
                f"seed {seed} after {sequence.log}, model calls {calls}: {error}\n{deployment.log_tails()}"
            ) from error
        finally:
            deployment.backend("/_ctl/release")
            for rollout in sequence.rollouts.values():
                if rollout.task is not None:
                    rollout.task.cancel()
