# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent server participant: park turn loops at boundaries and export agent sessions.

One participant serves both ways an episode can reach an agent:

- Through an environment server, the agent owns a session named by ``agent_session_id``. The
  environment server owns the protocol position and invokes one activation at a time.
- Through a legacy ``/run``, the agent owns the whole episode: it is both the episode protocol (seed,
  turn loop, verify) and the turn loop. The participant keys that episode by its logical rollout ID so a
  replacement attempt finds the restored state, and tracks its steps with the same ``EpisodeSteps`` an
  environment server uses.

Either way the agent keeps its own session state and exposes it through ``AgentSessionHooks``. The
participant tracks which session is inside an activation and the activation's latest boundary. An
activation reports a boundary before each model call and after each tool round. A session is
``at_boundary`` while parked there or while waiting on a policy call made right after it: the policy
model server holds that call's response during a checkpoint, so the loop cannot move past the boundary.
Prepare is ready once no activation is between boundaries.

A boundary is a snapshot function, not a snapshot. It runs only when a checkpoint exports the session,
so a rollout that is never checkpointed pays nothing for its boundaries.
"""

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Protocol

from pydantic import JsonValue

from nemo_gym._checkpoint.control import (
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    JsonPayload,
    PrepareReport,
    next_attempt,
)
from nemo_gym._checkpoint.errors import AdmissionClosedError, ControlError, RolloutIdRequiredError, StaleAttemptError
from nemo_gym._checkpoint.steps import Boundary, EpisodeSteps, StepMode
from nemo_gym.episode_types import EpisodeId


BoundarySnapshot = Callable[[], dict[str, JsonValue]]


class AgentSessionRecord(CheckpointRecord):
    """One agent session at a committed boundary."""

    session_key: str
    session: JsonPayload
    boundary: Optional[JsonPayload] = None
    # A legacy /run episode's own boundary (which protocol step is next); None for native sessions.
    episode: Optional[JsonPayload] = None


class LegacyRun:
    """Handle for one legacy ``/run`` episode: its restored boundary, and its steps."""

    def __init__(self, steps: EpisodeSteps, key: str) -> None:
        self._steps = steps
        self._key = key
        self.continuation = steps.continuation(key)

    async def boundary(self, state: Boundary) -> None:
        await self._steps.boundary(self._key, state)

    def step(self, mode: StepMode) -> AbstractAsyncContextManager[None]:
        return self._steps.step(self._key, mode)


@dataclass(frozen=True)
class RestoredAgentSession:
    session_key: str
    episode_id: EpisodeId
    session: dict[str, JsonValue]


class AgentSessionHooks(Protocol):
    """Agent-owned session state, exported and restored by the participant.

    The hooks are asynchronous and exporting takes every session at once, so an agent whose session state
    lives outside the process, such as a sandbox it runs tools or a harness in, can checkpoint all of it
    concurrently. Export runs at commit, while every session is parked at a boundary: no tool call is using
    the sandbox. Return what a restore needs to rebuild the sandbox as of the checkpoint, not a descriptor
    of the live sandbox, which keeps changing afterwards. A restore runs in a fresh process after a crash
    and rebuilds every session or raises; the crashed process's sandboxes are still running and can be
    stopped there. Retire stops a discarded attempt's sandbox.
    """

    async def export_agent_sessions(self, session_keys: list[str]) -> dict[str, dict[str, JsonValue]]:
        """Return the state of every session in ``session_keys``."""

    async def restore_agent_sessions(self, sessions: list[RestoredAgentSession]) -> None:
        """Validate every session, then install all of them; never install a partial set."""

    async def retire_agent_session(self, session_key: str) -> None: ...


@dataclass
class _Session:
    key: str
    episode_id: EpisodeId
    state: Literal["idle", "running", "at_boundary"] = "idle"
    park_requested: bool = False
    boundary: Optional[BoundarySnapshot] = None
    continuation: Optional[dict[str, JsonValue]] = None
    task: Optional[asyncio.Task] = None
    resume: asyncio.Event = field(default_factory=asyncio.Event)


class Activation:
    """Handle for one running activation of a session."""

    def __init__(self, participant: "AgentSessionParticipant", session: _Session) -> None:
        self._participant = participant
        self._session = session
        self.continuation, session.continuation = session.continuation, None

    async def boundary(self, snapshot: BoundarySnapshot) -> None:
        """Record a complete loop step and park here if a checkpoint asked for it.

        ``snapshot`` must return the loop state as of this call, even if it runs later while the loop
        waits on the next policy call.
        """
        await self._participant.at_boundary(self._session, snapshot)

    async def park(self) -> None:
        """Park at the latest boundary until resume; used when a policy call is refused as parked."""
        await self._participant.park(self._session)

    @asynccontextmanager
    async def awaiting_model(self) -> AsyncIterator[None]:
        """Wrap a policy call made right after a boundary; the session counts as at that boundary."""
        async with self._participant.awaiting_model(self._session):
            yield


class AgentSessionParticipant(CheckpointParticipant):
    kind = "agent"
    record_model = AgentSessionRecord

    def __init__(self, hooks: AgentSessionHooks) -> None:
        super().__init__()
        self.hooks = hooks
        self.accepting = True
        self._sessions: dict[str, _Session] = {}
        self.legacy_episodes = EpisodeSteps(self.notify)
        self._restored_episodes: dict[str, Boundary] = {}

    # -- sessions and activations ---------------------------------------------------------------

    def open_session(self, session_key: str, episode_id: EpisodeId) -> None:
        """Register a new session, or re-bind a restored one that its replacement attempt re-seeds."""
        self.retiring.check(episode_id)
        existing = self._sessions.get(session_key)
        if existing is not None:
            if existing.episode_id != episode_id:
                raise StaleAttemptError(f"agent session {session_key!r} belongs to {existing.episode_id.capture_key}")
            return
        if not self.accepting:
            raise AdmissionClosedError("agent admission is closed for a checkpoint")
        self._sessions[session_key] = _Session(key=session_key, episode_id=episode_id)

    def has_session(self, session_key: str) -> bool:
        return session_key in self._sessions

    def close_session(self, session_key: str) -> None:
        self._sessions.pop(session_key, None)

    @asynccontextmanager
    async def legacy_run(self, session_key: str, episode_id: EpisodeId) -> AsyncIterator[LegacyRun]:
        """Hold one legacy ``/run`` episode and its turn-loop session."""
        self.open_session(session_key, episode_id)
        session = self._sessions[session_key]
        self.legacy_episodes.begin(session_key, continuation=self._restored_episodes.pop(session_key, None))
        try:
            yield LegacyRun(self.legacy_episodes, session_key)
        finally:
            if self._sessions.get(session_key) is session:
                del self._sessions[session_key]
            await self.legacy_episodes.end(session_key)

    @asynccontextmanager
    async def activation(self, session_key: str, episode_id: EpisodeId) -> AsyncIterator[Activation]:
        self.retiring.check(episode_id)
        session = self._sessions.get(session_key)
        if session is None:
            raise ControlError(f"agent session {session_key!r} is not open")
        if session.episode_id != episode_id:
            raise StaleAttemptError(f"agent session {session_key!r} belongs to {session.episode_id.capture_key}")
        if session.state != "idle":
            raise ControlError(f"agent session {session_key!r} already has an active activation")
        # An activation may start while a checkpoint is preparing: the caller invoked it before
        # admission closed, so it parks at its first boundary.
        session.state = "running"
        session.park_requested = not self.accepting
        session.task = asyncio.current_task()
        try:
            yield Activation(self, session)
        finally:
            if self._live(session):
                session.state = "idle"
                session.boundary = None
                session.task = None
            await self.notify()

    async def at_boundary(self, session: _Session, snapshot: BoundarySnapshot) -> None:
        self._require_live(session)
        session.boundary = snapshot
        if session.park_requested:
            await self.park(session)

    async def park(self, session: _Session) -> None:
        self._require_live(session)
        session.state = "at_boundary"
        session.resume.clear()
        if self.accepting:
            session.resume.set()
        await self.notify()
        while True:
            await session.resume.wait()
            self._require_live(session)
            if self.accepting:
                break
            # A new checkpoint closed admission before this session woke: it stays parked for that one too.
            session.resume.clear()
        session.state = "running"

    @asynccontextmanager
    async def awaiting_model(self, session: _Session) -> AsyncIterator[None]:
        session.state = "at_boundary"
        await self.notify()
        try:
            yield
        finally:
            if self._live(session):
                session.state = "running"

    def _live(self, session: _Session) -> bool:
        return self._sessions.get(session.key) is session

    def _require_live(self, session: _Session) -> None:
        if not self._live(session):
            raise StaleAttemptError(f"agent session {session.key!r} was retired")

    # -- checkpoint -----------------------------------------------------------------------------

    async def close_admission(self, request: CheckpointRequest) -> None:
        self.accepting = False
        self.legacy_episodes.close()
        for session in self._sessions.values():
            session.park_requested = True

    async def open_admission(self) -> None:
        self.accepting = True
        self.legacy_episodes.open()
        for session in self._sessions.values():
            session.park_requested = False
            session.resume.set()

    def readiness(self) -> PrepareReport:
        legacy_blockers = {
            self._sessions[key].episode_id.capture_key
            for key in self.legacy_episodes.blockers()
            if key in self._sessions
        }
        blockers = sorted(
            {session.episode_id.capture_key for session in self._sessions.values() if session.state == "running"}
            | legacy_blockers
        )
        at_boundary = sum(session.state == "at_boundary" for session in self._sessions.values())
        return PrepareReport(
            ready=not blockers,
            blockers=blockers,
            counts={"sessions": len(self._sessions), "at_boundary": at_boundary},
        )

    async def retire(self, episode_id: EpisodeId) -> None:
        for session in list(self._sessions.values()):
            if (
                session.episode_id.rollout_id != episode_id.rollout_id
                or session.episode_id.attempt > episode_id.attempt
            ):
                continue
            del self._sessions[session.key]
            session.resume.set()
            if session.task is not None and session.task is not asyncio.current_task():
                session.task.cancel()
            self._restored_episodes.pop(session.key, None)
            await self.legacy_episodes.retire(session.key)
            await self.hooks.retire_agent_session(session.key)

    async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        legacy = self.legacy_episodes.exported()
        sessions = list(self._sessions.values())
        # Every session is parked, so its boundary cannot move while the hooks are awaited.
        boundaries = [self._export_boundary(session) for session in sessions]
        states = await self.hooks.export_agent_sessions([session.key for session in sessions])
        missing = [session.key for session in sessions if session.key not in states]
        if missing:
            raise ControlError(f"the agent did not export sessions {missing}")
        return [
            AgentSessionRecord(
                session_key=session.key,
                episode_id=session.episode_id,
                session=states[session.key],
                boundary=boundary,
                # A restored legacy episode whose replacement /run has not started keeps its restored step.
                episode=legacy.get(session.key, self._restored_episodes.get(session.key)),
            )
            for session, boundary in zip(sessions, boundaries)
        ]

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        raise NotImplementedError("agent sessions export through export(), which awaits the agent's hooks")

    @staticmethod
    def _export_boundary(session: _Session) -> Optional[dict[str, JsonValue]]:
        if session.state == "at_boundary" and session.boundary:
            return session.boundary()
        # A restored session that no activation has claimed yet still continues from its restored boundary.
        return session.continuation

    async def install(self, records: list[CheckpointRecord]) -> None:
        if self._sessions:
            raise ControlError("agent restore requires a process without live sessions")
        restored = [
            RestoredAgentSession(
                session_key=record.session_key, episode_id=next_attempt(record.episode_id), session=record.session
            )
            for record in records
        ]
        await self.hooks.restore_agent_sessions(restored)
        for record, session in zip(records, restored):
            self._sessions[session.session_key] = _Session(
                key=session.session_key, episode_id=session.episode_id, continuation=record.boundary
            )
            if record.episode is not None:
                self._restored_episodes[session.session_key] = record.episode

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        raise NotImplementedError("agent sessions restore through install(), which awaits the agent's hooks")

    def status_extra(self) -> dict[str, Any]:
        return {"accepting": self.accepting}


def require_rollout(capture_key: Optional[str]) -> str:
    """Refuse agent work that a checkpoint could neither record nor retire."""
    if capture_key is None:
        raise RolloutIdRequiredError(
            "checkpointing requires a rollout id on every agent call: send _ng_rollout_id or the /ng-rollout/<id> prefix"
        )
    return capture_key


class RestartOnlyAgentParticipant(CheckpointParticipant):
    """Participant for an agent that cannot continue a rollout.

    It still takes part in every checkpoint so that an unsupported agent fails closed: its in-flight
    ``/run`` and ``/v1/responses`` calls block prepare until the controller retires them, they export
    nothing, and the retired rollouts restart from their input.
    """

    kind = "agent"
    record_model = CheckpointRecord

    def __init__(self) -> None:
        super().__init__()
        self.accepting = True
        self._tasks: dict[str, set[asyncio.Task]] = {}

    @asynccontextmanager
    async def track(self, capture_key: Optional[str]) -> AsyncIterator[None]:
        key = require_rollout(capture_key)
        self.retiring.check(EpisodeId.from_capture_key(key))
        if not self.accepting:
            raise AdmissionClosedError("agent admission is closed for a checkpoint")
        task = asyncio.current_task()
        self._tasks.setdefault(key, set()).add(task)
        try:
            yield
        finally:
            tasks = self._tasks.get(key, set())
            tasks.discard(task)
            if not tasks:
                self._tasks.pop(key, None)
            await self.notify()

    async def close_admission(self, request: CheckpointRequest) -> None:
        self.accepting = False

    async def open_admission(self) -> None:
        self.accepting = True

    def readiness(self) -> PrepareReport:
        blockers = sorted(self._tasks)
        return PrepareReport(ready=not blockers, blockers=blockers, counts={"inflight": len(blockers)})

    async def retire(self, episode_id: EpisodeId) -> None:
        for key, tasks in list(self._tasks.items()):
            other = EpisodeId.from_capture_key(key)
            if other.rollout_id == episode_id.rollout_id and other.attempt <= episode_id.attempt:
                for task in list(tasks):
                    if task is not asyncio.current_task():
                        task.cancel()

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        return []

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        if records:
            raise ControlError("a restart-only agent cannot restore rollouts")
