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
from collections.abc import AsyncIterator, Awaitable, Callable
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
from nemo_gym.server_utils import current_session_id


BoundarySnapshot = Callable[[], dict[str, JsonValue]]


class AgentSessionRecord(CheckpointRecord):
    """One agent session at a committed boundary."""

    session_key: str
    session: JsonPayload
    boundary: Optional[JsonPayload] = None
    # A legacy /run episode's own boundary (which protocol step is next); None for native sessions.
    episode: Optional[JsonPayload] = None
    # With several workers, the worker ID in the session's cookie (see nemo_gym.session_routing).
    owner: Optional[str] = None
    # The session ID in the session's cookie, which routes a restored session whose cookie names no worker.
    session_id: Optional[str] = None


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

    async def mark_restart(self) -> None:
        """A server this episode uses cannot capture its part, such as a restart-only resources server: the
        episode never holds up a checkpoint, is never exported, and starts over from its input after a crash."""
        await self._steps.mark_restart(self._key)


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
    owner: Optional[str] = None
    session_id: Optional[str] = None
    state: Literal["idle", "running", "at_boundary"] = "idle"
    park_requested: bool = False
    boundary: Optional[BoundarySnapshot] = None
    continuation: Optional[dict[str, JsonValue]] = None
    task: Optional[asyncio.Task] = None
    resume: asyncio.Event = field(default_factory=asyncio.Event)
    # Installed by a restore and not activated since: a commit that no longer continues its episode retires it.
    restored_pending: bool = False
    # Set by retire, which keeps the session tracked until its activation has stopped.
    retired: bool = False


class Activation:
    """Handle for one running activation of a session."""

    def __init__(self, participant: "AgentSessionParticipant", session: _Session) -> None:
        self._participant = participant
        self._session = session
        self.continuation, session.continuation = session.continuation, None
        session.restored_pending = False

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
        # Seeds in progress, by session key; each blocks prepare until its session is registered.
        self._seeding: dict[str, EpisodeId] = {}
        self._open = asyncio.Event()
        self._open.set()
        # With several workers: this worker's routing owner, and where restored legacy episodes are claimed.
        self.owner: Optional[str] = None
        self.claim_restored: Optional[Callable[[str], Awaitable[Optional[dict[str, Any]]]]] = None

    @staticmethod
    def claim_key(record: CheckpointRecord) -> Optional[str]:
        # A legacy /run episode lives inside its /run; a native session spans requests.
        return record.session_key if record.episode is not None else None

    # -- sessions and activations ---------------------------------------------------------------

    @asynccontextmanager
    async def seeding(self, session_key: str, episode_id: EpisodeId) -> AsyncIterator[None]:
        """Hold one session seed, from before the agent builds its state until the session is registered.

        The environment server seeds inside a replay step that a checkpoint does not wait for, so a seed can meet
        a checkpoint either way, and the agent's session must match the environment's boundary. A seed that
        arrives while a checkpoint is open waits for resume: the environment's step is still in flight, so its
        boundary precedes the seed, and after a crash it seeds again. A seed already in progress when admission
        closes blocks prepare until its session is registered, and is then exported: the environment may record
        its boundary after the seed before the commit.
        """
        self.retired.check(episode_id)
        while not self.accepting:
            await self._open.wait()
        self._seeding[session_key] = episode_id
        try:
            yield
        finally:
            del self._seeding[session_key]
            await self.notify()

    def open_session(self, session_key: str, episode_id: EpisodeId, *, seed: bool = False) -> None:
        """Register a new session, or re-bind a restored one that its replacement attempt re-seeds.

        A ``seed`` inside ``seeding`` is registered even if admission closed after it began.
        """
        self.retired.check(episode_id)
        existing = self._sessions.get(session_key)
        if existing is not None:
            if existing.episode_id != episode_id:
                raise StaleAttemptError(f"agent session {session_key!r} belongs to {existing.episode_id.capture_key}")
            return
        if not self.accepting and not (seed and session_key in self._seeding):
            raise AdmissionClosedError("agent admission is closed for a checkpoint")
        self._sessions[session_key] = _Session(
            key=session_key, episode_id=episode_id, owner=self.owner, session_id=current_session_id()
        )

    def has_session(self, session_key: str) -> bool:
        return session_key in self._sessions

    def close_session(self, session_key: str) -> None:
        """Forget a closed session, and stop an activation still running for it.

        The environment server closes a session when its episode ends, including when it gave up waiting on the
        activation; nothing would use the activation's further model and tool calls.
        """
        session = self._sessions.pop(session_key, None)
        if session is None:
            return
        session.resume.set()
        if session.task is not None and session.task is not asyncio.current_task():
            session.task.cancel()

    @asynccontextmanager
    async def legacy_run(self, session_key: str, episode_id: EpisodeId) -> AsyncIterator[LegacyRun]:
        """Hold one legacy ``/run`` episode and its turn-loop session."""
        await self._claim(session_key, episode_id)
        self.open_session(session_key, episode_id)
        session = self._sessions[session_key]
        self.legacy_episodes.begin(session_key, continuation=self._restored_episodes.pop(session_key, None))
        try:
            yield LegacyRun(self.legacy_episodes, session_key)
        finally:
            if self._sessions.get(session_key) is session:
                del self._sessions[session_key]
            await self.legacy_episodes.end(session_key)

    async def _claim(self, session_key: str, episode_id: EpisodeId) -> None:
        """Take a restored legacy episode from the coordinator, if there is one, and install it here."""
        if self.claim_restored is None or session_key in self._sessions:
            return
        self.retired.check(episode_id)
        claimed = await self.claim_restored(session_key)
        if claimed is None:
            return
        record = AgentSessionRecord.model_validate(claimed)
        restored = RestoredAgentSession(
            session_key=session_key, episode_id=next_attempt(record.episode_id), session=record.session
        )
        # Registered before the agent's hook runs, so a checkpoint that closes meanwhile still exports it.
        self._sessions[session_key] = _Session(
            key=session_key, episode_id=restored.episode_id, owner=record.owner, continuation=record.boundary
        )
        self._restored_episodes[session_key] = record.episode
        try:
            await self.hooks.restore_agent_sessions([restored])
        except BaseException:
            self._sessions.pop(session_key, None)
            self._restored_episodes.pop(session_key, None)
            raise

    @asynccontextmanager
    async def activation(self, session_key: str, episode_id: EpisodeId) -> AsyncIterator[Activation]:
        self.retired.check(episode_id)
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
        self._open.clear()
        self.legacy_episodes.close()
        for session in self._sessions.values():
            session.park_requested = True

    async def open_admission(self) -> None:
        self.accepting = True
        self._open.set()
        self.legacy_episodes.open()
        for session in self._sessions.values():
            session.park_requested = False
            session.resume.set()

    def readiness(self) -> PrepareReport:
        legacy_blockers = {
            self._sessions[key].episode_id.capture_key if key in self._sessions else key
            for key in self.legacy_episodes.blockers(self.legacy_episodes.blocker_count())
        }
        restart_keys = set(self.legacy_episodes.restarts())
        # A session being retired blocks until its retire has finished, whatever its state. A restart's running
        # session does not: nothing of it is in the checkpoint.
        blockers = sorted(
            {
                session.episode_id.capture_key
                for key, session in self._sessions.items()
                if session.retired or (session.state == "running" and key not in restart_keys)
            }
            | legacy_blockers
            | {episode_id.capture_key for episode_id in self._seeding.values()}
        )
        at_boundary = sum(session.state == "at_boundary" for session in self._sessions.values())
        return PrepareReport(
            ready=not blockers,
            blockers=blockers,
            counts={"sessions": len(self._sessions), "at_boundary": at_boundary},
            restarts=sorted(
                self._sessions[key].episode_id.capture_key for key in restart_keys if key in self._sessions
            ),
        )

    async def retire(self, episode_id: EpisodeId) -> None:
        for session in list(self._sessions.values()):
            if (
                session.episode_id.rollout_id != episode_id.rollout_id
                or session.episode_id.attempt > episode_id.attempt
            ):
                continue
            # The session stays tracked until its activation has stopped, so a retire cut short by its deadline
            # leaves it for a retry to wait for again, without cancelling a second time.
            first = not session.retired
            session.retired = True
            session.resume.set()
            task = session.task
            if task is not None and task is not asyncio.current_task():
                if first:
                    task.cancel()
                # The activation has stopped calling the model and tools before the session is freed.
                await asyncio.wait([task])
            await self.legacy_episodes.retire(session.key)
            await self.hooks.retire_agent_session(session.key)
            # Freed last, so a retire cut short by its deadline in either await leaves the session for a retry.
            if self._sessions.get(session.key) is session:
                del self._sessions[session.key]
            self._restored_episodes.pop(session.key, None)

    async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        legacy = self.legacy_episodes.exported()
        # Unclaimed restored sessions outside the commit's scope are not exported: the commit retires them.
        scope = None if episode_ids is None else set(episode_ids)
        restart_keys = set(self.legacy_episodes.restarts())
        sessions = [
            session
            for key, session in self._sessions.items()
            if not session.retired
            and key not in restart_keys
            and (scope is None or not session.restored_pending or session.episode_id in scope)
        ]
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
                owner=session.owner,
                session_id=session.session_id,
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

    async def install(self, records: list[CheckpointRecord], scope: list[EpisodeId]) -> None:
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
                key=session.session_key,
                episode_id=session.episode_id,
                owner=record.owner,
                session_id=record.session_id,
                continuation=record.boundary,
                restored_pending=True,
            )
            if record.episode is not None:
                self._restored_episodes[session.session_key] = record.episode

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        raise NotImplementedError("agent sessions restore through install(), which awaits the agent's hooks")

    async def restored_pending(self) -> list[EpisodeId]:
        return [session.episode_id for session in self._sessions.values() if session.restored_pending]

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
    ``/run`` and ``/v1/responses`` calls are reported as restarts, which the controller leaves out of the
    commit and starts over from their input after a crash. Nothing of it is in a checkpoint, so it never
    closes admission and never holds up prepare: its rollouts keep running through a checkpoint. Its seed
    replies say so, so an environment server's episode that uses it is a restart too.
    """

    kind = "agent"
    record_model = CheckpointRecord

    def __init__(self) -> None:
        super().__init__()
        self.accepting = True
        self._tasks: dict[str, set[asyncio.Task]] = {}
        # Cleared while a checkpoint is open: a new session would add a restart the checkpoint did not report.
        self._open = asyncio.Event()
        self._open.set()

    @asynccontextmanager
    async def track(self, capture_key: Optional[str]) -> AsyncIterator[None]:
        key = require_rollout(capture_key)
        self.retired.check(EpisodeId.from_capture_key(key))
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
        # Nothing of this agent is in the checkpoint, so its rollouts keep running through it; only its seeds wait.
        self._open.clear()

    async def open_admission(self) -> None:
        self.accepting = True
        self._open.set()

    async def wait_open(self) -> None:
        """Return once no checkpoint is open, so a seed never adds a restart to an open checkpoint."""
        await self._open.wait()

    def readiness(self) -> PrepareReport:
        restarts = sorted(self._tasks)
        return PrepareReport(ready=True, restarts=restarts, counts={"inflight": len(restarts)})

    async def retire(self, episode_id: EpisodeId) -> None:
        for key, tasks in list(self._tasks.items()):
            other = EpisodeId.from_capture_key(key)
            if other.rollout_id == episode_id.rollout_id and other.attempt <= episode_id.attempt:
                stopping = [task for task in list(tasks) if task is not asyncio.current_task()]
                for task in stopping:
                    # Cancel once: a retry after a retire cut short must not interrupt the task's cleanup.
                    if not task.cancelling():
                        task.cancel()
                if stopping:
                    await asyncio.wait(stopping)

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        return []

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        if records:
            raise ControlError("a restart-only agent cannot restore rollouts")
