# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent server participant: park turn loops at boundaries and export agent sessions.

One participant serves both ways an episode can reach an agent:

- Through an environment server, the agent owns a session named by ``agent_session_id``.
  The environment server owns the protocol position and invokes one activation at a time.
- Through a legacy ``/run``, the agent owns the whole episode: it is both the episode protocol (seed,
  turn loop, verify) and the turn loop.
  The participant keys that episode by its logical rollout ID so a replacement attempt finds the restored state,
  and tracks its steps with the same ``EpisodeSteps`` an environment server uses.

Either way the agent keeps its own session state and exposes it through ``AgentSessionHooks``.
The participant tracks which session is inside an activation and the activation's latest boundary.
An activation reports a boundary before each model call and after each tool round.
A session is ``at_boundary`` while parked there or while waiting on a policy call made right after it:
the policy model server holds that call's response during a checkpoint, so the loop cannot move past the boundary.
A reply the model server delivered before its own prepare can still reach the agent while a checkpoint is open.
The session then holds that reply and stays parked at the boundary before the call until resume,
so a session that prepare counted at its boundary never leaves it.
Prepare is ready once no activation is between boundaries.

A boundary is a snapshot function, not a snapshot.
It runs only when a checkpoint exports the session,
so a rollout that is never checkpointed pays nothing for its boundaries.
"""

import asyncio
import json
from collections import Counter
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
from nemo_gym.rollout_correlation import RolloutContextMiddleware, maybe_rollout_id_from_run_body


BoundarySnapshot = Callable[[], dict[str, JsonValue]]


class AgentSessionRecord(CheckpointRecord):
    """One agent session at a committed boundary."""

    session_key: str
    session: JsonPayload
    boundary: Optional[JsonPayload] = None
    # A legacy /run episode's own boundary (which protocol step is next); None for native sessions.
    episode: Optional[JsonPayload] = None


class ActivationOutOfOrderError(ControlError):
    """An activation index that is neither the session's last completed activation nor the next one."""

    code = "activation_out_of_order"


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
        """A server this episode uses cannot capture its part, such as a restart-only resources server:
        the episode never holds up a checkpoint, is never exported, and starts over from its input after a crash."""
        await self._steps.mark_restart(self._key)


@dataclass(frozen=True)
class RestoredAgentSession:
    session_key: str
    episode_id: EpisodeId
    session: dict[str, JsonValue]


class AgentSessionHooks(Protocol):
    """Agent-owned session state, exported and restored by the participant.

    The hooks are asynchronous and exporting takes every session at once,
    so an agent whose session state lives outside the process, such as a sandbox it runs tools or a harness in,
    can checkpoint all of it concurrently.
    Export runs at commit, while every session is parked at a boundary: no tool call is using the sandbox.
    Return what a restore needs to rebuild the sandbox as of the checkpoint,
    not a descriptor of the live sandbox, which keeps changing afterwards.
    A restore runs in a fresh process after a crash and rebuilds every session or raises;
    the crashed process's sandboxes are still running and can be stopped there.
    Retire stops a discarded attempt's sandbox.
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
    # Installed by a restore and not activated since: a commit that no longer continues its episode retires it.
    restored_pending: bool = False
    # Set by retire, which keeps the session tracked until its activation has stopped.
    retired: bool = False
    # Set by close, which keeps the session tracked until its activation has stopped.
    closing: bool = False


class Activation:
    """Handle for one running activation of a session."""

    def __init__(self, participant: "AgentSessionParticipant", session: _Session) -> None:
        self._participant = participant
        self._session = session
        self.continuation, session.continuation = session.continuation, None
        session.restored_pending = False

    async def boundary(self, snapshot: BoundarySnapshot) -> None:
        """Record a complete loop step and park here if a checkpoint asked for it.

        ``snapshot`` must return the loop state as of this call,
        even if it runs later while the loop waits on the next policy call.
        """
        await self._participant.at_boundary(self._session, snapshot)

    @asynccontextmanager
    async def awaiting_model(self) -> AsyncIterator[None]:
        """Wrap a policy call made right after a boundary; the session counts as at that boundary.

        A reply that arrives while a checkpoint is open is held here until resume.
        """
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
        # Seeds in progress, counted by session key and attempt; each blocks prepare until its session is registered.
        # A retire waits for the seeds of its attempts, so a seed it refuses frees what it built first.
        self._seeding: dict[str, Counter[EpisodeId]] = {}
        self._open = asyncio.Event()
        self._open.set()

    # -- sessions and activations ---------------------------------------------------------------

    @asynccontextmanager
    async def seeding(self, session_key: str, episode_id: EpisodeId) -> AsyncIterator[None]:
        """Hold one session seed, from before the agent builds its state until the session is registered.

        The environment server seeds inside a replay step that a checkpoint does not wait for,
        so a seed can meet a checkpoint either way, and the agent's session must match the environment's boundary.
        A seed that arrives while a checkpoint is open waits for resume: the environment's step is still in flight,
        so its boundary precedes the seed, and after a crash it seeds again.
        A seed already in progress when admission closes blocks prepare until its session is registered,
        and is then exported: the environment may record its boundary after the seed before the commit.
        The same session can be seeded more than once at a time,
        for example by a retried request; each seed is counted.
        A seed whose attempt is retired before its session is registered frees the state it built.
        """
        self.retired.check(episode_id)
        while not self.accepting:
            await self._open.wait()
        # A retire may have marked the attempt while this seed waited for resume.
        self.retired.check(episode_id)
        seeds = self._seeding.setdefault(session_key, Counter())
        seeds[episode_id] += 1
        try:
            yield
        except StaleAttemptError:
            # The retire waits for this seed, so its reply still means that nothing of the attempt is left.
            if self._attempt_retired(episode_id) and session_key not in self._sessions:
                await self.hooks.retire_agent_session(session_key)
            raise
        finally:
            seeds[episode_id] -= 1
            if not seeds[episode_id]:
                del seeds[episode_id]
            if not seeds and self._seeding.get(session_key) is seeds:
                del self._seeding[session_key]
            await self.notify()

    def _attempt_retired(self, episode_id: EpisodeId) -> bool:
        try:
            self.retired.check(episode_id)
        except StaleAttemptError:
            return True
        return False

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
        self._sessions[session_key] = _Session(key=session_key, episode_id=episode_id)

    def has_session(self, session_key: str) -> bool:
        return session_key in self._sessions

    async def close_session(self, session_key: str) -> None:
        """Stop an activation still running for a closed session, then forget the session.

        The environment server closes a session when its episode ends,
        including when it gave up waiting on the activation;
        nothing would use the activation's further model and tool calls.
        The session stays tracked, and holds up prepare, until its activation has stopped.
        """
        session = self._sessions.get(session_key)
        if session is None:
            return
        session.closing = True
        task = session.task
        if task is not None and task is not asyncio.current_task():
            # Cancel once: a retire, possibly cut short by its deadline, may have cancelled it already,
            # and a second cancellation would interrupt its cleanup.
            if not session.retired and not task.cancelling():
                task.cancel()
            await asyncio.wait([task])
        if self._sessions.get(session_key) is session and not session.retired:
            del self._sessions[session_key]

    @asynccontextmanager
    async def legacy_run(self, session_key: str, episode_id: EpisodeId) -> AsyncIterator[LegacyRun]:
        """Hold one legacy ``/run`` episode and its turn-loop session."""
        self.open_session(session_key, episode_id)
        session = self._sessions[session_key]
        # This /run claims a restored episode whatever step it continues at, so a commit no longer retires it.
        session.restored_pending = False
        self.legacy_episodes.begin(session_key, continuation=self._restored_episodes.pop(session_key, None))
        try:
            yield LegacyRun(self.legacy_episodes, session_key)
        finally:
            if self._sessions.get(session_key) is session:
                del self._sessions[session_key]
            await self.legacy_episodes.end(session_key)

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
        # An activation may start while a checkpoint is preparing: the caller invoked it before admission closed,
        # so it parks at its first boundary.
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
                if session.closing and not session.retired:
                    # A close cut short leaves the session to its activation, which forgets it once stopped.
                    del self._sessions[session.key]
            if not self.accepting:
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
        if not self.accepting:
            await self.notify()
        try:
            yield
        except BaseException:
            if self._live(session):
                session.state = "running"
            raise
        if session.park_requested:
            # The reply arrived while a checkpoint is open, maybe after prepare counted the session at its boundary.
            # Hold it and stay parked at the boundary before the call until resume, so the session never leaves it.
            # After a crash the restored session calls again from that boundary;
            # if the model server's checkpoint kept this call, the token-capture builder treats the two as a retry.
            await self.park(session)
        elif self._live(session):
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
        # A session being retired or closed blocks until its activation has stopped, whatever its state.
        # A restart's running session does not: nothing of it is in the checkpoint.
        blockers = sorted(
            {
                session.episode_id.capture_key
                for key, session in self._sessions.items()
                if session.retired or session.closing or (session.state == "running" and key not in restart_keys)
            }
            | legacy_blockers
            | {episode_id.capture_key for seeds in self._seeding.values() for episode_id in seeds}
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

    def ready(self) -> bool:
        # A prepare asks after every change, so stop at the first blocker instead of listing and sorting them all.
        if self._seeding or self.legacy_episodes.blocker_count():
            return False
        restart_keys: Optional[set[str]] = None
        for key, session in self._sessions.items():
            if session.retired or session.closing:
                return False
            if session.state == "running":
                if restart_keys is None:
                    restart_keys = set(self.legacy_episodes.restarts())
                if key not in restart_keys:
                    return False
        return True

    async def retire(self, episode_id: EpisodeId) -> None:
        # A seed admitted before the attempt was marked registers its session or, refused, frees what it built.
        # Wait for it, so nothing of the attempt is left once this returns.
        while self._seeding_any(episode_id):
            await self.wait_changed(1.0)
        for session in list(self._sessions.values()):
            if (
                session.episode_id.rollout_id != episode_id.rollout_id
                or session.episode_id.attempt > episode_id.attempt
            ):
                continue
            # The session stays tracked until its activation has stopped,
            # so a retire cut short by its deadline leaves it for a retry to wait for again,
            # without cancelling a second time.
            first = not session.retired
            session.retired = True
            session.resume.set()
            task = session.task
            if task is not None and task is not asyncio.current_task():
                # A close may have cancelled it already.
                if first and not task.cancelling():
                    task.cancel()
                # The activation has stopped calling the model and tools before the session is freed.
                await asyncio.wait([task])
            await self.legacy_episodes.retire(session.key)
            await self.hooks.retire_agent_session(session.key)
            # Freed last, so a retire cut short by its deadline in either await leaves the session for a retry.
            if self._sessions.get(session.key) is session:
                del self._sessions[session.key]
            self._restored_episodes.pop(session.key, None)

    def _seeding_any(self, episode_id: EpisodeId) -> bool:
        """Whether a seed of ``episode_id`` or an earlier attempt of its rollout is in progress."""
        return any(
            seeding.rollout_id == episode_id.rollout_id and seeding.attempt <= episode_id.attempt
            for seeds in self._seeding.values()
            for seeding in seeds
        )

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
        if self._sessions or self._seeding:
            raise ControlError("agent restore requires a process without live sessions or seeds in progress")
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

    It still takes part in every checkpoint so that an unsupported agent fails closed:
    its in-flight ``/run`` and ``/v1/responses`` calls are reported as restarts,
    which the controller leaves out of the commit and starts over from their input after a crash.
    Nothing of it is in a checkpoint, so it never closes admission and never holds up prepare:
    its rollouts keep running through a checkpoint.
    Its seed replies say so, so an environment server's episode that uses it is a restart too.
    """

    kind = "agent"
    record_model = CheckpointRecord

    def __init__(self) -> None:
        super().__init__()
        self._tasks: dict[str, set[asyncio.Task]] = {}
        # Cleared while a checkpoint is open: a new session would add a restart the checkpoint did not report.
        self._open = asyncio.Event()
        self._open.set()

    def admit(self, capture_key: Optional[str]) -> str:
        """Refuse work that a checkpoint could neither report nor retire, and work of a retired attempt."""
        key = require_rollout(capture_key)
        self.retired.check(EpisodeId.from_capture_key(key))
        return key

    @asynccontextmanager
    async def track(self, capture_key: Optional[str]) -> AsyncIterator[None]:
        key = self.admit(capture_key)
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


class RestartOnlyTrackingMiddleware:
    """Count every ``/run`` and ``/v1/responses`` call of a restart-only agent against its participant.

    A middleware, not a route wrapper, so it also covers an agent that builds its own app and routes.
    A call is keyed by its ``/ng-rollout/<id>`` path prefix, and a ``/run`` without one by its request body.
    """

    def __init__(self, app: Any, participant: RestartOnlyAgentParticipant) -> None:
        self.app = app
        self.participant = participant

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        path = scope.get("path", "")
        prefix = RolloutContextMiddleware._PREFIX.match(path)
        route = prefix.group("rest") if prefix is not None else path
        if route != "/run" and not route.endswith("/v1/responses"):
            await self.app(scope, receive, send)
            return
        capture_key = prefix.group("rollout_id") if prefix is not None else None
        if capture_key is None and route == "/run":
            capture_key, receive = await _run_body_rollout_id(receive)
        try:
            key = self.participant.admit(capture_key)
        except ControlError as error:
            await error.response()(scope, receive, send)
            return
        async with self.participant.track(key):
            await self.app(scope, receive, send)


async def _run_body_rollout_id(receive: Any) -> tuple[Optional[str], Any]:
    """Read a ``/run`` request's body for its rollout ID, and return a receive that replays the body."""
    messages = []
    body = b""
    while True:
        message = await receive()
        messages.append(message)
        if message["type"] != "http.request":
            break
        body += message.get("body", b"")
        if not message.get("more_body"):
            break

    async def replay() -> dict[str, Any]:
        return messages.pop(0) if messages else await receive()

    try:
        payload = json.loads(body)
        capture_key = maybe_rollout_id_from_run_body(payload) if isinstance(payload, dict) else None
    except ValueError:
        # A malformed body or rollout ID; the request is refused as one without a rollout.
        capture_key = None
    return capture_key, replay
