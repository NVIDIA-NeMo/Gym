# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources server participant: quiesce tool traffic and export session state.

Resources servers keep per-rollout state under the cookie session ID.
The participant learns each session's lifecycle from the requests that already define it:
a successful session start binds the session to the episode named by the request's rollout prefix,
and a successful ``/close_session`` or ``/verify`` ends it.
A replayable ``/verify`` is the exception when its caller says it closes the session afterwards.
The caller says so with ``CHECKPOINT_CLOSE_FOLLOWS_HEADER``, and the session stays tracked until that close.
A checkpoint may keep the caller's boundary from before the verification even when the verification finishes first,
so the session must still be exported, and a restore runs the verification again against the restored state.
No route has to be classified as a mutation or read.
The session starts are the routes the server declares in ``checkpoint_session_starts``, ``/seed_session`` by default.
Every one of them passes the same admission boundary before its handler runs,
and one without a rollout prefix is refused, since nothing could track the session it creates.
A ``/close_session`` or ``/verify`` without a rollout prefix is refused too.
Only the owning attempt may use a session: any other request whose prefix names another episode is refused,
so a harness of a replaced attempt cannot end or verify the session a restore gave its replacement.
A request on a session a restore installed must name its attempt:
one that carries only the session cookie is refused, since it could come from the attempt the restore replaced.
An MCP tool call carries no prefix.
Its signed session token names the attempt whose seed minted it instead, and the same checks apply to that attempt,
so a harness of a retired or replaced attempt cannot call a tool after a retire or a restore.
A protocol that starts sessions under another name, such as Gymnasium's ``/reset``, declares that route;
one whose end no route reveals, such as a terminal ``/step``, reports it through ``checkpoint_session_ended``.

A server declares how its sessions checkpoint:

- ``stateless``: sessions hold nothing a continuation needs.
  A checkpoint still records which sessions are live, without state,
  so with several workers a restore places them on a worker that their cookies then reach,
  and frees them like any other session.
- ``exported``: the server exports and restores each session's state through ``ResourcesSessionHooks``.
- ``restart_only``: the default.
  Nothing of the server is in a checkpoint: its requests never hold up prepare,
  and every live session is reported as a restart.
  Its seed reply says so, so the episode that seeded it is a restart too:
  the controller leaves it out of the commit and starts it over from its input after a crash.
  Only its seeds wait out a checkpoint, so the restarts it reports never change during one;
  every other request is admitted.
  This fails closed for servers nobody has audited.

A retire frees each of the attempt's sessions through ``retire_session_state`` in every mode,
so a server that keeps per-session state without checkpointing it, such as a restart-only game, still releases it.

Prepare closes admission for data routes and is ready once no request is in flight.
A request on a session of an episode that an earlier prepare stage reported as a restart does not count:
the commit leaves that episode out, and its session is not exported.
A request that arrives while a checkpoint is open waits for resume instead of being refused:
a continued episode parks before it calls again, so only a restart,
which keeps running through the checkpoint, sends one.
A seed already in flight holds up prepare, so a session always matches the environment server's boundary.
"""

import asyncio
import logging
from collections import Counter
from collections.abc import Callable
from typing import Any, NamedTuple, Optional, Protocol

from pydantic import JsonValue

from nemo_gym._checkpoint.control import (
    CheckpointMode,
    CheckpointParticipant,
    CheckpointRecord,
    CheckpointRequest,
    JsonPayload,
    PrepareReport,
    next_attempt,
)
from nemo_gym._checkpoint.errors import (
    AdmissionClosedError,
    ControlError,
    RetireIncompleteError,
    RolloutIdRequiredError,
    StaleAttemptError,
)
from nemo_gym._checkpoint.steps import (
    CHECKPOINT_CLOSE_FOLLOWS_HEADER,
    CHECKPOINT_RESTART_HEADER,
    CHECKPOINT_VERIFY_HEADER,
    StepType,
)
from nemo_gym.core.episode import EpisodeId, current_episode_id
from nemo_gym.server_utils import SESSION_ID_KEY


LOGGER = logging.getLogger(__name__)

_SESSION_START = "/seed_session"
_SESSION_CLOSE = "/close_session"
_SESSION_ENDS = (_SESSION_CLOSE, "/verify")


class ResourcesSessionRecord(CheckpointRecord):
    session_id: str
    state: JsonPayload


class MCPTokenClaims(NamedTuple):
    """What a verified MCP session token says about a tool call."""

    session_id: str
    # The attempt whose seed minted the token.
    # ``None`` for a token minted without checkpointing, which names no attempt.
    episode_id: Optional[EpisodeId]


class ResourcesSessionHooks(Protocol):
    """Session state owned by an ``exported`` resources server.

    The hooks are asynchronous and exporting takes every session at once,
    so a server whose state lives outside the process, such as a sandbox per session,
    can checkpoint all of it concurrently.

    A sandbox per session plugs in here:

    - ``export_session_states`` runs at commit.
      Prepare stops agents before resources servers, so no episode step is using a sandbox by then.
      Return what a restore needs to rebuild each sandbox as of this checkpoint, such as a snapshot id.
      A descriptor of the live sandbox is not enough on its own: the sandbox keeps changing after the checkpoint.
    - ``restore_session_states`` runs in a fresh process after a crash.
      Rebuild every session's sandbox or raise; the controller then restarts those rollouts from their inputs.
      The crashed process's sandboxes are still running and can be stopped here.
    - ``retire_session_state`` runs when an attempt is discarded, whatever the server's mode.
      Stop its sandbox.
      It runs again for the same session if a retire was cut short, so it must tolerate state it already freed.

    Whatever a checkpoint keeps outside Gym, such as snapshots,
    is the server's to delete once no checkpoint the controller may restore refers to it.
    """

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        """Return the state of each session; leave out a session the server no longer holds."""

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        """Validate every state, then install all of them; never install a partial set."""

    async def retire_session_state(self, session_id: str) -> None: ...


class ResourcesParticipant(CheckpointParticipant):
    kind = "resources"
    record_model = ResourcesSessionRecord

    def __init__(
        self, hooks: ResourcesSessionHooks, mode: CheckpointMode, *, verify_step_type: StepType = "wait"
    ) -> None:
        super().__init__()
        self.hooks = hooks
        self.mode = mode
        self.verify_step_type = verify_step_type
        # Requests the checkpoint does not wait for:
        # a verification the server declared replayable does not change checkpointed state.
        # A seed is waited for, so the session matches the environment's boundary.
        self.replayable_paths = {"/verify"} if verify_step_type == "replay" else set()
        self.accepting = True
        self._open = asyncio.Event()
        self._open.set()
        self.inflight = 0
        self._sessions: dict[str, EpisodeId] = {}
        # The same sessions by rollout, so retiring a batch of rollouts does not scan every session for each one.
        self._rollout_sessions: dict[str, set[str]] = {}
        # Seeds admitted and not finished, by attempt: a retire waits for them, so a session they create is freed too.
        self._seeding: Counter[EpisodeId] = Counter()
        # Sessions a restore installed, until they end.
        # A request on one must name its attempt, since the attempt the restore replaced holds the same cookie.
        self._restored_sessions: set[str] = set()
        # Sessions of retired attempts, by rollout, refused until the controller forgets the rollout:
        # a request without a rollout prefix names only its session, which is gone once retire freed it.
        # And requests in flight per session.
        self._retired_sessions: dict[str, set[str]] = {}
        self._retired_session_rollouts: dict[str, str] = {}
        self._session_requests: Counter[str] = Counter()
        # The requests in flight per session that a checkpoint waits for, the ones ``inflight`` counts.
        self._session_waited: Counter[str] = Counter()
        # Restored sessions no request has used yet; a commit that no longer continues their episode retires them.
        self._unclaimed: set[str] = set()

    def admit(
        self,
        session_id: Optional[str],
        path: str,
        *,
        starts_session: bool = False,
        mcp_token: Optional[MCPTokenClaims] = None,
    ) -> None:
        # The attempts the request speaks for: the one its rollout prefix names, and for an MCP tool call,
        # the one its session token names.
        prefix = current_episode_id()
        episode_ids = [] if prefix is None else [prefix]
        if mcp_token is not None:
            if mcp_token.episode_id is None:
                # With checkpointing on, every token names the attempt whose seed minted it.
                # One that names none was minted by a process without checkpointing,
                # such as one from before a restart, so nothing can tell whether its attempt was retired.
                raise StaleAttemptError(f"the MCP session token for {mcp_token.session_id!r} names no attempt")
            episode_ids.append(mcp_token.episode_id)
        bound = self._sessions.get(session_id) if session_id is not None else None
        for episode_id in episode_ids:
            if not starts_session and bound is not None and bound != episode_id:
                # A session belongs to the attempt that started it, or that a restore installed it for.
                # A request from any other attempt, such as a harness of a replaced attempt that outlived a crash,
                # must not end or change it.
                raise StaleAttemptError(
                    f"resources session {session_id!r} belongs to {bound.capture_key!r}, not {episode_id.capture_key!r}"
                )
        # A close from the owning attempt is never refused, even once that attempt is retired:
        # it is how the server frees a session's state.
        if path != _SESSION_CLOSE:
            if session_id in self._retired_session_rollouts:
                raise StaleAttemptError(f"resources session {session_id!r} belongs to a retired attempt")
            for episode_id in episode_ids:
                self.retired.check(episode_id)
        if not starts_session and not episode_ids and session_id in self._restored_sessions:
            # A harness of the replaced attempt that outlived a crash holds the same cookie as the replacement.
            # The request is refused before it could change the restored state or claim the session.
            raise RolloutIdRequiredError(
                f"with checkpointing on, {path} needs the /ng-rollout/<id> prefix of the attempt that owns "
                f"restored resources session {session_id!r}"
            )
        # A replay-safe request may still arrive from an episode step the checkpoint did not wait for.
        # Refusing it would fail that episode; its episode re-runs it after a crash anyway.
        if not self.accepting and path not in self.replayable_paths:
            raise AdmissionClosedError("resources admission is closed for a checkpoint")
        self._unclaimed.discard(session_id)

    async def wait_open(self) -> None:
        """Return once admission is open: after resume, or when a lease expires.

        It checks again on waking: a new checkpoint may close admission before the waiting request resumes.
        """
        while not self._open.is_set():
            await self._open.wait()

    def seeded(self, session_id: str, episode_id: EpisodeId) -> None:
        """A request started ``session_id`` for ``episode_id``."""
        self._bind(session_id, episode_id)
        if self._is_retired(episode_id):
            # A seed can finish after its attempt was retired, for example when a retire was cut short while it waited.
            # Its session is retired too: refused, never exported, and kept until a retried retire frees it.
            self._retired_sessions.setdefault(episode_id.rollout_id, set()).add(session_id)
            self._retired_session_rollouts[session_id] = episode_id.rollout_id

    def _bind(self, session_id: str, episode_id: EpisodeId) -> None:
        # Recorded in every mode: a stateless session has nothing to export, but a retire still drains and refuses it.
        # A seed of a restored session's ID claims it: a commit must not delete it as unclaimed.
        self._unbind(session_id)
        self._unclaimed.discard(session_id)
        self._sessions[session_id] = episode_id
        self._rollout_sessions.setdefault(episode_id.rollout_id, set()).add(session_id)

    def _is_retired(self, episode_id: EpisodeId) -> bool:
        try:
            self.retired.check(episode_id)
        except StaleAttemptError:
            return True
        return False

    def ended(self, session_id: str) -> None:
        self._unbind(session_id)
        self._unclaimed.discard(session_id)
        self._restored_sessions.discard(session_id)

    def session_ended_by(self, session_id: str, path: str, *, close_follows: bool = False) -> None:
        """A request on ``path`` ended ``session_id``.

        A replayable ``/verify`` whose caller closes the session afterwards leaves it tracked until that close.
        """
        if not (close_follows and path in self.replayable_paths):
            self.ended(session_id)

    def _unbind(self, session_id: str) -> None:
        episode_id = self._sessions.pop(session_id, None)
        if episode_id is not None:
            sessions = self._rollout_sessions.get(episode_id.rollout_id)
            if sessions is not None:
                sessions.discard(session_id)
                if not sessions:
                    del self._rollout_sessions[episode_id.rollout_id]

    def seed_started(self, episode_id: EpisodeId) -> None:
        self._seeding[episode_id] += 1

    def seed_ended(self, episode_id: EpisodeId) -> None:
        self._seeding[episode_id] -= 1
        if not self._seeding[episode_id]:
            del self._seeding[episode_id]

    async def close_admission(self, request: CheckpointRequest) -> None:
        self._open.clear()
        if self.mode == "restart_only":
            # Nothing of this server is in the checkpoint, so its sessions' rollouts keep running through it;
            # only its seeds wait, so the restarts it reports do not change during the checkpoint.
            return
        self.accepting = False

    async def open_admission(self) -> None:
        self.accepting = True
        self._open.set()

    def readiness(self) -> PrepareReport:
        # A seed in flight is a restart too: its reply always says so, and prepare does not wait for it.
        restarts = (
            sorted(
                {episode_id.capture_key for _, episode_id in self._checkpointed_sessions()}
                | {episode_id.capture_key for episode_id in self._seeding}
            )
            if self.mode == "restart_only"
            else []
        )
        # A session whose retire was cut short is still being freed: a checkpoint waits for a retried retire.
        blockers = sorted(
            {
                self._sessions[session_id].capture_key
                for session_id in self._retired_session_rollouts
                if session_id in self._sessions
            }
        )
        # A restart_only server's requests are never captured, so none of them holds up a checkpoint.
        inflight = 0 if self.mode == "restart_only" else self.inflight - self._carried_inflight()
        return PrepareReport(
            ready=inflight == 0 and not blockers,
            blockers=blockers,
            counts={"inflight": self.inflight, "sessions": len(self._sessions)},
            restarts=restarts,
        )

    def ready(self) -> bool:
        if self.mode != "restart_only" and self.inflight > self._carried_inflight():
            return False
        # Retired sessions not yet freed are few; scanning them, not every session, keeps each prepare wake cheap.
        return not any(session_id in self._sessions for session_id in self._retired_session_rollouts)

    def _carried_inflight(self) -> int:
        """Requests in flight on the sessions of episodes an earlier prepare stage reported as restarts.

        The commit leaves those episodes out, so a checkpoint does not wait for their tool calls.
        """
        if not self.carried_restarts or not self.inflight:
            return 0
        return sum(self._session_waited[session_id] for session_id in self._carried_sessions())

    def _carried_sessions(self) -> set[str]:
        """The sessions of the episodes an earlier prepare stage reported as restarts."""
        carried = {EpisodeId.from_capture_key(key) for key in self.carried_restarts}
        return {
            session_id
            for episode_id in carried
            for session_id in self._rollout_sessions.get(episode_id.rollout_id, ())
            if self._sessions[session_id] == episode_id
        }

    async def mark_retired(self, episode_ids: list[EpisodeId]) -> None:
        await super().mark_retired(episode_ids)
        self.mark_retired_sessions(episode_ids)

    def mark_retired_sessions(self, episode_ids: list[EpisodeId]) -> None:
        """Refuse the sessions these attempts and their earlier ones already have, from now on.

        It runs as the attempts are marked, before anything awaits,
        so a retire that is cut short or arrives after its deadline still leaves those sessions refused,
        held up in readiness, and left out of exports until a retried retire frees them.
        """
        for episode_id in episode_ids:
            self._mark_sessions(episode_id)

    def _mark_sessions(self, episode_id: EpisodeId) -> list[str]:
        """Refuse the tracked sessions of ``episode_id`` and its earlier attempts until forget; return them."""
        retired = [
            session_id
            for session_id in self._rollout_sessions.get(episode_id.rollout_id, ())
            if self._sessions[session_id].attempt <= episode_id.attempt
        ]
        if retired:
            # Only rollouts with sessions get an entry: a restore marks every rollout it continues.
            self._retired_sessions.setdefault(episode_id.rollout_id, set()).update(retired)
            self._retired_session_rollouts.update(dict.fromkeys(retired, episode_id.rollout_id))
        return retired

    async def retire(self, episode_id: EpisodeId) -> None:
        """Refuse the attempts' sessions, wait for their requests in flight, then release them."""
        # The sessions known now are refused before the first await, whether or not this retire finishes.
        self._mark_sessions(episode_id)
        # A seed admitted before the retire marked its attempt registers its session when it finishes; wait for it,
        # so that session is freed here and not left behind.
        while any(
            seeding.rollout_id == episode_id.rollout_id and seeding.attempt <= episode_id.attempt
            for seeding in self._seeding
        ):
            await self.wait_changed(1.0)
        # Again, for the sessions those seeds created.
        retired = self._mark_sessions(episode_id)
        while any(self._session_requests[session_id] for session_id in retired):
            await self.wait_changed(1.0)
        for session_id in retired:
            # The session stays tracked until its state is freed, so a retire cut short leaves it for a retry.
            # Every mode: a late request of a retired session is refused, so nothing else would free it.
            await self.hooks.retire_session_state(session_id)
            self._freed(session_id)

    def _freed(self, session_id: str) -> None:
        """Stop tracking a session whose state was freed outside a request, which no route ends.

        A retired session stays refused until the controller forgets its rollout.
        With several workers its routing claim lasts until then too,
        so a late request that carries only its cookie still reaches this worker, which refuses it.
        """
        self.ended(session_id)
        if self.on_session_freed is not None:
            self.on_session_freed(session_id, until_forget=session_id in self._retired_session_rollouts)

    def is_open(self, session_id: str) -> bool:
        """Whether ``session_id`` is a live session of an attempt that has not been retired.

        A session is open from the seed that starts it, or the restore that installs it,
        until a ``/close_session`` or ``/verify`` ends it or a retire marks it.
        A replayable ``/verify`` whose caller closes the session afterwards leaves it open until that close.
        MCP exposure uses it while checkpointing is on to revoke a session's token once the session is closed.
        """
        return session_id in self._sessions and session_id not in self._retired_session_rollouts

    @staticmethod
    def record_identity(record: CheckpointRecord) -> Optional[str]:
        return record.session_id

    def unclaimed_sessions(self) -> list[str]:
        """Restored sessions no request has used yet."""
        return sorted(self._unclaimed)

    async def delete_unclaimed_session(self, session_id: str) -> None:
        """Free a restored session no request has used, without refusing anything later."""
        if session_id in self._unclaimed:
            if self.mode == "exported":
                await self.hooks.retire_session_state(session_id)
            self._freed(session_id)

    async def delete_restored(self, episode_id: EpisodeId) -> None:
        """Free the restored sessions of ``episode_id`` that no request has used, without refusing anything later:
        the controller may still start the rollout over as that attempt."""
        # Through the rollout index, so deleting many restored episodes does not scan every restored session for each.
        for session_id in [
            key
            for key in self._rollout_sessions.get(episode_id.rollout_id, ())
            if key in self._unclaimed and self._sessions[key] == episode_id
        ]:
            if self.mode == "exported":
                await self.hooks.retire_session_state(session_id)
            self._freed(session_id)

    def check_forget(self, rollout_ids: list[str]) -> None:
        # A seed of a retired attempt still in flight would bind its session as live once the mark is gone.
        seeding = sorted(
            episode_id.capture_key
            for episode_id in self._seeding
            if episode_id.rollout_id in rollout_ids and self._is_retired(episode_id)
        )
        if seeding:
            raise RetireIncompleteError(f"seeds of retired attempts {seeding[:10]} are still in flight")
        # Every tracked session of a retired attempt, marked or not,
        # so a mark a retire never reached cannot leave a session live once the attempt's mark is gone.
        unfreed = sorted(
            session_id
            for rollout_id in rollout_ids
            for session_id in self._rollout_sessions.get(rollout_id, ())
            if self._is_retired(self._sessions[session_id])
        )
        if unfreed:
            # Forgetting would make these sessions live again: exportable, restorable and admitted.
            # Nothing changes until a repeated retire has freed them.
            raise RetireIncompleteError(f"retire has not finished freeing resources sessions {unfreed[:10]}")

    async def forget(self, rollout_ids: list[str]) -> None:
        self.check_forget(rollout_ids)
        await super().forget(rollout_ids)
        for rollout_id in rollout_ids:
            for session_id in self._retired_sessions.pop(rollout_id, ()):
                self._retired_session_rollouts.pop(session_id, None)
                if self.on_session_freed is not None and session_id not in self._sessions:
                    # Its refusal ends here, and so does its routing claim, one close window later.
                    self.on_session_freed(session_id)

    def request_started(self, session_id: Optional[str], *, counted: bool) -> None:
        if session_id is not None:
            self._session_requests[session_id] += 1
            if counted:
                self._session_waited[session_id] += 1
        if counted:
            self.inflight += 1

    def request_ended(self, session_id: Optional[str], *, counted: bool) -> None:
        """Synchronous, so a cancellation cannot leave a request counted; the caller notifies afterwards."""
        if session_id is not None:
            _uncount(self._session_requests, session_id)
            if counted:
                _uncount(self._session_waited, session_id)
        if counted:
            self.inflight -= 1

    async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        if self.mode == "restart_only":
            return []
        # Only sessions within the commit's scope:
        # another participant may report one of these episodes as a restart,
        # and unclaimed restored sessions outside it are retired by the commit.
        scope = None if episode_ids is None else set(episode_ids)
        # Nor the sessions of episodes an earlier prepare stage reported as restarts, which start over after a crash.
        carried = self._carried_sessions() if self.carried_restarts else set()
        sessions = [
            (session_id, episode_id)
            for session_id, episode_id in self._checkpointed_sessions()
            if (scope is None or episode_id in scope) and session_id not in carried
        ]
        if self.mode == "stateless":
            # No state, only which sessions continue, so a restore places them where their requests are routed.
            return [
                ResourcesSessionRecord(session_id=session_id, episode_id=episode_id, state=None)
                for session_id, episode_id in sessions
            ]
        states = await self.hooks.export_session_states([session_id for session_id, _ in sessions])
        records = []
        for session_id, episode_id in sessions:
            if session_id not in states:
                # The server already dropped this session, for example when a failed verification cleaned it up.
                # There is nothing to continue, so stop tracking it rather than fail the commit.
                LOGGER.warning("resources session %s of %s is gone; not exported", session_id, episode_id.capture_key)
                self._freed(session_id)
                continue
            state = states[session_id]
            records.append(ResourcesSessionRecord(session_id=session_id, episode_id=episode_id, state=state))
        return records

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        raise NotImplementedError("resources sessions export through export(), which awaits the server's hooks")

    def _checkpointed_sessions(self) -> list[tuple[str, EpisodeId]]:
        return [
            (key, episode_id)
            for key, episode_id in self._sessions.items()
            if key not in self._retired_session_rollouts
        ]

    async def install(self, records: list[CheckpointRecord], scope: list[EpisodeId]) -> None:
        if self._sessions or self.inflight:
            raise ControlError("resources restore requires a process that has not served sessions")
        if records and self.mode == "restart_only":
            raise ControlError("a restart_only resources server cannot restore session state")
        if records and self.mode == "exported":
            await self.hooks.restore_session_states({record.session_id: record.state for record in records})
        for record in records:
            self._bind(record.session_id, next_attempt(record.episode_id))
            self._unclaimed.add(record.session_id)
            self._restored_sessions.add(record.session_id)

    async def unclaimed(self) -> list[EpisodeId]:
        return sorted(
            {self._sessions[session_id] for session_id in self._unclaimed},
            key=lambda episode_id: episode_id.capture_key,
        )

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        raise NotImplementedError("resources sessions restore through install(), which awaits the server's hooks")

    def status_extra(self) -> dict[str, Any]:
        return {
            "verify": self.verify_step_type,
            "retired_sessions": len(self._retired_session_rollouts),
        }


class ResourcesCheckpointMiddleware:
    """Pure ASGI middleware inside the session middleware, so ``scope["session"]`` is populated.

    A tool call over MCP is one POST to the MCP endpoint.
    It names its session with a signed token instead of the cookie,
    and the token also names the attempt whose seed minted it.
    The middleware admits, refuses, and counts that request as a whole, so direct MCP dispatch inside it loses nothing.
    While a checkpoint is open, an MCP call waits for resume instead of being refused:
    MCP clients are third-party agent harnesses that would show a refusal to the model as a tool error.
    A waiting call is not in flight, so it never holds up prepare.
    Once admission reopens, a call whose token names a retired or replaced attempt is refused before its tool runs.

    A ``/close_session`` waits the same way.
    It comes from an episode ending while the checkpoint is open, for example one a controller retired,
    and a refusal would lose the release: its caller closes once.

    ``session_starts`` are the routes that start a session.
    Each waits out a checkpoint in every mode, counts as a seed, and needs a rollout prefix.
    ``/close_session`` and ``/verify`` need a rollout prefix too, so the participant can check who sends them.
    """

    #: Read by MCP auto-exposure: this middleware does its job on the MCP request as a whole.
    applies_to_mcp_requests = True

    def __init__(
        self,
        app: Any,
        participant: ResourcesParticipant,
        mcp_token_claims: Optional[Callable[[dict[str, Any]], Optional[MCPTokenClaims]]] = None,
        mcp_path: str = "/mcp",
        session_starts: frozenset[str] = frozenset({_SESSION_START}),
    ) -> None:
        self.app = app
        self.participant = participant
        self.mcp_token_claims = mcp_token_claims
        self.mcp_path = mcp_path
        self.session_starts = session_starts

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        starts_session = path in self.session_starts
        if (
            scope.get("type") != "http"
            or (scope.get("method") in ("GET", "HEAD") and not starts_session)
            or path.startswith("/ng-control/")
        ):
            await self.app(scope, receive, send)
            return
        session_id = (scope.get("session") or {}).get(SESSION_ID_KEY)
        is_mcp = self.mcp_token_claims is not None and (path == self.mcp_path or path.startswith(self.mcp_path + "/"))
        # A token that does not verify is left to the MCP endpoint, which rejects it.
        mcp_token = self.mcp_token_claims(scope) if is_mcp else None
        if mcp_token is not None:
            session_id = mcp_token.session_id
        # Inside an episode every resources call carries the rollout prefix, seeds from an Environment Server included.
        seed = current_episode_id() if starts_session else None
        # A session start without the prefix creates a session that nothing tracks.
        # No checkpoint would export it, and no retire would free it.
        # A session end without it could come from any attempt, including one a restore replaced.
        if (starts_session or path in _SESSION_ENDS) and current_episode_id() is None:
            error = RolloutIdRequiredError(
                f"with checkpointing on, {path} needs the /ng-rollout/<id> prefix of the rollout the session belongs to"
            )
            await error.response()(scope, receive, send)
            return
        await self._serve(scope, receive, send, path, session_id, seed, mcp_token)

    async def _serve(
        self,
        scope: dict[str, Any],
        receive: Any,
        send: Any,
        path: str,
        session_id: Optional[str],
        seed: Optional[EpisodeId],
        mcp_token: Optional[MCPTokenClaims] = None,
    ) -> None:
        # A request that arrives while a checkpoint is open waits for resume.
        # A seed comes from a replay step the checkpoint does not wait for,
        # so its episode's boundary stays before the seed until resume; a close comes from an episode ending,
        # and its caller closes once; other requests come from a restart,
        # which keeps running through a checkpoint, often a third-party harness that would show a refusal to the model.
        if path not in self.participant.replayable_paths:
            while not self.participant.accepting:
                await self.participant.wait_open()
        if seed is not None:
            # A restart_only server stays open during a checkpoint, but a new session would add a restart to it.
            await self.participant.wait_open()
        # A seed counts from its admission, with no await in between,
        # so a retire that marks its attempt after this waits for the seed to finish;
        # a seed still waiting for resume is refused by the mark once it is admitted.
        if seed is not None:
            self.participant.seed_started(seed)
        try:
            self.participant.admit(session_id, path, starts_session=seed is not None, mcp_token=mcp_token)
        except ControlError as error:
            if seed is not None:
                self.participant.seed_ended(seed)
            await error.response()(scope, receive, send)
            return

        started = False
        close_follows = any(
            key.lower() == CHECKPOINT_CLOSE_FOLLOWS_HEADER.encode() and value == b"1"
            for key, value in scope.get("headers", ())
        )

        def apply_session_change(status: int) -> None:
            # Applied once the handler has changed the session, before the reply is sent:
            # a client that disconnects while it is sent cancels the request, which must not lose the change.
            # Registered while the request still counts as in flight, so a prepare that waits for it sees the session.
            if session_id is None or not 200 <= status < 300:
                return
            if seed is not None:
                # A seed handler may replace the session's ID,
                # for example with the caller's resources session ID; later requests carry the new one,
                # so that is the session to track.
                self.participant.seeded((scope.get("session") or {}).get(SESSION_ID_KEY) or session_id, seed)
            elif path in _SESSION_ENDS:
                self.participant.session_ended_by(session_id, path, close_follows=close_follows)

        async def capture_status(message: dict[str, Any]) -> None:
            nonlocal started
            if message["type"] == "http.response.start" and not started:
                started = True
                if seed is not None:
                    # Tell the caller, before it ever verifies, whether this server's /verify may be replayed,
                    # and whether its episode must start over after a crash because this server cannot capture it.
                    headers = [(CHECKPOINT_VERIFY_HEADER.encode(), self.participant.verify_step_type.encode())]
                    if self.participant.mode == "restart_only":
                        headers.append((CHECKPOINT_RESTART_HEADER.encode(), b"1"))
                    message = {**message, "headers": [*message.get("headers", []), *headers]}
                apply_session_change(message["status"])
            await send(message)

        counted = path not in self.participant.replayable_paths
        self.participant.request_started(session_id, counted=counted)
        try:
            await self.app(scope, receive, capture_status)
        finally:
            self.participant.request_ended(session_id, counted=counted)
            if seed is not None:
                self.participant.seed_ended(seed)
            await self.participant.notify()


def _uncount(counter: Counter[str], key: str) -> None:
    counter[key] -= 1
    if not counter[key]:
        del counter[key]
