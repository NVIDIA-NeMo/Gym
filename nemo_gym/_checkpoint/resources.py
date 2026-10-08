# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources server participant: quiesce tool traffic and export session state.

Resources servers keep per-rollout state under the cookie session ID. The participant learns each
session's lifecycle from the requests that already define it: a successful ``/seed_session`` binds the
session to the episode named by the request's rollout prefix, and a successful ``/close_session`` or
``/verify`` ends it. No route has to be classified as a mutation or read. A protocol whose lifecycle
no route reveals, such as Gymnasium's ``/reset`` and terminal ``/step``, reports it explicitly through
the resources server's ``checkpoint_session_started`` and ``checkpoint_session_ended``.

A server declares how its sessions checkpoint:

- ``stateless``: sessions hold nothing a continuation needs. A checkpoint still records which sessions are live,
  without state, so a restore keeps routing their cookies and frees them like any other session.
- ``exported``: the server exports and restores each session's state through ``ResourcesSessionHooks``.
- ``restart_only``: the default. Nothing of the server is in a checkpoint: its requests never hold up
  prepare, and every live session is reported as a restart. Its seed reply says so, so the episode that
  seeded it is a restart too: the controller leaves it out of the commit and starts it over from its input
  after a crash. Only its seeds wait out a checkpoint, so the restarts it reports never change during one;
  every other request is admitted. This fails closed for servers nobody has audited.

Prepare closes admission for data routes and is ready once no request is in flight. A request that arrives
while a checkpoint is open waits for resume instead of being refused: a continued episode parks before it
calls again, so only a restart, which keeps running through the checkpoint, sends one. A seed already in
flight holds up prepare, so a session always matches the environment server's boundary.
"""

import asyncio
import logging
from collections import Counter
from collections.abc import Callable
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
from nemo_gym._checkpoint.errors import ControlError, StaleAttemptError
from nemo_gym._checkpoint.steps import CHECKPOINT_RESTART_HEADER, CHECKPOINT_VERIFY_HEADER, StepMode
from nemo_gym.episode_types import EpisodeId
from nemo_gym.rollout_correlation import current_episode_id
from nemo_gym.server_utils import SESSION_ID_KEY
from nemo_gym.session_routing import SESSION_OWNER_KEY


LOGGER = logging.getLogger(__name__)

ResourcesCheckpointMode = Literal["stateless", "exported", "restart_only"]
_SESSION_START = "/seed_session"
_SESSION_CLOSE = "/close_session"
_SESSION_ENDS = (_SESSION_CLOSE, "/verify")


class ResourcesAdmissionClosedError(ControlError):
    code = "admission_closed"


class ResourcesSessionRecord(CheckpointRecord):
    session_id: str
    state: JsonPayload
    # With several workers, the worker ID in the session's cookie and MCP token (see nemo_gym.session_routing).
    owner: Optional[str] = None


class ResourcesSessionHooks(Protocol):
    """Session state owned by an ``exported`` resources server.

    The hooks are asynchronous and exporting takes every session at once, so a server whose state lives
    outside the process, such as a sandbox per session, can checkpoint all of it concurrently.

    A sandbox per session plugs in here:

    - ``export_session_states`` runs at commit. Prepare stops agents before resources servers, so no episode
      step is using a sandbox by then. Return what a restore needs to rebuild each sandbox as of this
      checkpoint, such as a snapshot id. A descriptor of the live sandbox is not enough on its own: the
      sandbox keeps changing after the checkpoint.
    - ``restore_session_states`` runs in a fresh process after a crash. Rebuild every session's sandbox or
      raise; the controller then restarts those rollouts from their inputs. The crashed process's sandboxes
      are still running and can be stopped here.
    - ``retire_session_state`` runs when an attempt is discarded. Stop its sandbox. It runs again for the same
      session if a retire was cut short, so it must tolerate state it already freed.

    Whatever a checkpoint keeps outside Gym, such as snapshots, is the server's to delete once no checkpoint
    the controller may restore refers to it.
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
        self, hooks: ResourcesSessionHooks, mode: ResourcesCheckpointMode, *, verify_mode: StepMode = "wait"
    ) -> None:
        super().__init__()
        self.hooks = hooks
        self.mode = mode
        self.verify_mode = verify_mode
        # Requests the checkpoint does not wait for: a verification the server declared replayable does not
        # change checkpointed state. A seed is waited for, so the session matches the environment's boundary.
        self.replayable_paths = {"/verify"} if verify_mode == "replay" else set()
        self.accepting = True
        self._open = asyncio.Event()
        self._open.set()
        self.inflight = 0
        self._sessions: dict[str, EpisodeId] = {}
        # Sessions of retired attempts, by rollout, refused until the controller forgets the rollout: an MCP tool
        # call names only its session, which is gone once retire freed it. And requests in flight per session.
        self._retired_sessions: dict[str, set[str]] = {}
        self._retired_session_rollouts: dict[str, str] = {}
        self._session_requests: Counter[str] = Counter()
        # The routing owner of each session, with several workers; ``owner`` is this worker's.
        self.owner: Optional[str] = None
        self._owners: dict[str, Optional[str]] = {}
        # Restored sessions no request has used yet; a commit that no longer continues their episode retires them.
        self._restored_pending: set[str] = set()

    def admit(self, session_id: Optional[str], path: str) -> None:
        # A close is never refused: it is how the server frees a session's state.
        if path != _SESSION_CLOSE:
            if session_id in self._retired_session_rollouts:
                raise StaleAttemptError(f"resources session {session_id!r} belongs to a retired attempt")
            episode_id = current_episode_id()
            if episode_id is not None:
                self.retired.check(episode_id)
        # A replay-safe request may still arrive from an episode step the checkpoint did not wait for.
        # Refusing it would fail that episode; its episode re-runs it after a crash anyway.
        if not self.accepting and path not in self.replayable_paths:
            raise ResourcesAdmissionClosedError("resources admission is closed for a checkpoint")
        self._restored_pending.discard(session_id)

    async def wait_open(self) -> None:
        """Return once admission is open: after resume, or when a lease expires."""
        await self._open.wait()

    def seeded(self, session_id: str, episode_id: EpisodeId, owner: Optional[str] = None) -> None:
        self._sessions[session_id] = episode_id
        # A restored session keeps the owner its cookie names.
        self._owners.setdefault(session_id, owner or self.owner)

    def ended(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._owners.pop(session_id, None)
        self._restored_pending.discard(session_id)

    async def close_admission(self, request: CheckpointRequest) -> None:
        self._open.clear()
        if self.mode == "restart_only":
            # Nothing of this server is in the checkpoint, so its sessions' rollouts keep running through it; only
            # its seeds wait, so the restarts it reports do not change during the checkpoint.
            return
        self.accepting = False

    async def open_admission(self) -> None:
        self.accepting = True
        self._open.set()

    def readiness(self) -> PrepareReport:
        restarts = (
            sorted({episode_id.capture_key for _, episode_id in self._checkpointed_sessions()})
            if self.mode == "restart_only"
            else []
        )
        # A session whose retire was cut short is still being freed: a checkpoint waits for a retried retire.
        blockers = sorted(
            {
                episode_id.capture_key
                for key, episode_id in self._sessions.items()
                if key in self._retired_session_rollouts
            }
        )
        # A restart_only server's requests are never captured, so none of them holds up a checkpoint.
        inflight = 0 if self.mode == "restart_only" else self.inflight
        return PrepareReport(
            ready=inflight == 0 and not blockers,
            blockers=blockers,
            counts={"inflight": self.inflight, "sessions": len(self._sessions)},
            restarts=restarts,
        )

    async def retire(self, episode_id: EpisodeId) -> None:
        """Refuse the attempts' sessions, wait for their requests in flight, then release them."""
        retired = [
            session_id
            for session_id, bound in self._sessions.items()
            if bound.rollout_id == episode_id.rollout_id and bound.attempt <= episode_id.attempt
        ]
        # Refused from now on, whether or not this retire finishes, until the controller forgets the rollout.
        self._retired_sessions.setdefault(episode_id.rollout_id, set()).update(retired)
        self._retired_session_rollouts.update(dict.fromkeys(retired, episode_id.rollout_id))
        while any(self._session_requests[session_id] for session_id in retired):
            await self.wait_changed(1.0)
        for session_id in retired:
            # The session stays tracked until its state is freed, so a retire cut short leaves it for a retry.
            if self.mode == "exported":
                await self.hooks.retire_session_state(session_id)
            self._sessions.pop(session_id, None)
            self._restored_pending.discard(session_id)
            self._owners.pop(session_id, None)

    async def forget(self, rollout_ids: list[str]) -> None:
        await super().forget(rollout_ids)
        for rollout_id in rollout_ids:
            for session_id in self._retired_sessions.pop(rollout_id, ()):
                self._retired_session_rollouts.pop(session_id, None)

    def request_started(self, session_id: Optional[str]) -> None:
        if session_id is not None:
            self._session_requests[session_id] += 1

    async def request_ended(self, session_id: Optional[str]) -> None:
        if session_id is not None:
            self._session_requests[session_id] -= 1
            if not self._session_requests[session_id]:
                del self._session_requests[session_id]
            await self.notify()

    async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        if self.mode == "restart_only":
            return []
        # Unclaimed restored sessions outside the commit's scope are not exported: the commit retires them.
        scope = None if episode_ids is None else set(episode_ids)
        sessions = [
            (session_id, episode_id)
            for session_id, episode_id in self._checkpointed_sessions()
            if scope is None or session_id not in self._restored_pending or episode_id in scope
        ]
        if self.mode == "stateless":
            # No state, only which sessions continue and the owner their cookies name, so a restore keeps them routed.
            return [
                ResourcesSessionRecord(
                    session_id=session_id, episode_id=episode_id, state=None, owner=self._owners.get(session_id)
                )
                for session_id, episode_id in sessions
            ]
        states = await self.hooks.export_session_states([session_id for session_id, _ in sessions])
        records = []
        for session_id, episode_id in sessions:
            if session_id not in states:
                # The server already dropped this session, for example when a failed verification cleaned
                # it up. There is nothing to continue, so stop tracking it rather than fail the commit.
                LOGGER.warning("resources session %s of %s is gone; not exported", session_id, episode_id.capture_key)
                self._sessions.pop(session_id, None)
                self._owners.pop(session_id, None)
                continue
            records.append(
                ResourcesSessionRecord(
                    session_id=session_id,
                    episode_id=episode_id,
                    state=states[session_id],
                    owner=self._owners.get(session_id),
                )
            )
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
            self._sessions[record.session_id] = next_attempt(record.episode_id)
            self._owners[record.session_id] = record.owner
            self._restored_pending.add(record.session_id)

    async def restored_pending(self) -> list[EpisodeId]:
        return sorted(
            {self._sessions[session_id] for session_id in self._restored_pending},
            key=lambda episode_id: episode_id.capture_key,
        )

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        raise NotImplementedError("resources sessions restore through install(), which awaits the server's hooks")

    def status_extra(self) -> dict[str, Any]:
        return {"mode": self.mode, "verify": self.verify_mode, "retired_sessions": len(self._retired_session_rollouts)}


class ResourcesCheckpointMiddleware:
    """Pure ASGI middleware inside the session middleware, so ``scope["session"]`` is populated.

    A tool call over MCP is one POST to the MCP endpoint that names its session with a signed token instead of
    the cookie. The middleware admits, fences, and counts that request as a whole, so direct MCP dispatch inside
    it loses nothing. While a checkpoint is open, an MCP call waits for resume instead of being refused: MCP
    clients are third-party agent harnesses that would show a refusal to the model as a tool error. A waiting
    call is not in flight, so it never holds up prepare.

    A ``/close_session`` waits the same way. It comes from an episode ending while the checkpoint is open, for
    example one a controller retired, and a refusal would lose the release: its caller closes once.
    """

    #: Read by MCP auto-exposure: this middleware does its job on the MCP request as a whole.
    applies_to_mcp_requests = True

    def __init__(
        self,
        app: Any,
        participant: ResourcesParticipant,
        mcp_session_id: Optional[Callable[[dict[str, Any]], Optional[str]]] = None,
        mcp_path: str = "/mcp",
    ) -> None:
        self.app = app
        self.participant = participant
        self.mcp_session_id = mcp_session_id
        self.mcp_path = mcp_path

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        path = scope.get("path", "")
        if scope.get("type") != "http" or scope.get("method") in ("GET", "HEAD") or path.startswith("/ng-control/"):
            await self.app(scope, receive, send)
            return
        session = scope.get("session") or {}
        session_id = session.get(SESSION_ID_KEY)
        is_mcp = self.mcp_session_id is not None and (path == self.mcp_path or path.startswith(self.mcp_path + "/"))
        if is_mcp:
            session_id = self.mcp_session_id(scope) or session_id
        # A request that arrives while a checkpoint is open waits for resume. A seed comes from a replay step the
        # checkpoint does not wait for, so its episode's boundary stays before the seed until resume; a close comes
        # from an episode ending, and its caller closes once; other requests come from a restart, which keeps
        # running through a checkpoint, often a third-party harness that would show a refusal to the model.
        if path not in self.participant.replayable_paths:
            while not self.participant.accepting:
                await self.participant.wait_open()
        if path == _SESSION_START:
            # A restart_only server stays open during a checkpoint, but a new session would add a restart to it.
            await self.participant.wait_open()
        try:
            self.participant.admit(session_id, path)
        except ControlError as error:
            await error.response()(scope, receive, send)
            return

        status: list[int] = []

        async def capture_status(message: dict[str, Any]) -> None:
            if message["type"] == "http.response.start":
                status.append(message["status"])
                if path == _SESSION_START:
                    # Tell the caller, before it ever verifies, whether this server's /verify may be replayed, and
                    # whether its episode must start over after a crash because this server cannot capture it.
                    headers = [(CHECKPOINT_VERIFY_HEADER.encode(), self.participant.verify_mode.encode())]
                    if self.participant.mode == "restart_only":
                        headers.append((CHECKPOINT_RESTART_HEADER.encode(), b"1"))
                    message = {**message, "headers": [*message.get("headers", []), *headers]}
            await send(message)

        counted = path not in self.participant.replayable_paths
        if counted:
            self.participant.inflight += 1
        self.participant.request_started(session_id)
        try:
            await self.app(scope, receive, capture_status)
            # Registered while the request still counts as in flight, so a prepare that waits for it sees the session.
            if session_id is not None and status and 200 <= status[0] < 300:
                if path == _SESSION_START:
                    # Inside an episode every resources call carries the rollout prefix, native seeds included.
                    episode_id = current_episode_id()
                    if episode_id is not None:
                        self.participant.seeded(session_id, episode_id, session.get(SESSION_OWNER_KEY))
                elif path in _SESSION_ENDS:
                    self.participant.ended(session_id)
        finally:
            await self.participant.request_ended(session_id)
            if counted:
                self.participant.inflight -= 1
                await self.participant.notify()
