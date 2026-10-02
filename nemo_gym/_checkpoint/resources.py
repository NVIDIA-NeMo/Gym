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

- ``stateless``: sessions hold nothing a continuation needs.
- ``exported``: the server exports and restores each session's state through ``ResourcesSessionHooks``.
- ``restart_only``: the default. A live session blocks prepare until the controller retires its
  rollout, which then restarts from its input. This fails closed for servers nobody has audited.

Prepare closes admission for data routes and is ready once no request is in flight and no
``restart_only`` session is live.
"""

import asyncio
import logging
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
from nemo_gym._checkpoint.steps import CHECKPOINT_VERIFY_HEADER, StepMode
from nemo_gym.episode_types import EpisodeId
from nemo_gym.rollout_correlation import current_episode_id
from nemo_gym.server_utils import SESSION_ID_KEY


LOGGER = logging.getLogger(__name__)

ResourcesCheckpointMode = Literal["stateless", "exported", "restart_only"]
_SESSION_START = "/seed_session"
_SESSION_CLOSE = "/close_session"
_SESSION_ENDS = (_SESSION_CLOSE, "/verify")


class ResourcesAdmissionClosedError(ControlError):
    code = "resources_admission_closed"


class ResourcesSessionRecord(CheckpointRecord):
    session_id: str
    state: JsonPayload


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
    - ``retire_session_state`` runs when an attempt is discarded. Stop its sandbox.

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
        # Requests the checkpoint does not wait for: seeds are idempotent, and a verification the
        # server declared replayable does not change checkpointed state.
        self.replayable_paths = {_SESSION_START} | ({"/verify"} if verify_mode == "replay" else set())
        self.accepting = True
        self._open = asyncio.Event()
        self._open.set()
        self.inflight = 0
        self._sessions: dict[str, EpisodeId] = {}
        self._retired_sessions: set[str] = set()
        # Sessions seeded after admission closed. Their episodes' boundaries precede the seed, so they
        # are not part of this checkpoint: they neither block it nor are exported by it.
        self._seeded_while_closed: set[str] = set()
        # Restored sessions no request has used yet; a commit that no longer continues their episode retires them.
        self._restored_unused: set[str] = set()

    def admit(self, session_id: Optional[str], path: str) -> None:
        # Closing a retired attempt's session is never stale: it is how the server releases that session's state,
        # which a restart_only server's retire cannot do.
        if session_id in self._retired_sessions and path != _SESSION_CLOSE:
            raise StaleAttemptError(f"resources session {session_id!r} belongs to a retired attempt")
        # A replay-safe request may still arrive from an episode step the checkpoint did not wait for.
        # Refusing it would fail that episode; its episode re-runs it after a crash anyway.
        if not self.accepting and path not in self.replayable_paths:
            raise ResourcesAdmissionClosedError("resources admission is closed for a checkpoint")
        self._restored_unused.discard(session_id)

    async def wait_open(self) -> None:
        """Return once admission is open: after resume, or when a lease expires."""
        await self._open.wait()

    def seeded(self, session_id: str, episode_id: EpisodeId) -> None:
        if self.mode != "stateless":
            self._sessions[session_id] = episode_id
            if not self.accepting:
                self._seeded_while_closed.add(session_id)

    def ended(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._seeded_while_closed.discard(session_id)
        self._restored_unused.discard(session_id)

    async def close_admission(self, request: CheckpointRequest) -> None:
        self.accepting = False
        self._open.clear()

    async def open_admission(self) -> None:
        self.accepting = True
        self._open.set()
        self._seeded_while_closed.clear()

    def readiness(self) -> PrepareReport:
        blockers = (
            sorted(episode_id.capture_key for key, episode_id in self._checkpointed_sessions())
            if self.mode == "restart_only"
            else []
        )
        return PrepareReport(
            ready=self.inflight == 0 and not blockers,
            blockers=blockers,
            counts={"inflight": self.inflight, "sessions": len(self._sessions)},
        )

    async def retire(self, episode_id: EpisodeId) -> None:
        for session_id, bound in list(self._sessions.items()):
            if bound.rollout_id == episode_id.rollout_id and bound.attempt <= episode_id.attempt:
                del self._sessions[session_id]
                self._restored_unused.discard(session_id)
                self._retired_sessions.add(session_id)
                if self.mode == "exported":
                    await self.hooks.retire_session_state(session_id)

    async def export(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        if self.mode != "exported":
            return []
        sessions = self._checkpointed_sessions()
        states = await self.hooks.export_session_states([session_id for session_id, _ in sessions])
        records = []
        for session_id, episode_id in sessions:
            if session_id not in states:
                # The server already dropped this session, for example when a failed verification cleaned
                # it up. There is nothing to continue, so stop tracking it rather than fail the commit.
                LOGGER.warning("resources session %s of %s is gone; not exported", session_id, episode_id.capture_key)
                self._sessions.pop(session_id, None)
                continue
            state = states[session_id]
            records.append(ResourcesSessionRecord(session_id=session_id, episode_id=episode_id, state=state))
        return records

    def export_records(self, episode_ids: Optional[list[EpisodeId]]) -> list[CheckpointRecord]:
        raise NotImplementedError("resources sessions export through export(), which awaits the server's hooks")

    def _checkpointed_sessions(self) -> list[tuple[str, EpisodeId]]:
        return [
            (key, episode_id) for key, episode_id in self._sessions.items() if key not in self._seeded_while_closed
        ]

    async def install(self, records: list[CheckpointRecord]) -> None:
        if self._sessions or self.inflight:
            raise ControlError("resources restore requires a process that has not served sessions")
        if records and self.mode != "exported":
            raise ControlError(f"a {self.mode} resources server cannot restore session state")
        if records:
            await self.hooks.restore_session_states({record.session_id: record.state for record in records})
        for record in records:
            self._sessions[record.session_id] = next_attempt(record.episode_id)
            self._restored_unused.add(record.session_id)

    async def restored_pending(self) -> list[EpisodeId]:
        return sorted(
            {self._sessions[session_id] for session_id in self._restored_unused},
            key=lambda episode_id: episode_id.capture_key,
        )

    def restore_records(self, records: list[CheckpointRecord]) -> None:
        raise NotImplementedError("resources sessions restore through install(), which awaits the server's hooks")

    def status_extra(self) -> dict[str, Any]:
        return {"mode": self.mode, "verify": self.verify_mode}


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
        session_id = (scope.get("session") or {}).get(SESSION_ID_KEY)
        is_mcp = self.mcp_session_id is not None and (path == self.mcp_path or path.startswith(self.mcp_path + "/"))
        if is_mcp:
            session_id = self.mcp_session_id(scope) or session_id
        if is_mcp or path == _SESSION_CLOSE:
            while not self.participant.accepting:
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
                    # Tell the caller, before it ever verifies, whether this server's /verify may be replayed.
                    header = (CHECKPOINT_VERIFY_HEADER.encode(), self.participant.verify_mode.encode())
                    message = {**message, "headers": [*message.get("headers", []), header]}
            await send(message)

        counted = path not in self.participant.replayable_paths
        if counted:
            self.participant.inflight += 1
        try:
            await self.app(scope, receive, capture_status)
        finally:
            if counted:
                self.participant.inflight -= 1
                await self.participant.notify()
        if session_id is None or not status or not 200 <= status[0] < 300:
            return
        if path == _SESSION_START:
            # Inside an episode every resources call carries the rollout prefix, native seeds included.
            episode_id = current_episode_id()
            if episode_id is not None:
                self.participant.seeded(session_id, episode_id)
        elif path in _SESSION_ENDS:
            self.participant.ended(session_id)
