# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A resources server that owns its sessions' sandboxes, lends one to the agent, and checkpoints them.

SWE-bench is the model: the typed ``/seed_session`` starts a sandbox for the session and sets the task up in it,
the agent runs its harness in that sandbox, and ``/verify`` captures what the agent did and grades it.
:class:`SandboxOwnerResourcesServer` writes the parts every such server shares once:

- the typed seed and close of an Environment Server session, fenced per #4679,
  with the seed reply's ``sandbox_access`` naming the session's ``main`` sandbox;
- what the task setup learned that grading needs (``session_data``), such as the commit the agent started from:
  JSON, exported and restored with the sandboxes;
- :meth:`~SandboxOwnerResourcesServer.capture_once`, which captures an artifact for grading once per session,
  so a ``/verify`` the checkpoint replays grades the same artifact (``checkpoint_verify = "replay"``);
- the partial-rollout checkpoint hooks, through ``nemo_gym._checkpoint.sandbox.SandboxSessionCheckpointer``:
  a commit snapshots every live sandbox of every session, a restore forks them all or none,
  and a release deletes the snapshots of checkpoints the controller no longer keeps;
- the lifecycle: shutdown stops every live sandbox, and a session nobody closed is released once its
  lifetime from its latest seed (the sandbox ``ttl_s``, else 24 hours) has passed.

A snapshot holds the sandbox's files, not its processes.
A grader that needs processes the agent started, rather than files the agent left, declares
``restore_policy = "before_agent_close"``: the Environment Server then starts such an episode over after a crash
once its agent closed, instead of continuing it in a fork.

With checkpointing on, the server exports its sessions only when its sandbox provider can snapshot;
otherwise it runs ``restart_only``, and its episodes start over after a crash.

This module is outside ``nemo_gym.sandbox``, which stays lifecycle-only.
"""

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, ClassVar, Literal, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, TypeAdapter, ValidationError

from nemo_gym._checkpoint.control import CheckpointMode, next_attempt
from nemo_gym._checkpoint.sandbox import (
    SandboxCheckpointConfig,
    SandboxSessionCheckpointer,
    checkpoint_state_dir,
)
from nemo_gym._checkpoint.settings import checkpoint_settings
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionConflictError,
    ResourcesSessionFence,
    SimpleResourcesServer,
)
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers import SandboxProvider, SandboxSpec, SupportsSandboxSnapshot, create_provider
from nemo_gym.server_utils import SESSION_ID_KEY


LOGGER = logging.getLogger(__name__)

# The sandbox of a session that the agent borrows.
MAIN_SANDBOX = "main"
# A session nobody closed is released this long after its latest seed when its sandbox spec sets no ``ttl_s``.
DEFAULT_SESSION_LIFETIME_S = 24 * 3600.0

# What a session's setup learned that grading needs. Kept small: every checkpoint writes it beside the snapshot.
SessionData = dict[str, JsonValue]
_SESSION_DATA = TypeAdapter(SessionData)

RestorePolicy = Literal["any_step", "before_agent_close"]


class SessionSetup(BaseModel):
    """What a session's task setup returns: the agent's working directory and the data grading needs."""

    model_config = ConfigDict(extra="forbid")

    workdir: str
    data: SessionData = Field(default_factory=dict)


class SandboxOwnerConfig(BaseResourcesServerConfig):
    # The top-level sandbox provider block every session's sandbox is created from.
    sandbox_provider: str
    sandbox_checkpoint: SandboxCheckpointConfig = Field(default_factory=SandboxCheckpointConfig)


class _OwnedSessionState(BaseModel):
    """The exported state of one session: each sandbox's checkpoint state, or ``None`` once it was released."""

    model_config = ConfigDict(extra="forbid")

    sandboxes: dict[str, Optional[dict[str, JsonValue]]]
    workdir: Optional[str] = None
    data: SessionData = Field(default_factory=dict)
    task_id: TaskId
    episode_id: EpisodeId


@dataclass
class _Session:
    """One session this worker owns."""

    episode_id: EpisodeId
    task_id: TaskId
    workdir: Optional[str] = None
    data: SessionData = field(default_factory=dict)
    # The sandboxes the session holds, by name; a released sandbox leaves the set.
    sandboxes: set[str] = field(default_factory=set)
    # When the latest seed of the session finished, on the monotonic clock, and how long it may live from then.
    seeded_at: float = field(default_factory=monotonic)
    lifetime_s: float = DEFAULT_SESSION_LIFETIME_S
    # Held by capture_once and by an export of this session, so neither sees the other half done.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _member_key(session_id: str, name: str) -> str:
    """The checkpointer's key of one sandbox of a session."""
    return session_id if name == MAIN_SANDBOX else f"{session_id}.{name}"


class SandboxOwnerResourcesServer(SimpleResourcesServer):
    """A resources server whose sessions own sandboxes; see the module docstring.

    A subclass implements its typed ``seed_session`` with :meth:`seed_task_sandbox_session`, passing a setup that
    starts the session's sandboxes with :meth:`start_session_sandbox`, and grades in ``verify``.
    """

    config: SandboxOwnerConfig
    checkpoint_mode: ClassVar[CheckpointMode] = "exported"
    # Whether a fork of the sandboxes can continue an episode at every step, or only before its agent closed.
    restore_policy: ClassVar[RestorePolicy] = "any_step"

    _provider: Optional[SandboxProvider] = PrivateAttr(default=None)
    _sandboxes: Optional[SandboxSessionCheckpointer] = PrivateAttr(default=None)
    _sessions: dict[str, _Session] = PrivateAttr(default_factory=dict)
    _expired: set[str] = PrivateAttr(default_factory=set)
    _fence: Optional[ResourcesSessionFence] = PrivateAttr(default=None)
    _mode_warning_logged: bool = PrivateAttr(default=False)

    # -- the provider, the checkpointer and the mode -----------------------------------------------------------

    def shared_provider(self) -> SandboxProvider:
        """The provider every sandbox of this server shares, built from ``sandbox_provider`` on first use.

        Use it for sandboxes that are not session state, such as a grading sandbox, with ``owns_provider=False``.
        """
        if self._provider is None:
            resolved = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
            self._provider = create_provider(resolved) if isinstance(resolved, Mapping) else resolved
        return self._provider

    def checkpoint_mode_in_effect(self) -> CheckpointMode:
        """``exported`` on a provider that can snapshot; ``restart_only`` otherwise, with one warning."""
        if self.checkpoint_mode != "exported":
            return self.checkpoint_mode
        provider = self.shared_provider()
        if isinstance(provider, SupportsSandboxSnapshot):
            return "exported"
        if not self._mode_warning_logged:
            self._mode_warning_logged = True
            LOGGER.warning(
                "%s: sandbox provider %r cannot snapshot; its episodes restart after a crash instead of continuing",
                self.config.name,
                getattr(provider, "name", type(provider).__name__),
            )
        return "restart_only"

    def _checkpointer(self) -> SandboxSessionCheckpointer:
        if self._sandboxes is None:
            state_dir: Optional[Path] = None
            if (
                checkpoint_settings(getattr(self.server_client, "global_config_dict", None)) is not None
                and self.checkpoint_mode_in_effect() == "exported"
            ):
                # Fails at startup, not at the first commit, when checkpointing is on without a state_dir.
                state_dir = checkpoint_state_dir(getattr(self.server_client, "global_config_dict", None))
            self._sandboxes = SandboxSessionCheckpointer.from_config(
                self.shared_provider(),
                self.config.sandbox_checkpoint,
                state_dir=state_dir,
                server_name=self.config.name or type(self).__name__,
            )
        return self._sandboxes

    def _session_fence(self) -> ResourcesSessionFence:
        if self._fence is None:
            self._fence = ResourcesSessionFence(self.config.session_close_retry_window_seconds)
        return self._fence

    def setup_resources_checkpoint(self, app: FastAPI) -> None:
        super().setup_resources_checkpoint(app)
        if self._checkpoint is not None:
            self._checkpoint.restore_policy = self.restore_policy
            if self._checkpoint.mode == "exported":
                self._checkpointer()

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: FastAPI) -> AsyncIterator[Any]:
            try:
                async with parent_lifespan(application) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan
        return app

    async def shutdown(self) -> None:
        """Stop every live sandbox of every session; their snapshots stay for the checkpoints that name them."""
        for session_id in list(self._sessions):
            try:
                await self._stop_sandboxes(session_id)
            except Exception:
                LOGGER.exception("Failed to stop the sandboxes of session %s at shutdown", session_id)
        self._sessions.clear()
        if self._sandboxes is not None:
            await self._sandboxes.settle()

    # -- for subclasses ----------------------------------------------------------------------------------------

    async def seed_task_sandbox_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest,
        setup: Callable[[str, ResourcesSeedSessionRequest], Awaitable[None]],
    ) -> ResourcesSeedSessionResponse:
        """Seed an Environment Server session: run ``setup`` once, and lend the session's ``main`` sandbox.

        ``setup(session_id, body)`` validates ``body.task_data`` (raise ``ResourcesSeedRejectedError`` for input no
        retry can fix) and starts the session's sandboxes with :meth:`start_session_sandbox`.
        An identical retry returns the same sandbox, and so does the replacement attempt's seed of a session a
        restore forked: it continues in the fork without setting the task up again.
        A seed that names the session with another episode or task is refused with 409, as is a seed after close.
        """
        session_id = body.resources_session_id
        fence = self._session_fence()
        async with fence.locked(session_id):
            fence.raise_if_closed(session_id)
            self._release_expired_sessions()
            session = self._sessions.get(session_id)
            if session is not None:
                if (session.episode_id, session.task_id) != (body.episode_id, body.task_id):
                    restored = self._checkpoint.restored_episode(session_id) if self._checkpoint is not None else None
                    detail = (
                        "a restore continues this session as another attempt"
                        if restored is not None
                        else "resources_session_id is already bound to another episode or task"
                    )
                    raise ResourcesSessionConflictError(detail)
                if MAIN_SANDBOX not in session.sandboxes:
                    raise ResourcesSessionConflictError(f"resources session {session_id!r} has no sandbox to lend")
                session.seeded_at = monotonic()
                response = await self._seed_response(session_id)
                request.session[SESSION_ID_KEY] = session_id
                return response
            await self.seed_sandbox_session(
                session_id, episode_id=body.episode_id, task_id=body.task_id, setup=lambda: setup(session_id, body)
            )
            try:
                response = await self._seed_response(session_id)
            except BaseException:
                # The agent cannot reach a sandbox it was never handed, so nothing of the seed is kept.
                await self._drop_session(session_id)
                raise
            request.session[SESSION_ID_KEY] = session_id
            return response

    async def seed_sandbox_session(
        self, session_id: str, *, episode_id: EpisodeId, task_id: TaskId, setup: Callable[[], Awaitable[None]]
    ) -> None:
        """Open session ``session_id`` and run ``setup``, which starts its sandboxes with ``start_session_sandbox``.

        For a seed outside the typed protocol, such as an agent's own ``/run`` that seeds with a flat task;
        the typed seed calls it too. A setup that raises, or starts no ``main`` sandbox, leaves nothing behind.
        """
        if session_id in self._sessions:
            raise ResourcesSessionConflictError(f"resources session {session_id!r} is already seeded")
        self._sessions[session_id] = _Session(episode_id=episode_id, task_id=task_id)
        try:
            await setup()
            if MAIN_SANDBOX not in self._sessions[session_id].sandboxes:
                raise RuntimeError(f"{type(self).__name__} setup started no {MAIN_SANDBOX!r} sandbox to lend")
        except BaseException:
            await self._drop_session(session_id)
            raise

    async def start_session_sandbox(
        self,
        session_id: str,
        spec: SandboxSpec,
        setup: Callable[[AsyncSandbox], Awaitable[SessionSetup]],
        *,
        name: str = MAIN_SANDBOX,
    ) -> AsyncSandbox:
        """Start one of the session's sandboxes on the shared provider and run ``setup`` against it.

        ``setup`` returns the agent's working directory and the data grading needs (the ``main`` sandbox's setup
        sets them for the session). A setup that raises stops the sandbox. The sandbox is tracked only once its
        setup succeeded, so a failed seed leaves nothing for a checkpoint to export.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"session {session_id!r} is not being seeded")
        if name in session.sandboxes:
            raise ValueError(f"session {session_id!r} already has a {name!r} sandbox")
        sandbox = AsyncSandbox(self.shared_provider(), spec, owns_provider=False)
        prepared: list[SessionSetup] = []

        async def run_setup(started: AsyncSandbox) -> None:
            prepared.append(SessionSetup.model_validate(await setup(started)))

        await sandbox.start_with_setup(spec, run_setup)
        try:
            self._checkpointer().add(_member_key(session_id, name), sandbox, spec)
        except BaseException:
            await sandbox.stop()
            raise
        session.sandboxes.add(name)
        if spec.ttl_s:
            session.lifetime_s = float(spec.ttl_s)
        if name == MAIN_SANDBOX:
            session.workdir = prepared[0].workdir
            session.data = dict(prepared[0].data)
        return sandbox

    def session_sandboxes(self, session_id: str) -> dict[str, AsyncSandbox]:
        """The session's live sandboxes by name; ``main`` is the one the agent borrows."""
        session = self._sessions.get(session_id)
        if session is None or self._sandboxes is None:
            return {}
        found = {name: self._sandboxes.get(_member_key(session_id, name)) for name in session.sandboxes}
        return {name: sandbox for name, sandbox in found.items() if sandbox is not None}

    def session_sandbox(self, session_id: str, name: str = MAIN_SANDBOX) -> Optional[AsyncSandbox]:
        """The session's sandbox ``name``, or ``None``; after a restore it is the fork, under a new id."""
        return self.session_sandboxes(session_id).get(name)

    def require_session_sandbox(self, session_id: str, name: str = MAIN_SANDBOX) -> AsyncSandbox:
        """The session's sandbox; HTTP 409 when it has none, so grading never scores a sandbox it does not hold."""
        sandbox = self.session_sandbox(session_id, name)
        if sandbox is None:
            raise ResourcesSessionConflictError(
                f"resources session {session_id!r} has no {name!r} sandbox: this worker never seeded it, "
                "or it was released"
            )
        return sandbox

    def session_data(self, session_id: str) -> SessionData:
        """What the session's setup returned and ``capture_once`` stored; ``{}`` for a session not held here."""
        session = self._sessions.get(session_id)
        return dict(session.data) if session is not None else {}

    def update_session_data(self, session_id: str, data: SessionData) -> None:
        """Merge ``data``, which must be JSON, into the session's data; it is exported with the session."""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"resources session {session_id!r} is not held here")
        session.data.update(_SESSION_DATA.validate_python(data))

    def session_workdir(self, session_id: str) -> Optional[str]:
        session = self._sessions.get(session_id)
        return session.workdir if session is not None else None

    def session_expired(self, session_id: str) -> bool:
        """Whether this worker released the session because nobody closed it within its lifetime."""
        return session_id in self._expired

    async def current_sandbox_access(self, session_id: str) -> Optional[SandboxAccess]:
        """Access to the session's ``main`` sandbox as it is now: after a restore, the fork."""
        session = self._sessions.get(session_id)
        if session is None or MAIN_SANDBOX not in session.sandboxes or session.workdir is None:
            return None
        return await self._checkpointer().access(
            _member_key(session_id, MAIN_SANDBOX),
            provider_config_ref=self.config.sandbox_provider,
            workdir=session.workdir,
        )

    async def end_session_sandbox(self, session_id: str) -> None:
        """Stop the session's sandboxes and keep its data, which grading may still need.

        A sandbox whose stop fails stays tracked, so a later close or retire stops it again.
        Its snapshots stay until a release drops the checkpoints that name them.
        """
        await self._stop_sandboxes(session_id)

    async def capture_once(
        self,
        session_id: str,
        name: str,
        capture: Callable[[], Awaitable[JsonValue]],
        *,
        release_sandbox_after: bool = False,
    ) -> JsonValue:
        """Capture an artifact for grading once per session, such as the agent's patch, and return it.

        ``capture`` runs at most once: its result is stored in the session's data under ``name``, which a
        checkpoint exports, and every later call returns the stored result, including one after a restore.
        A server that grades a captured artifact can therefore declare ``checkpoint_verify = "replay"``:
        a ``/verify`` the checkpoint replays grades the same artifact.
        With ``release_sandbox_after``, the session's sandboxes stop once the result is stored; a checkpoint
        taken after that exports the session's data alone. A capture that raises stores nothing and releases
        nothing, so a retry captures again.
        The capture holds the session's export lock: a commit never snapshots a sandbox the capture is releasing.
        """
        session = self._sessions.get(session_id)
        if session is None:
            raise ResourcesSessionConflictError(f"resources session {session_id!r} is not held here")
        async with session.lock:
            if name in session.data:
                return session.data[name]
            result = await capture()
            session.data[name] = _SESSION_DATA.validate_python({name: result})[name]
            if release_sandbox_after:
                try:
                    await self._stop_sandboxes(session_id)
                except Exception:
                    # The result is stored; a sandbox whose stop failed stays tracked for close or retire.
                    LOGGER.warning("Failed to release the sandboxes of session %s after capture", session_id)
            return session.data[name]

    # -- the typed close ---------------------------------------------------------------------------------------

    async def close_resources_session(
        self, request: Request, body: ResourcesCloseSessionRequest
    ) -> ResourcesCloseSessionResponse:
        """Close an Environment Server session: stop its sandboxes, forget it, and fence a late seed."""
        session_id = body.resources_session_id
        fence = self._session_fence()
        async with fence.locked(session_id):
            closed = fence.closed_episode(session_id)
            if closed is not None:
                if body.episode_id != closed:
                    raise ResourcesSessionConflictError("episode_id does not match the closed resources session")
            else:
                session = self._sessions.get(session_id)
                if session is not None and body.episode_id != session.episode_id:
                    raise ResourcesSessionConflictError("episode_id does not match the seeded resources session")
                await self._stop_sandboxes(session_id)
                self._sessions.pop(session_id, None)
                self._expired.discard(session_id)
                fence.mark_closed(session_id, body.episode_id)
            request.session.pop(SESSION_ID_KEY, None)
            return ResourcesCloseSessionResponse(resources_session_id=session_id)

    # -- partial-rollout checkpoint hooks ----------------------------------------------------------------------

    async def export_session_states(self, session_ids: list[str], *, checkpoint_id: str) -> dict[str, JsonValue]:
        """Snapshot every live sandbox of each session; a session past ``capture_once`` exports its data alone."""
        held = [session_id for session_id in session_ids if session_id in self._sessions]
        async with _all_locked([self._sessions[session_id].lock for session_id in held]):
            members = {
                session_id: sorted(self._sessions[session_id].sandboxes)
                for session_id in held
                if session_id in self._sessions
            }
            snapshots = await self._checkpointer().export(
                [_member_key(session_id, name) for session_id, names in members.items() for name in names],
                checkpoint_id=checkpoint_id,
            )
            states: dict[str, JsonValue] = {}
            for session_id, names in members.items():
                session = self._sessions[session_id]
                state = _OwnedSessionState(
                    sandboxes={name: snapshots.get(_member_key(session_id, name)) for name in names},
                    workdir=session.workdir,
                    data=session.data,
                    task_id=session.task_id,
                    episode_id=session.episode_id,
                )
                states[session_id] = state.model_dump(mode="json")
            return states

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        """Fork every session's sandboxes from their snapshots, all or none, then install the sessions."""
        parsed: dict[str, _OwnedSessionState] = {}
        for session_id, state in states.items():
            try:
                parsed[session_id] = _OwnedSessionState.model_validate(state)
            except ValidationError as error:
                raise ValueError(f"invalid {self.config.name} session state for {session_id!r}: {error}") from error
            if session_id in self._sessions:
                raise ValueError(f"resources session {session_id!r} is already held by this worker")
        await self._checkpointer().restore(
            {
                _member_key(session_id, name): sandbox_state
                for session_id, state in parsed.items()
                for name, sandbox_state in state.sandboxes.items()
                if sandbox_state is not None
            }
        )
        for session_id, state in parsed.items():
            self._sessions[session_id] = _Session(
                episode_id=next_attempt(state.episode_id),
                task_id=state.task_id,
                workdir=state.workdir,
                data=dict(state.data),
                sandboxes={name for name, sandbox_state in state.sandboxes.items() if sandbox_state is not None},
            )

    async def retire_session_state(self, session_id: str) -> None:
        # The attempt's snapshots stay: a checkpoint the controller keeps may still restore it.
        await self._stop_sandboxes(session_id)
        self._sessions.pop(session_id, None)

    async def release_external_state(self, kept: frozenset[str]) -> None:
        # Also in a process that has not created a sandbox yet: the index holds what a crashed process took.
        await self._checkpointer().release(kept)

    # -- internals ---------------------------------------------------------------------------------------------

    async def _drop_session(self, session_id: str) -> None:
        try:
            await self._stop_sandboxes(session_id)
        finally:
            self._sessions.pop(session_id, None)

    async def _seed_response(self, session_id: str) -> ResourcesSeedSessionResponse:
        return ResourcesSeedSessionResponse(
            resources_session_id=session_id, sandbox_access=await self.current_sandbox_access(session_id)
        )

    async def _stop_sandboxes(self, session_id: str) -> None:
        """Stop every sandbox of the session; one whose stop fails stays tracked, and the first failure is raised."""
        session = self._sessions.get(session_id)
        if session is None or self._sandboxes is None:
            return
        failures: list[BaseException] = []
        for name in sorted(session.sandboxes):
            try:
                await self._sandboxes.stop(_member_key(session_id, name))
            except Exception as error:
                failures.append(error)
                continue
            session.sandboxes.discard(name)
        if failures:
            raise failures[0]

    def _release_expired_sessions(self) -> None:
        """Release, in the background, the sessions whose lifetime from their latest seed has passed.

        An Environment Server that died before its close leaves its sessions behind; nothing else would stop them.
        """
        now = monotonic()
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if session.sandboxes and now - session.seeded_at > session.lifetime_s
        ]
        for session_id in expired:
            LOGGER.warning("Releasing resources session %s, which nobody closed within its lifetime", session_id)
            self._expired.add(session_id)
            task = asyncio.ensure_future(self._release_expired(session_id))
            task.add_done_callback(lambda done: done.cancelled() or done.exception())

    async def _release_expired(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session is None:
            return
        fence = self._session_fence()
        async with fence.locked(session_id):
            await self._stop_sandboxes(session_id)
            self._sessions.pop(session_id, None)
            fence.mark_closed(session_id, session.episode_id)


@asynccontextmanager
async def _all_locked(locks: list[asyncio.Lock]) -> AsyncIterator[None]:
    """Hold every lock, taken in one order, so two exports never deadlock."""
    taken: list[asyncio.Lock] = []
    try:
        for lock in sorted(locks, key=id):
            await lock.acquire()
            taken.append(lock)
        yield
    finally:
        for lock in reversed(taken):
            lock.release()
