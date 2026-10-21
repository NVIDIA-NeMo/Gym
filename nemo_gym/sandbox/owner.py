# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A resources server that owns one sandbox per session and lends it to the agent.

The SWE-bench server is the model: ``/seed_session`` starts a sandbox for the session, the agent runs its harness
in it, and ``/verify`` extracts what the agent did and ends the session. :class:`SandboxOwnerResourcesServer`
writes the parts every such server shares once:

- the provider every session's sandbox runs on, built from the ``sandbox_provider`` block the config names;
- the :class:`~nemo_gym.sandbox.checkpoint.SandboxSessionCheckpointer` that tracks the sandboxes and implements
  the partial-rollout checkpoint hooks (export, restore, retire, park, release);
- the per-session data a subclass's setup returns, such as the commit the agent started from, which is exported
  and restored with the sandbox, because a restore must not lose what verify needs;
- ``/sandbox_access``, which names the session's current sandbox to the agent: after a restore it is a fork of
  the checkpoint's snapshot under a new id;
- ``/close_session`` with an empty body, which ends the cookie's session, for an agent that leaves without
  a verdict.

A subclass starts the session's sandbox at seed with :meth:`~SandboxOwnerResourcesServer.start_session_sandbox`,
reuses :meth:`~SandboxOwnerResourcesServer.session_sandbox` when a replacement attempt re-seeds a session a
restore rebuilt, and ends the session with :meth:`~SandboxOwnerResourcesServer.end_session_sandbox` once it is
done with the sandbox.

With checkpointing on, the server checkpoints its sessions (``exported``) only when its provider can snapshot.
Otherwise, and for an instance that lends no sandbox (:meth:`~SandboxOwnerResourcesServer.lends_sandboxes`),
it runs ``restart_only``: its rollouts start over after a crash instead of continuing.

This module is not imported from ``nemo_gym.sandbox``: it builds on ``nemo_gym.base_resources_server``, which
imports from that package.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Awaitable, Callable, ClassVar, Optional

from fastapi import Body, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import JsonValue, PrivateAttr, TypeAdapter, ValidationError

from nemo_gym._checkpoint.resources import ResourcesCheckpointMode
from nemo_gym._checkpoint.settings import checkpoint_settings
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.checkpoint import SandboxCheckpointConfig, SandboxSessionCheckpointer, checkpoint_state_dir
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers import create_provider
from nemo_gym.sandbox.providers.base import SandboxProvider, SandboxSpec, SupportsSandboxSnapshot
from nemo_gym.server_utils import SESSION_ID_KEY


LOGGER = logging.getLogger(__name__)

# What a session's setup learned that verify needs, such as the commit the agent started from. Kept small: it is
# written into every checkpoint next to the snapshot.
SessionData = dict[str, JsonValue]
_SESSION_DATA = TypeAdapter(SessionData)


class SandboxOwnerConfig(BaseResourcesServerConfig):
    # The top-level sandbox block every session's sandbox is created from.
    sandbox_provider: str
    sandbox_checkpoint: SandboxCheckpointConfig = SandboxCheckpointConfig()


class SandboxOwnerResourcesServer(SimpleResourcesServer):
    """A resources server that owns one sandbox per session and lends it to the agent.

    Each session's sandbox lives from ``/seed_session`` until the subclass ends the session, typically in
    ``/verify`` once it has extracted what the agent did. A partial-rollout checkpoint snapshots the sandboxes and
    a restore forks them from the snapshots; see ``nemo_gym.sandbox.checkpoint``.
    """

    config: SandboxOwnerConfig
    checkpoint_mode: ClassVar[ResourcesCheckpointMode] = "exported"
    _provider: Optional[SandboxProvider] = PrivateAttr(default=None)
    _sandboxes: Optional[SandboxSessionCheckpointer] = PrivateAttr(default=None)
    # Where the checkpointer indexes the snapshots it takes; None unless the server checkpoints its sessions.
    _state_dir: Optional[Path] = PrivateAttr(default=None)
    _mode_in_effect: ResourcesCheckpointMode = PrivateAttr(default="exported")
    # The working directory of each session's sandbox, for the agent and the subclass.
    _workdirs: dict[str, str] = PrivateAttr(default_factory=dict)
    _session_data: dict[str, SessionData] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        global_config = getattr(self.server_client, "global_config_dict", None)
        if checkpoint_settings(global_config) is None:
            return
        self._mode_in_effect = self._checkpoint_mode_for_provider()
        if self._mode_in_effect != "exported":
            return
        num_workers = self.config.num_workers or 1
        if num_workers > 1:
            raise ValueError(
                f"{self.config.name}: checkpointing the sandboxes of a server with num_workers={num_workers} is not "
                "supported yet: release and park do not reach the workers, so their snapshots and sandboxes would "
                "leak. Run this server with one worker, or with checkpointing off"
            )
        # Fails at startup, not at the first commit, when checkpointing is on without checkpoint.state_dir.
        self._state_dir = checkpoint_state_dir(global_config)

    def _checkpoint_mode_for_provider(self) -> ResourcesCheckpointMode:
        if not self.lends_sandboxes():
            # It holds no session state, so there is nothing to export and no provider to check.
            return "restart_only"
        # Built now: a bad provider config fails the startup, not the first seed.
        provider = self.shared_provider()
        if not isinstance(provider, SupportsSandboxSnapshot):
            LOGGER.warning(
                "%s: sandbox provider %r cannot snapshot; its rollouts restart after a crash instead of continuing",
                self.config.name,
                getattr(provider, "name", type(provider).__name__),
            )
            return "restart_only"
        return "exported"

    def checkpoint_mode_in_effect(self) -> ResourcesCheckpointMode:
        """``exported`` when the server lends sandboxes on a provider that can snapshot; ``restart_only`` otherwise."""
        return self._mode_in_effect

    # -- for subclasses ----------------------------------------------------------------------------------------

    def lends_sandboxes(self) -> bool:
        """Whether this instance lends session sandboxes to agents; the default is ``True``.

        Return ``False`` for an instance that only grades, such as one that verifies a dataset's own patches: it
        holds no session state, so with checkpointing on it runs ``restart_only`` and needs no ``state_dir``.
        """
        return True

    def shared_provider(self) -> SandboxProvider:
        """The provider every sandbox of this server shares, built from the config on first use.

        Use it for sandboxes that are not session state, such as a grading sandbox, with ``owns_provider=False``.
        """
        if self._provider is None:
            resolved = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
            self._provider = create_provider(resolved) if isinstance(resolved, Mapping) else resolved
        return self._provider

    def session_sandbox(self, session_id: str) -> Optional[AsyncSandbox]:
        """The session's sandbox, or ``None``. At seed, a sandbox here was rebuilt by a restore for the
        replacement attempt that re-seeds the session: reuse it, and do not run the setup again."""
        return self._sandboxes.get(session_id) if self._sandboxes is not None else None

    async def start_session_sandbox(
        self,
        session_id: str,
        spec: SandboxSpec,
        *,
        setup: Optional[Callable[[AsyncSandbox], Awaitable[Optional[SessionData]]]] = None,
        workdir: Optional[str] = None,
    ) -> AsyncSandbox:
        """Start the session's sandbox on the shared provider, run ``setup`` against it, then track the sandbox and
        what ``setup`` returned (see :meth:`session_data`).

        A ``setup`` that raises, or returns something that is not a JSON object, stops the sandbox and tracks
        nothing. ``workdir`` defaults to the sandbox's ``pwd`` after the setup.
        """
        sandboxes = self._checkpointer()
        if session_id in sandboxes:
            raise ValueError(f"session {session_id!r} already has a sandbox")
        sandbox = AsyncSandbox(self.shared_provider(), spec, owns_provider=False)
        prepared: dict[str, Any] = {}

        async def run_setup(started: AsyncSandbox) -> None:
            data = await setup(started) if setup is not None else None
            prepared["data"] = _SESSION_DATA.validate_python(data or {})
            prepared["workdir"] = workdir if workdir is not None else (await started.exec("pwd")).stdout.strip()

        await sandbox.start_with_setup(spec, run_setup)
        # Tracked only once the setup succeeded: a failed seed leaves nothing for a checkpoint to export.
        sandboxes.add(session_id, sandbox, spec)
        self._workdirs[session_id] = prepared["workdir"]
        self._session_data[session_id] = prepared["data"]
        return sandbox

    def session_data(self, session_id: str) -> SessionData:
        """What the session's setup returned, ``{}`` if nothing; a restore brings it back with the sandbox."""
        return self._session_data.get(session_id, {})

    def require_session_sandbox(self, session_id: str) -> AsyncSandbox:
        """The session's sandbox; HTTP 409 when it has none.

        A resent ``/verify`` after the first one ended the session, or a session this process never seeded:
        grading without the agent's sandbox would score the model as if it had done nothing.
        """
        sandbox = self.session_sandbox(session_id)
        if sandbox is None:
            raise HTTPException(
                409, f"session {session_id!r} has no sandbox: this process never seeded it, or its episode ended"
            )
        return sandbox

    async def end_session_sandbox(self, session_id: str) -> None:
        """Stop the session's sandbox and forget the session; the episode is over, so a failure is logged.

        Its snapshots stay until a release drops the checkpoints that name them.
        """
        if self._sandboxes is not None:
            try:
                await self._sandboxes.stop(session_id)
            except Exception:
                LOGGER.warning("Failed to stop the sandbox of session %s", session_id, exc_info=True)
        self._forget(session_id)

    # -- the routes --------------------------------------------------------------------------------------------

    async def close_resources_session(
        self, request: Request, body: Annotated[Optional[dict[str, Any]], Body()] = None
    ) -> ResourcesCloseSessionResponse | dict[str, bool]:
        """With an empty body, end the cookie's session: an agent that seeded through ``/run`` and leaves
        without a verdict, such as a cancelled run, closes it this way. Succeeds when it is already closed.

        A typed body (``ResourcesCloseSessionRequest``) gets the base default.
        """
        if not body:
            session_id = request.session.get(SESSION_ID_KEY)
            if session_id is not None:
                await self.end_session_sandbox(session_id)
            return {"closed": True}
        try:
            typed = ResourcesCloseSessionRequest.model_validate(body)
        except ValidationError as error:
            raise RequestValidationError(error.errors()) from error
        return await super().close_resources_session(typed)

    async def current_sandbox_access(self, session_id: str) -> SandboxAccess | None:
        """Access to the session's sandbox for the agent, which runs its harness inside it."""
        sandbox = self.session_sandbox(session_id)
        if sandbox is None:
            return None
        workdir = self._workdirs.get(session_id)
        if workdir is None:
            workdir = (await sandbox.exec("pwd")).stdout.strip()
            self._workdirs[session_id] = workdir
        return await self._checkpointer().access(
            session_id, provider_config_ref=self.config.sandbox_provider, workdir=workdir
        )

    # -- partial-rollout checkpoints ---------------------------------------------------------------------------

    def _checkpointer(self) -> SandboxSessionCheckpointer:
        if self._sandboxes is None:
            self._sandboxes = SandboxSessionCheckpointer(
                self.shared_provider(),
                state_dir=self._state_dir,
                server_name=self.config.name,
                parallelism=self.config.sandbox_checkpoint.parallelism,
                on_stop=self.config.sandbox_checkpoint.on_stop,
                snapshot_attempts=self.config.sandbox_checkpoint.snapshot_attempts,
                snapshot_retry_delay_s=self.config.sandbox_checkpoint.snapshot_retry_delay_s,
            )
        return self._sandboxes

    def _forget(self, session_id: str) -> None:
        self._workdirs.pop(session_id, None)
        self._session_data.pop(session_id, None)

    async def export_session_states(self, session_ids: list[str], *, checkpoint_id: str) -> dict[str, JsonValue]:
        """Snapshot each session's sandbox; the state is the snapshot, the working directory, and the session's
        data when its setup returned any."""
        sandboxes = await self._checkpointer().export(session_ids, checkpoint_id=checkpoint_id)
        states: dict[str, JsonValue] = {}
        for session_id, sandbox_state in sandboxes.items():
            state: dict[str, JsonValue] = {"sandbox": sandbox_state, "workdir": self._workdirs.get(session_id)}
            data = self._session_data.get(session_id)
            if data:
                state["data"] = data
            states[session_id] = state
        return states

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        parsed: dict[str, tuple[dict[str, Any], Optional[str], SessionData]] = {}
        for session_id, state in states.items():
            if not isinstance(state, dict) or not isinstance(state.get("sandbox"), dict):
                raise ValueError(f"invalid {self.config.name} session state for {session_id!r}: {state!r}")
            workdir = state.get("workdir")
            if workdir is not None and not isinstance(workdir, str):
                raise ValueError(f"invalid workdir in the {self.config.name} session state of {session_id!r}")
            try:
                data = _SESSION_DATA.validate_python(state.get("data", {}))
            except ValidationError as error:
                raise ValueError(
                    f"invalid data in the {self.config.name} session state of {session_id!r}: {error}"
                ) from error
            parsed[session_id] = (state["sandbox"], workdir, data)
        # Validates every sandbox state, then forks all of them from their snapshots or none.
        await self._checkpointer().restore({session_id: sandbox for session_id, (sandbox, _, _) in parsed.items()})
        for session_id, (_, workdir, data) in parsed.items():
            if workdir is not None:
                self._workdirs[session_id] = workdir
            self._session_data[session_id] = data

    async def retire_session_state(self, session_id: str) -> None:
        # The attempt's snapshots stay: a checkpoint the controller keeps may still restore it.
        if self._sandboxes is not None:
            await self._sandboxes.stop(session_id)
        self._forget(session_id)

    async def park_session_states(self, session_ids: list[str]) -> None:
        if self._sandboxes is None:
            return
        await self._sandboxes.park(session_ids)
        for session_id in session_ids:
            if session_id not in self._sandboxes:
                self._forget(session_id)

    async def release_external_state(self, kept_checkpoint_ids: list[str]) -> None:
        # Also after a restart that has not created a sandbox yet: the index holds what the crashed process took.
        await self._checkpointer().release(kept_checkpoint_ids)
