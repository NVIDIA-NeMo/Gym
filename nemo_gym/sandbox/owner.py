# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A resources server that owns one sandbox per session and lends it to the agent.

The SWE-bench server is the model: ``/seed_session`` starts a sandbox for the session, the agent runs its harness
in it, and ``/verify`` extracts what the agent did and ends the session. :class:`SandboxOwnerResourcesServer`
writes the parts every such server shares once:

- the provider every session's sandbox runs on, built from the ``sandbox_provider`` block the config names;
- the :class:`~nemo_gym.sandbox.checkpoint.SandboxSessionCheckpointer` that tracks the sandboxes and implements
  the partial-rollout checkpoint hooks (export, restore, retire, park, release);
- ``/sandbox_access``, which names the session's current sandbox to the agent: after a restore it is a fork of
  the checkpoint's snapshot under a new id.

A subclass starts the session's sandbox at seed with :meth:`~SandboxOwnerResourcesServer.start_session_sandbox`,
reuses :meth:`~SandboxOwnerResourcesServer.session_sandbox` when a replacement attempt re-seeds a session a
restore rebuilt, and ends the session with :meth:`~SandboxOwnerResourcesServer.end_session_sandbox` once it is
done with the sandbox.

This module is not imported from ``nemo_gym.sandbox``: it builds on ``nemo_gym.base_resources_server``, which
imports from that package.
"""

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Awaitable, Callable, ClassVar, Optional

from fastapi import HTTPException
from pydantic import JsonValue, PrivateAttr

from nemo_gym._checkpoint.resources import ResourcesCheckpointMode
from nemo_gym.base_resources_server import BaseResourcesServerConfig, SimpleResourcesServer
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox.access import SandboxAccess
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.checkpoint import SandboxCheckpointConfig, SandboxSessionCheckpointer, checkpoint_state_dir
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers import create_provider
from nemo_gym.sandbox.providers.base import SandboxProvider, SandboxSpec


LOGGER = logging.getLogger(__name__)


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
    # Where the checkpointer indexes the snapshots it takes; None when checkpointing is off.
    _state_dir: Optional[Path] = PrivateAttr(default=None)
    # The working directory of each session's sandbox, for the agent and the subclass.
    _workdirs: dict[str, str] = PrivateAttr(default_factory=dict)

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        # Fails at startup, not at the first commit, when checkpointing is on without checkpoint.state_dir.
        self._state_dir = checkpoint_state_dir(getattr(self.server_client, "global_config_dict", None))

    # -- for subclasses ----------------------------------------------------------------------------------------

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
        replacement attempt that re-seeds the session: reuse it."""
        return self._checkpointer().get(session_id)

    async def start_session_sandbox(
        self,
        session_id: str,
        spec: SandboxSpec,
        *,
        setup: Optional[Callable[[AsyncSandbox], Awaitable[None]]] = None,
        workdir: Optional[str] = None,
    ) -> AsyncSandbox:
        """Start the session's sandbox on the shared provider, run ``setup`` against it, and track it.

        A ``setup`` that raises stops the sandbox. ``workdir`` defaults to the sandbox's ``pwd``.
        """
        sandbox = AsyncSandbox(self.shared_provider(), spec, owns_provider=False)
        await sandbox.start_with_setup(spec, setup or _no_setup)
        self._checkpointer().add(session_id, sandbox, spec)
        if workdir is None:
            workdir = (await sandbox.exec("pwd")).stdout.strip()
        self._workdirs[session_id] = workdir
        return sandbox

    def require_session_sandbox(self, session_id: str) -> AsyncSandbox:
        """The session's sandbox; HTTP 409 when it has none.

        A resent ``/verify`` after the first one ended the session, or a session this process never seeded:
        grading without the agent's sandbox would score the model as if it had done nothing.
        """
        sandbox = self._checkpointer().get(session_id)
        if sandbox is None:
            raise HTTPException(
                409, f"session {session_id!r} has no sandbox: this process never seeded it, or its episode ended"
            )
        return sandbox

    async def end_session_sandbox(self, session_id: str) -> None:
        """Stop the session's sandbox and forget the session; the episode is over, so a failure is logged.

        Its snapshots stay until a release drops the checkpoints that name them.
        """
        try:
            await self._checkpointer().stop(session_id)
        except Exception:
            LOGGER.warning("Failed to stop the sandbox of session %s", session_id, exc_info=True)
        self._workdirs.pop(session_id, None)

    # -- partial-rollout checkpoints ---------------------------------------------------------------------------

    def _checkpointer(self) -> SandboxSessionCheckpointer:
        if self._sandboxes is None:
            self._sandboxes = SandboxSessionCheckpointer(
                self.shared_provider(),
                state_dir=self._state_dir,
                server_name=self.config.name,
                parallelism=self.config.sandbox_checkpoint.parallelism,
                on_stop=self.config.sandbox_checkpoint.on_stop,
            )
        return self._sandboxes

    async def export_session_states(self, session_ids: list[str], *, checkpoint_id: str) -> dict[str, JsonValue]:
        """Snapshot each session's sandbox; the state is the snapshot and the working directory."""
        sandboxes = await self._checkpointer().export(session_ids, checkpoint_id=checkpoint_id)
        return {
            session_id: {"sandbox": state, "workdir": self._workdirs.get(session_id)}
            for session_id, state in sandboxes.items()
        }

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        parsed: dict[str, dict[str, Any]] = {}
        for session_id, state in states.items():
            if not isinstance(state, dict) or not isinstance(state.get("sandbox"), dict):
                raise ValueError(f"invalid {self.config.name} session state for {session_id!r}: {state!r}")
            parsed[session_id] = state
        # Validates every sandbox state, then forks all of them from their snapshots or none.
        await self._checkpointer().restore({session_id: state["sandbox"] for session_id, state in parsed.items()})
        for session_id, state in parsed.items():
            if isinstance(state.get("workdir"), str):
                self._workdirs[session_id] = state["workdir"]

    async def retire_session_state(self, session_id: str) -> None:
        # The attempt's snapshots stay: a checkpoint the controller keeps may still restore it.
        await self._checkpointer().stop(session_id)
        self._workdirs.pop(session_id, None)

    async def park_session_states(self, session_ids: list[str]) -> None:
        if self._sandboxes is not None:
            await self._sandboxes.park(session_ids)

    async def release_external_state(self, kept_checkpoint_ids: list[str]) -> None:
        # Also after a restart that has not created a sandbox yet: the index holds what the crashed process took.
        await self._checkpointer().release(kept_checkpoint_ids)

    async def current_sandbox_access(self, session_id: str) -> SandboxAccess | None:
        """Access to the session's sandbox for the agent, which runs its harness inside it."""
        sandboxes = self._checkpointer()
        if session_id not in sandboxes:
            return None
        workdir = self._workdirs.get(session_id)
        if workdir is None:
            workdir = (await sandboxes.get(session_id).exec("pwd")).stdout.strip()
            self._workdirs[session_id] = workdir
        return await sandboxes.access(session_id, provider_config_ref=self.config.sandbox_provider, workdir=workdir)


async def _no_setup(sandbox: AsyncSandbox) -> None:
    pass
