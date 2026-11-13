# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial-rollout checkpoints for episode sessions whose seed starts one task sandbox.

``CheckpointedSandboxSessionResourcesServer`` adds the checkpoint half to ``SandboxSessionResourcesServer``, whose
seed, verify, close, fences and expiry it keeps as they are:

- a commit snapshots each session's task sandbox, through
  ``nemo_gym.server_runtime._checkpoint.sandbox.SandboxSessionCheckpointer``, and exports the seed, the agent's working
  directory, the patch verify captured (if it has) and the task state the server keeps;
- a restore forks every task sandbox from its snapshot, all or none, and installs each session for the attempt that
  replaces the one the checkpoint saved. That attempt's episode seed gets the fork without setting the task up again,
  and a verify after the restore grades the patch the checkpoint saved instead of capturing it again;
- a release deletes the snapshots of checkpoints the controller no longer keeps.

A snapshot may hold the sandbox's files only, not its running processes; OpenSandbox's do. A server whose grader
needs processes the agent started declares ``restore_policy = "before_agent_close"``: the environment server then
starts such an episode over after a crash once its agent's turn ended, instead of continuing it in a fork.

With checkpointing on, a server exports its sessions only when its sandbox provider can snapshot; otherwise it runs
``restart_only``, and its episodes start over after a crash.

A subclass starts each task sandbox with ``new_task_sandbox``, which puts it on the provider every session shares and
registers it for the session and its checkpoints, sets ``task_model`` to the model of its episode seed's
``task_data``, and keeps the task state a restore needs through ``_export_task_sandbox_state`` and
``_restore_task_sandbox_state``. A commit fails for a session whose sandbox was registered any other way. What the
server passes to ``capture_task_sandbox_patch`` must be a JSON value: a checkpoint stores it.
"""

import asyncio
import logging
from contextlib import AsyncExitStack
from pathlib import Path
from time import monotonic
from typing import Any, ClassVar, Literal, Optional, get_args

import anyio
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from nemo_gym.config import get_global_config_dict
from nemo_gym.sandbox import (
    AsyncSandbox,
    SandboxProvider,
    SandboxSpec,
    SupportsSandboxSnapshot,
    create_provider,
    resolve_provider_config,
)
from nemo_gym.server_runtime import (
    CheckpointMode,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionLocks,
    SandboxCheckpointConfig,
    SandboxSessionCheckpointer,
    checkpoint_settings,
    checkpoint_state_dir,
    next_attempt,
)
from resources_servers.swebench.sandbox_sessions import SandboxSessionResourcesServer, _TaskSession


LOG = logging.getLogger(__name__)

RestorePolicy = Literal["any_step", "before_agent_close"]

_JSON = TypeAdapter(JsonValue)


class SandboxCheckpointFields(BaseModel):
    """The config field of a server that checkpoints its sessions' task sandboxes."""

    sandbox_checkpoint: SandboxCheckpointConfig = Field(default_factory=SandboxCheckpointConfig)


class _CheckpointedSession(BaseModel):
    """The exported state of one episode session."""

    model_config = ConfigDict(extra="forbid")

    seed: ResourcesSeedSessionRequest
    workdir: str
    # The task sandbox's checkpoint state, or ``None`` once verify stopped it after capturing the patch.
    sandbox: Optional[dict[str, JsonValue]] = None
    capture_ready: bool = False
    capture: JsonValue = None
    task_state: dict[str, JsonValue] = Field(default_factory=dict)


class CheckpointedSandboxSessionResourcesServer(SandboxSessionResourcesServer):
    """A ``SandboxSessionResourcesServer`` whose episode sessions take part in partial-rollout checkpoints.

    See the module docstring. The config must also have ``sandbox_checkpoint`` (``SandboxCheckpointFields``).
    """

    checkpoint_mode: ClassVar[CheckpointMode] = "exported"
    # Whether a fork of the task sandbox can continue an episode at every step, or only until its agent's turn ends.
    restore_policy: ClassVar[RestorePolicy] = "any_step"
    # The model of an episode seed's ``task_data``; a restore rebuilds the session's task from it.
    task_model: ClassVar[type[BaseModel]]

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._provider: Optional[SandboxProvider] = None
        self._sandbox_checkpointer: Optional[SandboxSessionCheckpointer] = None
        # Held by a commit's export and by every stop of a task sandbox, so a commit never snapshots one being stopped.
        # Separate from the session lock, which verify holds while it grades.
        self._export_locks = ResourcesSessionLocks()
        self._mode_warning_logged = False

    # -- the provider, the checkpointer and the mode -----------------------------------------------------------

    def shared_provider(self) -> SandboxProvider:
        """The provider every task sandbox shares, built from ``sandbox_provider`` on first use.

        Use it, with ``owns_provider=False``, for sandboxes that are not session state too, such as a grading sandbox.
        """
        if self._provider is None:
            resolved = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
            self._provider = create_provider(resolved)
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
            LOG.warning(
                "%s: sandbox provider %r cannot snapshot; its episodes restart after a crash instead of continuing",
                self.config.name,
                getattr(provider, "name", type(provider).__name__),
            )
        return "restart_only"

    def _checkpointer(self) -> SandboxSessionCheckpointer:
        if self._sandbox_checkpointer is None:
            global_config = getattr(self.server_client, "global_config_dict", None)
            state_dir: Optional[Path] = None
            if checkpoint_settings(global_config) is not None and self.checkpoint_mode_in_effect() == "exported":
                # Fails at startup, not at the first commit, when checkpointing is on without a usable state_dir.
                state_dir = checkpoint_state_dir(global_config)
            self._sandbox_checkpointer = SandboxSessionCheckpointer.from_config(
                self.shared_provider(),
                self.config.sandbox_checkpoint,
                state_dir=state_dir,
                server_name=self.config.name or type(self).__name__,
            )
        return self._sandbox_checkpointer

    def setup_resources_checkpoint(self, app: FastAPI) -> None:
        # At startup, like the base's class settings: a misspelt policy would be sent and read as "any_step".
        if self.restore_policy not in get_args(RestorePolicy):
            raise ValueError(
                f"{type(self).__name__}.restore_policy must be one of {get_args(RestorePolicy)}, "
                f"not {self.restore_policy!r}"
            )
        super().setup_resources_checkpoint(app)
        if self._checkpoint is not None:
            self._checkpoint.restore_policy = self.restore_policy
            if self._checkpoint.mode == "exported":
                task_model = getattr(type(self), "task_model", None)
                if not (isinstance(task_model, type) and issubclass(task_model, BaseModel)):
                    # Otherwise the first restore after a crash is what finds it missing.
                    raise TypeError(
                        f"{type(self).__name__}.task_model must be the pydantic model of its episode seed's "
                        f"task_data, not {task_model!r}"
                    )
                self._checkpointer()

    async def shutdown(self) -> None:
        """Stop the task sandboxes as the base does, then close the shared provider.

        Their snapshots stay for the checkpoints that name them.
        """
        await super().shutdown()
        if self._sandbox_checkpointer is not None:
            # What only the checkpointer still tracks, such as the fork of a restore whose rollback could not stop it.
            leftover = self._sandbox_checkpointer.keys
            outcomes = await asyncio.gather(
                *(self._sandbox_checkpointer.stop(key) for key in leftover), return_exceptions=True
            )
            for key, outcome in zip(leftover, outcomes):
                if isinstance(outcome, Exception):
                    LOG.error("Failed to stop the task sandbox of session %s at shutdown", key, exc_info=outcome)
            await self._sandbox_checkpointer.settle()
        if self._provider is not None:
            await self._provider.aclose()

    # -- for subclasses ----------------------------------------------------------------------------------------

    def new_task_sandbox(self, session_id: str, spec: SandboxSpec) -> AsyncSandbox:
        """A task sandbox for ``session_id`` on the shared provider, registered for the session and its checkpoints.

        Call it from ``_start_task_sandbox`` before starting the sandbox, so a failed setup still stops it.
        """
        sandbox = AsyncSandbox(self.shared_provider(), spec, owns_provider=False)
        # The checkpointer first: it refuses a session that already has a sandbox, and then nothing changed.
        self._checkpointer().add(session_id, sandbox, spec)
        self._session_id_to_sandbox[session_id] = sandbox
        return sandbox

    def _export_task_sandbox_state(self, session_id: str) -> dict[str, JsonValue]:
        """The task state, other than the sandbox, that a restored session needs, such as its starting commit."""
        return {}

    def _restore_task_sandbox_state(self, session_id: str, state: dict[str, JsonValue]) -> None:
        """Install the task state ``_export_task_sandbox_state`` returned, for a restored session."""

    # -- stopping ----------------------------------------------------------------------------------------------

    async def _stop_released_task_sandbox(self, session_id: str) -> None:
        # Not around ``_stop_task_sandbox``: its wait for a claim ends in ``_release_task_sandbox``, which takes
        # this lock too.
        async with self._export_locks.locked(session_id):
            await super()._stop_released_task_sandbox(session_id)
            if session_id not in self._session_id_to_sandbox:
                await self._untrack(session_id)

    async def _release_task_sandbox(self, session_id: str, sandbox: AsyncSandbox) -> None:
        async with self._export_locks.locked(session_id):
            await super()._release_task_sandbox(session_id, sandbox)
            if session_id not in self._session_id_to_sandbox:
                await self._untrack(session_id)

    async def _untrack(self, session_id: str) -> None:
        """Drop the session's sandbox from the checkpointer once the server no longer registers it.

        The checkpointer stops what it still tracks, which does nothing to a sandbox that already stopped.
        """
        if self._sandbox_checkpointer is not None and session_id in self._sandbox_checkpointer:
            await self._sandbox_checkpointer.stop(session_id)

    # -- partial-rollout checkpoint hooks ----------------------------------------------------------------------

    async def export_session_states(self, session_ids: list[str], *, checkpoint_id: str) -> dict[str, JsonValue]:
        """Snapshot each seeded session's task sandbox and export the session; one past capture exports no sandbox."""
        async with AsyncExitStack() as held:
            for session_id in sorted(set(session_ids)):
                if self._exportable(session_id):
                    await held.enter_async_context(self._export_locks.locked(session_id))
            # A stop that held the lock first may have released a sandbox meanwhile.
            exported = [session_id for session_id in sorted(set(session_ids)) if self._exportable(session_id)]
            checkpointer = self._checkpointer()
            for session_id in exported:
                if session_id in self._session_id_to_sandbox and session_id not in checkpointer:
                    raise RuntimeError(
                        f"{type(self).__name__} started the task sandbox of session {session_id!r} without "
                        "new_task_sandbox, so a checkpoint cannot snapshot it"
                    )
            # Before any snapshot is taken: a capture a checkpoint cannot store fails every commit.
            captures = {session_id: self._exported_capture(session_id) for session_id in exported}
            snapshots = await checkpointer.export(
                [session_id for session_id in exported if session_id in checkpointer], checkpoint_id=checkpoint_id
            )
            return {
                session_id: self._session_state(
                    session_id, snapshots.get(session_id), captures[session_id]
                ).model_dump(mode="json")
                for session_id in exported
            }

    def _exportable(self, session_id: str) -> bool:
        record = self._task_sessions.get(session_id)
        return record is not None and record.seed is not None and record.response is not None and not record.expired

    def _exported_capture(self, session_id: str) -> JsonValue:
        record = self._task_sessions[session_id]
        if not record.capture_ready:
            return None
        try:
            return _JSON.validate_python(record.capture)
        except ValidationError as error:
            raise TypeError(f"{type(self).__name__} captured a patch a checkpoint cannot store: {error}") from error

    def _session_state(
        self, session_id: str, sandbox: Optional[dict[str, JsonValue]], capture: JsonValue
    ) -> _CheckpointedSession:
        record = self._task_sessions[session_id]
        return _CheckpointedSession(
            seed=record.seed,
            workdir=record.workdir,
            sandbox=sandbox,
            capture_ready=record.capture_ready,
            capture=capture,
            task_state=self._export_task_sandbox_state(session_id),
        )

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        """Fork every task sandbox from its snapshot, all or none, then install each session for its next attempt."""
        parsed: dict[str, tuple[_CheckpointedSession, BaseModel]] = {}
        for session_id, state in states.items():
            try:
                session = _CheckpointedSession.model_validate(state)
                task = self.task_model.model_validate(session.seed.task_data)
            except ValidationError as error:
                raise ValueError(f"invalid {self.config.name} session state for {session_id!r}: {error}") from error
            if session_id in self._task_sessions or session_id in self._session_id_to_sandbox:
                raise ValueError(f"resources session {session_id!r} is already held by this worker")
            parsed[session_id] = (session, task)
        checkpointer = self._checkpointer()
        await checkpointer.restore(
            {session_id: session.sandbox for session_id, (session, _) in parsed.items() if session.sandbox is not None}
        )
        try:
            for session_id, (session, task) in parsed.items():
                await self._install_restored_session(session_id, session, task)
        except BaseException:
            # All or none: the forks are running, so a failed install must not leave any of them or of the records.
            with anyio.CancelScope(shield=True):
                await self._discard_restored_sessions(list(parsed))
            raise

    async def _install_restored_session(self, session_id: str, session: _CheckpointedSession, task: BaseModel) -> None:
        sandbox = self._checkpointer().get(session_id)
        if sandbox is not None:
            self._session_id_to_sandbox[session_id] = sandbox
        record = _TaskSession(
            seed=session.seed.model_copy(update={"episode_id": next_attempt(session.seed.episode_id)}),
            task=task,
            workdir=session.workdir,
            capture=session.capture,
            capture_ready=session.capture_ready,
        )
        # The replacement attempt's identical seed gets the fork. A session past capture lends no sandbox:
        # the base refuses a seed of it as one whose task sandbox is no longer available.
        record.response = (
            await self._task_sandbox_response(session_id, record)
            if sandbox is not None
            else ResourcesSeedSessionResponse(resources_session_id=session_id)
        )
        self._task_sessions[session_id] = record
        self._open_task_sessions[session_id] = monotonic() + self._session_lifetime_s()
        self._restore_task_sandbox_state(session_id, session.task_state)

    async def _discard_restored_sessions(self, session_ids: list[str]) -> None:
        """Stop the forks of a restore that failed after forking and drop what it installed, so a retry can run."""
        checkpointer = self._checkpointer()
        for session_id in session_ids:
            try:
                await checkpointer.stop(session_id)
            except Exception:
                LOG.exception("Failed to stop the fork of session %s after a failed restore", session_id)
            if session_id not in checkpointer:
                self._session_id_to_sandbox.pop(session_id, None)
            # A fork that did not stop stays registered, so shutdown stops it.
            self._task_sessions.pop(session_id, None)
            self._open_task_sessions.pop(session_id, None)
            self._forget_task_sandbox_state(session_id)

    async def retire_session_state(self, session_id: str) -> None:
        # The attempt's snapshots stay: a checkpoint the controller keeps may still restore it.
        async with self._task_session_locks.locked(session_id):
            await self._stop_session_sandboxes(session_id)
            self._task_sessions.pop(session_id, None)
            self._open_task_sessions.pop(session_id, None)

    async def delete_external_state(self, kept: frozenset[str]) -> None:
        # Also in a process that has not created a sandbox yet: the index holds what a crashed process took.
        await self._checkpointer().release(kept)
