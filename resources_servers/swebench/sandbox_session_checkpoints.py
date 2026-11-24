# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Partial-rollout checkpoints for episode sessions whose seed starts one task sandbox.

``CheckpointedSandboxSessionResourcesServer`` adds the checkpoint half to ``SandboxSessionResourcesServer``, whose
seed, verify, close, fences and expiry it keeps as they are:

- a commit snapshots each session's task sandbox, through
  ``nemo_gym.server_runtime.OwnedSandboxes``, and exports the seed, the agent's working
  directory, the patch verify captured (if it has) and the task state the server keeps;
- a restore forks every task sandbox from its snapshot, all or none, and installs each session for the attempt that
  replaces the one the checkpoint saved. That attempt's episode seed gets the fork without setting the task up again,
  and a verify after the restore grades the patch the checkpoint saved instead of capturing it again;
- a release deletes the snapshots of checkpoints the controller no longer keeps.

What a snapshot holds depends on the provider: always the sandbox's files, and its running processes too when the
provider's ``capabilities.snapshot_keeps`` says so (a VM memory snapshot).

With checkpointing on, a server exports its sessions only when its sandbox provider can snapshot; otherwise it runs
``restart_only``, and its episodes start over after a crash. A server whose grader needs something a snapshot of its
provider does not keep, such as processes the agent started, declares ``checkpoint_mode = "restart_only"`` itself.

Only a session an episode seed started is exported. A task sandbox an agent-run seed starts is not registered for
checkpoints: the seed's reply says its episode starts over after a crash, and a commit leaves the session out.

A subclass starts each task sandbox with ``new_task_sandbox``, which puts it on the provider every session shares and
registers it for the session and its checkpoints, sets ``task_model`` to the model of its episode seed's
``task_data``, and keeps the task state a restore needs through ``_export_task_sandbox_state`` and
``_restore_task_sandbox_state``. A commit fails for a session whose sandbox was registered any other way. What the
server passes to ``capture_task_sandbox_patch`` must be a JSON value: a checkpoint stores it.

A subclass may declare ``checkpoint_verify = "replay"`` when its ``/verify`` uses the task sandbox only inside
``capture_task_sandbox_patch`` and stops it only with ``stop_task_sandbox_after_capture``: both hold the lock a
commit holds while it snapshots the sandbox, so a verify admitted during a commit neither runs commands in a sandbox
being snapshotted nor changes what the commit exports.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack
from pathlib import Path
from time import monotonic
from typing import Any, ClassVar, Literal

import anyio
from fastapi import FastAPI, Request
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
    OwnedSandboxes,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    ResourcesSessionLocks,
    SandboxCheckpointConfig,
    checkpoint_state_dir,
    next_attempt,
)
from resources_servers.swebench.sandbox_sessions import SandboxSessionResourcesServer, _TaskSession


LOG = logging.getLogger(__name__)

# The name of a session's task sandbox among the sandboxes a checkpoint saves for the session.
TASK_SANDBOX = "task"


class SandboxCheckpointFields(BaseModel):
    """The config field of a server that checkpoints its sessions' task sandboxes."""

    sandbox_checkpoint: SandboxCheckpointConfig = Field(default_factory=SandboxCheckpointConfig)


class CheckpointedSandboxSessionResourcesServer(SandboxSessionResourcesServer):
    """A ``SandboxSessionResourcesServer`` whose episode sessions take part in partial-rollout checkpoints.

    See the module docstring. The config must also have ``sandbox_checkpoint`` (``SandboxCheckpointFields``).
    """

    checkpoint_mode: ClassVar[CheckpointMode] = "exported"
    # The model of an episode seed's ``task_data``; a restore rebuilds the session's task from it.
    task_model: ClassVar[type[BaseModel]]

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)
        self._provider: SandboxProvider | None = None
        self._sandbox_checkpointer: OwnedSandboxes | None = None
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

    def _exports_sessions(self) -> bool:
        """Whether checkpointing is on and this server exports its sessions; known once the app is set up."""
        return self._checkpoint is not None and self._checkpoint.mode == "exported"

    def _checkpointer(self) -> OwnedSandboxes:
        if self._sandbox_checkpointer is None:
            state_dir: Path | None = None
            if self._exports_sessions():
                # Fails at startup, not at the first commit, when checkpointing is on without a usable state_dir.
                state_dir = checkpoint_state_dir(getattr(self.server_client, "global_config_dict", None))
            self._sandbox_checkpointer = OwnedSandboxes.from_config(
                self.shared_provider(),
                self.config.sandbox_checkpoint,
                state_dir=state_dir,
                server_name=self.config.name or type(self).__name__,
            )
        return self._sandbox_checkpointer

    def setup_resources_checkpoint(self, app: FastAPI) -> None:
        super().setup_resources_checkpoint(app)
        if self._exports_sessions():
            task_model = getattr(type(self), "task_model", None)
            if not (isinstance(task_model, type) and issubclass(task_model, BaseModel)):
                # Otherwise the first restore after a crash is what finds it missing.
                raise TypeError(
                    f"{type(self).__name__}.task_model must be the pydantic model of its episode seed's "
                    f"task_data, not {task_model!r}"
                )
            # Built here, once the participant says whether sessions are exported: that decides its state_dir.
            self._checkpointer()

    async def shutdown(self) -> None:
        """Stop the task sandboxes as the base does, then close the shared provider.

        Their snapshots stay for the checkpoints that name them.
        """
        await super().shutdown()
        if self._sandbox_checkpointer is not None:
            # What only the checkpointer still tracks, such as the fork of a restore whose rollback could not stop it.
            # It closes first, so a restore that installs its forks during the shutdown has them stopped here.
            await self._sandbox_checkpointer.aclose()
        if self._provider is not None:
            await self._provider.aclose()

    # -- for subclasses ----------------------------------------------------------------------------------------

    async def seed_task_sandbox_session(
        self, request: Request, body: ResourcesSeedSessionRequest, task_model: type[BaseModel]
    ) -> ResourcesSeedSessionResponse:
        # A restore reads the saved task data with ``task_model``, so a seed must read it with the same model.
        declared = getattr(type(self), "task_model", None)
        if declared is not None and task_model is not declared:
            raise TypeError(
                f"{type(self).__name__} seeds sessions with {task_model.__name__} but declares "
                f"task_model = {getattr(declared, '__name__', declared)}; a restore would read the saved task data "
                "with another model than the seed"
            )
        return await super().seed_task_sandbox_session(request, body, task_model)

    def new_task_sandbox(self, session_id: str, spec: SandboxSpec) -> AsyncSandbox:
        """A task sandbox for ``session_id`` on the shared provider, registered for the session and its checkpoints.

        Call it from ``_start_task_sandbox`` before starting the sandbox, so a failed setup still stops it.

        Only the sandbox of a session an episode seed started is registered for checkpoints. One an agent-run seed
        started has no episode seed a restore could install it for: while the server exports sessions, that seed's
        reply says its episode starts over after a crash.
        """
        if session_id in self._session_id_to_sandbox:
            raise ValueError(f"{session_id!r} already has a sandbox")
        sandbox = AsyncSandbox(self.shared_provider(), spec, owns_provider=False)
        record = self._task_sessions.get(session_id)
        if record is not None and record.seed is not None:
            # Before the session registers it: the checkpointer refuses a key that still has a sandbox.
            self._checkpointer().add(self._sandbox_key(session_id), sandbox, spec)
        elif self._exports_sessions():
            self._checkpoint.restart_session()
        self._session_id_to_sandbox[session_id] = sandbox
        return sandbox

    def _export_task_sandbox_state(self, session_id: str) -> dict[str, JsonValue]:
        """The task state, other than the sandbox, that a restored session needs, such as its starting commit.

        ``{}`` for a session the server keeps no task state for, including one it never saw or has forgotten.
        """
        return {}

    def _restore_task_sandbox_state(self, session_id: str, state: dict[str, JsonValue]) -> None:
        """Install the task state ``_export_task_sandbox_state`` returned, for a restored session.

        It runs before ``_task_sandbox_response`` builds the restored session's seed reply, as a seed's setup does.
        """

    @staticmethod
    def _sandbox_key(session_id: str, name: str = TASK_SANDBOX) -> str:
        """The checkpointer's key for the sandbox ``name`` of ``session_id``."""
        return f"{session_id}/{name}"

    # -- stopping ----------------------------------------------------------------------------------------------

    async def _stop_unclaimed_task_sandbox(self, session_id: str) -> None:
        # Not around ``_stop_task_sandbox``: its wait for a claim ends in ``_release_task_sandbox``, which takes
        # this lock too.
        async with self._export_locks.locked(session_id):
            await super()._stop_unclaimed_task_sandbox(session_id)
            if session_id not in self._session_id_to_sandbox:
                await self._untrack(session_id)

    async def _release_task_sandbox(self, session_id: str, sandbox: AsyncSandbox) -> None:
        async with self._export_locks.locked(session_id):
            await super()._release_task_sandbox(session_id, sandbox)
            if session_id not in self._session_id_to_sandbox:
                await self._untrack(session_id)

    async def capture_task_sandbox_patch[CaptureT](
        self, session_id: str, capture: Callable[[], Awaitable[CaptureT]]
    ) -> CaptureT:
        # A replayable verify is admitted while a commit runs: under the export lock its capture neither runs
        # commands in a sandbox being snapshotted nor changes what the commit exports for the session.
        async with self._export_locks.locked(session_id):
            return await super().capture_task_sandbox_patch(session_id, capture)

    async def _untrack(self, session_id: str) -> None:
        """Drop the session's sandbox from the checkpointer once the server no longer registers it.

        The checkpointer stops what it still tracks, which does nothing to a sandbox that already stopped.
        """
        key = self._sandbox_key(session_id)
        if self._sandbox_checkpointer is not None and key in self._sandbox_checkpointer:
            await self._sandbox_checkpointer.stop(key)

    # -- partial-rollout checkpoint hooks ----------------------------------------------------------------------

    async def export_session_states(self, session_ids: list[str], *, checkpoint_id: str) -> dict[str, JsonValue]:
        """Snapshot each seeded session's task sandbox and export the session; one past capture exports no sandbox.

        Every session is checked, and what a checkpoint cannot store refused, before any snapshot is taken. Each
        session then keeps its export lock only until its own snapshot is done, so a slow snapshot of one session
        does not hold up a capture or a stop of another.
        """
        checkpointer = self._checkpointer()
        locks: dict[str, AsyncExitStack] = {}
        # Releases the locks no session released itself, such as after a refusal below.
        async with AsyncExitStack() as held:
            for session_id in sorted(set(session_ids)):
                if self._exportable(session_id):
                    locks[session_id] = AsyncExitStack()
                    held.push_async_exit(locks[session_id])
                    await locks[session_id].enter_async_context(self._export_locks.locked(session_id))
            # A stop that held the lock first may have released a sandbox meanwhile, and a session may have expired.
            exported = [session_id for session_id in locks if self._exportable(session_id)]
            for session_id in locks.keys() - set(exported):
                await locks[session_id].aclose()
            for session_id in exported:
                if session_id in self._session_id_to_sandbox and self._sandbox_key(session_id) not in checkpointer:
                    raise RuntimeError(
                        f"{type(self).__name__} started the task sandbox of session {session_id!r} without "
                        "new_task_sandbox, so a checkpoint cannot snapshot it"
                    )
            # Before any snapshot is taken: a capture or task state a checkpoint cannot store fails every commit.
            # Read here, under every lock, so what a session exports does not change once its own lock is released.
            sessions = {session_id: self._session_state(session_id) for session_id in exported}

            async def export_one(session_id: str) -> JsonValue:
                key = self._sandbox_key(session_id)
                try:
                    # A key without a live sandbox maps to None: verify stopped it after capturing the patch.
                    snapshot = (await checkpointer.export([key], checkpoint_id=checkpoint_id))[key]
                finally:
                    await locks[session_id].aclose()
                state = sessions[session_id].model_copy(update={"sandboxes": {TASK_SANDBOX: snapshot}})
                return state.model_dump(mode="json")

            # Every snapshot is awaited before the first failure is raised, as the checkpointer does for a batch.
            outcomes = await asyncio.gather(
                *(export_one(session_id) for session_id in exported), return_exceptions=True
            )
            failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
            if failures:
                raise failures[0]
            return dict(zip(exported, outcomes))

    def _exportable(self, session_id: str) -> bool:
        record = self._task_sessions.get(session_id)
        return (
            record is not None
            and record.seed is not None
            and record.response is not None
            and not record.expired
            # Past its deadline the provider may have ended the task sandbox, and the sweep that marks the session
            # expired runs only when a request arrives: without this, one such session would fail the commit for
            # all. A session whose verify stopped the sandbox has nothing to snapshot, and one still being graded
            # past the deadline has not expired, so its capture stays in the checkpoint.
            and (
                not self._has_task_sandbox_to_snapshot(session_id)
                or self._open_task_sessions.get(session_id, float("inf")) > monotonic()
            )
        )

    def _has_task_sandbox_to_snapshot(self, session_id: str) -> bool:
        """Whether a commit would snapshot a sandbox for the session.

        The checkpointer is asked too, not the registry alone: a verify that claimed the sandbox takes it out of the
        registry while it still runs, until ``_release_task_sandbox``.
        """
        tracked = self._checkpointer().get(self._sandbox_key(session_id))
        return session_id in self._session_id_to_sandbox or (tracked is not None and tracked.started)

    def _session_state(self, session_id: str) -> "_CheckpointedSession":
        """What a commit exports for the session, apart from its sandboxes; raises for what it cannot store."""
        record = self._task_sessions[session_id]
        capture: JsonValue = None
        try:
            if record.capture_ready:
                capture = _JSON.validate_python(record.capture)
        except ValidationError as error:
            raise TypeError(f"{type(self).__name__} captured a patch a checkpoint cannot store: {error}") from error
        try:
            task_state = _TASK_STATE.validate_python(self._export_task_sandbox_state(session_id))
        except ValidationError as error:
            raise TypeError(f"{type(self).__name__} exported task state a checkpoint cannot store: {error}") from error
        return _CheckpointedSession(
            seed=record.seed,
            workdir=record.workdir,
            capture_ready=record.capture_ready,
            capture=capture,
            task_state=task_state,
        )

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        """Fork every task sandbox from its snapshot, all or none, then install each session for its next attempt."""
        parsed: dict[str, tuple["_CheckpointedSession", BaseModel]] = {}
        for session_id, state in states.items():
            version = state.get("version", _SESSION_VERSION) if isinstance(state, Mapping) else _SESSION_VERSION
            if version != _SESSION_VERSION:
                raise ValueError(
                    f"{self.config.name} session state for {session_id!r} has version {version!r}; "
                    f"this server reads version {_SESSION_VERSION}"
                )
            try:
                session = _CheckpointedSession.model_validate(state)
                task = self.task_model.model_validate(session.seed.task_data)
            except ValidationError as error:
                raise ValueError(f"invalid {self.config.name} session state for {session_id!r}: {error}") from error
            unknown = sorted(set(session.sandboxes) - {TASK_SANDBOX})
            if unknown:
                raise ValueError(
                    f"invalid {self.config.name} session state for {session_id!r}: it saves sandboxes {unknown}, "
                    f"and this server restores only {TASK_SANDBOX!r}"
                )
            if session_id in self._task_sessions or session_id in self._session_id_to_sandbox:
                raise ValueError(f"resources session {session_id!r} is already held by this worker")
            parsed[session_id] = (session, task)
        checkpointer = self._checkpointer()
        await checkpointer.restore(
            {
                self._sandbox_key(session_id): sandbox
                for session_id, (session, _) in parsed.items()
                if (sandbox := session.sandboxes.get(TASK_SANDBOX)) is not None
            }
        )
        try:
            for session_id, (session, task) in parsed.items():
                await self._install_restored_session(session_id, session, task)
        except BaseException:
            # All or none: the forks are running, so a failed install must not leave any of them or of the records.
            # Shielded: a client that disconnects cancels the request through its scope, and every await in it.
            with anyio.CancelScope(shield=True):
                await self._discard_restored_sessions(list(parsed))
            raise

    async def _install_restored_session(
        self, session_id: str, session: "_CheckpointedSession", task: BaseModel
    ) -> None:
        sandbox = self._checkpointer().get(self._sandbox_key(session_id))
        if sandbox is not None:
            self._session_id_to_sandbox[session_id] = sandbox
        record = _TaskSession(
            seed=session.seed.model_copy(update={"episode_id": next_attempt(session.seed.episode_id)}),
            task=task,
            workdir=session.workdir,
            capture=session.capture,
            capture_ready=session.capture_ready,
        )
        self._task_sessions[session_id] = record
        # Nobody may seed the session again; then it is closed at this deadline, as one a seed started is.
        self._open_task_sessions[session_id] = monotonic() + self._session_lifetime_s()
        # Before the seed reply is built, as a seed's setup records the task state before its reply: a subclass's
        # ``_task_sandbox_response`` may read it.
        self._restore_task_sandbox_state(session_id, session.task_state)
        # The replacement attempt's identical seed gets the fork. A session past capture lends no sandbox:
        # the base refuses a seed of it as one whose task sandbox is no longer available.
        record.response = (
            await self._task_sandbox_response(session_id, record)
            if sandbox is not None
            else ResourcesSeedSessionResponse(resources_session_id=session_id)
        )

    async def _discard_restored_sessions(self, session_ids: list[str]) -> None:
        """Stop the forks of a restore that failed after forking and drop what it installed.

        Every session can be restored again at once. A fork whose stop failed is left to the checkpointer,
        which stops it again at the next restore and at shutdown.
        """
        await self._checkpointer().discard([self._sandbox_key(session_id) for session_id in session_ids])
        for session_id in session_ids:
            self._session_id_to_sandbox.pop(session_id, None)
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
        await self._checkpointer().delete_snapshots(kept)


_SESSION_VERSION = 1

_JSON = TypeAdapter(JsonValue)
_TASK_STATE = TypeAdapter(dict[str, JsonValue])


class _CheckpointedSession(BaseModel):
    """The exported state of one episode session.

    ``version`` is the version of this record's format; a restore refuses one it does not know.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = _SESSION_VERSION
    seed: ResourcesSeedSessionRequest
    workdir: str
    # The checkpoint state of each of the session's sandboxes, by name: ``TASK_SANDBOX`` for the task sandbox.
    # ``None`` for one that has stopped, as the task sandbox has once verify captured the patch.
    sandboxes: dict[str, dict[str, JsonValue] | None] = Field(default_factory=dict)
    capture_ready: bool = False
    capture: JsonValue = None
    task_state: dict[str, JsonValue] = Field(default_factory=dict)
