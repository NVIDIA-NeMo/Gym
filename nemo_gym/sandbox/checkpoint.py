# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint a server's sandboxes with partial-rollout checkpoints.

A sandbox is session state outside Gym's process. Its owner, the resources server or agent that created it,
implements its participant's checkpoint hooks (``export_session_states`` and friends from
``nemo_gym._checkpoint``) by delegating to a :class:`SandboxSessionCheckpointer`.

The restore point of a sandbox is an explicit snapshot taken at commit: never the live sandbox, which keeps
changing once the episode continues, and never a paused state, which the backend may consume on resume.

- At commit, :meth:`~SandboxSessionCheckpointer.export` snapshots every live session's sandbox and waits until
  each snapshot is ready. The sandbox keeps running; the episode continues in it after the resume. A snapshot
  that fails or does not become ready in time fails the commit, and the controller keeps its previous checkpoint.
- When the controller stops after a commit (a preemption, for example), :meth:`~SandboxSessionCheckpointer.park`
  frees the compute the exported sandboxes hold, as ``on_stop`` says. Gym then exits holding nothing.
- After a crash, :meth:`~SandboxSessionCheckpointer.restore` rebuilds each session's sandbox in a fresh process
  from its snapshot (a fork), whatever happened to the live sandbox since. It validates every state first and
  installs all or nothing, as the participant hooks require. The superseded sandbox is stopped, best effort.
- :meth:`~SandboxSessionCheckpointer.stop` frees a session's sandbox. When the episode is over (verified or
  closed) it also deletes the snapshots this process took of it: a finished episode is never restored. A retired
  attempt keeps its snapshots, because a retained checkpoint may still restore it; the operator sweep
  (``nemo_gym.sandbox.providers.opensandbox.snapshots --retain-from``) deletes what no retained checkpoint names.
"""

import asyncio
import dataclasses
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import (
    ConnectableProvider,
    SandboxProvider,
    SandboxSpec,
    SupportsSandboxSnapshot,
)


LOGGER = logging.getLogger(__name__)

# What ``park`` does with an exported sandbox when the controller stops after the commit: ``kill`` stops it, so
# the stopped Gym holds no compute and the restore forks from the snapshot; ``none`` leaves it running out its TTL.
OnStop = Literal["kill", "none"]


class SandboxCheckpointState(BaseModel):
    """What a restore needs to rebuild one session's sandbox as of a checkpoint: its snapshot and its spec.

    ``descriptor`` names the sandbox the snapshot was taken from, so a restore can stop it; it is never restored
    in place. ``snapshot_at`` is the Unix time the snapshot became ready.
    """

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    descriptor: dict[str, JsonValue]
    snapshot_id: str
    spec: dict[str, JsonValue]
    snapshot_at: float


class SandboxCheckpointError(RuntimeError):
    """A session's sandbox could not be exported or rebuilt."""

    def __init__(self, session_id: str, detail: str) -> None:
        super().__init__(f"sandbox of session {session_id!r} {detail}")
        self.session_id = session_id


def spec_to_json(spec: SandboxSpec) -> dict[str, JsonValue]:
    """Serialize a spec for a checkpoint record. Files were uploaded at creation; a snapshot already has them."""
    data = dataclasses.asdict(spec)
    data["ports"] = list(spec.ports)
    data["files"] = {}
    return data


def spec_from_json(data: Mapping[str, Any]) -> SandboxSpec:
    return SandboxSpec(**data)


@dataclass
class _Entry:
    sandbox: AsyncSandbox
    spec: SandboxSpec
    # Every snapshot this process took of the sandbox, deleted when the episode is over.
    snapshot_ids: list[str] = field(default_factory=list)


class SandboxSessionCheckpointer:
    """The sandboxes one server owns, by session, and their checkpoint operations.

    ``parallelism`` bounds concurrent snapshot, create, and stop calls against the backend: a commit snapshots
    every live session at once and a restore rebuilds them all at once. Size it with the commit deadline.
    """

    def __init__(self, provider: SandboxProvider, *, parallelism: int = 16, on_stop: OnStop = "kill") -> None:
        if parallelism < 1:
            raise ValueError("parallelism must be at least 1")
        if on_stop not in ("kill", "none"):
            raise ValueError(f"on_stop must be 'kill' or 'none', not {on_stop!r}")
        self._provider = provider
        self._semaphore = asyncio.Semaphore(parallelism)
        self._on_stop: OnStop = on_stop
        self._entries: dict[str, _Entry] = {}

    @property
    def provider(self) -> SandboxProvider:
        return self._provider

    @property
    def session_ids(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._entries

    def get(self, session_id: str) -> Optional[AsyncSandbox]:
        """The session's current sandbox, or ``None``; after a restore it is the fork, under a new id."""
        entry = self._entries.get(session_id)
        return entry.sandbox if entry is not None else None

    def add(self, session_id: str, sandbox: AsyncSandbox, spec: SandboxSpec) -> None:
        """Track a sandbox the server created itself; ``spec`` is what a restore re-creates it from."""
        if session_id in self._entries:
            raise ValueError(f"session {session_id!r} already has a sandbox")
        self._entries[session_id] = _Entry(sandbox=sandbox, spec=spec)

    async def create(self, session_id: str, spec: SandboxSpec) -> AsyncSandbox:
        """Create a session's sandbox with the shared provider and track it."""
        if session_id in self._entries:
            raise ValueError(f"session {session_id!r} already has a sandbox")
        sandbox = AsyncSandbox(self._provider, spec, owns_provider=False)
        async with self._semaphore:
            await sandbox.start()
        self._entries[session_id] = _Entry(sandbox=sandbox, spec=spec)
        return sandbox

    async def access(self, session_id: str, *, provider_config_ref: str, workdir: str) -> SandboxAccess:
        """Access to the session's current sandbox for a borrower, such as an agent that runs tools in it.

        A restore rebuilds the sandbox from its snapshot under a new id, so a borrower that kept an older access
        must ask again after a restore.
        """
        entry = self._entries.get(session_id)
        if entry is None:
            raise KeyError(f"session {session_id!r} has no sandbox")
        descriptor = await entry.sandbox.serialize()
        connection = DirectSandboxConnection(provider_config_ref=provider_config_ref, descriptor=descriptor)
        return SandboxAccess(connection=connection, workdir=workdir)

    # -- commit -------------------------------------------------------------------------------------------------

    async def export(self, session_ids: Iterable[str]) -> dict[str, JsonValue]:
        """Snapshot each session's sandbox and return its checkpoint state; leaves out sessions no longer held.

        Raises :class:`SandboxCheckpointError` if a snapshot fails or is not ready in time, which fails the
        commit. Every sandbox keeps running; snapshots already taken are orphans for the sweep.
        """
        exported = await asyncio.gather(*(self._export_one(session_id) for session_id in session_ids))
        return {session_id: state for session_id, state in exported if state is not None}

    async def _export_one(self, session_id: str) -> tuple[str, Optional[dict[str, JsonValue]]]:
        entry = self._entries.get(session_id)
        if entry is None or entry.sandbox.handle is None:
            return session_id, None
        if not isinstance(self._provider, SupportsSandboxSnapshot):
            raise SandboxCheckpointError(
                session_id, f"cannot be checkpointed: provider {self._provider_name()!r} does not support snapshots"
            )
        handle = entry.sandbox.handle
        if isinstance(self._provider, ConnectableProvider):
            descriptor = await entry.sandbox.serialize()
        else:
            descriptor = {"sandbox_id": handle.sandbox_id}
        async with self._semaphore:
            try:
                snapshot_id = await self._provider.snapshot(handle, name=f"ng-{session_id[:16]}-{int(time.time())}")
            except Exception as error:
                raise SandboxCheckpointError(session_id, f"could not be snapshotted: {error}") from error
        entry.snapshot_ids.append(snapshot_id)
        state = SandboxCheckpointState(
            provider_name=handle.provider_name,
            descriptor=descriptor,
            snapshot_id=snapshot_id,
            spec=spec_to_json(entry.spec),
            snapshot_at=time.time(),
        )
        return session_id, state.model_dump(mode="json")

    async def park(self, session_ids: Iterable[str]) -> None:
        """Free what the exported sandboxes hold, as ``on_stop`` says; for a commit the controller stops after.

        The checkpoint is already durable, so a failure is logged and the sandbox runs out its TTL. The sessions
        are forgotten either way: nothing in this process continues them.
        """
        if self._on_stop == "none":
            return
        entries = [(session_id, self._entries.pop(session_id)) for session_id in session_ids if session_id in self]
        outcomes = await asyncio.gather(
            *(self._stop_sandbox(entry.sandbox) for _, entry in entries), return_exceptions=True
        )
        for (session_id, entry), outcome in zip(entries, outcomes):
            if isinstance(outcome, BaseException):
                LOGGER.warning(
                    "sandbox %s of session %s was not stopped for the stop: %r",
                    entry.sandbox.handle,
                    session_id,
                    outcome,
                )

    # -- restore ------------------------------------------------------------------------------------------------

    async def restore(self, states: Mapping[str, JsonValue]) -> None:
        """Rebuild every session's sandbox from its snapshot, all or nothing.

        Every state is validated before the backend is touched. If any sandbox cannot be rebuilt, the ones this
        call created are stopped and :class:`SandboxCheckpointError` names the session.
        """
        provider_name = self._provider_name()
        parsed: dict[str, tuple[SandboxCheckpointState, SandboxSpec]] = {}
        for session_id, raw in states.items():
            try:
                state = SandboxCheckpointState.model_validate(raw)
                spec = spec_from_json(state.spec)
            except (ValidationError, TypeError, ValueError) as error:
                raise SandboxCheckpointError(session_id, f"has an invalid checkpoint state: {error}") from error
            if state.provider_name != provider_name:
                raise SandboxCheckpointError(
                    session_id, f"was created by provider {state.provider_name!r}, not {provider_name!r}"
                )
            if session_id in self._entries:
                raise SandboxCheckpointError(session_id, "already has a sandbox in this process")
            if not isinstance(self._provider, SupportsSandboxSnapshot):
                raise SandboxCheckpointError(
                    session_id, f"cannot be restored: provider {provider_name!r} does not support snapshots"
                )
            parsed[session_id] = (state, spec)

        created: dict[str, AsyncSandbox] = {}
        outcomes = await asyncio.gather(
            *(self._fork(session_id, state, spec, created) for session_id, (state, spec) in parsed.items()),
            return_exceptions=True,
        )
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if failures:
            await asyncio.gather(
                *(self._stop_sandbox(sandbox) for sandbox in created.values()), return_exceptions=True
            )
            raise failures[0]
        for session_id, sandbox in created.items():
            state, spec = parsed[session_id]
            # The fork inherits the snapshot it came from, so the episode's end deletes that one too.
            self._entries[session_id] = _Entry(sandbox=sandbox, spec=spec, snapshot_ids=[state.snapshot_id])
        # The crashed process's sandboxes are superseded: nothing continues them. Best effort, in parallel.
        await asyncio.gather(*(self._stop_superseded(session_id, state) for session_id, (state, _) in parsed.items()))

    async def _fork(
        self, session_id: str, state: SandboxCheckpointState, spec: SandboxSpec, created: dict[str, AsyncSandbox]
    ) -> None:
        # A snapshot replaces the image: OpenSandbox requires exactly one of the two.
        fork_spec = dataclasses.replace(
            spec,
            image=None,
            files={},
            provider_options={**spec.provider_options, "snapshot_id": state.snapshot_id},
        )
        sandbox = AsyncSandbox(self._provider, fork_spec, owns_provider=False)
        async with self._semaphore:
            try:
                await sandbox.start()
            except Exception as error:
                raise SandboxCheckpointError(
                    session_id, f"could not be re-created from snapshot {state.snapshot_id!r}: {error}"
                ) from error
        created[session_id] = sandbox

    async def _stop_superseded(self, session_id: str, state: SandboxCheckpointState) -> None:
        if not isinstance(self._provider, ConnectableProvider):
            return
        sandbox_id = state.descriptor.get("sandbox_id")
        try:
            async with self._semaphore:
                old = await AsyncSandbox.connect(state.descriptor, provider=self._provider, owns_provider=False)
                await old.stop()
        except Exception as error:
            # Already gone (parked at the stop, or reaped by its TTL), or unreachable: the sweep covers it.
            LOGGER.info("superseded sandbox %s of session %s was not stopped: %r", sandbox_id, session_id, error)

    # -- release ------------------------------------------------------------------------------------------------

    async def stop(self, session_id: str, *, forget_snapshots: bool = False) -> None:
        """Stop and forget a session's sandbox; a no-op for a session without one, so a retire may repeat it.

        If the stop fails the session stays tracked, so a retried retire stops it again. With
        ``forget_snapshots``, for an episode that is over, the snapshots this process took of the sandbox are
        deleted too, best effort: a snapshot that outlives its episode is only storage, which the sweep reclaims.
        """
        entry = self._entries.get(session_id)
        if entry is None:
            return
        await self._stop_sandbox(entry.sandbox)
        self._entries.pop(session_id, None)
        if forget_snapshots and entry.snapshot_ids and isinstance(self._provider, SupportsSandboxSnapshot):
            outcomes = await asyncio.gather(
                *(self._provider.delete_snapshot(snapshot_id) for snapshot_id in entry.snapshot_ids),
                return_exceptions=True,
            )
            for snapshot_id, outcome in zip(entry.snapshot_ids, outcomes):
                if isinstance(outcome, BaseException):
                    LOGGER.warning("snapshot %s of session %s was not deleted: %r", snapshot_id, session_id, outcome)

    async def _stop_sandbox(self, sandbox: AsyncSandbox) -> None:
        async with self._semaphore:
            await sandbox.stop()

    def _provider_name(self) -> str:
        return getattr(self._provider, "name", type(self._provider).__name__)
