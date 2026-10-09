# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint a server's sandboxes with partial-rollout checkpoints.

A sandbox is session state outside Gym's process. Its owner, the resources server or agent that created it,
implements its participant's checkpoint hooks (``export_session_states`` and friends from
``nemo_gym._checkpoint``) by delegating to a :class:`SandboxSessionCheckpointer`:

- At commit, :meth:`~SandboxSessionCheckpointer.export` pauses every live session's sandbox. The paused
  sandbox is frozen at the checkpoint, and the exported state names it (descriptor and spec) so a restore can
  resume it in place. The live sandbox keeps changing once it resumes, so a descriptor of a running sandbox
  alone would restore work the agent then replays.
- What the restore point is depends on the backend, recorded as ``restore_point``:

  - ``"snapshot"``: the pause also leaves a snapshot that survives the resume, and the provider declares
    ``snapshot_survives_resume = True``. The state names the snapshot too, and a restore can re-create the
    sandbox from it even after the live one moved on (a fork).
  - ``"paused"``: the pause only freezes the sandbox, or its snapshot is consumed by the resume (OpenSandbox
    on Kubernetes). The checkpoint can restore the sandbox only while it is still paused. A snapshot id is
    still recorded when the provider can look one up, for the operator tooling, but it is best effort and
    never a correctness input: the listing may be partial or stale.

- The sandbox stays paused until its next use: :meth:`~SandboxSessionCheckpointer.ensure_running` resumes it
  once. Prepare stops every caller before commit, so nothing uses the sandbox in between.
- After a crash, :meth:`~SandboxSessionCheckpointer.restore` rebuilds each session's sandbox in a fresh
  process. It resumes the old sandbox when that is still paused at the checkpoint; otherwise it forks from
  the snapshot on a ``"snapshot"`` backend and raises a typed error on a ``"paused"`` one. It validates every
  state first and installs all or nothing, as the participant hooks require.
- :meth:`~SandboxSessionCheckpointer.stop` frees a session's sandbox, on close and on retire.

Snapshots are never deleted here. One lives as long as a checkpoint that can restore it, which only the
controller knows; ``nemo_gym.sandbox.providers.opensandbox.snapshots`` reaps the ones no retained checkpoint
names.
"""

import asyncio
import dataclasses
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import (
    ConnectableProvider,
    SandboxHandle,
    SandboxProvider,
    SandboxSpec,
    SandboxStatus,
)


LOGGER = logging.getLogger(__name__)


@runtime_checkable
class SupportsSandboxSnapshotLookup(Protocol):
    """Optional provider capability: find the snapshot a pause left behind.

    A provider whose pause keeps no snapshot, such as one that only freezes a container, does not implement
    this; its sandboxes can then be restored only while they are still paused.
    """

    async def latest_snapshot_id(self, handle: SandboxHandle) -> Optional[str]:
        """The newest snapshot of this sandbox, or ``None`` when it has none."""
        ...


RestorePoint = Literal["paused", "snapshot"]

# How long a commit waits for the best-effort snapshot lookup of one sandbox before recording none.
SNAPSHOT_LOOKUP_TIMEOUT_S = 15.0


class SandboxCheckpointState(BaseModel):
    """What a restore needs to rebuild one session's sandbox as of a checkpoint.

    ``restore_point`` says what the checkpoint can be restored from: the paused sandbox only, or also the
    snapshot named by ``snapshot_id``. ``expires_at`` is when the sandbox's TTL runs out (a Unix timestamp),
    estimated from its creation and ``spec.ttl_s``; ``None`` when the spec sets no TTL.
    """

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    descriptor: dict[str, JsonValue]
    snapshot_id: Optional[str]
    spec: dict[str, JsonValue]
    paused_at: float
    restore_point: RestorePoint
    expires_at: Optional[float] = None

    @property
    def expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at


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
    # Set from the moment a checkpoint starts pausing the sandbox until ensure_running saw it run again.
    paused_by_checkpoint: bool = False
    created_at: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class SandboxSessionCheckpointer:
    """The sandboxes one server owns, by session, and their checkpoint operations.

    ``parallelism`` bounds concurrent pause, create, and resume calls against the backend: a commit pauses
    every live session at once and a restore rebuilds them all at once. Size it with the commit deadline: at
    OpenSandbox's measured rate, ``parallelism`` sandboxes pause every 20 to 30 seconds.
    """

    def __init__(self, provider: SandboxProvider, *, parallelism: int = 16) -> None:
        if parallelism < 1:
            raise ValueError("parallelism must be at least 1")
        self._provider = provider
        self._semaphore = asyncio.Semaphore(parallelism)
        self._entries: dict[str, _Entry] = {}

    @property
    def provider(self) -> SandboxProvider:
        return self._provider

    @property
    def restore_point(self) -> RestorePoint:
        """What this provider's checkpoints can be restored from; see the module docstring."""
        survives = bool(getattr(self._provider, "snapshot_survives_resume", False))
        return "snapshot" if survives and isinstance(self._provider, SupportsSandboxSnapshotLookup) else "paused"

    @property
    def session_ids(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, session_id: object) -> bool:
        return session_id in self._entries

    def get(self, session_id: str) -> Optional[AsyncSandbox]:
        """The session's sandbox without resuming it; use :meth:`ensure_running` before using it."""
        entry = self._entries.get(session_id)
        return entry.sandbox if entry is not None else None

    def add(
        self, session_id: str, sandbox: AsyncSandbox, spec: SandboxSpec, *, created_at: Optional[float] = None
    ) -> None:
        """Track a sandbox the server created itself; ``spec`` is what a restore re-creates it from.

        ``created_at`` (a Unix timestamp) dates the TTL estimate in the exported state; it defaults to now.
        """
        if session_id in self._entries:
            raise ValueError(f"session {session_id!r} already has a sandbox")
        self._entries[session_id] = _Entry(sandbox=sandbox, spec=spec, created_at=created_at or time.time())

    async def create(self, session_id: str, spec: SandboxSpec) -> AsyncSandbox:
        """Create a session's sandbox with the shared provider and track it."""
        if session_id in self._entries:
            raise ValueError(f"session {session_id!r} already has a sandbox")
        sandbox = AsyncSandbox(self._provider, spec, owns_provider=False)
        async with self._semaphore:
            await sandbox.start()
        self._entries[session_id] = _Entry(sandbox=sandbox, spec=spec)
        return sandbox

    async def ensure_running(self, session_id: str) -> AsyncSandbox:
        """The session's sandbox, resumed if a checkpoint left it paused. Call before every use."""
        entry = self._entries.get(session_id)
        if entry is None:
            raise KeyError(f"session {session_id!r} has no sandbox")
        if entry.paused_by_checkpoint:
            async with entry.lock:
                if entry.paused_by_checkpoint:
                    # A pause that failed part way may have left the sandbox running; only a paused one resumes.
                    if await entry.sandbox.status() == SandboxStatus.PAUSED:
                        async with self._semaphore:
                            await entry.sandbox.resume()
                    entry.paused_by_checkpoint = False
        return entry.sandbox

    async def export(self, session_ids: list[str]) -> dict[str, JsonValue]:
        """Pause each session's sandbox and return its checkpoint state; leaves out sessions no longer held.

        Raises :class:`SandboxCheckpointError` if a pause fails, which fails the commit. Sandboxes paused before
        the failure stay paused until their next ``ensure_running``.
        """
        exported = await asyncio.gather(*(self._export_one(session_id) for session_id in session_ids))
        states = {session_id: state for session_id, state in exported if state is not None}
        if states and self.restore_point == "paused":
            LOGGER.info(
                "checkpointed %d sandbox(es) by pausing them; this backend keeps no snapshot past the resume, so "
                "the checkpoint can restore them only while they are still paused",
                len(states),
            )
        return states

    async def _export_one(self, session_id: str) -> tuple[str, Optional[dict[str, JsonValue]]]:
        entry = self._entries.get(session_id)
        if entry is None:
            return session_id, None
        async with entry.lock:
            handle = entry.sandbox.handle
            if handle is None:
                return session_id, None
            if isinstance(self._provider, ConnectableProvider):
                descriptor = await entry.sandbox.serialize()
            else:
                descriptor = {"sandbox_id": handle.sandbox_id}
            entry.paused_by_checkpoint = True
            async with self._semaphore:
                try:
                    await entry.sandbox.pause()
                except Exception as error:
                    raise SandboxCheckpointError(session_id, f"could not be paused: {error}") from error
                snapshot_id = await self._latest_snapshot_id(handle, session_id=session_id)
            ttl_s = entry.spec.ttl_s
            state = SandboxCheckpointState(
                provider_name=handle.provider_name,
                descriptor=descriptor,
                snapshot_id=snapshot_id,
                spec=spec_to_json(entry.spec),
                paused_at=time.time(),
                restore_point=self.restore_point,
                expires_at=entry.created_at + float(ttl_s) if ttl_s is not None else None,
            )
            return session_id, state.model_dump(mode="json")

    async def restore(self, states: Mapping[str, JsonValue]) -> None:
        """Rebuild every session's sandbox from its checkpoint state, all or nothing.

        Every state is validated before the backend is touched. If any sandbox cannot be rebuilt, the ones this
        call created are stopped and :class:`SandboxCheckpointError` names the session.
        """
        provider_name = getattr(self._provider, "name", type(self._provider).__name__)
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
            parsed[session_id] = (state, spec)

        created: dict[str, AsyncSandbox] = {}
        outcomes = await asyncio.gather(
            *(self._rebuild(session_id, state, spec, created) for session_id, (state, spec) in parsed.items()),
            return_exceptions=True,
        )
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if failures:
            await asyncio.gather(*(sandbox.stop() for sandbox in created.values()), return_exceptions=True)
            raise failures[0]
        for session_id, sandbox in created.items():
            self._entries[session_id] = _Entry(sandbox=sandbox, spec=parsed[session_id][1])

    async def _rebuild(
        self, session_id: str, state: SandboxCheckpointState, spec: SandboxSpec, created: dict[str, AsyncSandbox]
    ) -> None:
        async with self._semaphore:
            old = await self._reconnect(session_id, state)
            resume_error: Optional[BaseException] = None
            if old is not None:
                try:
                    paused_here = await old.status() == SandboxStatus.PAUSED
                    if paused_here and state.restore_point == "snapshot" and state.snapshot_id is not None:
                        # The snapshot listing is authoritative here: a pause at a later checkpoint means the
                        # frozen sandbox is not this checkpoint's, and the fork below is.
                        paused_here = await self._latest_snapshot_id(old.handle) == state.snapshot_id
                    if paused_here:
                        # Still frozen at this checkpoint: the crash came before anything resumed it.
                        await old.resume()
                        created[session_id] = old
                        return
                except Exception as error:
                    resume_error = error
                    LOGGER.warning(
                        "sandbox %s of session %s cannot be resumed (%r)",
                        state.descriptor.get("sandbox_id"),
                        session_id,
                        error,
                    )
            if state.restore_point == "paused" or state.snapshot_id is None:
                raise SandboxCheckpointError(session_id, self._unrestorable(state, old, resume_error))
            # A snapshot replaces the image: OpenSandbox requires exactly one of the two.
            fork_spec = dataclasses.replace(
                spec,
                image=None,
                files={},
                provider_options={**spec.provider_options, "snapshot_id": state.snapshot_id},
            )
            sandbox = AsyncSandbox(self._provider, fork_spec, owns_provider=False)
            try:
                await sandbox.start()
            except Exception as error:
                raise SandboxCheckpointError(
                    session_id, f"could not be re-created from snapshot {state.snapshot_id!r}: {error}"
                ) from error
            created[session_id] = sandbox
            if old is not None:
                # The crashed process's sandbox has moved past the checkpoint; nothing continues it.
                try:
                    await old.stop()
                except Exception as error:
                    LOGGER.warning(
                        "failed to stop superseded sandbox %s of session %s: %r", old.handle, session_id, error
                    )

    @staticmethod
    def _unrestorable(
        state: SandboxCheckpointState, old: Optional[AsyncSandbox], resume_error: Optional[BaseException]
    ) -> str:
        """Why a session's sandbox cannot be rebuilt, for the typed error."""
        sandbox_id = state.descriptor.get("sandbox_id")
        if state.expired:
            expiry = datetime.fromtimestamp(state.expires_at, tz=timezone.utc).isoformat(timespec="seconds")
            return f"sandbox {sandbox_id} reached its TTL at {expiry} and the checkpoint keeps no snapshot of it"
        if old is None:
            return f"sandbox {sandbox_id} is unreachable and the checkpoint keeps no snapshot to re-create it from"
        if resume_error is not None:
            return f"sandbox {sandbox_id} is paused but did not resume: {resume_error}"
        if state.restore_point == "paused":
            return (
                f"sandbox {sandbox_id} resumed after the checkpoint; this backend keeps no durable snapshot, so "
                "the checkpoint could restore it only while it was still paused"
            )
        return f"has no snapshot to re-create from and sandbox {sandbox_id} is no longer paused at the checkpoint"

    async def _reconnect(self, session_id: str, state: SandboxCheckpointState) -> Optional[AsyncSandbox]:
        if not isinstance(self._provider, ConnectableProvider):
            return None
        try:
            return await AsyncSandbox.connect(state.descriptor, provider=self._provider, owns_provider=False)
        except Exception as error:
            LOGGER.info(
                "sandbox %s of session %s is unreachable (%r)",
                state.descriptor.get("sandbox_id"),
                session_id,
                error,
            )
            return None

    async def stop(self, session_id: str) -> None:
        """Stop and forget a session's sandbox; a no-op for a session without one, so a retire may repeat it.

        If the stop fails the session stays tracked, so a retried retire stops it again.
        """
        entry = self._entries.get(session_id)
        if entry is None:
            return
        async with entry.lock:
            await entry.sandbox.stop()
        self._entries.pop(session_id, None)

    async def _latest_snapshot_id(
        self, handle: Optional[SandboxHandle], *, session_id: Optional[str] = None
    ) -> Optional[str]:
        """The sandbox's newest snapshot, or ``None`` when the provider has no lookup.

        At export (``session_id`` given) the lookup is best effort: a failure or a slow listing records no
        snapshot rather than failing the commit. At restore it raises, and the caller decides.
        """
        if handle is None or not isinstance(self._provider, SupportsSandboxSnapshotLookup):
            return None
        if session_id is None:
            return await self._provider.latest_snapshot_id(handle)
        try:
            async with asyncio.timeout(SNAPSHOT_LOOKUP_TIMEOUT_S):
                return await self._provider.latest_snapshot_id(handle)
        except Exception as error:
            LOGGER.warning(
                "snapshot lookup for sandbox %s of session %s failed (%r); the checkpoint records no snapshot id",
                handle.sandbox_id,
                session_id,
                error,
            )
            return None
