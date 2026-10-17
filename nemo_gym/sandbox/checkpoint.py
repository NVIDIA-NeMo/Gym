# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint a server's sandboxes with partial-rollout checkpoints.

A sandbox is session state outside Gym's process. Its owner, the resources server or agent that created it,
implements its participant's checkpoint hooks (``export_session_states`` and friends from
``nemo_gym._checkpoint``) by delegating to a :class:`SandboxSessionCheckpointer`.

The restore point of a sandbox is an explicit snapshot taken at commit: never the live sandbox, which keeps
changing once the episode continues, and never a paused state, which the backend may consume on resume.

A sandbox and a snapshot are two objects outside Gym with two lifetimes.

A sandbox lives while the process that created it uses it.
:meth:`~SandboxSessionCheckpointer.stop` frees it when its episode is verified, closed, or retired;
:meth:`~SandboxSessionCheckpointer.park` frees the exported ones when the controller stops after a commit;
a restore stops the crashed process's sandboxes it supersedes. What a crashed process leaves running ends with
the provider's lease: on OpenSandbox the shipped config creates every sandbox with ``create.renew_ttl_s`` and
the creating process renews it every ``create.renew_interval_s``, so the service deletes a dead process's
sandboxes within one TTL. Forks made by a restore go through the same create and get the same lease.

A snapshot lives while a checkpoint the controller may restore names it. No episode event deletes one: a
checkpoint taken before a rollout finished still restores that rollout. Instead the checkpointer keeps an index
on disk, an empty file ``<state_dir>/<server>/snapshots/<checkpoint id>/<snapshot id>`` per snapshot, and the
controller's ``release`` decides:

- At commit, :meth:`~SandboxSessionCheckpointer.export` snapshots every live session's sandbox. The moment the
  service returns a snapshot's id, its file is written, before the snapshot is awaited; then the export waits
  until each snapshot is ready. The sandbox keeps running; the episode continues in it after the resume. A
  snapshot that fails or does not become ready in time fails the commit, and the controller keeps its previous
  checkpoint. The participant's prepare guarantees no tool call is in flight when ``export`` runs, so the
  snapshot is consistent: the owner must not call it while a sandbox is in use.
- :meth:`~SandboxSessionCheckpointer.release` receives every checkpoint the controller may still restore and
  deletes the snapshots indexed under any other checkpoint, removing each file once the service confirms the
  snapshot is gone. The index outlives the process, so a release deletes what a crashed process created too,
  including the snapshots of a checkpoint whose commit never finished. A failed delete keeps its file, and the
  next release tries again.
- After a crash, :meth:`~SandboxSessionCheckpointer.restore` rebuilds each session's sandbox in a fresh process
  from its snapshot (a fork), whatever happened to the live sandbox since. It validates every state first and
  installs all or nothing, as the participant hooks require. The superseded sandboxes are stopped afterwards,
  best effort and off the restore's critical path. A restore deletes no snapshot.

``state_dir`` must be on a filesystem every restart of the server reads, outside anything the controller
deletes, and the same across restarts: the index is only useful to a process that can read it.
"""

import asyncio
import contextlib
import dataclasses
import logging
import os
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym._checkpoint.settings import checkpoint_settings
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import (
    ConnectableProvider,
    SandboxHandle,
    SandboxProvider,
    SandboxSpec,
    SupportsSandboxSnapshot,
)


LOGGER = logging.getLogger(__name__)

# What ``park`` does with an exported sandbox when the controller stops after the commit: ``kill`` stops it, so
# the stopped Gym holds no compute and the restore forks from the snapshot; ``none`` leaves it running out its TTL.
OnStop = Literal["kill", "none"]

# A fork that the backend answers with "snapshot not found" is retried this many times, this far apart.
FORK_NOT_FOUND_RETRIES = 4
FORK_NOT_FOUND_BACKOFF_S = 5.0

# Spec entries a fork never reads: the snapshot replaces the image, and with it the registry credentials.
_UNRESTORED_PROVIDER_OPTIONS = ("image_auth", "snapshot_id")


class SandboxCheckpointConfig(BaseModel):
    """How a server checkpoints the sandboxes its sessions own: the ``sandbox_checkpoint`` block of its config."""

    # Concurrent snapshot, re-create, and stop calls against the sandbox backend during a commit or restore.
    parallelism: int = 16
    # What happens to the agents' sandboxes when the controller stops after a commit: "kill" frees them (the
    # restore forks from the snapshots), "none" leaves them to their TTL.
    on_stop: OnStop = "kill"
    # Snapshots of one sandbox per commit before its export fails: the backend's commit job can fail
    # (OpenSandbox: CommitJobFailed after its own retries) while the sandbox keeps running, and one failed
    # snapshot otherwise fails the whole checkpoint. Every attempt is indexed, so a release deletes it.
    snapshot_attempts: int = 3
    # Pause between those attempts.
    snapshot_retry_delay_s: float = 5.0


class SandboxCheckpointState(BaseModel):
    """What a restore needs to rebuild one session's sandbox as of a checkpoint: its snapshot and its spec.

    ``snapshot_id`` is the snapshot a restore forks.
    ``descriptor`` names the sandbox the snapshot was taken from, so a restore can stop it; it is never restored
    in place. ``snapshot_at`` is the Unix time the snapshot became ready.

    ``spec`` is written as the fork needs it, which includes the sandbox environment (``env``) in plain text:
    a checkpoint directory holds whatever the sandboxes were started with.
    """

    model_config = ConfigDict(extra="forbid")

    provider_name: str
    descriptor: dict[str, JsonValue]
    snapshot_id: str
    spec: dict[str, JsonValue]
    snapshot_at: float


class _RetryableSnapshotError(Exception):
    """One snapshot attempt failed; ``error`` is what the export raises once no attempt is left."""

    def __init__(self, error: "SandboxCheckpointError") -> None:
        super().__init__(str(error))
        self.error = error


class SandboxCheckpointError(RuntimeError):
    """A session's sandbox could not be exported or rebuilt."""

    def __init__(self, session_id: str, detail: str) -> None:
        super().__init__(f"sandbox of session {session_id!r} {detail}")
        self.session_id = session_id


def checkpoint_state_dir(global_config: Any) -> Optional[Path]:
    """``checkpoint.state_dir`` for a server that checkpoints sandboxes; ``None`` when checkpointing is off.

    Raises ``ValueError`` when checkpointing is on without it: the server could not delete the snapshots its
    checkpoints take, so it refuses to start rather than leak them.
    """
    settings = checkpoint_settings(global_config)
    if settings is None:
        return None
    if not settings.state_dir:
        raise ValueError(
            "checkpoint.state_dir is required to checkpoint sandboxes: it is where the server records the "
            "snapshots its checkpoints take, so that a release can delete them after a crash too"
        )
    return Path(settings.state_dir)


def spec_to_json(spec: SandboxSpec) -> dict[str, JsonValue]:
    """Serialize a spec for a checkpoint record, leaving out what a fork never reads.

    Files were uploaded at creation and the snapshot has them; the snapshot also replaces the image, so the
    image and its registry credentials (``provider_options["image_auth"]``) are dropped too.
    """
    data = dataclasses.asdict(spec)
    data["image"] = None
    data["files"] = {}
    data["ports"] = list(spec.ports)
    data["sidecars"] = list(data["sidecars"])  # JSON has no tuple; the spec takes the list back
    options = dict(data["provider_options"])
    for key in _UNRESTORED_PROVIDER_OPTIONS:
        options.pop(key, None)
    data["provider_options"] = options
    return data


def spec_from_json(data: Mapping[str, Any]) -> SandboxSpec:
    """Rebuild a spec from a checkpoint record written by :func:`spec_to_json`.

    Raises ``TypeError`` for a key the current :class:`SandboxSpec` does not know and ``ValueError`` for an
    invalid ``resources`` or ``ports`` value; a restore reports either as an invalid checkpoint state.
    """
    return SandboxSpec(**data)


def _path_component(name: str, what: str) -> str:
    """``name`` as one path component of the index, which it must be used as unchanged."""
    if not name or name in (".", "..") or "/" in name or "\0" in name:
        raise ValueError(f"{what} {name!r} cannot name a file")
    return name


class _SnapshotIndex:
    """The snapshots one server created, by the checkpoint whose export created them, as empty files.

    ``<root>/<checkpoint id>/<snapshot id>`` exists from the moment the service returned the snapshot's id until
    the service confirmed it deleted. Every operation is a filesystem call or two and nothing is written into a
    file, so several processes, such as a crashed process's replacement, can share one index without locks; the
    worst case is a snapshot deleted twice.
    """

    def __init__(self, root: Path) -> None:
        self.root = root

    def mark(self, checkpoint_id: str, snapshot_id: str) -> None:
        folder = self.root / _path_component(checkpoint_id, "checkpoint id")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / _path_component(snapshot_id, "snapshot id")).touch()

    def unmark(self, checkpoint_id: str, snapshot_id: str) -> None:
        (self.root / checkpoint_id / snapshot_id).unlink(missing_ok=True)
        self.remove_if_empty(checkpoint_id)

    def remove_if_empty(self, checkpoint_id: str) -> None:
        with contextlib.suppress(OSError):  # not empty, or already gone
            (self.root / checkpoint_id).rmdir()

    def dropped(self, kept_checkpoint_ids: set[str]) -> dict[str, list[str]]:
        """The snapshots indexed under each checkpoint outside ``kept_checkpoint_ids``."""
        if not self.root.is_dir():
            return {}
        return {
            folder.name: sorted(path.name for path in folder.iterdir())
            for folder in self.root.iterdir()
            if folder.is_dir() and folder.name not in kept_checkpoint_ids
        }

    def snapshot_ids(self) -> dict[str, list[str]]:
        """Every indexed snapshot, by checkpoint."""
        return self.dropped(set())


@dataclass
class _Entry:
    sandbox: AsyncSandbox
    spec: SandboxSpec


class SandboxSessionCheckpointer:
    """The sandboxes one server owns, by session, and their checkpoint operations.

    ``state_dir`` and ``server_name`` place the snapshot index at ``<state_dir>/<server_name>/snapshots``; see
    the module docstring for what ``state_dir`` must be. Without a ``state_dir``, for a server whose
    checkpointing is off, sessions are tracked and stopped but cannot be exported. ``parallelism`` bounds concurrent snapshot, create,
    stop, and delete calls against the backend: a commit snapshots every live session at once and a restore
    rebuilds them all at once. Size it with the commit deadline.
    """

    def __init__(
        self,
        provider: SandboxProvider,
        *,
        state_dir: str | os.PathLike[str] | None,
        server_name: str,
        parallelism: int = 16,
        on_stop: OnStop = "kill",
        snapshot_attempts: int = 1,
        snapshot_retry_delay_s: float = 0.0,
    ) -> None:
        if parallelism < 1:
            raise ValueError("parallelism must be at least 1")
        if snapshot_attempts < 1:
            raise ValueError("snapshot_attempts must be at least 1")
        self._snapshot_attempts = snapshot_attempts
        self._snapshot_retry_delay_s = max(0.0, snapshot_retry_delay_s)
        if on_stop not in ("kill", "none"):
            raise ValueError(f"on_stop must be 'kill' or 'none', not {on_stop!r}")
        self._provider = provider
        self._semaphore = asyncio.Semaphore(parallelism)
        self._on_stop: OnStop = on_stop
        self._entries: dict[str, _Entry] = {}
        self._index: Optional[_SnapshotIndex] = None
        if state_dir is not None:
            self._index = _SnapshotIndex(Path(state_dir) / _path_component(server_name, "server name") / "snapshots")
            # Fails at server start, not at the first commit, if the directory cannot be created.
            self._index.root.mkdir(parents=True, exist_ok=True)
        self._background: set[asyncio.Task[None]] = set()
        # The latest release; the next one waits for it, so two never delete the same snapshots at once.
        self._releasing: Optional[asyncio.Task[None]] = None

    @property
    def provider(self) -> SandboxProvider:
        """The shared provider every tracked sandbox was created with."""
        return self._provider

    @property
    def session_ids(self) -> list[str]:
        """The sessions that currently have a sandbox, including one whose stop failed and awaits a retry."""
        return list(self._entries)

    @property
    def indexed_snapshots(self) -> dict[str, list[str]]:
        """Every snapshot in the index, by checkpoint: the ones no release has deleted yet. Reads the disk."""
        return self._index.snapshot_ids() if self._index is not None else {}

    def __contains__(self, session_id: object) -> bool:
        """Whether ``session_id`` currently has a sandbox."""
        return session_id in self._entries

    def get(self, session_id: str) -> Optional[AsyncSandbox]:
        """The session's current sandbox, or ``None``; after a restore it is the fork, under a new id."""
        entry = self._entries.get(session_id)
        return entry.sandbox if entry is not None else None

    def add(self, session_id: str, sandbox: AsyncSandbox, spec: SandboxSpec) -> None:
        """Track a sandbox the server created itself; ``spec`` is what a restore re-creates it from.

        The sandbox must run on this checkpointer's provider without owning it
        (``AsyncSandbox(checkpointer.provider, spec, owns_provider=False)``): the provider is shared by every
        session, and a sandbox that owned it would close it for all of them when stopped.
        """
        if session_id in self._entries:
            raise ValueError(f"session {session_id!r} already has a sandbox")
        if sandbox.owns_provider:
            raise ValueError(
                f"the sandbox of session {session_id!r} owns its provider; create it with owns_provider=False "
                "on the checkpointer's provider"
            )
        if sandbox.provider is not self._provider:
            raise ValueError(f"the sandbox of session {session_id!r} is not on the checkpointer's provider")
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

    async def settle(self) -> None:
        """Wait for the background work: superseded sandboxes a restore stops, and snapshots a release deletes."""
        while self._background:
            await asyncio.gather(*self._background, return_exceptions=True)

    # -- commit -------------------------------------------------------------------------------------------------

    async def export(self, session_ids: Iterable[str], *, checkpoint_id: str) -> dict[str, JsonValue]:
        """Snapshot each session's sandbox for checkpoint ``checkpoint_id`` and return its checkpoint state;
        leaves out sessions no longer held.

        Each snapshot is indexed under ``checkpoint_id`` as soon as it exists, and every snapshot is awaited
        before the result or the first failure is returned. Raises :class:`SandboxCheckpointError` if a
        snapshot fails, cannot be indexed, or is not ready in time, which fails the commit; every sandbox keeps
        running, and the snapshots already taken are deleted by a release that does not keep ``checkpoint_id``.
        """
        outcomes = await asyncio.gather(
            *(self._export_one(session_id, checkpoint_id) for session_id in session_ids), return_exceptions=True
        )
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if failures:
            raise failures[0]
        return {session_id: state for session_id, state in outcomes if state is not None}

    async def _export_one(self, session_id: str, checkpoint_id: str) -> tuple[str, Optional[dict[str, JsonValue]]]:
        entry = self._entries.get(session_id)
        if entry is None or entry.sandbox.handle is None:
            return session_id, None
        if not isinstance(self._provider, SupportsSandboxSnapshot):
            raise SandboxCheckpointError(
                session_id, f"cannot be checkpointed: provider {self._provider_name()!r} does not support snapshots"
            )
        index = self._index
        if index is None:
            raise SandboxCheckpointError(
                session_id, "cannot be checkpointed without a state_dir to index its snapshot"
            )
        handle = entry.sandbox.handle
        if isinstance(self._provider, ConnectableProvider):
            descriptor = await entry.sandbox.serialize()
        else:
            descriptor = {"sandbox_id": handle.sandbox_id}
        async with self._semaphore:
            attempt = 1
            while True:
                try:
                    snapshot_id = await self._snapshot_once(session_id, handle, index, checkpoint_id)
                    break
                except _RetryableSnapshotError as failure:
                    if attempt >= self._snapshot_attempts:
                        raise failure.error from failure.__cause__
                    LOGGER.warning(
                        "sandbox checkpoint snapshot attempt %d/%d failed; retrying: session_id=%s sandbox_id=%s %s",
                        attempt,
                        self._snapshot_attempts,
                        session_id,
                        handle.sandbox_id,
                        failure.error,
                    )
                    attempt += 1
                    await asyncio.sleep(self._snapshot_retry_delay_s)
        state = SandboxCheckpointState(
            provider_name=handle.provider_name,
            descriptor=descriptor,
            snapshot_id=snapshot_id,
            spec=spec_to_json(entry.spec),
            snapshot_at=time.time(),
        )
        return session_id, state.model_dump(mode="json")

    async def _snapshot_once(
        self, session_id: str, handle: SandboxHandle, index: "_SnapshotIndex", checkpoint_id: str
    ) -> str:
        """Take one snapshot and wait until it is ready. A failed submit or snapshot raises
        :class:`_RetryableSnapshotError`; failing to index one raises :class:`SandboxCheckpointError`."""
        try:
            snapshot_id = await self._provider.snapshot(handle, name=f"ng-{session_id[:16]}-{int(time.time())}")
        except Exception as error:
            raise _RetryableSnapshotError(
                SandboxCheckpointError(session_id, f"could not be snapshotted: {error}")
            ) from error
        # Indexed before anything else happens with it: a snapshot that fails, times out, or outlives a
        # cancelled export or a crash exists on the backend all the same, and a release must find it.
        try:
            await asyncio.to_thread(index.mark, checkpoint_id, snapshot_id)
        except (OSError, ValueError) as error:
            raise SandboxCheckpointError(
                session_id, f"snapshot {snapshot_id!r} could not be indexed: {error}"
            ) from error
        try:
            await self._provider.wait_snapshot_ready(snapshot_id)
        except Exception as error:
            raise _RetryableSnapshotError(
                SandboxCheckpointError(session_id, f"snapshot {snapshot_id!r} did not become ready: {error}")
            ) from error
        return snapshot_id

    async def park(self, session_ids: Iterable[str]) -> None:
        """Free what the exported sandboxes hold, as ``on_stop`` says; for a commit the controller stops after.

        This participant's records are durable by now, so a failure is logged and the sandbox runs out its TTL.
        With ``kill`` the sessions are forgotten whether or not the stop succeeded: nothing in this process
        continues them. With ``none`` nothing changes.
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
        call created are stopped and :class:`SandboxCheckpointError` names the session. Once the forks are in
        place the crashed process's sandboxes are stopped in the background, best effort; a failure there is
        logged and the sandbox runs out its TTL.
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
            self._entries[session_id] = _Entry(sandbox=sandbox, spec=parsed[session_id][1])
        # The crashed process's sandboxes are superseded: nothing continues them. Off the critical path: the
        # install is deadline-bound and the forks are already in place.
        self._spawn(self._stop_superseded_all(parsed))

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
        for attempt in range(FORK_NOT_FOUND_RETRIES + 1):
            try:
                async with self._semaphore:
                    await sandbox.start()
                break
            except Exception as error:
                # A snapshot store that is not fully shared across replicas answers "not found" from the
                # replicas that did not make the snapshot; a later request may land on one that did.
                if "NOT_FOUND" in repr(error) and attempt < FORK_NOT_FOUND_RETRIES:
                    LOGGER.warning(
                        "snapshot %s of session %s not found on attempt %d; retrying in %.0fs",
                        state.snapshot_id,
                        session_id,
                        attempt + 1,
                        FORK_NOT_FOUND_BACKOFF_S,
                    )
                    await asyncio.sleep(FORK_NOT_FOUND_BACKOFF_S)
                    continue
                raise SandboxCheckpointError(
                    session_id, f"could not be re-created from snapshot {state.snapshot_id!r}: {error}"
                ) from error
        created[session_id] = sandbox

    async def _stop_superseded_all(self, parsed: Mapping[str, tuple[SandboxCheckpointState, SandboxSpec]]) -> None:
        await asyncio.gather(*(self._stop_superseded(session_id, state) for session_id, (state, _) in parsed.items()))

    async def _stop_superseded(self, session_id: str, state: SandboxCheckpointState) -> None:
        if not isinstance(self._provider, ConnectableProvider):
            return
        sandbox_id = state.descriptor.get("sandbox_id")
        try:
            async with self._semaphore:
                old = await AsyncSandbox.connect(state.descriptor, provider=self._provider, owns_provider=False)
                await old.stop()
        except Exception as error:
            # Already gone (parked at the stop, or ended by its lifetime), or unreachable: its lifetime ends it.
            LOGGER.info("superseded sandbox %s of session %s was not stopped: %r", sandbox_id, session_id, error)

    def _spawn(self, operation: Any) -> asyncio.Task[None]:
        task: asyncio.Task[None] = asyncio.ensure_future(operation)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    # -- release ------------------------------------------------------------------------------------------------

    async def release(self, kept_checkpoint_ids: Iterable[str]) -> None:
        """Delete the snapshots indexed under every checkpoint outside ``kept_checkpoint_ids``; returns at once.

        ``kept_checkpoint_ids`` is every checkpoint the controller may still restore; an empty list deletes every
        indexed snapshot. Call it only while no export runs, as the participant does by admitting a release only
        when idle: the checkpoints to drop are read from the index now, and the export of a later checkpoint
        must not be among them. Their snapshots are deleted in the background, after any earlier release
        finished, under the same parallelism as everything else. A snapshot the service deleted, or reports
        gone, leaves the index; one whose delete fails stays, and the next release tries again.
        """
        index = self._index
        if index is None:
            return
        dropped = await asyncio.to_thread(index.dropped, set(kept_checkpoint_ids))
        self._releasing = self._spawn(self._delete_dropped(index, dropped, self._releasing))

    async def _delete_dropped(
        self, index: _SnapshotIndex, dropped: dict[str, list[str]], previous: Optional[asyncio.Task[None]]
    ) -> None:
        if previous is not None:
            await asyncio.wait([previous])
        if isinstance(self._provider, SupportsSandboxSnapshot):
            await asyncio.gather(
                *(
                    self._delete_snapshot(index, checkpoint_id, snapshot_id)
                    for checkpoint_id, snapshot_ids in dropped.items()
                    for snapshot_id in snapshot_ids
                )
            )
        for checkpoint_id in dropped:
            await asyncio.to_thread(index.remove_if_empty, checkpoint_id)

    async def _delete_snapshot(self, index: _SnapshotIndex, checkpoint_id: str, snapshot_id: str) -> None:
        try:
            async with self._semaphore:
                await self._provider.delete_snapshot(snapshot_id)
        except Exception as error:
            LOGGER.warning(
                "snapshot %s of dropped checkpoint %s was not deleted; the next release tries again: %r",
                snapshot_id,
                checkpoint_id,
                error,
            )
            return
        await asyncio.to_thread(index.unmark, checkpoint_id, snapshot_id)

    async def stop(self, session_id: str) -> None:
        """Stop and forget a session's sandbox; a no-op for a session without one, so a retire may repeat it.

        If the stop fails the session stays tracked, so a retried retire stops it again. The sandbox's snapshots
        stay: a checkpoint the controller may still restore can name them, and only a release knows which.
        """
        entry = self._entries.get(session_id)
        if entry is None:
            return
        await self._stop_sandbox(entry.sandbox)
        self._entries.pop(session_id, None)

    async def _stop_sandbox(self, sandbox: AsyncSandbox) -> None:
        async with self._semaphore:
            await sandbox.stop()

    def _provider_name(self) -> str:
        return getattr(self._provider, "name", type(self._provider).__name__)
