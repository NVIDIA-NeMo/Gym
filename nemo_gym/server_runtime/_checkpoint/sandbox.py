# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint the sandboxes a server owns with partial-rollout checkpoints.

A sandbox is session state outside Gym's process. Its owner, a resources server, an agent or an environment server,
implements its participant's checkpoint hooks (``export_session_states`` and friends from ``nemo_gym.server_runtime._checkpoint``)
by delegating to a :class:`SandboxSessionCheckpointer`. The checkpointer knows sandboxes by an opaque key, the
owner's session ID or a name within it, and nothing about the owner's protocol.

The restore point of a sandbox is an explicit snapshot taken at commit: never the live sandbox, which keeps changing
once the episode continues, and never a paused state, which the backend may consume on resume.

A sandbox and a snapshot are two objects outside Gym with two lifetimes.

A sandbox lives while its owner uses it. :meth:`~SandboxSessionCheckpointer.stop` stops it when its session ends
or is retired, and :meth:`~SandboxSessionCheckpointer.aclose` stops every one left at the server's shutdown; a
restore stops the crashed process's sandboxes it supersedes. What a crashed process leaves running ends with the
lifetime the provider gives a sandbox, the spec's ``ttl_s`` at the latest. A provider whose sandboxes would outlive
a crashed process by that much should end them sooner, for example with a lease that only the creating process
renews; the provider's documentation says how. Forks made by a restore go through the same create and get the same
lifetime.

A snapshot lives while a checkpoint the controller may restore names it. No episode event deletes one: a checkpoint
taken before a rollout finished still restores that rollout. Instead the checkpointer keeps an index on disk, an
empty file ``<state_dir>/<server>/snapshots/<checkpoint id>/<snapshot id>`` per snapshot, and the controller's
release decides, through :meth:`~SandboxSessionCheckpointer.delete_snapshots`:

- At commit, :meth:`~SandboxSessionCheckpointer.export` snapshots every live sandbox it is asked for. The moment the
  service returns a snapshot's id, its file is written, before the snapshot is awaited; then the export waits until
  each snapshot is ready. A snapshot that fails or does not become ready in time is taken again, up to
  ``snapshot_attempts`` times, and every attempt is indexed. The sandbox keeps running; the episode continues in it
  after the resume. The owner calls ``export`` only while no command uses the sandbox. A participant's prepare
  guarantees that for a resources server whose ``checkpoint_verify`` is ``"wait"``, the default. With ``"replay"``
  its ``/verify`` is admitted while a checkpoint is open, so an owner whose ``/verify`` runs commands in the
  sandbox, or stops it, must keep ``checkpoint_verify`` at ``"wait"``.
- :meth:`~SandboxSessionCheckpointer.delete_snapshots` receives every checkpoint the controller may still restore
  and deletes the snapshots indexed under any other checkpoint, removing each file once the service confirms the
  snapshot is gone. It returns only once it has tried every delete, so the participant that awaits it admits no
  commit meanwhile and a later checkpoint is never among the ones it drops. The index outlives the process, so a
  release deletes what a crashed process created too, including the snapshots of a commit that never finished. A
  failed delete keeps its file, the call raises, and the next release tries again. A release also lists this
  server's snapshots by name and deletes those of dropped checkpoints that never reached the index, such as one
  whose process died between the service's reply and the file write. That sweep relies on the provider keeping
  the name a snapshot is taken with (see :class:`~nemo_gym.sandbox.providers.base.SupportsSandboxSnapshot`): on a
  provider that drops it, such a snapshot is never found.
- After a crash, :meth:`~SandboxSessionCheckpointer.restore` rebuilds each sandbox in a fresh process from its
  snapshot (a fork), whatever happened to the live sandbox since. It validates every state first and installs all or
  nothing. The superseded sandboxes are stopped afterwards, best effort and off the restore's critical path. A
  restore deletes no snapshot.

Nothing but a release deletes a snapshot: the checkpointer cannot tell which checkpoints the controller may still
restore, and it does not assume the provider expires snapshots. Under a controller that never releases, every
commit therefore leaves one snapshot per live sandbox, for good, and :meth:`~SandboxSessionCheckpointer.export`
warns once the index holds more than a few checkpoints. A job that ended without a last release leaves the
snapshots of the checkpoints it had not dropped. Either way they are found by their name prefix and deleted with
the provider's own tooling.

Snapshot names are ``ng-<state dir>-<server>-<checkpoint>-<key>-<attempt>``, each part but the attempt a short hash,
so names stay within provider limits and a listing can tell the state directory, the server and the checkpoint of
each snapshot. ``<state dir>`` is read from ``<state_dir>/state_dir_id``, written once by the first server to use
the directory, so every server and every restart that shares a ``state_dir`` shares it too. It identifies the
directory, not a collection run: it is unrelated to the ``run_id`` of a rollouts manifest or an exporter.

A ``state_dir`` belongs to one job and its restarts. Two jobs that run at the same time must not share one: they
would share the index and the name prefix, each one's release would treat the other's checkpoints as dropped, and
each would delete the other's snapshots. Give every job its own, for example with the job's id in the path. It
must be on a filesystem every restart of the server reads, outside anything the controller deletes, and the same
across restarts: the index is only useful to a process that can read it. The filesystem's ``flock`` must hold
across every node that runs the server, since it serializes releases across the processes that share the index.
"""

import asyncio
import contextlib
import dataclasses
import fcntl
import functools
import hashlib
import logging
import os
import re
import secrets
import time
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import (
    ConnectableProvider,
    SandboxProvider,
    SandboxSnapshotInUseError,
    SandboxSnapshotNotFoundError,
    SandboxSpec,
    SupportsSandboxSnapshot,
)
from nemo_gym.server_runtime._checkpoint.settings import checkpoint_settings


LOGGER = logging.getLogger(__name__)

# A fork that the backend answers with "snapshot not found" is retried this many times, this far apart: a snapshot
# store that is not fully shared across the service's replicas can answer from one that has not seen it yet.
FORK_NOT_FOUND_RETRIES = 4
FORK_NOT_FOUND_BACKOFF_S = 5.0

# Spec entries a fork never reads: the snapshot replaces the image, and with it the registry credentials.
_UNRESTORED_PROVIDER_OPTIONS = ("image_auth", "snapshot_id")
_STATE_DIR_ID_FILE = "state_dir_id"
_HASH_CHARS = 8
_MARK_ATTEMPTS = 10
# What follows the prefix in a name this checkpointer gives: the checkpoint's hash, the key's hash and the attempt.
_SNAPSHOT_NAME_SUFFIX = re.compile(rf"([0-9a-f]{{{_HASH_CHARS}}})-[0-9a-f]{{{_HASH_CHARS}}}-[1-9][0-9]*")
# Pause between tries for the index lock while another release holds it.
_LOCK_POLL_S = 0.1
# An export warns once the index holds more checkpoints than this, and again each time that number doubles: a
# controller that releases keeps the one or two it may restore, so more means no release is dropping any.
_UNRELEASED_CHECKPOINTS_WARNING = 5
_STATE_VERSION = 1
_SPEC_JSON = TypeAdapter(dict[str, JsonValue])


class SandboxCheckpointConfig(BaseModel):
    """How a server checkpoints the sandboxes it owns."""

    model_config = ConfigDict(extra="forbid")

    parallelism: int = Field(
        default=16,
        ge=1,
        description="Concurrent snapshot submits, creates, stops and deletes against the sandbox backend. "
        "Readiness waits are not counted: a commit waits for every snapshot at once.",
    )
    snapshot_attempts: int = Field(
        default=3,
        ge=1,
        description="Snapshots of one sandbox per commit before its export fails. The backend's commit job can "
        "fail while the sandbox keeps running, and one failed snapshot otherwise fails the whole checkpoint. "
        "Every attempt is indexed, so a release deletes it. A snapshot that is not ready in time is taken again "
        "too, so the commit deadline must cover this many times the provider's snapshot timeout.",
    )
    snapshot_retry_delay_s: float = Field(default=5.0, ge=0, description="Pause between those attempts.")


class SandboxCheckpointState(BaseModel):
    """What a restore needs to rebuild one sandbox as of a checkpoint: its snapshot and its spec.

    ``snapshot_id`` is the snapshot a restore forks. ``descriptor`` names the sandbox the snapshot was taken from, so
    a restore can stop it; it is never restored in place. ``snapshot_ready_ts`` is the Unix time the snapshot became
    ready. ``version`` is the version of this record's format; a restore refuses one it does not know.

    ``spec`` is written as the fork needs it, which includes the sandbox environment (``env``) in plain text: a
    checkpoint directory holds whatever the sandboxes were started with.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = _STATE_VERSION
    provider_name: str
    descriptor: dict[str, JsonValue]
    snapshot_id: str
    spec: dict[str, JsonValue]
    snapshot_ready_ts: float


class SandboxCheckpointError(RuntimeError):
    """A sandbox could not be exported or rebuilt."""

    def __init__(self, key: str, detail: str) -> None:
        super().__init__(f"sandbox {key!r} {detail}")
        self.key = key


class SnapshotDeleteIncompleteError(RuntimeError):
    """A release could not delete every snapshot it dropped; ``failed`` counts those left for the next release."""

    def __init__(self, failed: int, kept: frozenset[str]) -> None:
        super().__init__(
            f"{failed} snapshots of checkpoints outside {sorted(kept)} were not deleted; the next release tries again"
        )
        self.failed = failed


class _RetryableSnapshotError(Exception):
    """One snapshot attempt failed; ``error`` is what the export raises once no attempt is left."""

    def __init__(self, error: SandboxCheckpointError, snapshot_id: Optional[str]) -> None:
        super().__init__(str(error))
        self.error = error
        self.snapshot_id = snapshot_id


def checkpoint_state_dir(global_config: Any) -> Optional[Path]:
    """``checkpoint.state_dir`` for a server that checkpoints sandboxes; ``None`` when checkpointing is off.

    Raises ``ValueError`` when checkpointing is on without it: the server could not delete the snapshots its
    checkpoints take, so it refuses to start rather than leak them.
    It must be absolute: a relative path would name a different directory for each server and each restart
    started from another working directory.
    """
    settings = checkpoint_settings(global_config)
    if settings is None:
        return None
    if not settings.state_dir:
        raise ValueError(
            "checkpoint.state_dir is required to checkpoint sandboxes: it is where the server records the "
            "snapshots its checkpoints take, so that a release can delete them after a crash too"
        )
    state_dir = Path(settings.state_dir)
    if not state_dir.is_absolute():
        raise ValueError(f"checkpoint.state_dir must be an absolute path, got {settings.state_dir!r}")
    return state_dir


def spec_to_json(spec: SandboxSpec) -> dict[str, JsonValue]:
    """Serialize a spec for a checkpoint record, leaving out what a fork never reads.

    Files were uploaded at creation and the snapshot has them; the snapshot also replaces the image, so the image and
    its registry credentials (``provider_options["image_auth"]``) are dropped too.
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

    Raises ``TypeError`` for a key the current :class:`SandboxSpec` does not know and ``ValueError`` for an invalid
    ``resources`` or ``ports`` value; a restore reports either as an invalid checkpoint state.
    """
    return SandboxSpec(**data)


def name_hash(value: str) -> str:
    """The short hash that stands for ``value`` in a snapshot name."""
    return hashlib.sha256(value.encode()).hexdigest()[:_HASH_CHARS]


def _path_component(name: str, what: str) -> str:
    """``name`` as one path component of the index, which it must be used as unchanged."""
    if not name or name in (".", "..") or "/" in name or "\0" in name:
        raise ValueError(f"{what} {name!r} cannot name a file")
    return name


def read_state_dir_id(state_dir: Path) -> str:
    """The ID of ``state_dir``, written once by the first process to ask; every later process reads it.

    It identifies the directory, and so every server, restart and job that uses it; it is not the ``run_id`` of a
    rollouts manifest or an exporter.
    """
    state_dir.mkdir(parents=True, exist_ok=True)
    path = state_dir / _STATE_DIR_ID_FILE
    candidate = secrets.token_hex(_HASH_CHARS // 2)
    temporary = state_dir / f".{_STATE_DIR_ID_FILE}.{os.getpid()}.{candidate}"
    temporary.write_text(candidate)
    try:
        # A hard link never replaces an existing file, so of two processes racing here exactly one wins.
        os.link(temporary, path)
    except FileExistsError:
        pass
    finally:
        temporary.unlink(missing_ok=True)
    state_dir_id = path.read_text().strip()
    if len(state_dir_id) != _HASH_CHARS or any(char not in "0123456789abcdef" for char in state_dir_id):
        raise ValueError(f"{path} does not hold a state directory ID of {_HASH_CHARS} hex digits: {state_dir_id!r}")
    return state_dir_id


class _SnapshotIndex:
    """The snapshots one server created, by the checkpoint whose export created them, as empty files.

    ``<root>/<checkpoint id>/<snapshot id>`` exists from the moment the service returned the snapshot's id until the
    service confirmed it deleted. Marks and unmarks are one or two filesystem calls, so several processes, such as a
    crashed process's replacement or another worker, can mark without locks; releases hold ``<root>.lock``, so two
    never delete the same snapshots at once.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        self.lock_path = root.with_name(root.name + ".lock")

    def mark(self, checkpoint_id: str, snapshot_id: str) -> None:
        folder = self.root / _path_component(checkpoint_id, "checkpoint id")
        path = folder / _path_component(snapshot_id, "snapshot id")
        # An unmark of another snapshot of the checkpoint may remove the folder while it is empty, between the
        # mkdir and the touch here: make it again.
        for attempt in range(_MARK_ATTEMPTS):
            try:
                folder.mkdir(parents=True, exist_ok=True)
                path.touch()
                return
            except (FileExistsError, FileNotFoundError):
                if attempt == _MARK_ATTEMPTS - 1:
                    raise

    def unmark(self, checkpoint_id: str, snapshot_id: str) -> None:
        (self.root / checkpoint_id / snapshot_id).unlink(missing_ok=True)
        self.remove_if_empty(checkpoint_id)

    def remove_if_empty(self, checkpoint_id: str) -> None:
        with contextlib.suppress(OSError):  # not empty, or already gone
            (self.root / checkpoint_id).rmdir()

    def checkpoint_count(self) -> int:
        """How many checkpoints have an indexed snapshot: the ones no release has dropped yet."""
        try:
            return sum(1 for folder in self.root.iterdir() if folder.is_dir())
        except FileNotFoundError:
            return 0

    def snapshot_ids(self, *, excluding: frozenset[str] = frozenset()) -> dict[str, list[str]]:
        """Every indexed snapshot, by checkpoint; the folders of the checkpoints in ``excluding`` are not read."""
        if not self.root.is_dir():
            return {}
        indexed: dict[str, list[str]] = {}
        for folder in self.root.iterdir():
            if folder.name in excluding:
                continue
            try:
                indexed[folder.name] = sorted(path.name for path in folder.iterdir())
            except (FileNotFoundError, NotADirectoryError):
                # Not a checkpoint's folder, or one an unmark emptied and removed since it was listed: nothing
                # is indexed under it.
                continue
        return indexed

    @contextlib.asynccontextmanager
    async def locked(self) -> AsyncIterator[None]:
        """Hold the release lock for the block.

        The lock is tried without blocking and tried again after a pause while another release holds it, so the
        event loop keeps running and a cancelled wait leaves nothing behind: closing the descriptor gives the lock
        back wherever the block ends. Raises ``RuntimeError`` naming the lock file if the filesystem cannot lock.
        """
        try:
            descriptor = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        except OSError as error:
            raise RuntimeError(f"cannot open the snapshot index lock {self.lock_path}: {error}") from error
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    await asyncio.sleep(_LOCK_POLL_S)
                except OSError as error:
                    raise RuntimeError(
                        f"cannot lock the snapshot index {self.lock_path}: {error}; checkpoint.state_dir must be on "
                        "a filesystem that supports flock (Lustre: mount with -o flock)"
                    ) from error
            yield
        finally:
            os.close(descriptor)


@dataclass
class _Entry:
    sandbox: AsyncSandbox
    spec: SandboxSpec


class SandboxSessionCheckpointer:
    """The sandboxes one server owns, by key, and their checkpoint operations.

    ``state_dir`` and ``server_name`` place the snapshot index at ``<state_dir>/<server_name>/snapshots``; see the
    module docstring for what ``state_dir`` must be. Without a ``state_dir``, for a server whose checkpointing is off,
    sandboxes are tracked and stopped but cannot be exported. ``parallelism`` bounds concurrent snapshot submits,
    creates, stops and deletes against the backend; size it with the commit deadline.
    """

    def __init__(
        self,
        provider: SandboxProvider,
        *,
        state_dir: str | os.PathLike[str] | None,
        server_name: str,
        parallelism: int = 16,
        snapshot_attempts: int = 3,
        snapshot_retry_delay_s: float = 5.0,
    ) -> None:
        config = SandboxCheckpointConfig(
            parallelism=parallelism,
            snapshot_attempts=snapshot_attempts,
            snapshot_retry_delay_s=snapshot_retry_delay_s,
        )
        self._provider = provider
        self._server_name = _path_component(server_name, "server name")
        self._snapshot_attempts = config.snapshot_attempts
        self._snapshot_retry_delay_s = config.snapshot_retry_delay_s
        self._semaphore = asyncio.Semaphore(config.parallelism)
        self._entries: dict[str, _Entry] = {}
        self._index: Optional[_SnapshotIndex] = None
        self._state_dir_id: Optional[str] = None
        if state_dir is not None:
            # Fails at server start, not at the first commit, if the directory cannot be created.
            self._state_dir_id = read_state_dir_id(Path(state_dir))
            self._index = _SnapshotIndex(Path(state_dir) / self._server_name / "snapshots")
            self._index.root.mkdir(parents=True, exist_ok=True)
        # Keys whose sandbox is being created or restored: no other create, add or restore may take them meanwhile.
        self._pending: set[str] = set()
        # Forks a restore created and will not install, from before their stop until it succeeds: a stop that
        # fails or is cancelled leaves its fork here for ``settle``, the next restore and ``aclose`` to stop again.
        self._unstopped_forks: list[AsyncSandbox] = []
        # One stop per fork at a time, whichever of those retries it.
        self._fork_stops = asyncio.Lock()
        self._background: set[asyncio.Task[Any]] = set()
        self._unreleased_warning_at = _UNRELEASED_CHECKPOINTS_WARNING

    @classmethod
    def from_config(
        cls,
        provider: SandboxProvider,
        config: SandboxCheckpointConfig,
        *,
        state_dir: str | os.PathLike[str] | None,
        server_name: str,
    ) -> "SandboxSessionCheckpointer":
        """Build a checkpointer from a server's ``SandboxCheckpointConfig``; the rest as in the constructor."""
        return cls(provider, state_dir=state_dir, server_name=server_name, **config.model_dump())

    @property
    def provider(self) -> SandboxProvider:
        """The shared provider every tracked sandbox was created with."""
        return self._provider

    @property
    def keys(self) -> list[str]:
        """The keys that currently have a sandbox, including one whose stop failed and awaits a retry."""
        return list(self._entries)

    @property
    def indexed_snapshots(self) -> dict[str, list[str]]:
        """Every snapshot in the index, by checkpoint: the ones no release has deleted yet. Reads the disk."""
        return self._index.snapshot_ids() if self._index is not None else {}

    @property
    def snapshot_name_prefix(self) -> Optional[str]:
        """The name prefix of every snapshot this server takes under its ``state_dir``; ``None`` without one."""
        if self._state_dir_id is None:
            return None
        return f"ng-{self._state_dir_id}-{name_hash(self._server_name)}-"

    def snapshot_name(self, checkpoint_id: str, key: str, attempt: int) -> str:
        """The name of attempt ``attempt`` to snapshot ``key``'s sandbox for ``checkpoint_id``."""
        prefix = self.snapshot_name_prefix
        if prefix is None:
            raise ValueError("snapshot names need a state_dir")
        return f"{prefix}{name_hash(checkpoint_id)}-{name_hash(key)}-{attempt}"

    def __contains__(self, key: object) -> bool:
        """Whether ``key`` currently has a sandbox."""
        return key in self._entries

    def get(self, key: str) -> Optional[AsyncSandbox]:
        """``key``'s current sandbox, or ``None``; after a restore it is the fork, under a new id."""
        entry = self._entries.get(key)
        return entry.sandbox if entry is not None else None

    def add(self, key: str, sandbox: AsyncSandbox, spec: SandboxSpec) -> None:
        """Track a sandbox the owner started itself; ``spec`` is what a restore re-creates it from.

        The sandbox must run on this checkpointer's provider without owning it
        (``AsyncSandbox(checkpointer.provider, spec, owns_provider=False)``): the provider is shared by every
        sandbox, and a sandbox that owned it would close it for all of them when stopped.
        """
        if key in self._entries or key in self._pending:
            raise ValueError(f"{key!r} already has a sandbox")
        if sandbox.owns_provider:
            raise ValueError(
                f"the sandbox of {key!r} owns its provider; create it with owns_provider=False on the checkpointer's "
                "provider"
            )
        if sandbox.provider is not self._provider:
            raise ValueError(f"the sandbox of {key!r} is not on the checkpointer's provider")
        self._entries[key] = _Entry(sandbox=sandbox, spec=spec)

    async def create(self, key: str, spec: SandboxSpec) -> AsyncSandbox:
        """Start a sandbox for ``key`` on the shared provider and track it.

        The key is taken from the first moment, so a second create for it is refused while the first is starting.
        """
        if key in self._entries or key in self._pending:
            raise ValueError(f"{key!r} already has a sandbox")
        self._pending.add(key)
        try:
            sandbox = AsyncSandbox(self._provider, spec, owns_provider=False)
            async with self._semaphore:
                await sandbox.start()
            self._entries[key] = _Entry(sandbox=sandbox, spec=spec)
        finally:
            self._pending.discard(key)
        return sandbox

    async def access(self, key: str, *, provider_config_ref: str, workdir: str) -> SandboxAccess:
        """Access to ``key``'s current sandbox for a borrower, such as an agent that runs a harness in it.

        A restore rebuilds the sandbox from its snapshot under a new id, so a borrower that kept an older access must
        be given the new one after a restore. The access lets the borrower operate the sandbox but not destroy it,
        on a provider that scopes descriptors: the owner still has to checkpoint it and stops it itself.
        """
        entry = self._entries.get(key)
        if entry is None:
            raise KeyError(f"{key!r} has no sandbox")
        descriptor = await entry.sandbox.serialize(scope="operate")
        connection = DirectSandboxConnection(provider_config_ref=provider_config_ref, descriptor=descriptor)
        return SandboxAccess(connection=connection, workdir=workdir)

    async def settle(self) -> None:
        """Wait for the background work, then stop again the forks an earlier restore could not stop.

        The background work is the superseded sandboxes and forks a restore stops and the failed snapshots an
        export deletes; cancelling the wait leaves that work running. Each fork whose stop failed is then tried
        once more, so a process that never restores again still stops them; one that fails again is logged and
        stays for the next call.
        """
        while self._background:
            await asyncio.wait(set(self._background))
        await self._stop_forks()

    async def aclose(self) -> None:
        """Stop every sandbox this checkpointer is responsible for; call it at the server's shutdown.

        That is every key's sandbox and every fork a failed restore left running, after the background work has
        ended. A sandbox whose stop fails is logged and stays tracked (its key stays in :attr:`keys`), the others
        are still stopped, and calling this again tries the rest again. Snapshots are not touched: the controller
        may still restore from them, and only its release deletes them.
        """
        keys = list(self._entries)
        outcomes = await asyncio.gather(*(self.stop(key) for key in keys), return_exceptions=True)
        for key, outcome in zip(keys, outcomes):
            if isinstance(outcome, BaseException):
                LOGGER.warning("sandbox of %s was not stopped at shutdown; its lifetime ends it: %r", key, outcome)
        await self.settle()

    # -- commit -------------------------------------------------------------------------------------------------

    async def export(self, keys: Iterable[str], *, checkpoint_id: str) -> dict[str, Optional[dict[str, JsonValue]]]:
        """Snapshot each key's sandbox for checkpoint ``checkpoint_id`` and return its checkpoint state.

        A key without a live sandbox, never started or already stopped (for example after the owner captured what it
        grades), maps to ``None``: a restore then gives it no sandbox. That tells an owner which of its keys have
        nothing to restore, but it is not what the ``export_session_states`` hook returns for such a session: the
        hook leaves out a session the server no longer holds. An owner keyed by session ID must drop the ``None``
        entries before it returns this result from the hook, or a restore brings those sessions back without a
        sandbox. Each snapshot is indexed under
        ``checkpoint_id`` as soon as it exists, and every snapshot is awaited before the result or the first failure
        is returned. Raises :class:`SandboxCheckpointError` if a sandbox cannot be snapshotted in
        ``snapshot_attempts`` attempts, or a snapshot cannot be indexed, which fails the commit. Every sandbox keeps
        running, and the snapshots already taken are deleted by a release that does not keep ``checkpoint_id``.
        """
        keys = list(keys)
        outcomes = await asyncio.gather(
            *(self._export_one(key, checkpoint_id) for key in keys), return_exceptions=True
        )
        await self._warn_of_unreleased_checkpoints()
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if failures:
            raise failures[0]
        return dict(zip(keys, outcomes))

    async def _warn_of_unreleased_checkpoints(self) -> None:
        """Warn when the index keeps growing: nothing but a release deletes a snapshot.

        A controller that releases keeps the index at the one or two checkpoints it may restore. One that never
        does leaves the snapshots of every commit on the provider for good, which only shows in the provider's
        storage unless it is said here.
        """
        if self._index is None:
            return
        try:
            count = await asyncio.to_thread(self._index.checkpoint_count)
        except OSError:
            return  # the export itself reports an index that cannot be read
        if count <= _UNRELEASED_CHECKPOINTS_WARNING:
            self._unreleased_warning_at = _UNRELEASED_CHECKPOINTS_WARNING
        elif count > self._unreleased_warning_at:
            self._unreleased_warning_at = 2 * count
            LOGGER.warning(
                "the snapshot index of server %s holds %d checkpoints that no release has dropped. Nothing but a "
                "release deletes a snapshot, so each of them keeps its snapshots on the provider: check that the "
                "controller releases the checkpoints it no longer restores, and delete what a finished job left "
                "by the name prefix %s",
                self._server_name,
                count,
                self.snapshot_name_prefix,
            )

    async def _export_one(self, key: str, checkpoint_id: str) -> Optional[dict[str, JsonValue]]:
        entry = self._entries.get(key)
        if entry is None or not entry.sandbox.started:
            return None
        provider = self._provider
        if not isinstance(provider, SupportsSandboxSnapshot):
            raise SandboxCheckpointError(
                key, f"cannot be checkpointed: provider {self._provider_name()!r} does not support snapshots"
            )
        index = self._index
        if index is None:
            raise SandboxCheckpointError(key, "cannot be checkpointed without a state_dir to index its snapshot")
        sandbox = entry.sandbox
        # The exec defaults as they are now, not as the sandbox was created: setup may have changed them
        # (``set_default_user``, ``set_default_workdir``), and the fork must run commands as the same user in the
        # same directory.
        current_spec = dataclasses.replace(entry.spec, user=sandbox.default_user, workdir=sandbox.default_workdir)
        # Before any snapshot is taken: a spec that cannot be written fails every commit the same way.
        try:
            spec = _SPEC_JSON.validate_python(spec_to_json(current_spec))
        except ValidationError as error:
            raise SandboxCheckpointError(key, f"has a spec that cannot be checkpointed: {error}") from error
        if isinstance(provider, ConnectableProvider):
            # The full descriptor, not a borrower's: a restore stops the superseded sandbox with it.
            descriptor = await sandbox.serialize()
        else:
            descriptor = {"sandbox_id": sandbox.sandbox_id}
        attempt = 1
        while True:
            try:
                snapshot_id = await self._snapshot_once(key, sandbox, index, checkpoint_id, attempt)
                break
            except _RetryableSnapshotError as failure:
                if failure.snapshot_id is not None:
                    # Best effort: a failed snapshot stays indexed, and a release deletes it if this does not.
                    self._spawn(self._delete_snapshot(index, checkpoint_id, failure.snapshot_id))
                if attempt >= self._snapshot_attempts:
                    raise failure.error from failure.__cause__
                LOGGER.warning(
                    "sandbox snapshot attempt %d/%d failed; retrying: key=%s sandbox_id=%s %s",
                    attempt,
                    self._snapshot_attempts,
                    key,
                    sandbox.sandbox_id,
                    failure.error,
                )
                attempt += 1
                await asyncio.sleep(self._snapshot_retry_delay_s)
        state = SandboxCheckpointState(
            provider_name=self._provider_name(),
            descriptor=descriptor,
            snapshot_id=snapshot_id,
            spec=spec,
            snapshot_ready_ts=time.time(),
        )
        return state.model_dump(mode="json")

    async def _snapshot_once(
        self, key: str, sandbox: AsyncSandbox, index: _SnapshotIndex, checkpoint_id: str, attempt: int
    ) -> str:
        """Take one snapshot and wait until it is ready.

        Only the submit holds the backend semaphore: a commit waits for every snapshot to become ready at once. A
        failed submit or snapshot raises :class:`_RetryableSnapshotError`; failing to index one raises
        :class:`SandboxCheckpointError`.
        """
        name = self.snapshot_name(checkpoint_id, key, attempt)
        try:
            async with self._semaphore:
                snapshot_id = await sandbox.snapshot(name=name)
        except Exception as error:
            raise _RetryableSnapshotError(
                SandboxCheckpointError(key, f"could not be snapshotted: {error}"), None
            ) from error
        # Indexed before anything else happens with it: a snapshot that fails, times out, or outlives a cancelled
        # export or a crash exists on the backend all the same, and a release must find it.
        try:
            await asyncio.to_thread(index.mark, checkpoint_id, snapshot_id)
        except (OSError, ValueError) as error:
            raise SandboxCheckpointError(key, f"snapshot {snapshot_id!r} could not be indexed: {error}") from error
        try:
            await self._provider.wait_snapshot_ready(snapshot_id)
        except Exception as error:
            raise _RetryableSnapshotError(
                SandboxCheckpointError(key, f"snapshot {snapshot_id!r} did not become ready: {error}"), snapshot_id
            ) from error
        return snapshot_id

    # -- restore ------------------------------------------------------------------------------------------------

    async def restore(self, states: Mapping[str, Optional[JsonValue]]) -> None:
        """Rebuild every key's sandbox from its snapshot, all or nothing; a ``None`` state rebuilds nothing.

        Every state is validated before the backend is touched. If any sandbox cannot be rebuilt, the forks this call
        created are stopped and :class:`SandboxCheckpointError` names the key; if the call is cancelled, they are
        stopped in the background, which :meth:`settle` waits for. A fork whose stop fails or is cancelled stays
        tracked, and :meth:`settle`, the next restore and :meth:`aclose` stop it again. Once the forks are in place
        the crashed process's sandboxes are stopped in the background, best effort; a failure there is logged and
        the sandbox runs out its lifetime.
        """
        await self._stop_forks()
        provider_name = self._provider_name()
        parsed: dict[str, tuple[SandboxCheckpointState, SandboxSpec]] = {}
        for key, raw in states.items():
            if raw is None:
                continue
            version = raw.get("version", _STATE_VERSION) if isinstance(raw, Mapping) else _STATE_VERSION
            if version != _STATE_VERSION:
                raise SandboxCheckpointError(
                    key, f"has checkpoint state version {version!r}; this server reads version {_STATE_VERSION}"
                )
            try:
                state = SandboxCheckpointState.model_validate(raw)
                spec = spec_from_json(state.spec)
            except (ValidationError, TypeError, ValueError) as error:
                raise SandboxCheckpointError(key, f"has an invalid checkpoint state: {error}") from error
            if state.provider_name != provider_name:
                raise SandboxCheckpointError(
                    key, f"was created by provider {state.provider_name!r}, not {provider_name!r}"
                )
            if key in self._entries or key in self._pending:
                raise SandboxCheckpointError(key, "already has a sandbox in this process")
            if not isinstance(self._provider, SupportsSandboxSnapshot):
                raise SandboxCheckpointError(
                    key, f"cannot be restored: provider {provider_name!r} does not support snapshots"
                )
            parsed[key] = (state, spec)

        created: dict[str, AsyncSandbox] = {}
        self._pending.update(parsed)
        try:
            try:
                outcomes = await asyncio.gather(
                    *(self._fork(key, state, spec, created) for key, (state, spec) in parsed.items()),
                    return_exceptions=True,
                )
            except asyncio.CancelledError:
                # The gather cancelled the forks still starting and waited for them, so ``created`` is complete.
                self._track_unstopped(created)
                self._spawn(self._stop_forks())
                raise
            failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
            if failures:
                # Tracked before the first stop is awaited: a cancel from here on leaves every fork recorded.
                self._track_unstopped(created)
                await self._stop_forks()
                raise failures[0]
            for key, sandbox in created.items():
                self._entries[key] = _Entry(sandbox=sandbox, spec=parsed[key][1])
        finally:
            self._pending.difference_update(parsed)
        # The crashed process's sandboxes are superseded: nothing continues them. Off the critical path: the install
        # is deadline-bound and the forks are already in place.
        self._spawn(self._stop_superseded_all(parsed))

    async def _fork(
        self, key: str, state: SandboxCheckpointState, spec: SandboxSpec, created: dict[str, AsyncSandbox]
    ) -> None:
        # The snapshot replaces the image and holds the files the spec delivered.
        fork_spec = dataclasses.replace(spec, image=None, files={})
        for attempt in range(FORK_NOT_FOUND_RETRIES + 1):
            sandbox = AsyncSandbox(self._provider, fork_spec, owns_provider=False)
            try:
                async with self._semaphore:
                    await sandbox.start(snapshot_id=state.snapshot_id)
                break
            except SandboxSnapshotNotFoundError as error:
                if attempt == FORK_NOT_FOUND_RETRIES:
                    raise SandboxCheckpointError(
                        key, f"could not be re-created from snapshot {state.snapshot_id!r}: {error}"
                    ) from error
                LOGGER.warning(
                    "snapshot %s of %s not found on attempt %d; retrying in %.0fs",
                    state.snapshot_id,
                    key,
                    attempt + 1,
                    FORK_NOT_FOUND_BACKOFF_S,
                )
                await asyncio.sleep(FORK_NOT_FOUND_BACKOFF_S)
            except Exception as error:
                raise SandboxCheckpointError(
                    key, f"could not be re-created from snapshot {state.snapshot_id!r}: {error}"
                ) from error
        created[key] = sandbox

    def _track_unstopped(self, created: Mapping[str, AsyncSandbox]) -> None:
        """Record the forks of a restore that will not install them, before anything stops them."""
        self._unstopped_forks.extend(created.values())

    async def _stop_forks(self) -> None:
        """Stop every tracked fork once; one leaves the list only when its stop succeeded.

        A stop that fails is logged and its fork stays for the next call; one that is cancelled leaves it too.
        """
        async with self._fork_stops:
            forks = list(self._unstopped_forks)
            outcomes = await asyncio.gather(*(self._stop_fork(sandbox) for sandbox in forks), return_exceptions=True)
        for sandbox, outcome in zip(forks, outcomes):
            if isinstance(outcome, BaseException):
                LOGGER.warning(
                    "fork %s of a restore that failed was not stopped; settle, the next restore or aclose stops it "
                    "again: %r",
                    sandbox.sandbox_id,
                    outcome,
                )

    async def _stop_fork(self, sandbox: AsyncSandbox) -> None:
        await self._stop_sandbox(sandbox)
        self._unstopped_forks.remove(sandbox)

    async def _stop_superseded_all(self, parsed: Mapping[str, tuple[SandboxCheckpointState, SandboxSpec]]) -> None:
        await asyncio.gather(*(self._stop_superseded(key, state) for key, (state, _) in parsed.items()))

    async def _stop_superseded(self, key: str, state: SandboxCheckpointState) -> None:
        if not isinstance(self._provider, ConnectableProvider):
            return
        sandbox_id = state.descriptor.get("sandbox_id")
        try:
            async with self._semaphore:
                old = await AsyncSandbox.connect(state.descriptor, provider=self._provider, owns_provider=False)
                await old.stop()
        except Exception as error:
            # Already gone (ended by its lifetime), or unreachable: its lifetime ends it.
            LOGGER.info("superseded sandbox %s of %s was not stopped: %r", sandbox_id, key, error)

    def _spawn(self, operation: Any) -> asyncio.Task[Any]:
        task: asyncio.Task[Any] = asyncio.ensure_future(operation)
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    # -- release ------------------------------------------------------------------------------------------------

    async def delete_snapshots(self, kept: frozenset[str]) -> None:
        """Delete the snapshots of every checkpoint outside ``kept``; returns once every delete was tried.

        This is what an owner's ``delete_external_state`` hook calls for the controller's release. It stops no
        sandbox. ``kept`` is every checkpoint the controller may still restore, not the ones to delete; an empty
        set deletes every snapshot. The checkpoints to drop are read from the index here, so call it only while no
        export runs, as a participant does by admitting a release only when idle and awaiting it before it admits
        a commit. The deletion holds
        the index's file lock, so releases of other processes that share the index wait for it. It deletes the
        indexed snapshots of dropped checkpoints, then lists this server's snapshots by name and deletes those of
        dropped checkpoints the index never recorded. A snapshot the service deleted, or confirms gone, leaves the
        index; one whose delete fails stays.

        Raises :class:`SnapshotDeleteIncompleteError` if any delete failed, once every delete was tried, and
        ``RuntimeError`` if the index cannot be locked; the provider's error if the listing fails. Whatever a
        failed or cancelled call left is still indexed, and the next release deletes it.
        """
        if self._index is None:
            return
        kept = frozenset(kept)
        async with self._index.locked():
            failed = await self._delete_snapshots_locked(self._index, kept)
        if failed:
            raise SnapshotDeleteIncompleteError(failed, kept)

    async def _delete_snapshots_locked(self, index: _SnapshotIndex, kept: frozenset[str]) -> int:
        """Delete what ``kept`` drops; returns how many snapshots could not be deleted."""
        # Kept checkpoints are not read: the sweep leaves their snapshots by the checkpoint hash in the name.
        dropped = await asyncio.to_thread(functools.partial(index.snapshot_ids, excluding=kept))
        provider = self._provider
        if not isinstance(provider, SupportsSandboxSnapshot):
            return 0
        deleted = await asyncio.gather(
            *(
                self._delete_snapshot(index, checkpoint_id, snapshot_id)
                for checkpoint_id, snapshot_ids in dropped.items()
                for snapshot_id in snapshot_ids
            )
        )
        for checkpoint_id in dropped:
            await asyncio.to_thread(index.remove_if_empty, checkpoint_id)
        # Every dropped checkpoint's id, deleted or not: a delete that failed above is not tried again here.
        swept = await self._sweep_unindexed(
            index, kept, {snapshot_id for ids in dropped.values() for snapshot_id in ids}
        )
        return deleted.count(False) + swept.count(False)

    async def _sweep_unindexed(self, index: _SnapshotIndex, kept: frozenset[str], indexed: set[str]) -> list[bool]:
        """Delete this server's snapshots that the index never recorded, unless their checkpoint is kept."""
        prefix = self.snapshot_name_prefix
        assert prefix is not None
        kept_hashes = {name_hash(checkpoint_id) for checkpoint_id in kept}
        listed = await self._provider.list_snapshots(prefix)
        unindexed = []
        for snapshot in listed:
            name = _SNAPSHOT_NAME_SUFFIX.fullmatch((snapshot.name or "")[len(prefix) :])
            if name is None:
                continue  # not a name this checkpointer gives: leave it alone
            if name.group(1) in kept_hashes or snapshot.snapshot_id in indexed:
                continue
            unindexed.append(snapshot.snapshot_id)
        if unindexed:
            LOGGER.warning("deleting %d snapshots of dropped checkpoints the index never recorded", len(unindexed))
        return await asyncio.gather(*(self._delete_snapshot(index, None, snapshot_id) for snapshot_id in unindexed))

    async def _delete_snapshot(self, index: _SnapshotIndex, checkpoint_id: Optional[str], snapshot_id: str) -> bool:
        """Delete one snapshot and its index file; ``False`` if the delete failed, which keeps the file."""
        try:
            async with self._semaphore:
                await self._provider.delete_snapshot(snapshot_id)
        except SandboxSnapshotInUseError as error:
            LOGGER.info(
                "snapshot %s of checkpoint %s is still in use; the next release tries again: %s",
                snapshot_id,
                checkpoint_id,
                error,
            )
            return False
        except Exception as error:
            LOGGER.warning(
                "snapshot %s of checkpoint %s was not deleted; the next release tries again: %r",
                snapshot_id,
                checkpoint_id,
                error,
            )
            return False
        if checkpoint_id is not None:
            await asyncio.to_thread(index.unmark, checkpoint_id, snapshot_id)
        return True

    # -- stop ---------------------------------------------------------------------------------------------------

    async def stop(self, key: str) -> None:
        """Stop and forget ``key``'s sandbox; a no-op for a key without one, so a retire may repeat it.

        If the stop fails the key stays tracked, so a retried retire stops it again. The sandbox's snapshots stay: a
        checkpoint the controller may still restore can name them, and only a release knows which.
        """
        entry = self._entries.get(key)
        if entry is None:
            return
        await self._stop_sandbox(entry.sandbox)
        self._entries.pop(key, None)

    async def _stop_sandbox(self, sandbox: AsyncSandbox) -> None:
        async with self._semaphore:
            await sandbox.stop()

    def _provider_name(self) -> str:
        return getattr(self._provider, "name", type(self._provider).__name__)
