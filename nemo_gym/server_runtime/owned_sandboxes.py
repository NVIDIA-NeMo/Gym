# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The sandboxes a server owns: their start, lending, stop and partial-rollout checkpoints.

Every sandbox has one owner, the server that starts it. Each resources, environment and agent server holds an
:class:`OwnedSandboxes` as ``SimpleServer.sandboxes`` to start, lend and stop the sandboxes it owns; servers move onto
it one family at a time. It knows sandboxes by an opaque key, the owner's session ID or a name within it such as ``session/grader``, and nothing
about the owner's protocol. Anyone else only borrows a sandbox, through the operate-scope access ``access`` gives.

A sandbox is also session state outside Gym's process. An owner that checkpoints implements its participant's
checkpoint hooks (``export_session_states`` and friends from ``nemo_gym.server_runtime._checkpoint``) by delegating
to the same instance.

The restore point of a sandbox is an explicit snapshot taken at commit: never the live sandbox, which keeps changing
once the episode continues, and never a paused state, which the backend may consume on resume.

A sandbox and a snapshot are two objects outside Gym with two lifetimes.

A sandbox lives while its owner uses it. :meth:`~OwnedSandboxes.stop` stops it when its session ends
or is retired, and :meth:`~OwnedSandboxes.aclose` stops every one left at the server's shutdown; a
restore stops the crashed process's sandboxes it supersedes. What a crashed process leaves running ends with the
lifetime the provider gives a sandbox, the spec's ``ttl_s`` at the latest. A provider whose sandboxes would outlive
a crashed process by that much should end them sooner, for example with a lease that only the creating process
renews; the provider's documentation says how. Forks made by a restore go through the same create and get the same
lifetime.

A snapshot lives while a checkpoint the controller may restore names it. No episode event deletes one: a checkpoint
taken before a rollout finished still restores that rollout. Instead the owner keeps an index on disk, an
empty file ``<state_dir>/<server>/snapshots/<checkpoint id>/<snapshot id>`` per snapshot, and the controller's
release decides, through :meth:`~OwnedSandboxes.delete_snapshots`:

- At commit, :meth:`~OwnedSandboxes.export` snapshots every live sandbox it is asked for. The moment the
  service returns a snapshot's id, its file is written, before the snapshot is awaited; then the export waits until
  each snapshot is ready. A snapshot that fails or does not become ready in time is taken again, up to
  ``snapshot_attempts`` times, and every attempt is indexed. The sandbox keeps running; the episode continues in it
  after the resume. The owner calls ``export`` only while no command uses the sandbox. A participant's prepare
  guarantees that for a resources server whose ``checkpoint_verify`` is ``"wait"``, the default. With ``"replay"``
  its ``/verify`` is admitted while a checkpoint is open, so an owner whose ``/verify`` runs commands in the
  sandbox, or stops it, must keep ``checkpoint_verify`` at ``"wait"``.
- :meth:`~OwnedSandboxes.delete_snapshots` receives every checkpoint the controller may still restore
  and deletes the snapshots indexed under any other checkpoint, removing each file once the service confirms the
  snapshot is gone. It returns only once it has tried every delete, so the participant that awaits it admits no
  commit meanwhile and a later checkpoint is never among the ones it drops. The index outlives the process, so a
  release deletes what a crashed process created too, including the snapshots of a commit that never finished. A
  failed delete keeps its file, the call raises, and the next release tries again. A release also lists this
  server's snapshots by name and deletes those of dropped checkpoints that never reached the index, such as one
  whose process died between the service's reply and the file write. That sweep relies on the provider keeping
  the name a snapshot is taken with (see :class:`~nemo_gym.sandbox.providers.base.SupportsSandboxSnapshot`): on a
  provider that drops it, such a snapshot is never found.
- After a crash, :meth:`~OwnedSandboxes.restore` rebuilds each sandbox in a fresh process from its
  snapshot (a fork), whatever happened to the live sandbox since. It validates every state first and installs all or
  nothing. The superseded sandboxes are stopped afterwards, best effort and off the restore's critical path. A
  restore deletes no snapshot.

Nothing but a release deletes a snapshot: the owner cannot tell which checkpoints the controller may still
restore, and it does not assume the provider expires snapshots. Under a controller that never releases, every
commit therefore leaves one snapshot per live sandbox, for good, and :meth:`~OwnedSandboxes.export`
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
from typing import Any, Callable, Literal

import anyio
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError

from nemo_gym.config.config_types import BaseRunServerInstanceConfig
from nemo_gym.sandbox.access import DirectSandboxConnection, SandboxAccess
from nemo_gym.sandbox.api import STOP_TIMEOUT_S, AsyncSandbox
from nemo_gym.sandbox.config import OwnerConcurrency, resolve_provider_config, resolve_provider_owner_concurrency
from nemo_gym.sandbox.providers.base import (
    ConnectableProvider,
    SandboxProvider,
    SandboxSnapshotInUseError,
    SandboxSnapshotNotFoundError,
    SandboxSpec,
    SupportsSandboxSnapshot,
    provider_capabilities,
)
from nemo_gym.sandbox.providers.registry import create_provider
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
# What follows the prefix in a name this owner gives: the checkpoint's hash, the key's hash and the attempt.
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
        description="Concurrent snapshot submits and deletes against the sandbox backend. "
        "Readiness waits are not counted: a commit waits for every snapshot at once. "
        "A server's self.sandboxes takes its start and stop limits from the provider block's owner_concurrency; "
        "an instance built with OwnedSandboxes.from_config bounds starts and stops by this value too.",
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

    ``keeps_processes`` says the provider's snapshot kept the sandbox's running processes as well as its files, so
    the fork starts with them running (``"processes"`` in the provider's ``capabilities.snapshot_keeps``). States
    written before the field existed read as files only.
    """

    model_config = ConfigDict(extra="forbid")

    version: Literal[1] = _STATE_VERSION
    provider_name: str
    descriptor: dict[str, JsonValue]
    snapshot_id: str
    spec: dict[str, JsonValue]
    snapshot_ready_ts: float
    keeps_processes: bool = False


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

    def __init__(self, error: SandboxCheckpointError, snapshot_id: str | None) -> None:
        super().__init__(str(error))
        self.error = error
        self.snapshot_id = snapshot_id


def checkpoint_state_dir(global_config: Any) -> Path | None:
    """``checkpoint.state_dir`` for a server that checkpoints sandboxes; ``None`` when checkpointing is off.

    Raises ``ValueError`` when checkpointing is on without it: the server could not delete the snapshots its
    checkpoints take, so it refuses to start rather than leak them.
    It must be absolute: a relative path would name a different directory for each server and each restart
    started from another working directory.
    """
    state_dir = _configured_state_dir(global_config)
    if state_dir is None and checkpoint_settings(global_config) is not None:
        raise ValueError(
            "checkpoint.state_dir is required to checkpoint sandboxes: it is where the server records the "
            "snapshots its checkpoints take, so that a release can delete them after a crash too"
        )
    return state_dir


def _configured_state_dir(global_config: Any) -> Path | None:
    """``checkpoint.state_dir`` when checkpointing is on and sets one; ``None`` otherwise. Refuses a relative path."""
    settings = checkpoint_settings(global_config)
    if settings is None or not settings.state_dir:
        return None
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


def _open_index(state_dir: Path, server_name: str) -> tuple[str, "_SnapshotIndex"]:
    """The ID of ``state_dir`` and ``server_name``'s snapshot index in it, whose directory this creates if needed."""
    state_dir_id = read_state_dir_id(state_dir)
    index = _SnapshotIndex(state_dir / server_name / "snapshots")
    index.root.mkdir(parents=True, exist_ok=True)
    return state_dir_id, index


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


def _optional_semaphore(limit: int | None) -> asyncio.Semaphore | None:
    return asyncio.Semaphore(limit) if limit is not None else None


def _settle_s(spec: SandboxSpec) -> float:
    """How long a cancelled caller waits for a start to end: the spec's readiness wait, then a stop's limit."""
    return (spec.ready_timeout_s or 0) + STOP_TIMEOUT_S


async def _start_to_the_end(sandbox: AsyncSandbox, *, settle_s: float, **start_kwargs: Any) -> None:
    """Start ``sandbox``; if the caller is cancelled meanwhile, let the start end and stop what it started.

    A provider create cut short can leave a sandbox with no handle to stop, and not every provider applies ``ttl_s``
    (Enroot does not). So a cancelled caller waits for the start, shielded through any further cancellation and
    bounded by ``settle_s``, stops the sandbox it started, and only then lets the cancellation go on. A start still
    running after ``settle_s`` is cancelled, and its ``ttl_s`` is all that reclaims what it made.
    """
    start = asyncio.ensure_future(sandbox.start(**start_kwargs))
    try:
        await asyncio.shield(start)
    except asyncio.CancelledError:
        deadline = time.monotonic() + settle_s
        while not start.done() and (remaining := deadline - time.monotonic()) > 0:
            with contextlib.suppress(asyncio.CancelledError), anyio.CancelScope(shield=True):
                await asyncio.wait({start}, timeout=remaining)
        with anyio.CancelScope(shield=True):
            if not start.done():
                start.cancel()
                with anyio.move_on_after(STOP_TIMEOUT_S):
                    await asyncio.wait({start})
            elif not start.cancelled() and start.exception() is None:
                try:
                    await sandbox.stop(timeout_s=STOP_TIMEOUT_S)
                except Exception as error:
                    # The caller's own rollback stops it again where it still tracks the sandbox.
                    LOGGER.warning("a sandbox started for a cancelled caller was not stopped: %r", error)
        if start.done() and not start.cancelled() and start.exception() is not None:
            # Retrieved so it is not reported as never retrieved; the caller's cancellation is what goes on.
            LOGGER.info("a cancelled caller's start failed: %r", start.exception())
        raise


@dataclass
class _Entry:
    sandbox: AsyncSandbox
    spec: SandboxSpec
    # False for a sandbox whose start failed and whose stop failed too: it stays tracked until a stop succeeds, but
    # an export never snapshots it, since its setup never finished.
    exportable: bool = True


@dataclass
class _KeyLock:
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)
    users: int = 0


class _KeyLocks:
    """One lock per key, which exists only while someone holds it or waits for it."""

    def __init__(self) -> None:
        self._locks: dict[str, _KeyLock] = {}

    @contextlib.asynccontextmanager
    async def locked(self, key: str) -> AsyncIterator[None]:
        entry = self._locks.setdefault(key, _KeyLock())
        entry.users += 1
        try:
            async with entry.lock:
                yield
        finally:
            entry.users -= 1
            if entry.users == 0:
                del self._locks[key]

    def __len__(self) -> int:
        return len(self._locks)


class OwnedSandboxes:
    """The sandboxes one server owns on one provider, by key, and their checkpoint operations.

    A server reaches its instance as ``self.sandboxes`` (see :meth:`for_server`); tests and tools may build one
    directly. ``state_dir`` and ``server_name`` place the snapshot index at ``<state_dir>/<server_name>/snapshots``;
    see the module docstring for what ``state_dir`` must be. Without a ``state_dir``, for a server whose checkpointing
    is off, sandboxes are tracked and stopped but cannot be exported. ``parallelism`` bounds concurrent snapshot
    submits and deletes against the backend; size it with the commit deadline. ``max_concurrent_starts`` bounds
    concurrent starts and forks, and ``max_concurrent_stops`` concurrent stops, each unbounded by default, so a server
    that seeds hundreds of episodes at once starts all of their sandboxes at once, and a stop never waits behind
    starts. A server sets both per provider with its block's ``owner_concurrency``.
    ``provider_config_ref`` names the provider's config block, which :meth:`access` hands to a borrower.
    With ``owns_provider``, :meth:`aclose` closes the provider too; otherwise the caller that made it closes it.
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
        provider_config_ref: str | None = None,
        owns_provider: bool = False,
        max_concurrent_starts: int | None = None,
        max_concurrent_stops: int | None = None,
    ) -> None:
        concurrency = OwnerConcurrency(starts=max_concurrent_starts, stops=max_concurrent_stops)
        config = SandboxCheckpointConfig(
            parallelism=parallelism,
            snapshot_attempts=snapshot_attempts,
            snapshot_retry_delay_s=snapshot_retry_delay_s,
        )
        self._provider = provider
        self._owns_provider = owns_provider
        self._provider_config_ref = provider_config_ref
        # Set by the server that holds this instance, so :meth:`on` can give its instance for another provider.
        self._on: Callable[[str], "OwnedSandboxes"] | None = None
        self._server_name = _path_component(server_name, "server name")
        self._snapshot_attempts = config.snapshot_attempts
        self._snapshot_retry_delay_s = config.snapshot_retry_delay_s
        self._semaphore = asyncio.Semaphore(config.parallelism)
        self._start_limit = _optional_semaphore(concurrency.starts)
        self._stop_limit = _optional_semaphore(concurrency.stops)
        # Held by a create, a stop, an export and a restore of one key, so an export never snapshots a sandbox that is
        # still starting or being stopped, a stop waits for the key's restore and then stops its fork, and each waits
        # for nothing but its own keys.
        self._key_locks = _KeyLocks()
        self._entries: dict[str, _Entry] = {}
        self._index: _SnapshotIndex | None = None
        self._state_dir_id: str | None = None
        if state_dir is not None:
            # Fails at server start, not at the first commit, if the directory cannot be created.
            self._state_dir_id, self._index = _open_index(Path(state_dir), self._server_name)
        # Keys whose sandbox is being created or restored: no other create, add or restore may take them meanwhile.
        self._pending: set[str] = set()
        # Set while no key is pending, which is what ``aclose`` waits for before it stops the sandboxes.
        self._none_pending = asyncio.Event()
        self._none_pending.set()
        # Set by ``aclose``: from then on no sandbox is created, added or restored.
        self._closed = False
        # Forks a restore created and will not install, from before their stop until it succeeds: a stop that
        # fails or is cancelled leaves its fork here for ``settle``, the next restore and ``aclose`` to stop again.
        self._unstopped_forks: list[AsyncSandbox] = []
        # One stop per fork at a time, whichever of those retries it.
        self._fork_stops = asyncio.Lock()
        self._background: set[asyncio.Task[Any]] = set()
        self._unreleased_warning_at = _UNRELEASED_CHECKPOINTS_WARNING
        # The checkpoint whose export last checked the index for checkpoints no release dropped.
        self._unreleased_checked_for: str | None = None

    @classmethod
    def from_config(
        cls,
        provider: SandboxProvider,
        config: SandboxCheckpointConfig,
        *,
        state_dir: str | os.PathLike[str] | None,
        server_name: str,
    ) -> "OwnedSandboxes":
        """Build an instance from a server's ``SandboxCheckpointConfig``; the rest as in the constructor.

        ``parallelism`` also bounds its starts, forks and stops, as it did before they had limits of their own, since
        such a server has no ``owner_concurrency`` to set them.
        """
        return cls(
            provider,
            state_dir=state_dir,
            server_name=server_name,
            max_concurrent_starts=config.parallelism,
            max_concurrent_stops=config.parallelism,
            **config.model_dump(),
        )

    @classmethod
    def for_server(
        cls,
        *,
        config: BaseRunServerInstanceConfig,
        provider_field: str,
        global_config: Mapping[str, Any],
        on: Callable[[str], "OwnedSandboxes"] | None = None,
    ) -> "OwnedSandboxes":
        """The instance a server holds for the provider its config names in ``provider_field``.

        ``config`` is the server's config and ``global_config`` the merged global config, which resolves a named
        provider and ``checkpoint.state_dir``. An instance for a field other than ``sandbox_provider`` gets its own
        server name, ``<server>.<field>``, so its snapshot index and snapshot names stay apart. Without a
        ``checkpoint.state_dir`` the instance has no snapshot index: it creates, lends and stops sandboxes, and only
        :meth:`export` refuses, so a server that never exports, such as an agent, needs none. The instance owns the
        provider it creates and closes it in :meth:`aclose`. ``on`` gives the server's instance for another field,
        which :meth:`on` calls.

        Raises:
            ValueError: If the config has no field ``provider_field`` or leaves it unset, or ``checkpoint.state_dir``
                is relative.
            OSError: If ``checkpoint.state_dir`` cannot be written.
        """
        if provider_field not in type(config).model_fields:
            raise ValueError(f"{config.name} has no config field {provider_field!r} naming a sandbox provider")
        reference = getattr(config, provider_field, None)
        if reference is None:
            raise ValueError(f"{config.name} sets no {provider_field}, so it cannot own sandboxes on it")
        # Everything that can refuse the config runs before the provider exists, so a refusal leaks no provider.
        provider_config = resolve_provider_config(reference, global_config)
        concurrency = resolve_provider_owner_concurrency(reference, global_config)
        state_dir = _configured_state_dir(global_config)
        suffix = "" if provider_field == "sandbox_provider" else f".{provider_field}"
        server_name = _path_component(f"{config.name}{suffix}", "server name")
        if state_dir is not None:
            _open_index(state_dir, server_name)
        if "local" in provider_config:
            LOGGER.warning(
                "%s owns sandboxes on the local provider, which runs their commands as processes on this host; "
                "use a container provider for anything that runs model-generated code",
                config.name,
            )
        provider = create_provider(provider_config)
        owned = cls(
            provider,
            state_dir=state_dir,
            server_name=server_name,
            provider_config_ref=reference if isinstance(reference, str) else None,
            owns_provider=True,
            max_concurrent_starts=concurrency.starts,
            max_concurrent_stops=concurrency.stops,
        )
        owned._on = on
        return owned

    def on(self, provider_field: str) -> "OwnedSandboxes":
        """The same server's instance for the provider its config names in ``provider_field``, created on first use.

        Raises:
            RuntimeError: If no server holds this instance.
        """
        if self._on is None:
            raise RuntimeError("only an instance a server holds as self.sandboxes knows the server's other providers")
        return self._on(provider_field)

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
    def snapshot_name_prefix(self) -> str | None:
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

    def get(self, key: str) -> AsyncSandbox | None:
        """``key``'s current sandbox, or ``None``; after a restore it is the fork, under a new id."""
        entry = self._entries.get(key)
        return entry.sandbox if entry is not None else None

    def add(self, key: str, sandbox: AsyncSandbox, spec: SandboxSpec) -> None:
        """Track a sandbox the owner started itself; ``spec`` is what a restore re-creates its fork from.

        The default user and working directory are the exception: an export reads them from the sandbox, so a fork
        runs commands as the sandbox did when it was exported, and takes only the rest of its spec from ``spec``.
        The sandbox must run on this instance's provider without owning it
        (``AsyncSandbox(owned.provider, spec, owns_provider=False)``): the provider is shared by every
        sandbox, and a sandbox that owned it would close it for all of them when stopped.
        """
        self._require_open()
        if key in self._entries or key in self._pending:
            raise ValueError(f"{key!r} already has a sandbox")
        if sandbox.owns_provider:
            raise ValueError(
                f"the sandbox of {key!r} owns its provider; create it with owns_provider=False on the instance's "
                "provider"
            )
        if sandbox.provider is not self._provider:
            raise ValueError(f"the sandbox of {key!r} is not on this instance's provider")
        self._entries[key] = _Entry(sandbox=sandbox, spec=spec)

    async def create(self, key: str, spec: SandboxSpec) -> AsyncSandbox:
        """Start a sandbox for ``key`` on the shared provider and track it.

        The key is taken from the first moment, so a second create for it is refused while the first is starting. A
        create of a key that an export is snapshotting waits for that export; creates of other keys do not.

        The sandbox is tracked before it starts. A start that fails or is cancelled is stopped here, shielded from the
        cancellation. A start cancelled before the provider returned the sandbox is first let finish, shielded and
        bounded by ``STOP_TIMEOUT_S``: a provider create cut short can leave a sandbox with no handle to stop, and not
        every provider applies ``ttl_s`` (Enroot does not). Only a start still running after that is cut short, and
        the spec's ``ttl_s`` is all that reclaims what it made. If the stop fails too, the key stays tracked, so the
        next :meth:`stop` of the key or :meth:`aclose` stops it again, an export leaves it out, and the start's error
        is raised with the stop's as its context.

        Raises:
            ValueError: If ``key`` already has a sandbox, or one is starting, or ``spec`` sets no ``ttl_s``.
        """
        self._require_open()
        if spec.ttl_s is None:
            # The shutdown sweep is bounded and a crash runs none: ttl_s is what ends a sandbox its owner never stops.
            raise ValueError(f"the sandbox of {key!r} needs a ttl_s, so the provider ends it if its owner never does")
        if key in self._entries or key in self._pending:
            raise ValueError(f"{key!r} already has a sandbox")
        self._take_keys([key])
        try:
            async with self._key_locks.locked(key):
                sandbox = AsyncSandbox(self._provider, spec, owns_provider=False)
                entry = _Entry(sandbox=sandbox, spec=spec)
                self._entries[key] = entry
                try:
                    async with self._start_slot():
                        # A create queued for a slot when the shutdown began starts nothing.
                        self._require_open()
                        await _start_to_the_end(sandbox, settle_s=_settle_s(spec))
                except BaseException as error:
                    entry.exportable = False
                    try:
                        await self._stop_key(key)
                    except Exception as stop_error:
                        LOGGER.warning(
                            "sandbox of %s failed to start and was not stopped; the next stop of the key or aclose "
                            "stops it again: %r",
                            key,
                            stop_error,
                        )
                        # The start's error, with the failed stop as its context.
                        raise error
                    raise
        finally:
            self._give_back_keys([key])
        return sandbox

    def _start_slot(self) -> contextlib.AbstractAsyncContextManager[Any]:
        return self._start_limit if self._start_limit is not None else contextlib.nullcontext()

    def _stop_slot(self) -> contextlib.AbstractAsyncContextManager[Any]:
        return self._stop_limit if self._stop_limit is not None else contextlib.nullcontext()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("the owned sandboxes are closed: no sandbox is started, added or restored any more")

    def _take_keys(self, keys: Iterable[str]) -> None:
        self._pending.update(keys)
        if self._pending:
            self._none_pending.clear()

    def _give_back_keys(self, keys: Iterable[str]) -> None:
        self._pending.difference_update(keys)
        if not self._pending:
            self._none_pending.set()

    async def access(self, key: str, *, workdir: str, provider_config_ref: str | None = None) -> SandboxAccess:
        """Access to ``key``'s current sandbox for a borrower, such as an agent that runs a harness in it.

        A restore rebuilds the sandbox from its snapshot under a new id, so a borrower that kept an older access must
        be given the new one after a restore. The access lets the borrower operate the sandbox but not destroy it,
        on a provider that scopes descriptors: the owner still has to checkpoint it and stops it itself.
        ``provider_config_ref`` defaults to the one this instance was built with.

        Raises:
            KeyError: If ``key`` has no running sandbox: none, one still starting, or one whose start failed and whose
                stop failed too.
            ValueError: If neither this call nor the instance names the provider's config block, as for a server
                whose ``sandbox_provider`` is an inline block rather than a named one.
        """
        entry = self._entries.get(key)
        # A key is tracked from before its start, so being tracked does not mean there is a sandbox to lend.
        if entry is None or not entry.sandbox.started or not entry.exportable:
            raise KeyError(f"{key!r} has no running sandbox")
        provider_config_ref = provider_config_ref or self._provider_config_ref
        if provider_config_ref is None:
            raise ValueError("lending needs provider_config_ref: the borrower resolves the provider by that name")
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
        """Stop every sandbox this instance owns; call it at the server's shutdown.

        That is every key's sandbox and every fork a failed restore left running, after the background work has
        ended. The instance is closed first: ``create``, ``add`` and ``restore`` raise ``RuntimeError`` from
        then on, and the ones already running are waited for, so a restore that installs its forks during the
        shutdown has them stopped here. A sandbox whose stop fails or takes longer than ``STOP_TIMEOUT_S`` is
        logged and stays tracked (its key stays in :attr:`keys`), the others are still stopped, and calling this
        again tries the rest again. An instance that owns its provider closes it once every sandbox is stopped.
        Snapshots are not touched: the controller may still restore from them, and only its release deletes them.
        """
        self._closed = True
        # What runs now is stopped first: a create cancelled at shutdown may take its whole start to settle.
        running = [key for key in self._entries if key not in self._pending]
        await self._stop_at_shutdown(running)
        await self._none_pending.wait()
        # Each key is tried once per call; one whose stop failed stays for the next call.
        await self._stop_at_shutdown([key for key in self._entries if key not in running])
        await self.settle()
        if self._owns_provider and not self._entries and not self._unstopped_forks:
            # Last, and only once nothing is left to stop, so a later call can still retry a failed stop. Closing
            # runs the provider's own cleanup too, such as reaping a create the shutdown abandoned.
            self._owns_provider = False
            await self._provider.aclose()

    async def _stop_at_shutdown(self, keys: list[str]) -> None:
        outcomes = await asyncio.gather(*(self.stop(key) for key in keys), return_exceptions=True)
        for key, outcome in zip(keys, outcomes):
            if isinstance(outcome, BaseException):
                LOGGER.warning("sandbox of %s was not stopped at shutdown; its lifetime ends it: %r", key, outcome)

    # -- commit -------------------------------------------------------------------------------------------------

    async def export(self, keys: Iterable[str], *, checkpoint_id: str) -> dict[str, dict[str, JsonValue] | None]:
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
            *(self._export_locked(key, checkpoint_id) for key in keys), return_exceptions=True
        )
        await self._warn_of_unreleased_checkpoints(checkpoint_id)
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        if failures:
            raise failures[0]
        return dict(zip(keys, outcomes))

    async def _warn_of_unreleased_checkpoints(self, checkpoint_id: str) -> None:
        """Warn when the index keeps growing: nothing but a release deletes a snapshot.

        A controller that releases keeps the index at the one or two checkpoints it may restore. One that never
        does leaves the snapshots of every commit on the provider for good, which only shows in the provider's
        storage unless it is said here. The index is listed once per checkpoint, however many exports the owner
        makes for it: the index is on a filesystem every node reads.
        """
        if self._index is None or checkpoint_id == self._unreleased_checked_for:
            return
        self._unreleased_checked_for = checkpoint_id
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

    async def _export_locked(self, key: str, checkpoint_id: str) -> dict[str, JsonValue] | None:
        # After the key's create or stop, if one is running; a create or stop of the key waits for the export.
        async with self._key_locks.locked(key):
            return await self._export_one(key, checkpoint_id)

    async def _export_one(self, key: str, checkpoint_id: str) -> dict[str, JsonValue] | None:
        entry = self._entries.get(key)
        if entry is None or not entry.exportable or not entry.sandbox.started:
            return None
        provider = self._provider
        if not isinstance(provider, SupportsSandboxSnapshot):
            raise SandboxCheckpointError(
                key, f"cannot be checkpointed: provider {self._provider_name()!r} does not support snapshots"
            )
        index = self._index
        if index is None:
            raise SandboxCheckpointError(
                key,
                "cannot be checkpointed without a state_dir to index its snapshot: set checkpoint.state_dir, where "
                "the server records its snapshots so that a release can delete them",
            )
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
            keeps_processes="processes" in provider_capabilities(self._provider).snapshot_keeps,
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

    async def restore(self, states: Mapping[str, JsonValue | None]) -> None:
        """Rebuild every key's sandbox from its snapshot, all or nothing; a ``None`` state rebuilds nothing.

        Every state is validated before the backend is touched. If any sandbox cannot be rebuilt, the forks this call
        created are stopped and :class:`SandboxCheckpointError` names the key; if the call is cancelled, they are
        stopped in the background, which :meth:`settle` waits for. A fork whose stop fails or is cancelled stays
        tracked, and :meth:`settle`, the next restore and :meth:`aclose` stop it again. Once the forks are in place
        the crashed process's sandboxes are stopped in the background, best effort; a failure there is logged and
        the sandbox runs out its lifetime.
        """
        self._require_open()
        await self._stop_forks()
        # Again, since ``aclose`` may have run during the stops: nothing awaits from here until the keys are taken.
        self._require_open()
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
        self._take_keys(parsed)
        try:
            async with contextlib.AsyncExitStack() as held:
                # Every key's lock, in one order so two restores cannot deadlock: a stop of a key waits for its fork
                # to be installed, then stops it.
                for key in sorted(parsed):
                    await held.enter_async_context(self._key_locks.locked(key))
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
            self._give_back_keys(parsed)
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
                async with self._start_slot():
                    await _start_to_the_end(sandbox, settle_s=_settle_s(fork_spec), snapshot_id=state.snapshot_id)
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
            # Bounded as a whole, the wait for a slot included: a superseded sandbox's stop is best effort, and its
            # lifetime ends it if this one gives up.
            async with asyncio.timeout(STOP_TIMEOUT_S), self._stop_slot():
                old = await AsyncSandbox.connect(state.descriptor, provider=self._provider, owns_provider=False)
                await old.stop(timeout_s=STOP_TIMEOUT_S)
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
                continue  # not a name this owner gives: leave it alone
            if name.group(1) in kept_hashes or snapshot.snapshot_id in indexed:
                continue
            unindexed.append(snapshot.snapshot_id)
        if unindexed:
            LOGGER.warning("deleting %d snapshots of dropped checkpoints the index never recorded", len(unindexed))
        return await asyncio.gather(*(self._delete_snapshot(index, None, snapshot_id) for snapshot_id in unindexed))

    async def _delete_snapshot(self, index: _SnapshotIndex, checkpoint_id: str | None, snapshot_id: str) -> bool:
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

        It first waits for a create, an export or a restore of the same key to finish, and stops the fork a restore
        installed. The whole stop is shielded from the caller's cancellation, so a client that disconnects cannot
        leave the sandbox running. The wait for the key's lock is bounded by ``STOP_TIMEOUT_S``, and so is the
        provider's stop once it has a stop slot; the wait for a slot has no limit of its own, since every stop ahead
        holds its slot for at most ``STOP_TIMEOUT_S``. If the stop fails, or times out and raises ``TimeoutError``,
        the key stays tracked, so a retried retire stops it again. The sandbox's snapshots stay: a checkpoint the
        controller may still restore can name them, and only a release knows which.
        """
        with anyio.CancelScope(shield=True):
            async with contextlib.AsyncExitStack() as held:
                with anyio.fail_after(STOP_TIMEOUT_S):
                    await held.enter_async_context(self._key_locks.locked(key))
                await self._stop_key(key)

    async def _stop_key(self, key: str) -> None:
        entry = self._entries.get(key)
        if entry is None:
            return
        if entry.sandbox.started:
            await self._stop_sandbox(entry.sandbox)
        # Without a wait for a sandbox its owner already stopped: a caller cancelled there would leave the key
        # tracked with nothing left to stop.
        self._entries.pop(key, None)

    async def discard(self, keys: Iterable[str]) -> None:
        """Give up the sandboxes of ``keys`` and stop them; for the forks of a restore its owner could not install.

        Unlike :meth:`stop`, every key can be restored again at once, whether or not the stops succeed. A sandbox whose stop fails or takes longer than ``STOP_TIMEOUT_S`` is logged and joins the
        forks that :meth:`settle`, the next restore and :meth:`aclose` stop again. A key without a sandbox is skipped.
        """
        for key in keys:
            entry = self._entries.pop(key, None)
            if entry is not None:
                self._unstopped_forks.append(entry.sandbox)
        await self._stop_forks()

    async def _stop_sandbox(self, sandbox: AsyncSandbox) -> None:
        # Shielded, so a cancelled request cannot leave the sandbox running. The provider's stop is bounded once it has
        # its slot (AsyncSandbox.stop shields and bounds itself); the wait for the slot is not, so a stop queued behind
        # others under load is not cut short and left to ttl_s.
        with anyio.CancelScope(shield=True):
            async with self._stop_slot():
                await sandbox.stop(timeout_s=STOP_TIMEOUT_S)

    def _provider_name(self) -> str:
        return getattr(self._provider, "name", type(self._provider).__name__)
