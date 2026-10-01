# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Checkpoint an evaluation run from rollout collection, the process that dispatches its rows.

The controller of a checkpoint has to be the process that dispatches episodes: only it knows which episodes
it will continue, and only it can send their replacements after a restore. For an evaluation run that is
rollout collection, so it drives Gym's coordination here.

A checkpoint:

1. stops starting new rows;
2. prepares every participant;
3. commits the rows whose ``/run`` has not replied, which are the rows the run would continue;
4. writes this run's manifest, then points ``LATEST`` at the new checkpoint, which publishes it;
5. resumes, or stops the run when the checkpoint was asked for by a preemption signal.

A run restarted with ``resume_from_cache`` restores the latest checkpoint before dispatching. Its
continued rows are sent as their next attempt, which continues them from their checkpoint; every other
unfinished row starts from its input as before. Finished rows are already in the output file, which
rollout collection writes as each row completes.

Triggers: ``SIGUSR1`` checkpoints and continues, ``SIGTERM`` checkpoints and stops (Slurm sends it before
preempting a job, for example with ``--signal=B:TERM@120``), and ``checkpoint_every_s`` checkpoints on a timer.
"""

import asyncio
import contextlib
import logging
import os
import shutil
import signal
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Optional

import orjson
from pydantic import BaseModel

from nemo_gym._checkpoint import coordination
from nemo_gym._checkpoint.settings import checkpoint_settings
from nemo_gym.episode_types import EpisodeId
from nemo_gym.server_utils import ServerClient


LOGGER = logging.getLogger(__name__)

LATEST = "LATEST"
MANIFEST = "collection.json"
# The latest published checkpoint and the one before it; a checkpoint is deleted once two newer ones exist.
KEEP_CHECKPOINTS = 2


class CollectionStopped(Exception):
    """Rollout collection checkpointed and stopped, as a preemption signal asked. Rerun to continue."""

    def __init__(self, checkpoint_dir: Optional[Path]) -> None:
        self.checkpoint_dir = checkpoint_dir
        where = f" at {checkpoint_dir}" if checkpoint_dir is not None else " (the checkpoint failed)"
        super().__init__(f"rollout collection stopped after checkpointing{where}; rerun with resume_from_cache")


class ContinuedRow(BaseModel):
    """A row whose episode a checkpoint continues."""

    rollout_id: str
    attempt: int


class CollectionManifest(BaseModel):
    checkpoint_id: str
    created_at: float
    continued: list[ContinuedRow]


class CollectionCheckpointer:
    """Checkpoint, and restore, the in-flight rows of one rollout collection run."""

    def __init__(
        self,
        checkpoint_dir: Path,
        server_client: ServerClient,
        *,
        every_s: Optional[float] = None,
        timeout_s: float = 300,
    ) -> None:
        settings = checkpoint_settings(server_client.global_config_dict)
        if settings is None:
            raise ValueError("checkpoint_dir needs checkpointing enabled in the global `checkpoint:` block")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.server_client = server_client
        self.auth_token = settings.control_auth_token
        self.every_s = every_s
        self.timeout_s = timeout_s
        self.stopped = False
        self.last_published: Optional[Path] = None
        self._dispatching = asyncio.Event()
        self._dispatching.set()
        self._lock = asyncio.Lock()
        # Rows whose /run has been sent and has not replied, by rollout ID: the episodes a checkpoint continues.
        self._in_flight: dict[str, int] = {}
        self._participants: Optional[coordination.Participants] = None
        self._background: set[asyncio.Task] = set()
        self._timer: Optional[asyncio.Task] = None
        self._collection: Optional[asyncio.Task] = None

    # -- dispatch hooks ---------------------------------------------------------------------------

    async def before_dispatch(self, rollout_id: str, attempt: int) -> None:
        """Wait while a checkpoint is open, then count the row as in flight."""
        await self._dispatching.wait()
        self._in_flight[rollout_id] = attempt

    def after_dispatch(self, rollout_id: str) -> None:
        """The row's /run replied (or failed): a checkpoint no longer continues it."""
        self._in_flight.pop(rollout_id, None)

    # -- checkpoint ---------------------------------------------------------------------------------

    async def checkpoint(self, *, stop: bool = False) -> Optional[Path]:
        """Checkpoint the in-flight rows. Returns the published checkpoint, or ``None`` if it failed.

        With ``stop``, the run stops afterwards instead of resuming; the Gym servers are expected to stop
        with it, and their leases resume them if they do not.
        """
        async with self._lock:
            if self.stopped:
                return None
            checkpoint_id = f"ckpt-{time.strftime('%Y%m%dT%H%M%S')}-{uuid.uuid4().hex[:8]}"
            deadline = time.time() + self.timeout_s
            self._dispatching.clear()
            published: Optional[Path] = None
            participants: Optional[coordination.Participants] = None
            try:
                participants = await self._discover()
                clock = time.monotonic()
                prepared = await coordination.prepare(participants, checkpoint_id, deadline_ts=deadline)
                if not prepared.prepared:
                    LOGGER.warning("checkpoint %s did not prepare in time: %s", checkpoint_id, prepared.blockers())
                else:
                    continued = [ContinuedRow(rollout_id=key, attempt=value) for key, value in self._in_flight.items()]
                    target = self.checkpoint_dir / checkpoint_id
                    await coordination.commit(
                        participants,
                        checkpoint_id,
                        str(target),
                        [EpisodeId(rollout_id=row.rollout_id, attempt=row.attempt) for row in continued],
                        deadline_ts=deadline,
                    )
                    manifest = CollectionManifest(
                        checkpoint_id=checkpoint_id, created_at=time.time(), continued=continued
                    )
                    await asyncio.to_thread(self._publish, target, manifest)
                    published = target
                    print(
                        f"Checkpoint {checkpoint_id} published: {len(continued)} in-flight rows in "
                        f"{time.monotonic() - clock:.1f}s",
                        flush=True,
                    )
            except Exception:
                LOGGER.exception("checkpoint %s failed; the run continues from its previous checkpoint", checkpoint_id)
            if stop:
                self.stopped = True
                return published
            if participants is not None:
                with contextlib.suppress(Exception):
                    # Resume also aborts a checkpoint that was not published.
                    await coordination.resume(participants, checkpoint_id, deadline_ts=time.time() + self.timeout_s)
            self._dispatching.set()
            return published

    def _publish(self, target: Path, manifest: CollectionManifest) -> None:
        """Write the manifest, then point LATEST at the checkpoint, then drop checkpoints two behind it."""
        _atomic_write(target / MANIFEST, orjson.dumps(manifest.model_dump(mode="json")))
        _atomic_write(self.checkpoint_dir / LATEST, manifest.checkpoint_id.encode())
        published = sorted(
            (path for path in self.checkpoint_dir.iterdir() if (path / MANIFEST).exists()),
            key=lambda path: CollectionManifest.model_validate_json((path / MANIFEST).read_bytes()).created_at,
        )
        for old in published[:-KEEP_CHECKPOINTS]:
            shutil.rmtree(old, ignore_errors=True)

    async def _discover(self) -> coordination.Participants:
        if self._participants is None:
            self._participants = await coordination.discover(self.server_client, auth_token=self.auth_token)
        return self._participants

    def reset(self) -> None:
        """Forget the checkpoints of an earlier run in this directory; a fresh run must not restore them."""
        (self.checkpoint_dir / LATEST).unlink(missing_ok=True)
        if self.checkpoint_dir.exists():
            for path in self.checkpoint_dir.iterdir():
                if (path / MANIFEST).exists():
                    shutil.rmtree(path, ignore_errors=True)

    # -- restore ------------------------------------------------------------------------------------

    def latest(self) -> Optional[tuple[Path, CollectionManifest]]:
        pointer = self.checkpoint_dir / LATEST
        if not pointer.exists():
            return None
        target = self.checkpoint_dir / pointer.read_text().strip()
        return target, CollectionManifest.model_validate_json((target / MANIFEST).read_bytes())

    async def restore(self, unfinished: Iterable[str]) -> dict[str, int]:
        """Restore the latest checkpoint's rows that have not finished; return the attempt to send each as.

        A restored row continues as its next attempt. If the restore fails, coordination has already retired
        that attempt everywhere, so the row starts from its input one attempt later.
        """
        found = self.latest()
        if found is None:
            return {}
        target, manifest = found
        pending = set(unfinished)
        continued = [row for row in manifest.continued if row.rollout_id in pending]
        if not continued:
            return {}
        participants = await self._discover()
        restore_id = f"restore-{uuid.uuid4().hex[:12]}"
        deadline = time.time() + self.timeout_s
        episodes = [EpisodeId(rollout_id=row.rollout_id, attempt=row.attempt) for row in continued]
        try:
            await coordination.restore(participants, restore_id, str(target), episodes, deadline_ts=deadline)
            await coordination.resume(participants, restore_id, deadline_ts=deadline)
        except coordination.CoordinationError:
            LOGGER.exception("restoring %s failed; its %d rows restart from their input", target, len(continued))
            return {row.rollout_id: row.attempt + 2 for row in continued}
        print(
            f"Restored {len(continued)} in-flight rows from {target}; they continue as their next attempt", flush=True
        )
        return {row.rollout_id: row.attempt + 1 for row in continued}

    # -- triggers -----------------------------------------------------------------------------------

    def start(self, collection: asyncio.Task) -> None:
        """Install the signal and timer triggers for the collection task."""
        self._collection = collection
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGUSR1, self._spawn, lambda: self.checkpoint())
        loop.add_signal_handler(signal.SIGTERM, self._spawn, self._checkpoint_and_stop)
        if self.every_s is not None:
            self._timer = asyncio.create_task(self._every())
        _atomic_write(self.checkpoint_dir / "collector.pid", str(os.getpid()).encode())

    async def close(self) -> None:
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            loop.remove_signal_handler(signum)
        if self._timer is not None:
            self._timer.cancel()
        for task in list(self._background):
            # A checkpoint already under way finishes; a run that ends normally does not need it.
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await task

    def _spawn(self, make: Callable[[], Any]) -> None:
        task = asyncio.create_task(make())
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    async def _checkpoint_and_stop(self) -> None:
        self.last_published = await self.checkpoint(stop=True)
        if self._collection is not None:
            self._collection.cancel()

    async def _every(self) -> None:
        while True:
            await asyncio.sleep(self.every_s)
            if self._in_flight:
                await self.checkpoint()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
