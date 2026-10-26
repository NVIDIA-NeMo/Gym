# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Least-loaded endpoint placement shared by the uvicorn workers of one vllm_model server.

A rollout's first call goes to the endpoint with the fewest live rollouts, and every later call
of that rollout goes to the same endpoint so its prefix cache is reused. Any worker of a
multi-worker server may receive any call, so the placements live in a small SQLite file that all
workers of the server open. A rollout counts as live until it has made no call for ``live_s``.
"""

import asyncio
import hashlib
import logging
import os
import sqlite3
import tempfile
from dataclasses import dataclass
from threading import Lock
from time import monotonic, time
from typing import Dict, Optional


LOG = logging.getLogger(__name__)


def default_state_path(server_name: str, port: int) -> str:
    """A state file private to one server process tree.

    The workers of a multi-worker server are children of the same parent, so the parent's pid
    keeps a restarted server from inheriting the placements of an earlier one.
    """
    directory = "/dev/shm" if os.path.isdir("/dev/shm") else tempfile.gettempdir()
    safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in server_name)
    return os.path.join(directory, f"nemo_gym_least_loaded_{safe_name}_{port}_{os.getppid()}.sqlite")


@dataclass
class _Placement:
    endpoint: int
    n_endpoints: int
    written_at: float


class LeastLoadedPlacement:
    """Places rollouts on the endpoint with the fewest live rollouts, shared across worker processes."""

    def __init__(
        self, path: str, *, live_s: float, refresh_s: Optional[float] = None, busy_timeout_s: float = 0.1
    ) -> None:
        self._path = path
        self._live_s = live_s
        # A rollout's liveness only needs refreshing a few times per live window; every other call
        # of the rollout is served from this worker's cache without touching the file.
        self._refresh_s = refresh_s if refresh_s is not None else live_s / 4
        self._busy_timeout_s = busy_timeout_s
        self._cache: Dict[str, _Placement] = {}
        self._connection: Optional[sqlite3.Connection] = None
        self._lock = Lock()
        self._last_warning_at: Optional[float] = None

    async def endpoint_for(self, rollout_id: str, n_endpoints: int) -> Optional[int]:
        """The endpoint index for this rollout, or None when the shared state is unavailable."""
        cached = self._cache.get(rollout_id)
        now = monotonic()
        if cached is not None and cached.n_endpoints == n_endpoints and now - cached.written_at < self._refresh_s:
            return cached.endpoint
        try:
            endpoint = await asyncio.to_thread(self.place, rollout_id, n_endpoints)
        except sqlite3.Error as e:
            if self._last_warning_at is None or now - self._last_warning_at > 60:
                self._last_warning_at = now
                LOG.warning("least-loaded placement unavailable (%s); hashing rollouts onto endpoints", e)
            return None
        if len(self._cache) > 4096:
            self._cache = {k: v for k, v in self._cache.items() if now - v.written_at < self._live_s}
        self._cache[rollout_id] = _Placement(endpoint, n_endpoints, now)
        return endpoint

    def place(self, rollout_id: str, n_endpoints: int) -> int:
        """Record the rollout as live and return its endpoint, placing it if it has none yet.

        Blocking; call it off the event loop.
        """
        with self._lock:
            connection = self._connect()
            now = time()
            connection.execute("begin immediate")
            try:
                row = connection.execute(
                    "select endpoint, n_endpoints from placements where rollout_id = ?", (rollout_id,)
                ).fetchone()
                if row is not None and row[1] == n_endpoints:
                    connection.execute("update placements set last_seen = ? where rollout_id = ?", (now, rollout_id))
                    connection.execute("commit")
                    return row[0]
                live = dict(
                    connection.execute(
                        "select endpoint, count(*) from placements"
                        " where last_seen > ? and n_endpoints = ? group by endpoint",
                        (now - self._live_s, n_endpoints),
                    ).fetchall()
                )
                # Ties go by the rollout's hash so that concurrent placements spread instead of
                # all taking the lowest index.
                digest = int.from_bytes(hashlib.sha256(rollout_id.encode("utf-8")).digest()[:8], "big")
                endpoint = min(range(n_endpoints), key=lambda e: (live.get(e, 0), (e - digest) % n_endpoints))
                connection.execute(
                    "insert or replace into placements values (?, ?, ?, ?)", (rollout_id, endpoint, n_endpoints, now)
                )
                if digest % 512 == 0:
                    connection.execute("delete from placements where last_seen < ?", (now - 4 * self._live_s,))
                connection.execute("commit")
                return endpoint
            except BaseException:
                connection.execute("rollback")
                raise

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None:
            connection = sqlite3.connect(
                self._path, timeout=self._busy_timeout_s, isolation_level=None, check_same_thread=False
            )
            try:
                # WAL lets readers proceed while one worker writes. The file is scratch state for one
                # run, so durability across a crash is not needed.
                connection.execute("pragma journal_mode = wal")
                connection.execute("pragma synchronous = off")
                connection.execute(
                    "create table if not exists placements (rollout_id text primary key, endpoint integer,"
                    " n_endpoints integer, last_seen real)"
                )
                connection.execute("create index if not exists placements_last_seen on placements (last_seen)")
            except sqlite3.Error:
                # Workers start together and may race to set up a new file; the next call retries.
                connection.close()
                raise
            self._connection = connection
        return self._connection
