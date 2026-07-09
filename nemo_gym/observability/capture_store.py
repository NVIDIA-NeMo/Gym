# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LocalJsonlCaptureStore — per-rollout, append-only capture files on the
model server's local disk. This is a plain library plus files (the HTTP read
routes live in capture_reader.py); it is not a separate service or actor.

What it guarantees:
  - append() is atomic and totally ordered within a rollout (a file lock on
    <rid>.seq assigns the sequence number)
  - MANIFEST.json records an instance id so a reader can confirm a local
    directory is the same store it is talking to over HTTP before reading it
    directly
  - string payloads larger than 64 KiB are written to blobs/<sha256> and
    referenced by hash, so they stay resolvable over HTTP
  - WriteTracker.drain(rid) lets a reader wait for all in-flight writes for a
    rollout to finish: writes can happen in the background off the request
    path, but the trajectory builder and rollout-record merge await drain()
    before reading.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import time
import uuid
from typing import Any, Iterator


BLOB_THRESHOLD_BYTES = 64 * 1024
SCHEMA_VERSION = 1


class LocalJsonlCaptureStore:
    def __init__(self, capture_dir: str, run_id: str = "run", fsync_per_record: bool = True):
        self.dir = capture_dir
        self.fsync_per_record = fsync_per_record
        os.makedirs(os.path.join(self.dir, "blobs"), exist_ok=True)
        self.manifest_path = os.path.join(self.dir, "MANIFEST.json")
        if not os.path.exists(self.manifest_path):
            tmp = self.manifest_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"run_id": run_id, "instance_id": uuid.uuid4().hex, "schema_version": SCHEMA_VERSION}, f)
            os.replace(tmp, self.manifest_path)
        with open(self.manifest_path) as f:
            self.manifest: dict = json.load(f)

    def _jsonl(self, rid: str) -> str:
        return os.path.join(self.dir, f"{rid}.jsonl")

    def _seqf(self, rid: str) -> str:
        return os.path.join(self.dir, f"{rid}.seq")

    def blob_path(self, sha: str) -> str:
        return os.path.join(self.dir, "blobs", sha)

    def _spill(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._spill(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._spill(v) for v in obj]
        if isinstance(obj, str):
            raw = obj.encode()
            if len(raw) > BLOB_THRESHOLD_BYTES:
                sha = hashlib.sha256(raw).hexdigest()
                p = self.blob_path(sha)
                if not os.path.exists(p):
                    tmp = f"{p}.{uuid.uuid4().hex}.tmp"
                    with open(tmp, "wb") as f:
                        f.write(raw)
                    os.replace(tmp, p)
                return {"$blob": sha, "bytes": len(raw)}
        return obj

    def append(self, record) -> int:
        """Synchronous, atomic, per-rollout-ordered. Safe across processes."""
        rid = record.rollout_id
        data = self._spill(record.model_dump(exclude={"seq", "kind"}))
        seq_fd = os.open(self._seqf(rid), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(seq_fd, fcntl.LOCK_EX)
            raw = os.read(seq_fd, 32)
            seq = int(raw) + 1 if raw else 0
            env = {"v": SCHEMA_VERSION, "kind": record.kind, "seq": seq, "ts": time.time(), "data": data}
            with open(self._jsonl(rid), "a") as f:
                f.write(json.dumps(env, separators=(",", ":"), default=str) + "\n")
                f.flush()
                if self.fsync_per_record:
                    os.fsync(f.fileno())
            os.lseek(seq_fd, 0, os.SEEK_SET)
            os.ftruncate(seq_fd, 0)
            os.write(seq_fd, str(seq).encode())
            if self.fsync_per_record:
                os.fsync(seq_fd)
            return seq
        finally:
            fcntl.flock(seq_fd, fcntl.LOCK_UN)
            os.close(seq_fd)

    def read_envelopes(self, rid: str, kinds: set[str] | None = None, after_seq: int = -1) -> Iterator[dict]:
        path = self._jsonl(rid)
        if not os.path.exists(path):
            return
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                env = json.loads(line)
                if env["seq"] <= after_seq:
                    continue
                if kinds and env["kind"] not in kinds:
                    continue
                yield env

    def resolve_blob(self, sha: str) -> bytes:
        with open(self.blob_path(sha), "rb") as f:
            return f.read()

    def summary(self, rid: str) -> dict:
        counts: dict[str, int] = {}
        last = -1
        tokens_in = tokens_out = 0
        for env in self.read_envelopes(rid):
            counts[env["kind"]] = counts.get(env["kind"], 0) + 1
            last = max(last, env["seq"])
            if env["kind"] == "model_call":
                tokens_in += int(env["data"].get("tokens_in") or 0)
                tokens_out += int(env["data"].get("tokens_out") or 0)
        return {
            "rollout_id": rid,
            "counts": counts,
            "last_seq": last,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "instance_id": self.manifest["instance_id"],
        }

    def delete(self, rid: str) -> None:
        for p in (self._jsonl(rid), self._seqf(rid)):
            if os.path.exists(p):
                os.remove(p)


class WriteTracker:
    """Tracks in-flight background appends per rollout so a reader can wait for
    them to finish. Writes happen off the request path; a reader calls
    `await tracker.drain(rid)` before reading to be sure the capture is
    complete."""

    def __init__(self):
        self._pending: dict[str, set[asyncio.Task]] = {}

    def track(self, rid: str, coro) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        self._pending.setdefault(rid, set()).add(task)
        task.add_done_callback(lambda t, rid=rid: self._pending.get(rid, set()).discard(t))
        return task

    async def drain(self, rid: str) -> None:
        pending = list(self._pending.get(rid, ()))
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
