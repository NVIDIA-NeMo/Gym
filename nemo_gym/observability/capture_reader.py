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

"""Capture readers and the /ng-capture HTTP routes.

The reader's contract is the HTTP API. Reading the local files directly is
just an optimization used when the reader is co-located with the store, and
it is gated on a manifest identity check so it stays equivalent to the HTTP
path. The builder consumes the same iterator either way and cannot tell which
path served it.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

from nemo_gym.observability.capture_store import LocalJsonlCaptureStore
from nemo_gym.observability.records import record_from_envelope
from nemo_gym.server_utils import request


class CaptureReader:
    def envelopes(self, rid: str, kinds: set[str] | None = None) -> Iterator[dict]:
        raise NotImplementedError

    def records(self, rid: str, kinds: set[str] | None = None) -> list:
        out = [record_from_envelope(e) for e in self.envelopes(rid, kinds)]
        out.sort(key=lambda r: r.seq)
        return out


class LocalCaptureReader(CaptureReader):
    def __init__(self, store: LocalJsonlCaptureStore):
        self.store = store

    def envelopes(self, rid, kinds=None):
        yield from self.store.read_envelopes(rid, kinds)


class HttpCaptureReader(CaptureReader):
    """Reads via the model server's /ng-capture routes with an optional,
    manifest-verified local fast path.

    HTTP fetches go through the shared aiohttp client (nemo_gym.server_utils),
    never a per-instance sync client, so reading capture at high rollout
    concurrency does not stall the event loop. Call ``prefetch`` (async) once per
    rollout to pull its envelopes + summary over HTTP into an in-memory cache;
    the sync record/summary accessors the trajectory builder consumes then read
    from that cache (or the local files, when the fast path is active)."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "dummy_key",
        local_dir: str | None = None,
        local_read: str = "auto",
        timeout_s: float = 30.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._timeout_s = timeout_s
        self._local_dir = local_dir
        self._local_read = local_read
        self._local: LocalCaptureReader | None = None
        self._local_checked = False
        self._envelopes: dict[str, list[dict]] = {}
        self._summaries: dict[str, dict] = {}

    async def _ensure_local(self) -> None:
        """Resolve the local fast path once (manifest identity check over HTTP)."""
        if self._local_checked:
            return
        self._local_checked = True
        if not (self._local_dir and self._local_read in ("auto", "always")):
            return
        try:
            resp = await request("GET", f"{self._base_url}/ng-capture/manifest", headers=self._headers)
            resp.raise_for_status()
            remote = await resp.json()
            with open(os.path.join(self._local_dir, "MANIFEST.json")) as f:
                mine = json.load(f)
            if (remote.get("run_id"), remote.get("instance_id")) == (mine.get("run_id"), mine.get("instance_id")):
                self._local = LocalCaptureReader(LocalJsonlCaptureStore(self._local_dir, run_id=mine["run_id"]))
        except Exception:
            if self._local_read == "always":
                raise

    async def prefetch(self, rid: str) -> None:
        """Pull this rollout's envelopes + summary over HTTP into the cache. A
        no-op on the local fast path (files are read directly)."""
        await self._ensure_local()
        if self._local is not None:
            return
        resp = await request("GET", f"{self._base_url}/ng-capture/rollouts/{rid}", headers=self._headers)
        resp.raise_for_status()
        text = await resp.text()
        self._envelopes[rid] = [json.loads(line) for line in text.splitlines() if line.strip()]
        sresp = await request("GET", f"{self._base_url}/ng-capture/rollouts/{rid}/summary", headers=self._headers)
        sresp.raise_for_status()
        self._summaries[rid] = await sresp.json()

    def envelopes(self, rid, kinds=None):
        if self._local is not None:
            yield from self._local.envelopes(rid, kinds)
            return
        if rid not in self._envelopes:
            raise RuntimeError(f"capture for {rid!r} not prefetched; call `await reader.prefetch(rid)` first")
        for e in self._envelopes[rid]:
            if kinds is None or e.get("kind") in kinds:
                yield e

    def summary(self, rid: str) -> dict:
        if self._local is not None:
            return self._local.store.summary(rid)
        return self._summaries.get(rid, {"rollout_id": rid})


def build_capture_router(store: LocalJsonlCaptureStore):
    """FastAPI router exposing the read API on the model-server app.
    Mount OUTSIDE the /ng-rollout namespace; the gate middleware additionally
    404s any /ng-rollout/*/ng-capture/* construction (namespace guard)."""
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import PlainTextResponse, Response

    router = APIRouter(prefix="/ng-capture")

    @router.get("/manifest")
    def manifest() -> dict:
        return store.manifest

    @router.get("/rollouts/{rid}", response_class=PlainTextResponse)
    def rollout(rid: str, kinds: str | None = None, after_seq: int = -1) -> str:
        kind_set = set(kinds.split(",")) if kinds else None
        lines = [json.dumps(e, separators=(",", ":")) for e in store.read_envelopes(rid, kind_set, after_seq)]
        return "\n".join(lines)

    @router.get("/rollouts/{rid}/summary")
    def summary(rid: str) -> dict:
        return store.summary(rid)

    @router.get("/blobs/{sha}")
    def blob(sha: str) -> Response:
        try:
            return Response(content=store.resolve_blob(sha), media_type="application/octet-stream")
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="unknown blob")

    return router
