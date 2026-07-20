# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sandbox server: owns physical sandboxes and lends them across Gym servers.

It fronts exactly one sandbox provider (docker, opensandbox, ...) and exposes
its lifecycle and operations over HTTP. The value it adds over an in-process
``nemo_gym.sandbox.AsyncSandbox`` is that a sandbox created by one server can be
operated by another by reference: the client holds a small signed ``SandboxRef``,
not provider-owned in-process state.

Ownership model:
- create mints an OWNER lease; only an owner lease may DELETE (destroy) the box.
- a co-lease (POST /sandboxes/{id}/leases, scope=operate) may exec/upload/
  download but not destroy; this is what a verifier reattaches with.
- leases are signed capabilities bound to the sandbox's rollout id, so a ref
  leaked from one rollout can't touch another rollout's box.
- ttl_s reaps orphaned boxes regardless of leases (crash safety).
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from itsdangerous import BadSignature, URLSafeTimedSerializer
from pydantic import ConfigDict

from nemo_gym.config_types import BaseRunServerInstanceConfig
from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import (
    SCOPE_OPERATE,
    SCOPE_OWNER,
    AsyncSandbox,
    SandboxRef,
    SandboxResources,
    SandboxSpec,
    create_provider,
    resolve_provider_config,
)
from nemo_gym.sandbox.providers.remote.schemas import (
    CreateSandboxRequest,
    DeleteResponse,
    DownloadResponse,
    ExecRequest,
    ExecResponse,
    LeaseRequest,
    ReleaseResponse,
    StatusResponse,
    UploadRequest,
)
from nemo_gym.server_utils import SimpleServer, is_nemo_gym_fastapi_entrypoint


LEASE_HEADER = "X-NeMo-Gym-Sandbox-Lease"
_LEASE_SALT = "nemo-gym-sandbox-lease"
ROLLOUT_METADATA_KEY = "ng_rollout_id"


class SandboxServerConfig(BaseRunServerInstanceConfig):
    model_config = ConfigDict(extra="allow")

    # The single provider this server fronts: an inline single-key mapping
    # ({docker: {}}) or the name of a composed provider block. Resolved via
    # nemo_gym.sandbox.resolve_provider_config against the merged global config.
    sandbox_provider: Any
    # Pool cap: max concurrently-live sandboxes. Admission control in one place
    # instead of every consumer inventing its own limit.
    max_concurrent: int = 64
    # Orphan backstop applied when a create request omits ttl_s.
    default_ttl_s: Optional[float] = 1800.0
    # Lease-signing secret. Left unset -> a per-process random secret (fine for
    # a single-process local server; set it for multi-worker/persistent runs).
    lease_secret: Optional[str] = None


class _Entry:
    """One live sandbox and its bookkeeping."""

    def __init__(self, sandbox: AsyncSandbox, rollout_id: str, workdir: Optional[str], expires_at: float) -> None:
        self.sandbox = sandbox
        self.rollout_id = rollout_id
        self.workdir = workdir
        self.expires_at = expires_at
        self.leases = 1  # the owner lease


class SandboxServer(SimpleServer):
    config: SandboxServerConfig

    def model_post_init(self, context: Any) -> None:
        self._entries: dict[str, _Entry] = {}
        self._lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(self.config.max_concurrent)
        self._signer = URLSafeTimedSerializer(self.config.lease_secret or uuid4().hex, salt=_LEASE_SALT)
        self._provider_config = resolve_provider_config(self.config.sandbox_provider, get_global_config_dict())
        return super().model_post_init(context)

    # -- lease helpers -----------------------------------------------------

    def _mint(self, sandbox_id: str, rollout_id: str, scope: str) -> str:
        return self._signer.dumps({"sid": sandbox_id, "rid": rollout_id, "scope": scope})

    def _check(self, request: Request, sandbox_id: str, *, require_owner: bool = False) -> _Entry:
        token = request.headers.get(LEASE_HEADER)
        if not token:
            raise HTTPException(status_code=401, detail="missing sandbox lease token")
        try:
            payload = self._signer.loads(token, max_age=24 * 3600)
        except BadSignature as e:
            raise HTTPException(status_code=401, detail="invalid sandbox lease token") from e
        if payload.get("sid") != sandbox_id:
            raise HTTPException(status_code=403, detail="lease token does not match this sandbox")
        entry = self._entries.get(sandbox_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"sandbox {sandbox_id!r} not found")
        if payload.get("rid") != entry.rollout_id:
            raise HTTPException(status_code=403, detail="lease token rollout id does not match the sandbox")
        if require_owner and payload.get("scope") != SCOPE_OWNER:
            raise HTTPException(status_code=403, detail="this operation requires an owner lease")
        return entry

    def _server_url(self) -> str:
        host = self.config.host or "127.0.0.1"
        if host in ("0.0.0.0", "::", ""):
            host = "127.0.0.1"
        return f"http://{host}:{self.config.port}"

    def _ref(self, sandbox_id: str, rollout_id: str, scope: str, workdir: Optional[str]) -> dict:
        return SandboxRef(
            server_url=self._server_url(),
            sandbox_id=sandbox_id,
            lease_token=self._mint(sandbox_id, rollout_id, scope),
            provider_name=next(iter(self._provider_config)),
            scope=scope,
            workdir=workdir,
        ).to_dict()

    # -- routes ------------------------------------------------------------

    def setup_webserver(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            reaper = asyncio.ensure_future(self._reap_loop())
            try:
                yield
            finally:
                reaper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await reaper
                await self._shutdown_all()

        app = FastAPI(lifespan=lifespan)
        self.setup_liveness(app)
        app.post("/sandboxes")(self.create_sandbox)
        app.post("/sandboxes/{sandbox_id}/exec")(self.exec_sandbox)
        app.post("/sandboxes/{sandbox_id}/upload")(self.upload_sandbox)
        app.get("/sandboxes/{sandbox_id}/download")(self.download_sandbox)
        app.get("/sandboxes/{sandbox_id}/status")(self.status_sandbox)
        app.post("/sandboxes/{sandbox_id}/leases")(self.grant_lease)
        app.delete("/sandboxes/{sandbox_id}/leases/release")(self.release_lease)
        app.delete("/sandboxes/{sandbox_id}")(self.delete_sandbox)
        return app

    async def create_sandbox(self, body: CreateSandboxRequest) -> dict:
        rollout_id = str(body.metadata.get(ROLLOUT_METADATA_KEY) or "")
        ttl_s = body.ttl_s if body.ttl_s is not None else self.config.default_ttl_s
        spec = SandboxSpec(
            image=body.image,
            ttl_s=ttl_s,
            ready_timeout_s=body.ready_timeout_s,
            workdir=body.workdir,
            env=dict(body.env),
            metadata=dict(body.metadata),
            resources=SandboxResources.from_mapping(body.resources),
            entrypoint=body.entrypoint,
            provider_options=dict(body.provider_options),
        )
        await self._sem.acquire()
        try:
            provider = create_provider(self._provider_config)
            sandbox = await AsyncSandbox(provider, spec).start()
        except Exception:
            self._sem.release()
            raise
        sandbox_id = sandbox._handle.sandbox_id  # provider-neutral id
        expires_at = time.monotonic() + ttl_s if ttl_s else float("inf")
        async with self._lock:
            self._entries[sandbox_id] = _Entry(sandbox, rollout_id, body.workdir, expires_at)
        return self._ref(sandbox_id, rollout_id, SCOPE_OWNER, body.workdir)

    async def exec_sandbox(self, sandbox_id: str, body: ExecRequest, request: Request) -> ExecResponse:
        entry = self._check(request, sandbox_id)
        result = await entry.sandbox.exec(
            body.command, cwd=body.cwd, env=body.env, timeout_s=body.timeout_s, user=body.user
        )
        return ExecResponse(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
            error_type=result.error_type,
        )

    async def upload_sandbox(self, sandbox_id: str, body: UploadRequest, request: Request) -> dict:
        entry = self._check(request, sandbox_id)
        with tempfile.TemporaryDirectory(prefix="nemo-gym-sbxsrv-up-") as tmp:
            src = Path(tmp) / "payload"
            src.write_bytes(base64.b64decode(body.contents_b64))
            await entry.sandbox.upload(src, body.target_path)
        return {}

    async def download_sandbox(self, sandbox_id: str, remote_path: str, request: Request) -> DownloadResponse:
        entry = self._check(request, sandbox_id)
        with tempfile.TemporaryDirectory(prefix="nemo-gym-sbxsrv-dl-") as tmp:
            dst = Path(tmp) / "payload"
            await entry.sandbox.download(remote_path, dst)
            return DownloadResponse(contents_b64=base64.b64encode(dst.read_bytes()).decode("ascii"))

    async def status_sandbox(self, sandbox_id: str, request: Request) -> StatusResponse:
        entry = self._check(request, sandbox_id)
        return StatusResponse(status=(await entry.sandbox.status()).value)

    async def grant_lease(self, sandbox_id: str, body: LeaseRequest, request: Request) -> dict:
        entry = self._check(request, sandbox_id)
        scope = SCOPE_OWNER if body.scope == SCOPE_OWNER else SCOPE_OPERATE
        async with self._lock:
            entry.leases += 1
        return self._ref(sandbox_id, entry.rollout_id, scope, entry.workdir)

    async def release_lease(self, sandbox_id: str, request: Request) -> ReleaseResponse:
        entry = self._check(request, sandbox_id)
        async with self._lock:
            entry.leases = max(0, entry.leases - 1)
        return ReleaseResponse(released=True, remaining_leases=entry.leases)

    async def delete_sandbox(self, sandbox_id: str, request: Request) -> DeleteResponse:
        self._check(request, sandbox_id, require_owner=True)
        await self._destroy(sandbox_id)
        return DeleteResponse(deleted=True)

    # -- lifecycle ---------------------------------------------------------

    async def _destroy(self, sandbox_id: str) -> None:
        async with self._lock:
            entry = self._entries.pop(sandbox_id, None)
        if entry is None:
            return
        try:
            await entry.sandbox.stop()
        finally:
            self._sem.release()

    async def _reap_loop(self) -> None:
        while True:
            await asyncio.sleep(10.0)
            now = time.monotonic()
            expired = [sid for sid, e in list(self._entries.items()) if e.expires_at <= now]
            for sid in expired:
                with contextlib.suppress(Exception):
                    await self._destroy(sid)

    async def _shutdown_all(self) -> None:
        for sid in list(self._entries):
            with contextlib.suppress(Exception):
                await self._destroy(sid)


if __name__ == "__main__":
    SandboxServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SandboxServer.run_webserver()  # noqa: F401
