# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Sandbox provider that forwards operations to a sandbox server over HTTP.

This is the client half of the sandbox server. It implements the ordinary
``SandboxProvider`` protocol, so ``AsyncSandbox`` drives a server-owned sandbox
with the same code that drives an in-process one; the only difference is that
``handle.raw`` is a serializable ``SandboxRef`` instead of provider-owned state.
It also implements ``ConnectableProvider``, so a handle can be serialized to a
ref and rebuilt in another process.

The sandbox server owns the real provider (docker, opensandbox, ...). This class
never imports one; it only speaks the server's small HTTP surface.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.remote.schemas import (
    CreateSandboxRequest,
    DownloadResponse,
    ExecRequest,
    ExecResponse,
    LeaseRequest,
    StatusResponse,
    UploadRequest,
)
from nemo_gym.sandbox.ref import SCOPE_OWNER, SandboxRef
from nemo_gym.sandbox.transport import SandboxHttpTransport


class RemoteSandboxProvider:
    """Forwards sandbox operations to a sandbox server.

    Depends only on an injected :class:`SandboxHttpTransport`, so it stays in the
    sandbox library without importing the server framework. The server layer
    builds it with a Gym-wired transport (``nemo_gym.sandbox_client``).
    """

    name = "remote"

    def __init__(self, *, server_url: str, transport: SandboxHttpTransport, api_key: str | None = None) -> None:
        if not server_url:
            raise ValueError("RemoteSandboxProvider requires a server_url")
        self._server_url = server_url.rstrip("/")
        self._transport = transport
        self._api_key = api_key

    def _headers(self, lease_token: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if lease_token:
            headers["X-NeMo-Gym-Sandbox-Lease"] = lease_token
        return headers

    def _ref(self, handle_or_descriptor: Any) -> SandboxRef:
        raw = getattr(handle_or_descriptor, "raw", handle_or_descriptor)
        if isinstance(raw, SandboxRef):
            return raw
        if isinstance(raw, Mapping):
            return SandboxRef.from_dict(dict(raw))
        raise TypeError(f"RemoteSandboxProvider expected a SandboxRef, got {type(raw).__name__}")

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Ask the server to create a sandbox; return an owner-scoped handle.

        Files are not sent here: ``AsyncSandbox.start`` uploads them afterward
        through :meth:`upload_file`, matching the in-process providers.
        """
        body = CreateSandboxRequest(
            image=spec.image,
            ttl_s=spec.ttl_s,
            ready_timeout_s=spec.ready_timeout_s,
            workdir=spec.workdir,
            env=dict(spec.env),
            metadata=dict(spec.metadata),
            resources={
                "cpu": spec.resources.cpu,
                "memory_mib": spec.resources.memory_mib,
                "disk_gib": spec.resources.disk_gib,
                "gpu": spec.resources.gpu,
                "gpu_type": spec.resources.gpu_type,
            },
            entrypoint=spec.entrypoint,
            provider_options=dict(spec.provider_options),
        )
        resp = await self._transport.request(
            "POST",
            f"{self._server_url}/sandboxes",
            json=body.model_dump(mode="json"),
            headers=self._headers(),
        )
        await self._transport.raise_for_status(resp)
        ref = SandboxRef.from_dict(await resp.json())
        if not ref.can_close:
            raise SandboxCreateError("sandbox server did not return an owner-scoped ref for create()")
        return SandboxHandle(sandbox_id=ref.sandbox_id, provider_name=self.name, raw=ref)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        ref = self._ref(handle)
        body = ExecRequest(command=command, cwd=cwd, env=env, timeout_s=timeout_s, user=user)
        resp = await self._transport.request(
            "POST",
            f"{self._server_url}/sandboxes/{ref.sandbox_id}/exec",
            json=body.model_dump(mode="json"),
            headers=self._headers(ref.lease_token),
        )
        await self._transport.raise_for_status(resp)
        result = ExecResponse.model_validate(await resp.json())
        return SandboxExecResult(
            stdout=result.stdout,
            stderr=result.stderr,
            return_code=result.return_code,
            error_type=result.error_type,
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        ref = self._ref(handle)
        body = UploadRequest(
            target_path=target_path,
            contents_b64=base64.b64encode(Path(source_path).read_bytes()).decode("ascii"),
        )
        resp = await self._transport.request(
            "POST",
            f"{self._server_url}/sandboxes/{ref.sandbox_id}/upload",
            json=body.model_dump(mode="json"),
            headers=self._headers(ref.lease_token),
        )
        await self._transport.raise_for_status(resp)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        ref = self._ref(handle)
        resp = await self._transport.request(
            "GET",
            f"{self._server_url}/sandboxes/{ref.sandbox_id}/download",
            params={"remote_path": source_path},
            headers=self._headers(ref.lease_token),
        )
        await self._transport.raise_for_status(resp)
        data = DownloadResponse.model_validate(await resp.json())
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(base64.b64decode(data.contents_b64))

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        ref = self._ref(handle)
        resp = await self._transport.request(
            "GET",
            f"{self._server_url}/sandboxes/{ref.sandbox_id}/status",
            headers=self._headers(ref.lease_token),
        )
        if resp.status != 200:
            return SandboxStatus.UNKNOWN
        data = StatusResponse.model_validate(await resp.json())
        try:
            return SandboxStatus(data.status)
        except ValueError:
            return SandboxStatus.UNKNOWN

    async def close(self, handle: SandboxHandle) -> None:
        """End the lifecycle for an owner ref, or release the lease for a co-lessee."""
        ref = self._ref(handle)
        path = f"{self._server_url}/sandboxes/{ref.sandbox_id}"
        if ref.scope != SCOPE_OWNER:
            path = f"{path}/leases/release"
        resp = await self._transport.request("DELETE", path, headers=self._headers(ref.lease_token))
        await self._transport.raise_for_status(resp)

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        """Return a ``SandboxRef`` descriptor for this handle.

        With no ``scope`` (or the handle's own scope) the current ref is returned.
        A different scope mints a new co-lease on the server and returns that ref,
        which is how a verifier is handed operate rights on the owner's box.
        """
        ref = self._ref(handle)
        if scope is None or scope == ref.scope:
            return ref.to_dict()
        body = LeaseRequest(scope=scope)
        resp = await self._transport.request(
            "POST",
            f"{self._server_url}/sandboxes/{ref.sandbox_id}/leases",
            json=body.model_dump(mode="json"),
            headers=self._headers(ref.lease_token),
        )
        await self._transport.raise_for_status(resp)
        return await resp.json()

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        """Rebuild a handle from a ``SandboxRef`` descriptor. No network call:
        the ref already carries the id and lease, and calls are lazy."""
        ref = self._ref(descriptor)
        return SandboxHandle(sandbox_id=ref.sandbox_id, provider_name=self.name, raw=ref)

    async def aclose(self) -> None:
        return None
