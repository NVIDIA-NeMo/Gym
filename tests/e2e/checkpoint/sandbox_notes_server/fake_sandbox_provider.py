# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sandbox provider over the e2e suite's fake sandbox backend, registered as ``fake_remote``.

It implements the capabilities the checkpointer relies on: explicit snapshots (create, wait until ready,
delete), create from a snapshot, and connect by descriptor, all over Gym's shared aiohttp client. With
``lease_ttl_s`` and ``renew_interval_s`` it keeps each sandbox it created on a lease the way the OpenSandbox
provider does: created with the lease TTL, renewed to now plus the TTL every interval (jittered by 10%) until
closed, so the backend expires the sandboxes of a process that died.
"""

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Any, Optional

from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.registry import register_provider
from nemo_gym.server_utils import request


LOGGER = logging.getLogger(__name__)


class RemoteFakeSandboxProvider:
    name = "fake_remote"

    def __init__(
        self, base_url: str, *, lease_ttl_s: Optional[float] = None, renew_interval_s: Optional[float] = None
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.lease_ttl_s = lease_ttl_s
        self.renew_interval_s = renew_interval_s
        self._renewals: dict[str, asyncio.Task[None]] = {}

    async def _call(
        self, method: str, path: str, body: Any = None, *, tolerate: tuple[int, ...] = ()
    ) -> dict[str, Any]:
        response = await request(method, f"{self.base_url}{path}", json=body, _max_num_tries=1)
        async with response:
            payload = await response.json(content_type=None)
            if response.status >= 400 and response.status not in tolerate:
                raise RuntimeError(f"fake sandbox backend {method} {path} -> HTTP {response.status}: {payload}")
            return payload

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        ttl_s = self.lease_ttl_s if self.lease_ttl_s is not None else spec.ttl_s
        payload = await self._call(
            "POST",
            "/sandboxes",
            {"image": spec.image, "snapshot_id": spec.provider_options.get("snapshot_id"), "timeout": ttl_s},
        )
        sandbox_id = payload["id"]
        if self.renew_interval_s is not None and ttl_s is not None:
            self._renewals[sandbox_id] = asyncio.create_task(self._renew(sandbox_id, ttl_s))
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def _renew(self, sandbox_id: str, ttl_s: float) -> None:
        while True:
            await asyncio.sleep(self.renew_interval_s * random.uniform(0.9, 1.1))
            try:
                await self._call(
                    "POST", f"/sandboxes/{sandbox_id}/renew-expiration", {"expiresAt": time.time() + ttl_s}
                )
            except Exception as error:
                LOGGER.warning("renewal of sandbox %s failed; retrying at the next interval: %r", sandbox_id, error)

    async def _stop_renewal(self, sandbox_id: str) -> None:
        task = self._renewals.pop(sandbox_id, None)
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

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
        payload = await self._call("POST", f"/sandboxes/{handle.sandbox_id}/exec", {"command": command})
        return SandboxExecResult(
            stdout=payload["stdout"], stderr=payload["stderr"], return_code=int(payload["return_code"])
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        for line in source_path.read_text().splitlines():
            await self.exec(handle, f"append {target_path} {line}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        result = await self.exec(handle, f"read {source_path}")
        target_path.write_text(result.stdout or "")

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        try:
            payload = await self._call("GET", f"/sandboxes/{handle.sandbox_id}")
        except RuntimeError:
            return SandboxStatus.UNKNOWN
        return SandboxStatus.RUNNING if payload["state"] == "running" else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        await self._stop_renewal(handle.sandbox_id)
        await self._call("DELETE", f"/sandboxes/{handle.sandbox_id}")

    async def aclose(self) -> None:
        for sandbox_id in list(self._renewals):
            await self._stop_renewal(sandbox_id)

    async def snapshot(self, handle: SandboxHandle, *, name: str | None = None) -> str:
        created = await self._call("POST", f"/sandboxes/{handle.sandbox_id}/snapshots", {"name": name})
        return created["id"]

    async def wait_snapshot_ready(self, snapshot_id: str) -> None:
        # Asynchronous on the server, as on OpenSandbox: poll until the snapshot is ready to fork from.
        async with asyncio.timeout(30):
            while True:
                info = await self._call("GET", f"/v1/snapshots/{snapshot_id}")
                if info["status"]["state"] == "Ready":
                    return
                await asyncio.sleep(0.05)

    async def delete_snapshot(self, snapshot_id: str) -> None:
        await self._call("DELETE", f"/v1/snapshots/{snapshot_id}", tolerate=(404,))

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: dict[str, Any]) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        payload = await self._call("GET", f"/sandboxes/{sandbox_id}")
        if payload["state"] == "stopped":
            raise RuntimeError(f"sandbox {sandbox_id} is stopped")
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)


register_provider(RemoteFakeSandboxProvider.name, RemoteFakeSandboxProvider, override=True)
