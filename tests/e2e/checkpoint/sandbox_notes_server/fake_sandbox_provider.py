# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sandbox provider over the e2e suite's fake sandbox backend, registered as ``fake_remote``.

It implements the capabilities the checkpointer relies on: pause/resume, connect by descriptor, and the
snapshot lookup, all over Gym's shared aiohttp client.
"""

from pathlib import Path
from typing import Any

from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.registry import register_provider
from nemo_gym.server_utils import request


class RemoteFakeSandboxProvider:
    name = "fake_remote"
    # The backend keeps every snapshot until it is deleted, so a checkpoint can fork from one after a resume.
    snapshot_survives_resume = True

    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    async def _call(self, method: str, path: str, body: Any = None) -> dict[str, Any]:
        response = await request(method, f"{self.base_url}{path}", json=body, _max_num_tries=1)
        async with response:
            payload = await response.json(content_type=None)
            if response.status >= 400:
                raise RuntimeError(f"fake sandbox backend {method} {path} -> HTTP {response.status}: {payload}")
            return payload

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        payload = await self._call(
            "POST", "/sandboxes", {"image": spec.image, "snapshot_id": spec.provider_options.get("snapshot_id")}
        )
        return SandboxHandle(sandbox_id=payload["id"], provider_name=self.name, raw=None)

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
        return {"running": SandboxStatus.RUNNING, "paused": SandboxStatus.PAUSED}.get(
            payload["state"], SandboxStatus.STOPPED
        )

    async def close(self, handle: SandboxHandle) -> None:
        await self._call("DELETE", f"/sandboxes/{handle.sandbox_id}")

    async def aclose(self) -> None:
        pass

    async def pause(self, handle: SandboxHandle) -> None:
        await self._call("POST", f"/sandboxes/{handle.sandbox_id}/pause")

    async def resume(self, handle: SandboxHandle) -> None:
        await self._call("POST", f"/sandboxes/{handle.sandbox_id}/resume")

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: dict[str, Any]) -> SandboxHandle:
        sandbox_id = str(descriptor["sandbox_id"])
        payload = await self._call("GET", f"/sandboxes/{sandbox_id}")
        if payload["state"] == "stopped":
            raise RuntimeError(f"sandbox {sandbox_id} is stopped")
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def latest_snapshot_id(self, handle: SandboxHandle) -> str | None:
        payload = await self._call("GET", f"/v1/snapshots?sandboxId={handle.sandbox_id}")
        items = payload["items"]
        if not items:
            return None
        return max(items, key=lambda item: (item["createdAt"], item["id"]))["id"]


register_provider(RemoteFakeSandboxProvider.name, RemoteFakeSandboxProvider, override=True)
