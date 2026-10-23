# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted runtime cache and private, read-only workload mounts.

The execution manifest contains local paths and hashes. Only this helper sees
the shared storage root; workload sandboxes receive their own role directory.
"""

import asyncio
import hashlib
import json
import tempfile
from pathlib import Path
from shlex import quote
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox, SandboxResources, SandboxSpec, create_provider
from nemo_gym.sandbox.providers.base import SupportsSandboxSharedStorage


RUNTIME_ROOT = "/opt/gym-runtime"
PYTHON = RUNTIME_ROOT + "/python/bin/python3"
STORAGE_ROOT = "/mnt/gym-runtime-storage"
SUBMISSION_ROOT = "/opt/gym-submission"
ROLES = {"candidate": ("python", "opencode", "rg"), "judge": ("python",)}


def load_runtime_manifest(path: Path) -> dict:
    """Validate every host artifact before allocating any workload sandbox."""
    manifest = json.loads(path.read_text())
    if manifest.get("version") != 1 or "python" not in manifest.get("artifacts", {}):
        raise ValueError("Runtime manifest must be version 1 and contain the task Python runtime")
    manifest["artifacts"] = {name: manifest["artifacts"][name] for name in ("python", "opencode", "rg")}
    for name, artifact in manifest["artifacts"].items():
        source = Path(artifact["path"])
        if not source.is_absolute():
            source = path.parent / source
        with source.open("rb") as binary:
            digest = hashlib.file_digest(binary, "sha256").hexdigest()
        if digest != artifact["sha256"]:
            raise ValueError(f"Runtime SHA-256 mismatch: {name}")
        if name == "python" and artifact.get("format") != "tar.gz":
            raise ValueError("Python runtime must contain the full tar.gz install-only archive")
        artifact["path"] = str(source.resolve())
    return manifest


class RuntimeProvisioner:
    """Reuse a verified cache while isolating each session's mounted binaries."""

    def __init__(self, *, provider_config: dict, manifest_path: Path, helper_image: str):
        self.manifest = load_runtime_manifest(manifest_path)
        self._provider_config = dict(provider_config)
        self.provider = create_provider(self._provider_config)
        if not isinstance(self.provider, SupportsSandboxSharedStorage):
            raise ValueError("SWE-Together requires provider shared storage for private runtime mounts")
        self.helper = AsyncSandbox(self.provider)
        self.helper_image = helper_image
        self._lock = asyncio.Lock()
        self._start_task: asyncio.Task | None = None
        self._bundles: dict[str, str] = {}
        self._owner = uuid4().hex
        self._worker = "/tmp/gym-swet-runtime-" + self._owner + ".py"

    async def _exec(self, action: str, *args: str):
        command = " ".join(quote(value) for value in ["python3", self._worker, action, *args])
        result = await self.helper.exec(command, timeout_s=600)
        if result.return_code or result.error_type:
            raise RuntimeError("Trusted runtime preparation failed: " + (result.stderr or result.stdout or ""))
        return result

    async def _start(self):
        await self.helper.start(
            SandboxSpec(
                image=self.helper_image,
                entrypoint=["sh", "-c", "while :; do sleep 3600; done"],
                ttl_s=86400,
                ready_timeout_s=300,
                resources=SandboxResources(cpu=1, memory_mib=1024),
                provider_options=self.provider.shared_volume_options(None, STORAGE_ROOT)
                | {"network_policy": {"defaultAction": "deny", "egress": []}},
                metadata=self.provider.shared_volume_metadata() | {"benchmark": "swe-together", "role": "runtime"},
            )
        )
        await self.helper.upload(Path(__file__).with_name("runtime_worker.py"), self._worker)
        await self._exec("init", STORAGE_ROOT + "/gym-swet-runtime-cache")
        for name, artifact in self.manifest["artifacts"].items():
            await self._cache(name, artifact)

    async def _cache(self, name: str, artifact: dict):
        target = STORAGE_ROOT + "/gym-swet-runtime-cache/" + artifact["sha256"]
        checked = await self._exec("verify", target, artifact["sha256"])
        if (checked.stdout or "").strip() == "1":
            return
        # Filesystem API payloads are bounded. Upload once per hash in chunks,
        # then atomically publish after the helper independently checks bytes.
        staging = target + ".upload-" + self._owner
        with (
            tempfile.TemporaryDirectory(prefix="gym-swet-upload-") as temp,
            Path(artifact["path"]).open("rb") as source,
        ):
            chunks = 0
            while block := source.read(4 * 1024 * 1024):
                local = Path(temp) / "chunk"
                local.write_bytes(block)
                await self.helper.upload(local, staging + f".{chunks}")
                chunks += 1
        await self._exec("publish", target, staging, str(chunks), artifact["sha256"])

    async def prepare(self, session_id: str) -> str:
        async with self._lock:
            if self._start_task is None:
                self._start_task = asyncio.create_task(self._start())
            try:
                await asyncio.shield(self._start_task)
            except Exception:
                # A failed install is retryable by a later episode. Release the
                # failed helper before allocating a replacement.
                await self.helper.stop()
                self.provider = create_provider(self._provider_config)
                self.helper = AsyncSandbox(self.provider)
                self._start_task = None
                raise
            if session_id in self._bundles:
                return self._bundles[session_id]
            relative = "gym-swet-runtime-" + self._owner + "/" + uuid4().hex
            # Record ownership before preparation so failed setup can remove it.
            self._bundles[session_id] = relative
            cache = STORAGE_ROOT + "/gym-swet-runtime-cache/"
            roles = {
                role: {name: cache + self.manifest["artifacts"][name]["sha256"] for name in names}
                for role, names in ROLES.items()
            }
            await self._exec("prepare", STORAGE_ROOT + "/" + relative, json.dumps(roles), self._owner)
            return relative

    def mount_options(self, session_id: str, role: str) -> dict:
        runtime = self.provider.shared_volume_options(
            self._bundles[session_id] + "/" + role, RUNTIME_ROOT, read_only=True
        )
        submission = self.provider.shared_volume_options(
            self._bundles[session_id] + "/submission-" + role, SUBMISSION_ROOT, read_only=role == "judge"
        )
        return {"volumes": runtime["volumes"] + submission["volumes"]}

    async def seal_submission(self, session_id: str, manifest: dict) -> None:
        """Copy the candidate archive into a separate read-only judge mount and verify its digest."""
        await self._exec("seal", STORAGE_ROOT + "/" + self._bundles[session_id], json.dumps(manifest))

    def metadata(self) -> dict:
        return self.provider.shared_volume_metadata()

    async def release(self, session_id: str):
        async with self._lock:
            if relative := self._bundles.get(session_id):
                await self._exec("release", STORAGE_ROOT + "/" + relative, "gym-swet-runtime-" + self._owner)
                self._bundles.pop(session_id)

    async def close(self):
        if self._bundles:
            raise RuntimeError("Runtime bundles remain owned by task sandboxes")
        if self._start_task is not None:
            try:
                await asyncio.shield(self._start_task)
            except Exception:
                pass
        await self.helper.stop()
