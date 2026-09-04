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


import asyncio
import contextlib
import logging
import os
import shutil
import signal
import tempfile
import uuid
from pathlib import Path

from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)


LOGGER = logging.getLogger(__name__)

# Sandbox-runtime failure, not the command's exit code.
SANDBOX_RUNTIME_RETURN_CODE = 125
# Bounded so a child that escaped the process group cannot wedge the caller.
REAP_TIMEOUT_S = 5.0


class LocalProvider:
    """Runs commands on the host. No isolation: use a container provider for untrusted code."""

    name = "local"
    _semaphores: dict[int, asyncio.Semaphore] = {}

    def __init__(
        self,
        workspace_root: str | None = None,
        shell: str = "/bin/bash",
        default_timeout_s: float = 180.0,
        concurrency: int = 8,
    ) -> None:
        self._workspace_root = Path(workspace_root).expanduser() if workspace_root else None
        self._shell = shell
        self._default_timeout_s = default_timeout_s
        self._semaphore = LocalProvider._semaphores.setdefault(concurrency, asyncio.Semaphore(concurrency))

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """A caller-supplied workdir is theirs to keep. One we allocate is ours to delete."""
        ignored = [
            name
            for name, value in (
                ("image", spec.image),
                ("ttl_s", spec.ttl_s),
                ("ready_timeout_s", spec.ready_timeout_s),
                ("ports", tuple(spec.ports)),
            )
            if value
        ]
        if ignored:
            LOGGER.warning(
                f"Local sandbox provider ignores {', '.join(ignored)} and all resource limits. "
                f"Commands run unconfined on the host."
            )
        sandbox_id = f"local-{uuid.uuid4().hex[:12]}"
        if spec.workdir:
            workspace, owned = Path(spec.workdir).expanduser(), False
        else:
            root = self._workspace_root or Path(tempfile.gettempdir())
            workspace, owned = root / sandbox_id, True
        workspace.mkdir(parents=True, exist_ok=True)
        return SandboxHandle(
            sandbox_id=sandbox_id,
            provider_name=self.name,
            raw={"workspace": workspace, "env": dict(spec.env), "owned": owned},
        )

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
        """Own session, so a timeout kills the whole process tree."""
        if user is not None:
            raise ValueError("The local sandbox provider cannot run commands as another user")
        inst = handle.raw
        timeout = timeout_s if timeout_s is not None else self._default_timeout_s
        async with self._semaphore:
            process = await asyncio.create_subprocess_exec(
                self._shell,
                "-c",
                command,
                cwd=str(cwd or inst["workspace"]),
                env={**os.environ, **inst["env"], **(env or {})},
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            except (TimeoutError, asyncio.TimeoutError):
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), timeout=REAP_TIMEOUT_S)
                return SandboxExecResult(
                    stdout=None,
                    stderr=f"local command timed out after {timeout:g}s",
                    return_code=SANDBOX_RUNTIME_RETURN_CODE,
                    error_type="timeout",
                )
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
            error_type=None,
        )

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        target = Path(target_path)
        if not target.is_absolute():
            target = handle.raw["workspace"] / target
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, source_path, target)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        source = Path(source_path)
        if not source.is_absolute():
            source = handle.raw["workspace"] / source
        target_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, source, target_path)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.RUNNING if handle.raw["workspace"].is_dir() else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        inst = handle.raw
        if inst["owned"]:
            await asyncio.to_thread(shutil.rmtree, inst["workspace"], ignore_errors=True)

    async def aclose(self) -> None:
        return None
