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
import uuid
from pathlib import Path

from nemo_gym.core.worker_resources import record_process_group, worker_workspace_dir
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
            # Without a configured root, the workspace sits under this worker's directory.
            # A peer can then remove it if this worker dies.
            root = self._workspace_root or worker_workspace_dir()
            workspace, owned = root / sandbox_id, True
        workspace.mkdir(parents=True, exist_ok=True)
        return SandboxHandle(
            sandbox_id=sandbox_id,
            provider_name=self.name,
            # Records of process groups a command left running in the background; close ends them.
            raw={"workspace": workspace, "env": dict(spec.env), "owned": owned, "process_groups": []},
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
        """Kill the process group on timeout or cancellation."""
        if user is not None:
            raise ValueError("The local sandbox provider cannot run commands as another user")
        inst = handle.raw
        exec_cwd = Path(cwd) if cwd else inst["workspace"]
        # A relative cwd stays inside a workspace this provider allocated. A caller's workdir is a host
        # path, and the sandbox facade passes it as the default cwd, so it is not joined again.
        if cwd and inst["owned"] and not exec_cwd.is_absolute():
            exec_cwd = inst["workspace"] / exec_cwd
        timeout = timeout_s if timeout_s is not None else self._default_timeout_s
        # Each command confirms the groups earlier commands left running, so their records keep trusting them.
        # A group that has ended drops its record.
        # The list is rewritten in place with no await, so a concurrent command's appended record is never lost.
        # Confirming costs a few system calls and one small write to the node-local record file.
        groups = inst.setdefault("process_groups", [])
        for index, group in reversed(list(enumerate(groups))):
            confirmed = group.confirm()
            if confirmed is None:
                group.forget()
                del groups[index]
            else:
                groups[index] = confirmed
        async with self._semaphore:
            # The shell waits for one line on stdin, sent once its process group is recorded, then runs the command in
            # place with stdin from /dev/null.
            # Recording after the command started would race a command that exits at once and leaves a process
            # running in the background.
            process = await asyncio.create_subprocess_exec(
                self._shell,
                "-c",
                'read -r _ && exec "$0" -c "$1" </dev/null',
                self._shell,
                command,
                cwd=str(exec_cwd),
                env={**os.environ, **inst["env"], **(env or {})},
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            # Recorded under this worker, so a peer can kill the group if this worker dies.
            record = record_process_group(process.pid)
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(b"\n"), timeout=timeout)
            except (TimeoutError, asyncio.TimeoutError, asyncio.CancelledError) as exc:
                with contextlib.suppress(ProcessLookupError):
                    # start_new_session makes the PID the group ID, even after the shell exits.
                    os.killpg(process.pid, signal.SIGKILL)
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(process.wait(), timeout=REAP_TIMEOUT_S)
                if isinstance(exc, asyncio.CancelledError):
                    raise
                return SandboxExecResult(
                    stdout=None,
                    stderr=f"local command timed out after {timeout:g}s",
                    return_code=SANDBOX_RUNTIME_RETURN_CODE,
                    error_type="timeout",
                )
            finally:
                if record is not None:
                    # A process the command left running in the background keeps the group alive.
                    # The record then tracks those processes, and the sandbox keeps it until close ends the group.
                    tracked = record.track_members()
                    if tracked is not None:
                        inst.setdefault("process_groups", []).append(tracked)
                    else:
                        record.forget()
        return SandboxExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
            error_type=None,
        )

    def uploads_independent_of_execs(self) -> bool:
        # Uploads copy in a worker thread; only commands take the exec semaphore.
        return True

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
        """End the processes commands left running in the background, then remove an owned workspace."""
        inst = handle.raw
        groups = inst.get("process_groups", [])
        # No await between ending a record and removing it, so a record a concurrent command appends is not skipped.
        # A record whose end raises stays for the next close.
        while groups:
            groups[-1].end()
            groups.pop()
        if inst["owned"]:
            await asyncio.to_thread(shutil.rmtree, inst["workspace"], ignore_errors=True)

    async def aclose(self) -> None:
        return None
