# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox backends for the VCQA agent.

Three implementations of the same `Sandbox` protocol:

- `ApptainerSandbox` (production): boots an `apptainer instance` per rollout,
  bind-mounts the working tree at `/codebase:ro`, runs every tool call via
  `apptainer exec instance://... bash -lc <cmd>`. This is what the YAML
  config defaults to.
- `ApptainerDirectSandbox` (nested-container fallback): runs each tool call
  with direct `apptainer exec`, bind-mounting the working tree at
  `/codebase:ro` and a host scratch dir at `/tmp/scratch`. Use this when
  `apptainer exec instance://...` cannot enter the instance namespace.
- `LocalSandbox` (dev / macOS / no-apptainer hosts): runs commands directly
  with `cwd=<working_tree>`, no container at all. Cheaper to spin up and
  needs no daemon, but obviously gives the model access to the host
  filesystem outside `/codebase` if the path-containment helpers in
  `tools.py` ever miss a case. Use only against trusted artifact sources.

Both expose the same surface: `codebase_path`, `scratch_path`, `todos_path`,
`async start()`, `async exec(command, timeout_s) -> ExecResult`,
`async stop()`. Callers in `tools.py` and `app.py` only see the protocol,
not the concrete class.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol


CODEBASE_PATH = "/codebase"
SCRATCH_PATH = "/tmp/scratch"
TODOS_PATH = f"{SCRATCH_PATH}/todos.md"


@dataclass
class ExecResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


# Backwards-compat alias so existing tests / external callers keep working.
ApptainerExecResult = ExecResult


class Sandbox(Protocol):
    """Structural type used by `tools.py` and `app.py`."""

    codebase_path: str
    scratch_path: str
    todos_path: str

    async def start(self) -> None: ...

    async def exec(self, command: str, timeout_s: Optional[int] = None) -> ExecResult: ...

    async def stop(self) -> None: ...


########################################
# Apptainer backend
########################################


@dataclass
class ApptainerSandbox:
    """A running Apptainer instance scoped to a single rollout."""

    instance_name: str
    container_image: str
    working_tree: Path
    scratch_dir: Path
    exec_timeout_s: int = 30
    setup_done: bool = False
    started: bool = False
    cleaned_up: bool = False
    extra_setup_command: Optional[str] = None
    _started_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    codebase_path: str = CODEBASE_PATH
    scratch_path: str = SCRATCH_PATH
    todos_path: str = TODOS_PATH

    async def start(self) -> None:
        async with self._started_lock:
            if self.started:
                return
            self.scratch_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                "apptainer",
                "instance",
                "start",
                "--writable-tmpfs",
                "--bind",
                f"{self.working_tree}:{self.codebase_path}:ro",
                self.container_image,
                self.instance_name,
            ]
            try:
                await _run_subprocess(cmd, timeout_s=300)
                self.started = True

                await self._exec_checked(
                    f"mkdir -p {shlex.quote(self.scratch_path)} && touch {shlex.quote(self.todos_path)}",
                    timeout_s=30,
                )

                if self.extra_setup_command and not self.setup_done:
                    await self._exec_checked(self.extra_setup_command, timeout_s=600)
                    self.setup_done = True
            except Exception:
                if self.started:
                    await self.stop()
                raise

    async def _exec_checked(self, command: str, timeout_s: int) -> ExecResult:
        result = await self.exec(command, timeout_s=timeout_s)
        if result.timed_out:
            raise RuntimeError(f"sandbox command timed out after {timeout_s}s: {command}")
        if result.exit_code != 0:
            raise RuntimeError(
                f"sandbox command failed (exit={result.exit_code}): {command}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result

    async def exec(self, command: str, timeout_s: Optional[int] = None) -> ExecResult:
        timeout_s = timeout_s if timeout_s is not None else self.exec_timeout_s
        cmd = [
            "apptainer",
            "exec",
            "--pwd",
            self.codebase_path,
            f"instance://{self.instance_name}",
            "bash",
            "-lc",
            command,
        ]
        return await _exec_with_timeout(cmd, timeout_s=timeout_s, cwd=None)

    async def stop(self) -> None:
        if self.cleaned_up:
            return
        self.cleaned_up = True
        if self.started:
            try:
                await _run_subprocess(
                    ["apptainer", "instance", "stop", self.instance_name],
                    timeout_s=60,
                )
            except Exception:
                pass
            finally:
                self.started = False
        try:
            shutil.rmtree(self.scratch_dir, ignore_errors=True)
        except Exception:
            pass


########################################
# Direct Apptainer exec backend
########################################


@dataclass
class ApptainerDirectSandbox:
    """Apptainer sandbox without persistent instances.

    Each command runs in a fresh `apptainer exec`, so filesystem mutations in
    the container layer do not persist across calls. The image must already
    contain the command-line tools exposed by `tools.py` (`bash`, `git`, `rg`,
    `fd`, `tree`, coreutils). A host scratch directory is bind-mounted so
    `write_todos` persists across calls.
    """

    container_image: str
    working_tree: Path
    scratch_dir: Path
    exec_timeout_s: int = 30
    started: bool = False
    cleaned_up: bool = False
    extra_setup_command: Optional[str] = None
    _started_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    codebase_path: str = CODEBASE_PATH
    scratch_path: str = SCRATCH_PATH
    todos_path: str = TODOS_PATH

    def _exec_cmd(self, command: str) -> List[str]:
        return [
            "apptainer",
            "exec",
            "--writable-tmpfs",
            "--bind",
            f"{self.working_tree}:{self.codebase_path}:ro",
            "--bind",
            f"{self.scratch_dir}:{self.scratch_path}:rw",
            "--pwd",
            self.codebase_path,
            self.container_image,
            "bash",
            "-lc",
            command,
        ]

    async def start(self) -> None:
        async with self._started_lock:
            if self.started:
                return
            if self.extra_setup_command:
                raise RuntimeError(
                    "sandbox_backend='apptainer_exec' requires a container image with tools preinstalled; "
                    "apptainer_setup_command is not persistent without an Apptainer instance"
                )
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
            await _run_subprocess(
                self._exec_cmd(f"mkdir -p {shlex.quote(self.scratch_path)} && touch {shlex.quote(self.todos_path)}"),
                timeout_s=60,
            )
            self.started = True

    async def exec(self, command: str, timeout_s: Optional[int] = None) -> ExecResult:
        timeout_s = timeout_s if timeout_s is not None else self.exec_timeout_s
        return await _exec_with_timeout(self._exec_cmd(command), timeout_s=timeout_s, cwd=None)

    async def stop(self) -> None:
        if self.cleaned_up:
            return
        self.cleaned_up = True
        try:
            shutil.rmtree(self.scratch_dir, ignore_errors=True)
        except Exception:
            pass


########################################
# Local backend (no container)
########################################


@dataclass
class LocalSandbox:
    """No-container sandbox: runs commands with `cwd=working_tree` directly.

    Useful for development on hosts without apptainer (e.g. macOS) and for
    fast unit-test loops. Path-containment is enforced by `tools._resolve_*`.
    No bind-mount, no read-only protection; only point this at trusted
    artifact sources.
    """

    working_tree: Path
    scratch_dir: Path
    exec_timeout_s: int = 30
    started: bool = False
    cleaned_up: bool = False
    _started_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def codebase_path(self) -> str:
        return str(self.working_tree)

    @property
    def scratch_path(self) -> str:
        return str(self.scratch_dir)

    @property
    def todos_path(self) -> str:
        return str(self.scratch_dir / "todos.md")

    async def start(self) -> None:
        async with self._started_lock:
            if self.started:
                return
            self.scratch_dir.mkdir(parents=True, exist_ok=True)
            (self.scratch_dir / "todos.md").touch(exist_ok=True)
            self.started = True

    async def exec(self, command: str, timeout_s: Optional[int] = None) -> ExecResult:
        timeout_s = timeout_s if timeout_s is not None else self.exec_timeout_s
        cmd = ["bash", "-c", command]
        return await _exec_with_timeout(
            cmd,
            timeout_s=timeout_s,
            cwd=str(self.working_tree) if self.working_tree.exists() else None,
        )

    async def stop(self) -> None:
        if self.cleaned_up:
            return
        self.cleaned_up = True
        try:
            shutil.rmtree(self.scratch_dir, ignore_errors=True)
        except Exception:
            pass


########################################
# Shared subprocess helpers
########################################


async def _exec_with_timeout(cmd: List[str], *, timeout_s: int, cwd: Optional[str]) -> ExecResult:
    """Run a subprocess. Never raises on non-zero exit / timeout."""
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=cwd,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        return ExecResult(
            exit_code=proc.returncode if proc.returncode is not None else -1,
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr_b.decode(errors="replace"),
            timed_out=False,
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        stdout_b, stderr_b = b"", b""
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=5)
        except asyncio.TimeoutError:
            pass
        return ExecResult(
            exit_code=-1,
            stdout=stdout_b.decode(errors="replace"),
            stderr=stderr_b.decode(errors="replace"),
            timed_out=True,
        )


async def _run_subprocess(cmd: List[str], timeout_s: int) -> ExecResult:
    """Like `_exec_with_timeout` but raises on non-zero exit.

    Used for agent-internal operations (e.g. `apptainer instance start`)
    where a failure should abort the rollout, not be reflected back to the
    model as a tool-result error.
    """
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        raise RuntimeError(f"command timed out after {timeout_s}s: {' '.join(cmd)}")

    stdout = stdout_b.decode(errors="replace")
    stderr = stderr_b.decode(errors="replace")
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed (exit={proc.returncode}): {' '.join(cmd)}\nstdout: {stdout}\nstderr: {stderr}"
        )
    return ExecResult(exit_code=0, stdout=stdout, stderr=stderr, timed_out=False)


def make_instance_name(prefix: str = "vcqa") -> str:
    """Apptainer instance names must be unique on the host."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


# Re-exported in case callers want to override the apptainer search path.
def apptainer_on_path() -> bool:
    return shutil.which("apptainer") is not None or os.environ.get("VCQA_FAKE_APPTAINER") == "1"
