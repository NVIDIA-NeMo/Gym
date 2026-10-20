# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stirrup ``CodeExecToolProvider`` backed by a Gym ``AsyncSandbox``.

The GDPVal resources server owns the sandbox (it creates it, stages the
reference files under ``/workspace/reference_files`` and collects
``/workspace/deliverables`` and ``/trace`` at verify time). This provider only
*uses* an already-attached sandbox, so ``__aenter__`` returns the ``code_exec``
tool without creating anything and ``__aexit__`` leaves the sandbox alone.

Command semantics mirror the AA v2 Apptainer provider the GDPVal eval harness
runs: every command is an independent ``bash -c`` invocation in the working
directory (no shell state carries over), bounded by ``shell_timeout`` seconds,
and a timeout is reported to the model as an ``error_kind=timeout`` result
rather than raised.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from pathlib import Path
from shlex import quote

from stirrup.core.models import ImageContentBlock, Tool, ToolUseCountMetadata
from stirrup.tools.code_backends.base import (
    SHELL_TIMEOUT,
    CodeExecToolProvider,
    CodeExecutionParams,
    CommandResult,
)

logger = logging.getLogger(__name__)

# Same text the eval harness's Apptainer provider shows the model.
CODE_EXEC_DESCRIPTION = "Execute a shell command in the execution environment. Returns exit code, stdout, and stderr as XML."

# Bound on the tool's view of a single sandbox file (read_file_bytes / view_image).
MAX_READ_BYTES = 64 * 1024 * 1024


class SandboxCodeExecToolProvider(CodeExecToolProvider):
    """Run Stirrup ``code_exec`` calls inside a Gym ``AsyncSandbox``."""

    def __init__(
        self,
        sandbox,
        *,
        working_dir: str = "/workspace",
        shell_timeout: int = SHELL_TIMEOUT,
        description: str = CODE_EXEC_DESCRIPTION,
        allowed_commands: list[str] | None = None,
    ) -> None:
        super().__init__(allowed_commands=allowed_commands, shell_timeout=shell_timeout)
        self._sandbox = sandbox
        self._working_dir = working_dir.rstrip("/") or "/"
        self._description = description
        self.commands_run = 0
        self.commands_timed_out = 0

    # ------------------------------------------------------------------
    # Lifecycle: the resources server owns the sandbox.
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        return self.get_code_exec_tool(description=self._description)

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    @property
    def working_dir(self) -> str:
        return self._working_dir

    def _resolve_path(self, path: str) -> str:
        path = str(path)
        if path.startswith("/"):
            return path
        return f"{self._working_dir}/{path}"

    async def _exec(self, cmd: str, *, timeout: int | None) -> CommandResult:
        """Run ``cmd`` as one ``bash -c`` invocation in the working directory."""
        try:
            result = await self._sandbox.exec(
                f"bash -c {quote(cmd)}", cwd=self._working_dir, timeout_s=timeout
            )
        except (TimeoutError, asyncio.TimeoutError):
            raise
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        if getattr(result, "error_type", None):
            return CommandResult(
                exit_code=result.return_code if isinstance(result.return_code, int) else 1,
                stdout=stdout,
                stderr=stderr or str(result.error_type),
                error_kind="execution_error",
            )
        return CommandResult(exit_code=result.return_code, stdout=stdout, stderr=stderr)

    # ------------------------------------------------------------------
    # CodeExecToolProvider interface
    # ------------------------------------------------------------------

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        if timeout is None:
            timeout = self._shell_timeout
        if not self._check_allowed(cmd):
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command not allowed: '{cmd}' does not match any allowed patterns",
                error_kind="command_not_allowed",
                advice="Only commands matching the allowlist patterns are permitted.",
            )
        self.commands_run += 1
        try:
            return await self._exec(cmd, timeout=timeout)
        except (TimeoutError, asyncio.TimeoutError):
            self.commands_timed_out += 1
            logger.warning("Sandbox command timed out after %ss: %s", timeout, cmd[:100])
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                error_kind="timeout",
            )

    async def read_file_bytes(self, path: str) -> bytes:
        remote = self._resolve_path(path)
        if not await self.file_exists(remote):
            raise FileNotFoundError(f"File not found: {path}")
        size = await self._exec(f"stat -c %s {quote(remote)}", timeout=30)
        try:
            n = int((size.stdout or "0").strip().splitlines()[-1])
        except (ValueError, IndexError):
            n = 0
        if n > MAX_READ_BYTES:
            raise OSError(f"File too large to read through the tool ({n} bytes): {path}")
        with tempfile.TemporaryDirectory(prefix="gdpval-sbx-read-") as tmp:
            local = Path(tmp) / "file"
            await self._sandbox.download(remote, local)
            return local.read_bytes()

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        remote = self._resolve_path(path)
        parent = remote.rsplit("/", 1)[0] or "/"
        made = await self._exec(f"mkdir -p {quote(parent)}", timeout=30)
        if made.exit_code:
            raise OSError(f"Failed to create {parent}: {made.stderr}")
        with tempfile.TemporaryDirectory(prefix="gdpval-sbx-write-") as tmp:
            local = Path(tmp) / "file"
            local.write_bytes(content)
            await self._sandbox.upload(local, remote)

    async def file_exists(self, path: str) -> bool:
        result = await self._exec(f"test -f {quote(self._resolve_path(path))}", timeout=30)
        return result.exit_code == 0

    async def is_directory(self, path: str) -> bool:
        result = await self._exec(f"test -d {quote(self._resolve_path(path))}", timeout=30)
        return result.exit_code == 0

    async def list_files(self, path: str) -> list[str]:
        remote = self._resolve_path(path).rstrip("/") or "/"
        result = await self._exec(f"find {quote(remote)} -type f 2>/dev/null", timeout=60)
        if result.exit_code:
            return []
        files = []
        for line in (result.stdout or "").splitlines():
            line = line.strip()
            if not line:
                continue
            files.append(line[len(remote) + 1 :] if line.startswith(remote + "/") else line)
        return sorted(files)

    async def view_image(self, path: str) -> ImageContentBlock:
        return ImageContentBlock(data=await self.read_file_bytes(path))
