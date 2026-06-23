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

"""Async SWE environment adapter over ``nemo_gym.sandbox``.

Provides a thin async wrapper around a sandbox that any agent or the verifier
can use to run commands and move files in and out of the sandbox.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from nemo_gym.sandbox import AsyncSandbox, SandboxProvider, SandboxSpec


class AsyncSweEnvironment:
    """Thin async wrapper around a started ``AsyncSandbox``.

    Agents drive their own loop with ``execute``/``upload``/``download``; the
    verifier uses the same surface to run eval recipes. The environment never
    owns trajectory capture or grading logic — only sandbox I/O.
    """

    def __init__(self, sandbox: AsyncSandbox) -> None:
        """Wrap an already-started sandbox.

        Args:
            sandbox (AsyncSandbox): A started sandbox to drive I/O against.
        """
        self._sandbox = sandbox
        self._closed = False

    @classmethod
    async def start(
        cls,
        provider: Mapping[str, Any] | SandboxProvider,
        spec: SandboxSpec,
    ) -> "AsyncSweEnvironment":
        """Create and start a fresh sandbox and return the environment.

        Args:
            provider (Mapping[str, Any] | SandboxProvider): The sandbox provider
                config or instance to launch the sandbox with.
            spec (SandboxSpec): The sandbox spec describing image, workdir, env,
                and other launch options.

        Returns:
            AsyncSweEnvironment: An environment wrapping the started sandbox.
        """
        sandbox = AsyncSandbox(provider, spec)
        await sandbox.start()
        return cls(sandbox)

    @property
    def sandbox(self) -> AsyncSandbox:
        """The wrapped sandbox.

        Returns:
            AsyncSandbox: The underlying sandbox instance.
        """
        return self._sandbox

    @property
    def sandbox_id(self) -> str | None:
        """The provider-assigned sandbox identifier.

        Returns:
            str | None: The sandbox id, or ``None`` if the sandbox has no handle.
        """
        handle = getattr(self._sandbox, "_handle", None)
        return handle.sandbox_id if handle is not None else None

    @property
    def provider_name(self) -> str | None:
        """The name of the provider backing the sandbox.

        Returns:
            str | None: The provider name, or ``None`` if the sandbox has no handle.
        """
        handle = getattr(self._sandbox, "_handle", None)
        return handle.provider_name if handle is not None else None

    async def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        user: str | int | None = "root",
        timeout_s: int | float | None = None,
        is_eval: bool = False,
    ) -> dict[str, Any]:
        """Run a command in the sandbox and return a normalized result.

        Args:
            command (str): The shell command to execute.
            cwd (str | None): Working directory for the command, or ``None`` to
                use the sandbox default.
            user (str | int | None): User to run the command as. Defaults to
                ``"root"``.
            timeout_s (int | float | None): Optional timeout in seconds.
            is_eval (bool): Marks the command as part of evaluation.

        Returns:
            dict[str, Any]: A dict with ``output`` (combined stdout and stderr),
                ``returncode``, ``stdout``, ``stderr``, and ``error_type``.
        """
        result = await self._sandbox.exec(command, cwd=cwd, env=None, timeout_s=timeout_s, user=user)
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        output = "\n".join(part for part in (stdout, stderr) if part)
        return {
            "output": output,
            "returncode": result.return_code,
            "stdout": stdout,
            "stderr": stderr,
            "error_type": result.error_type,
        }

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        """Upload a local file into the sandbox.

        Args:
            local_path (Path | str): Path to the file on the host.
            remote_path (str): Destination path inside the sandbox.
        """
        await self._sandbox.upload(local_path, remote_path)

    async def download(self, remote_path: str, local_path: Path | str) -> None:
        """Download a file from the sandbox to the host.

        Args:
            remote_path (str): Source path inside the sandbox.
            local_path (Path | str): Destination path on the host.
        """
        await self._sandbox.download(remote_path, local_path)

    async def write_text(self, remote_path: str, content: str) -> None:
        """Write a string to a file inside the sandbox via a temporary upload.

        Args:
            remote_path (str): Destination path inside the sandbox.
            content (str): The text content to write.
        """
        tmp = tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8")
        try:
            tmp.write(content)
            tmp.flush()
            tmp.close()
            await self._sandbox.upload(tmp.name, remote_path)
        finally:
            os.unlink(tmp.name)

    async def cleanup(self) -> None:
        """Stop the sandbox. Idempotent: subsequent calls are no-ops."""
        if self._closed:
            return
        self._closed = True
        await self._sandbox.stop()

    async def __aenter__(self) -> "AsyncSweEnvironment":
        """Enter the async context manager.

        Returns:
            AsyncSweEnvironment: This environment instance.
        """
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit the async context manager and stop the sandbox.

        Args:
            exc_type (Any): The exception type, if one was raised.
            exc_val (Any): The exception instance, if one was raised.
            exc_tb (Any): The traceback, if an exception was raised.
        """
        await self.cleanup()
