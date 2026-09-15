# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sandbox-backed NOOA ShellTools for Gym-hosted rollouts.

The dev/tui BenchAgent expects host-side ShellTools. In a Gym rollout the
agent must operate inside the resources server's seeded sandbox instead, and
the swebench verifier extracts the model patch via uncommitted ``git diff`` in
that same sandbox. This module re-points ShellTools execution and file IO at a
Gym :class:`AsyncSandbox` so every command, read, and edit lands in the
container.

Design constraints honored here:

- ``cd`` persists across ``exec`` calls because every call passes ``cwd=`` and
  the session refreshes its tracked working directory from ``pwd`` after each
  command. Environment exports do NOT persist (fresh shell per exec).
- File IO travels as base64 payloads through ``tee`` so only the base64
  alphabet crosses the process boundary.
- The sandbox lifecycle belongs to the resources server; ``close()`` is a no-op.
- When ripgrep is absent (it is absent in swebench images), the pure-search
  anchor harvest degrades to ``None`` rather than running a doomed probe.
"""

from __future__ import annotations

import base64
import posixpath
import shlex
from typing import Any

from nooa.tools.shell_tools import FileWrite, Match, ShellTools


class SandboxBashSession:
    """Duck-typed NOOA ``BashSession`` backed by a Gym ``AsyncSandbox``."""

    def __init__(
        self,
        sandbox: Any,
        cwd: str = ".",
        init_command: str | None = None,
    ) -> None:
        self._sandbox = sandbox
        self._cwd = cwd
        self._init_command = init_command
        self._started = False
        self._has_rg: bool | None = None

    async def start(self) -> None:
        if self._started:
            return
        pwd = await self._sandbox.exec("pwd", timeout_s=5)
        if pwd.return_code == 0 and pwd.stdout and pwd.stdout.strip():
            self._cwd = pwd.stdout.strip()
        base64_probe = await self._sandbox.exec("command -v base64", timeout_s=5)
        if base64_probe.return_code != 0:
            raise RuntimeError("sandbox does not provide base64; sandbox file IO is unavailable")
        rg_probe = await self._sandbox.exec("command -v rg", timeout_s=5)
        self._has_rg = rg_probe.return_code == 0
        if self._init_command:
            await self._sandbox.exec(self._init_command, cwd=self._cwd, timeout_s=60)
        self._started = True

    async def run_with_timeout_flag(self, command: str, timeout: float = 30.0) -> tuple[str, str, int, bool]:
        if not self._started:
            await self.start()
        result = await self._sandbox.exec(command, cwd=self._cwd, timeout_s=timeout)
        timed_out = getattr(result, "error_type", None) == "timeout"
        stdout = result.stdout if result.stdout is not None else ""
        stderr = result.stderr if result.stderr is not None else ""
        pwd = await self._sandbox.exec("pwd", timeout_s=5)
        if pwd.return_code == 0 and pwd.stdout and pwd.stdout.strip():
            self._cwd = pwd.stdout.strip()
        return stdout, stderr, int(result.return_code), timed_out

    async def run(self, command: str, timeout: float = 30.0) -> tuple[str, str, int]:
        stdout, stderr, code, _timed_out = await self.run_with_timeout_flag(command, timeout=timeout)
        return stdout, stderr, code

    async def run_stream(self, command: str, timeout: float = 30.0):
        """Buffered stream: docker exec has no incremental transport.

        The sentinel protocol matches the host ``BashSession`` contract.
        """
        stdout, stderr, code, timed_out = await self.run_with_timeout_flag(command, timeout=timeout)
        for line in stdout.splitlines():
            yield ("stdout", line)
        for line in stderr.splitlines():
            yield ("stderr", line)
        yield ("__done__", f"{code},{1 if timed_out else 0}")

    async def close(self) -> None:
        # The resources server owns the sandbox lifecycle and stops it during
        # verification. This session must never stop it.
        return None


class SandboxShellTools(ShellTools):
    """ShellTools whose commands and file edits land inside one Gym sandbox."""

    def __init__(
        self,
        sandbox: Any,
        cwd: str = ".",
        init_command: str | None = None,
        **kwargs: Any,
    ) -> None:
        # Deliberately skip ShellTools.__init__: it would construct a host
        # BashSession. Re-implement its small body against the sandbox session.
        from nooa.skill import Skill

        Skill.__init__(self, **kwargs)
        self.cwd = cwd  # container-side path string
        self._session = SandboxBashSession(sandbox, cwd=cwd, init_command=init_command)

    def _resolve_path(self, path: str) -> str:
        if not posixpath.isabs(path):
            resolved = posixpath.normpath(posixpath.join(self.cwd, path))
        else:
            resolved = posixpath.normpath(path)
        return resolved

    async def _read_bytes(self, resolved_path: str) -> bytes:
        result = await self._session.run_with_timeout_flag(f"base64 -w 0 {shlex.quote(resolved_path)}", timeout=30.0)
        stdout, stderr, code, _timed_out = result
        if code != 0:
            raise FileNotFoundError(f"could not read {resolved_path} in the sandbox: {stderr.strip()}")
        return base64.b64decode(stdout)

    async def _write_bytes(self, resolved_path: str, data: bytes) -> None:
        parent = posixpath.dirname(resolved_path)
        if parent:
            mkdir = await self._session.run_with_timeout_flag(f"mkdir -p {shlex.quote(parent)}", timeout=30.0)
            if mkdir[2] != 0:
                raise OSError(f"could not create {parent} in the sandbox: {mkdir[1].strip()}")
        quoted_path = shlex.quote(resolved_path)
        append = False
        for offset in range(0, len(data), 61440):
            chunk = base64.b64encode(data[offset : offset + 61440]).decode()
            tee = "tee -a" if append else "tee"
            command = f"printf %s {shlex.quote(chunk)} | base64 -d | {tee} {quoted_path} >/dev/null"
            result = await self._session.run_with_timeout_flag(command, timeout=30.0)
            if result[2] != 0:
                raise OSError(f"could not write {resolved_path} in the sandbox: {result[1].strip()}")
            append = True
        if not append:
            command = f"base64 -d </dev/null | tee {quoted_path} >/dev/null"
            result = await self._session.run_with_timeout_flag(command, timeout=30.0)
            if result[2] != 0:
                raise OSError(f"could not write {resolved_path} in the sandbox: {result[1].strip()}")

    async def read(self, path: str, lines: tuple[int, int] | None = None) -> Match:
        resolved = self._resolve_path(path)
        content = (await self._read_bytes(resolved)).decode(errors="replace")
        all_lines = content.splitlines(keepends=True)
        total = len(all_lines)
        if lines is not None:
            start = max(1, lines[0])
            end = min(total, lines[1])
            text = "".join(all_lines[start - 1 : end])
            return Match(str(path), start, end, text, resolved_path=resolved)
        return Match(str(path), 1, total, content, resolved_path=resolved)

    async def replace(self, target: Any, old_or_new: str = "", new: str | None = None) -> FileWrite:
        if isinstance(target, Match):
            new_text = old_or_new
            resolved = target.resolved_path
            content = (await self._read_bytes(resolved)).decode(errors="replace")
            all_lines = content.splitlines(keepends=True)
            before = all_lines[: target.start - 1]
            after = all_lines[target.end :]
            if new_text and not new_text.endswith("\n") and after:
                new_text += "\n"
            new_content = "".join(before) + new_text + "".join(after)
            await self._write_bytes(resolved, new_content.encode())
            return FileWrite(
                path=target.path,
                message=f"Edited {target.path} (replaced lines {target.start}-{target.end})",
                diff=f"--- a/{target.path}\n+++ b/{target.path}",
                new_text=new_text,
            )
        if isinstance(target, str):
            if new is None:
                raise ValueError("replace(path, old, new) requires 3 arguments.")
            resolved = self._resolve_path(target)
            content = (await self._read_bytes(resolved)).decode(errors="replace")
            count = content.count(old_or_new)
            if count == 0:
                raise ValueError(f"old text not found in {target}.")
            if count > 1:
                raise ValueError(f"old text matched {count} times in {target}.")
            new_content = content.replace(old_or_new, new, 1)
            await self._write_bytes(resolved, new_content.encode())
            return FileWrite(
                path=target,
                message=f"Edited {target}",
                diff=f"--- a/{target}\n+++ b/{target}",
                new_text=new,
            )
        raise TypeError(f"target must be a Match or file path str, got {type(target).__name__}")

    async def write_file(self, path: str, content: str) -> FileWrite:
        resolved = self._resolve_path(path)
        await self._write_bytes(resolved, content.encode())
        line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return FileWrite(path=path, message=f"Created {path} ({line_count} lines)", new_text=content)

    async def _harvest_matches(self, command: str, displayed_stdout: str) -> list[Match] | None:
        session = self._session
        if getattr(session, "_has_rg", None) is not True:
            return None
        return await super()._harvest_matches(command, displayed_stdout)
