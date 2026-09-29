# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Codex ``exec_command`` / ``write_stdin`` semantics (codex-rs core/src/unified_exec).

Derived from openai/codex (Apache-2.0, Copyright OpenAI); modified by NVIDIA: no sandbox,
approvals, shell snapshots, or network proxy.

Commands run as ``<shell> -lc <cmd>`` (``-c`` without login) in their own process group, with pipes
or a PTY. Each call waits for the yield time or until the process exits, then returns the output
produced since the previous call in Codex's text format. Processes that are still running keep a
numeric session ID for ``write_stdin``. There is no sandbox: commands run with this server's
permissions.
"""

from __future__ import annotations

import asyncio
import fcntl
import os
import random
import signal
import termios
import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


MIN_YIELD_TIME_MS = 250
MIN_EMPTY_YIELD_TIME_MS = 5_000
MAX_YIELD_TIME_MS = 30_000
MAX_BACKGROUND_TERMINAL_TIMEOUT_MS = 300_000
DEFAULT_EXEC_YIELD_TIME_MS = 10_000
DEFAULT_WRITE_STDIN_YIELD_TIME_MS = 250
DEFAULT_MAX_OUTPUT_TOKENS = 10_000
OUTPUT_MAX_BYTES = 1024 * 1024
MAX_PROCESSES = 64
_PROTECTED_RECENT_PROCESSES = 8
_POST_EXIT_CLOSE_WAIT_S = 0.05
_APPROX_BYTES_PER_TOKEN = 4

UNIFIED_EXEC_ENV = {
    "NO_COLOR": "1",
    "TERM": "dumb",
    "LANG": "C.UTF-8",
    "LC_CTYPE": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "COLORTERM": "",
    "PAGER": "cat",
    "GIT_PAGER": "cat",
    "GH_PAGER": "cat",
    "CODEX_CI": "1",
}


class ExecError(Exception):
    """A model-visible tool error; ``str()`` is the full text returned to the model."""


# ---- Output truncation (codex-rs utils/string/src/truncate.rs, core/src/tools/context.rs) ----


def approx_token_count(byte_count: int) -> int:
    return (byte_count + _APPROX_BYTES_PER_TOKEN - 1) // _APPROX_BYTES_PER_TOKEN


def _truncate_middle(text: str, max_tokens: int) -> str:
    """``truncate_middle_with_token_budget``: keep the head and tail on UTF-8 boundaries."""
    data = text.encode("utf-8")
    max_bytes = max_tokens * _APPROX_BYTES_PER_TOKEN
    if not data or (max_tokens > 0 and len(data) <= max_bytes):
        return text
    if max_bytes == 0:
        return f"…{approx_token_count(len(data))} tokens truncated…"
    left_budget = max_bytes // 2
    tail_start_target = len(data) - (max_bytes - left_budget)
    prefix_end, suffix_start, offset = 0, len(data), 0
    for char in text:
        char_end = offset + len(char.encode("utf-8"))
        if char_end <= left_budget:
            prefix_end = char_end
        elif offset >= tail_start_target and suffix_start == len(data):
            suffix_start = offset
        offset = char_end
    suffix_start = max(suffix_start, prefix_end)
    marker = f"…{approx_token_count(len(data) - max_bytes)} tokens truncated…"
    return data[:prefix_end].decode("utf-8") + marker + data[suffix_start:].decode("utf-8")


def _formatted_truncate(text: str, max_tokens: int) -> str:
    """``formatted_truncate_text`` with a token policy."""
    if len(text.encode("utf-8")) <= max_tokens * _APPROX_BYTES_PER_TOKEN:
        return text
    original = approx_token_count(len(text.encode("utf-8")))
    total_lines = len(text.splitlines())
    return (
        f"Warning: truncated output (original token count: {original})\n"
        f"Total output lines: {total_lines}\n\n{_truncate_middle(text, max_tokens)}"
    )


class HeadTailBuffer:
    """Keeps the first and last halves of at most ``OUTPUT_MAX_BYTES`` of output."""

    def __init__(self) -> None:
        self.head = bytearray()
        self.tail = bytearray()
        self.omitted = 0
        self.total = 0

    def push(self, data: bytes) -> None:
        self.total += len(data)
        head_room = OUTPUT_MAX_BYTES // 2 - len(self.head)
        if head_room > 0:
            self.head += data[:head_room]
            data = data[head_room:]
        self.tail += data
        excess = len(self.tail) - (OUTPUT_MAX_BYTES - OUTPUT_MAX_BYTES // 2)
        if excess > 0:
            del self.tail[:excess]
            self.omitted += excess

    def text(self) -> str:
        if not self.omitted:
            return (bytes(self.head) + bytes(self.tail)).decode("utf-8", "replace")
        marker = f"\n... {self.omitted} bytes omitted ...\n".encode()
        return (bytes(self.head) + marker + bytes(self.tail)).decode("utf-8", "replace")


def _render_output(output: HeadTailBuffer, max_tokens: int) -> str:
    """``truncated_output_with_policy`` with a token policy."""
    text = output.text()
    if not output.omitted:
        return _formatted_truncate(text, max_tokens)
    marker = f"... {output.omitted} bytes omitted ..."
    if len(text.encode("utf-8")) <= max_tokens * _APPROX_BYTES_PER_TOKEN:
        return text if marker in text else f"{marker}\n{text}"
    truncated = _truncate_middle(text, max_tokens)
    notice = "" if marker in truncated else f"{marker}\n"
    return (
        f"Warning: truncated output (original token count: {approx_token_count(output.total)})\n{notice}\n{truncated}"
    )


def format_exec_response(
    *,
    output: HeadTailBuffer,
    wall_time_s: float,
    exit_code: Optional[int],
    session_id: Optional[int],
    max_output_tokens: int,
) -> str:
    """``ExecCommandToolOutput::response_text``."""
    sections = [f"Chunk ID: {random.randrange(16**6):06x}", f"Wall time: {wall_time_s:.4f} seconds"]
    if exit_code is not None:
        sections.append(f"Process exited with code {exit_code}")
    if session_id is not None:
        sections.append(f"Process running with session ID {session_id}")
    sections.append(f"Original token count: {approx_token_count(output.total)}")
    sections.append("Output:")
    header = "\n".join(sections)

    # History applies a 20% serialization allowance to the whole response; shrink the token budget
    # until header + body fit in it, so the output is not truncated a second time.
    budget = int(max_output_tokens * _APPROX_BYTES_PER_TOKEN * 1.2) - len(header) - 1
    tokens = max_output_tokens
    body = _render_output(output, tokens)
    while len(body.encode("utf-8")) > budget and tokens > 0:
        tokens = max(0, tokens - approx_token_count(len(body.encode("utf-8")) - budget))
        body = _render_output(output, tokens)
    return f"{header}\n{body}"


# ---- Processes ----


@dataclass
class ExecProcess:
    process: asyncio.subprocess.Process
    fd: int
    tty: bool
    last_used: float = field(default_factory=time.monotonic)
    pending: HeadTailBuffer = field(default_factory=HeadTailBuffer)
    output_closed: bool = False
    exit_code: Optional[int] = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def on_readable(self) -> None:
        try:
            data = os.read(self.fd, 65536)
        except BlockingIOError:
            return
        except OSError:  # EIO: every PTY slave closed.
            data = b""
        if data:
            self.pending.push(data)
        else:
            self.output_closed = True
            asyncio.get_running_loop().remove_reader(self.fd)
        self.changed.set()

    async def wait_exit(self) -> None:
        code = await self.process.wait()
        # Like codex-rs utils/pty exit_code_from_status: death by signal N reports 128 + N.
        self.exit_code = 128 - code if code < 0 else code
        self.changed.set()

    async def collect_until(self, deadline: float) -> HeadTailBuffer:
        """``collect_output_until_deadline``: stop at the deadline, or once the process has exited
        and its output closed (waiting at most 50 ms for background writers after exit)."""
        post_exit_deadline: Optional[float] = None
        while True:
            self.changed.clear()
            now = time.monotonic()
            if self.exit_code is not None:
                if self.output_closed:
                    break
                if post_exit_deadline is None:
                    post_exit_deadline = min(deadline, now + _POST_EXIT_CLOSE_WAIT_S)
            stop_at = deadline if post_exit_deadline is None else post_exit_deadline
            if now >= stop_at:
                break
            try:
                await asyncio.wait_for(self.changed.wait(), stop_at - now)
            except asyncio.TimeoutError:
                pass
        collected, self.pending = self.pending, HeadTailBuffer()
        return collected

    def signal_group(self, signum: int) -> None:
        try:
            os.killpg(self.process.pid, signum)
        except ProcessLookupError:
            pass

    def close(self, *, kill_group: bool) -> None:
        """Stop reading; with ``kill_group``, also kill the command and anything it left running."""
        if kill_group:
            self.signal_group(signal.SIGKILL)
        if not self.output_closed:
            self.output_closed = True
            try:
                asyncio.get_running_loop().remove_reader(self.fd)
            except RuntimeError:
                pass
        try:
            os.close(self.fd)
        except OSError:
            pass


def _set_controlling_tty() -> None:  # pragma: no cover - runs in the forked child
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)


def _parse_int(arguments: Mapping[str, Any], name: str, default: int) -> int:
    value = arguments.get(name)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ExecError(f"failed to parse function arguments: invalid value for `{name}`: expected a number >= 0")
    return int(value)


def _parse_bool(arguments: Mapping[str, Any], name: str, default: bool) -> bool:
    value = arguments.get(name, default)
    if value is None:
        return default
    if not isinstance(value, bool):
        raise ExecError(f"failed to parse function arguments: invalid type for `{name}`: expected a boolean")
    return value


class ExecRuntime:
    """The running commands of one workspace session."""

    def __init__(self, *, cwd: str, shell: str, login: bool, env: Mapping[str, str]) -> None:
        self.cwd = cwd
        self.shell = shell
        self.login = login
        self.env = dict(env)
        self.processes: dict[int, ExecProcess] = {}
        # Every spawned process group, including exited commands whose background jobs may linger.
        self.process_groups: set[int] = set()
        self.closed = False

    def _allocate_id(self) -> int:
        while True:
            process_id = random.randrange(1_000, 100_000)
            if process_id not in self.processes:
                return process_id

    def _prune(self) -> None:
        """Upstream LRU pruning: keep the 8 most recently used; prefer exited processes."""
        if len(self.processes) < MAX_PROCESSES:
            return
        by_recency = sorted(self.processes.items(), key=lambda item: item[1].last_used, reverse=True)
        protected = {process_id for process_id, _ in by_recency[:_PROTECTED_RECENT_PROCESSES]}
        candidates = [item for item in reversed(by_recency) if item[0] not in protected]
        victim = next((item for item in candidates if item[1].exit_code is not None), None) or (
            candidates[0] if candidates else None
        )
        if victim is not None:
            self.processes.pop(victim[0]).close(kill_group=True)

    def _resolve_workdir(self, workdir: object) -> str:
        if workdir is None or workdir == "":
            return self.cwd
        if not isinstance(workdir, str):
            raise ExecError("failed to parse function arguments: invalid type for `workdir`: expected a string")
        path = os.path.normpath(os.path.join(self.cwd, workdir))
        if not os.path.isdir(path):
            raise ExecError(f"exec_command failed: workdir {path} is not a directory")
        return path

    async def _spawn(self, argv: list[str], cwd: str, tty: bool) -> ExecProcess:
        if tty:
            master, slave = os.openpty()
            stdin = stdout = slave
            preexec = _set_controlling_tty
        else:
            master, slave = os.pipe()
            stdin, stdout, preexec = asyncio.subprocess.DEVNULL, slave, None
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=self.env,
                stdin=stdin,
                stdout=stdout,
                stderr=stdout,
                start_new_session=True,
                preexec_fn=preexec,
            )
        except OSError as error:
            os.close(master)
            raise ExecError(f"exec_command failed: Failed to create unified exec process: {error}") from None
        finally:
            os.close(slave)
        os.set_blocking(master, False)
        self.process_groups.add(process.pid)
        exec_process = ExecProcess(process=process, fd=master, tty=tty)
        asyncio.get_running_loop().add_reader(master, exec_process.on_readable)
        asyncio.get_running_loop().create_task(exec_process.wait_exit())
        return exec_process

    async def _respond(
        self, process_id: int, process: ExecProcess, deadline: float, started: float, max_tokens: int
    ) -> str:
        output = await process.collect_until(deadline)
        process.last_used = time.monotonic()
        exited = process.exit_code is not None
        if exited and self.processes.get(process_id) is process:
            del self.processes[process_id]
            process.close(kill_group=False)
        return format_exec_response(
            output=output,
            wall_time_s=time.monotonic() - started,
            exit_code=process.exit_code,
            session_id=None if exited else process_id,
            max_output_tokens=max_tokens,
        )

    async def exec_command(self, arguments: Mapping[str, Any]) -> str:
        cmd = arguments.get("cmd")
        if not isinstance(cmd, str):
            raise ExecError("failed to parse function arguments: missing field `cmd`")
        shell = arguments.get("shell") or self.shell
        if not isinstance(shell, str):
            raise ExecError("failed to parse function arguments: invalid type for `shell`: expected a string")
        login = _parse_bool(arguments, "login", self.login)
        tty = _parse_bool(arguments, "tty", False)
        yield_ms = min(
            max(_parse_int(arguments, "yield_time_ms", DEFAULT_EXEC_YIELD_TIME_MS), MIN_YIELD_TIME_MS),
            MAX_YIELD_TIME_MS,
        )
        max_tokens = _parse_int(arguments, "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)
        cwd = self._resolve_workdir(arguments.get("workdir"))
        if self.closed:
            raise ExecError("exec_command failed: the session is closed")

        self._prune()
        started = time.monotonic()
        process = await self._spawn([shell, "-lc" if login else "-c", cmd], cwd, tty)
        process_id = self._allocate_id()
        self.processes[process_id] = process
        async with process.lock:
            return await self._respond(process_id, process, started + yield_ms / 1000, started, max_tokens)

    async def write_stdin(self, arguments: Mapping[str, Any]) -> str:
        session_id = arguments.get("session_id")
        if isinstance(session_id, bool) or not isinstance(session_id, (int, float)):
            raise ExecError("failed to parse function arguments: missing field `session_id`")
        process_id = int(session_id)
        chars = arguments.get("chars") or ""
        if not isinstance(chars, str):
            raise ExecError("failed to parse function arguments: invalid type for `chars`: expected a string")
        requested_ms = max(
            _parse_int(arguments, "yield_time_ms", DEFAULT_WRITE_STDIN_YIELD_TIME_MS), MIN_YIELD_TIME_MS
        )
        if chars:
            yield_ms = min(requested_ms, MAX_YIELD_TIME_MS)
        else:
            yield_ms = min(max(requested_ms, MIN_EMPTY_YIELD_TIME_MS), MAX_BACKGROUND_TERMINAL_TIMEOUT_MS)
        max_tokens = _parse_int(arguments, "max_output_tokens", DEFAULT_MAX_OUTPUT_TOKENS)

        process = self.processes.get(process_id)
        if process is None:
            raise ExecError(f"write_stdin failed: Unknown process id {process_id}")
        async with process.lock:
            if self.processes.get(process_id) is not process:
                raise ExecError(f"write_stdin failed: Unknown process id {process_id}")
            if chars:
                if not process.tty:
                    if chars != "\x03":
                        raise ExecError(
                            "write_stdin failed: stdin is closed for this session; "
                            "rerun exec_command with tty=true to keep stdin open"
                        )
                    process.signal_group(signal.SIGINT)
                else:
                    try:
                        os.write(process.fd, chars.encode("utf-8"))
                    except OSError:
                        if process.exit_code is None:
                            raise ExecError("write_stdin failed: failed to write to stdin") from None
                    else:
                        # Give the process a moment to react so the poll below sees its output.
                        await asyncio.sleep(0.1)
            started = time.monotonic()
            return await self._respond(process_id, process, started + yield_ms / 1000, started, max_tokens)

    def close(self) -> None:
        self.closed = True
        for process in self.processes.values():
            process.close(kill_group=True)
        self.processes.clear()
        for process_group in self.process_groups:
            try:
                os.killpg(process_group, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.process_groups.clear()
