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
"""Code sandbox of the ARC code agent: a pool of fork-per-run Python servers, plus code and answer extraction.

Each pool member is a long-lived ``sandbox/sandbox_server.py`` process (line protocol on stdin/stdout) that preloads the
helper functions once. Every code run forks a child that applies resource limits, swaps in restricted builtins (no
``open``, import allowlist), runs the puzzle header, silently replays the earlier code blocks of the episode, and then
runs the new block with its stdout captured. The parent kills the child at the deadline. Episode state is rebuilt by
replay, so no process outlives a run.

Isolation this gives: wall-clock, CPU, address-space, file-size and open-file limits; a fresh scratch directory per
run; a scrubbed environment (no inherited API keys); no ``open``/``input``/``exit``; imports limited to an allowlist
(no ``os``, ``subprocess`` or ``socket``). What it does not give: the restricted builtins are not a security boundary
(a determined program can escape them through object introspection), and there is no network or filesystem namespace
unless ``command_prefix`` wraps the server in one (for example ``["unshare", "-rn"]`` or a bubblewrap/container
command). Run the agent in a container for untrusted models.
"""

import asyncio
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from resources_servers.arc_code.grid import is_grid


SERVER_SCRIPT = Path(__file__).parent / "sandbox" / "sandbox_server.py"
TOOLS_DIR = SERVER_SCRIPT.parent

_CODE = re.compile(r"<code>(.*?)</code>", re.DOTALL)
# The code of a native tool call ends at the first closing tag. Some models mix formats and close with </code>.
_TOOL_CALL = re.compile(
    r"<tool_call>\s*<function=[^>\n]*>\s*(?:<parameter=[^>\n]*>\s*)?(.*?)\s*(?:</parameter>|</function>|</code>|</tool_call>|<\|im_end\|>|$)",
    re.S,
)
_TAG_LINE = re.compile(r"^\s*</?(?:parameter|function|tool_call)(?:=[^>\n]*)?>\s*$", re.M)
CHECK_PASSED = re.compile(r"check: passed (\d+)/(\d+)")
_CHECK_CALL = re.compile(r"check\(\s*([A-Za-z_][A-Za-z_0-9]*)")
AUTO_PREDICT = (
    "import json as _json\n_g = _to_grid(_run_with_timeout({fn}, np.asarray(test_inputs[0], dtype=int), CHECK_TIMEOUT_SECONDS))\n"
    "print(_json.dumps([[int(v) for v in row] for row in _g]))"
)


def _strip_tag_lines(code: Optional[str]) -> Optional[str]:
    """Drop lines that are only tool-call tags (a model may open with <code> and still close with </parameter>)."""
    return None if code is None else _TAG_LINE.sub("", code)


def extract_code(text: str, parse_tool_calls: bool = False) -> Optional[str]:
    """Code of the last <code> block; an unterminated final <code> (generation stopped on the stop string) counts.

    With ``parse_tool_calls`` a native ``<tool_call><function=...>`` block counts as a code block too.
    """
    if parse_tool_calls and "<tool_call>" in text and "<code>" not in text:
        calls = _TOOL_CALL.findall(text)
        if calls:
            return _strip_tag_lines(calls[-1])
    if parse_tool_calls:
        return _strip_tag_lines(extract_code(text))
    found = _CODE.findall(text)
    tail = text.rsplit("</code>", 1)[-1] if found else text
    if "<code>" in tail:
        return tail.rsplit("<code>", 1)[-1]
    return found[-1] if found else None


def checked_function(code: str) -> Optional[str]:
    """Name passed to the last ``check(...)`` call in ``code``."""
    names = _CHECK_CALL.findall(code)
    return names[-1] if names else None


def full_check_pass(output: str) -> bool:
    """True when the last ``check`` line of a run's output reports every example pair passing."""
    found = CHECK_PASSED.findall(output)
    return bool(found) and found[-1][0] == found[-1][1] and found[-1][1] != "0"


def tool_response_text(reply: Dict[str, Any]) -> str:
    """The text of the tool turn for a sandbox reply: stdout, then ``[error] ...`` when the run failed."""
    body = reply.get("output", "")
    if reply.get("error"):
        body = (body + "\n" if body else "") + f"[error] {reply['error']}"
    return f"<tool_response>\n{body}\n</tool_response>"


START_TIMEOUT_S = 300


class _Server:
    """One sandbox server process (line protocol)."""

    def __init__(self, python: str, command_prefix: List[str]) -> None:
        self.python = python
        self.command_prefix = command_prefix
        self.proc: Optional[asyncio.subprocess.Process] = None

    async def start(self) -> None:
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "PYTHONHASHSEED": "0",
            "PYTHONNOUSERSITE": "1",
        }
        self.proc = await asyncio.create_subprocess_exec(
            *self.command_prefix,
            self.python,
            "-I",
            str(SERVER_SCRIPT),
            str(TOOLS_DIR),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            env=env,
            limit=1 << 26,
        )
        # A fresh interpreter imports numpy/scipy/networkx; on a node busy with hundreds of rollouts that can take
        # minutes, so allow START_TIMEOUT_S before giving up.
        ready = await asyncio.wait_for(self.proc.stdout.readline(), timeout=START_TIMEOUT_S)
        if b'"ready"' not in ready:
            raise RuntimeError(f"sandbox server failed to start: {ready!r}")

    async def call(self, req: Dict[str, Any], wait_s: float) -> Dict[str, Any]:
        """One request; a failure (or a server that could not be restarted) comes back as an error reply, never as
        an exception, so one bad sandbox never ends the rollout or poisons the pool."""
        if self.proc is None:  # an earlier restart failed: try again now
            try:
                await self.start()
            except Exception as exc:
                await self.close()
                return {
                    "output": "",
                    "timed_out": False,
                    "error": f"sandbox unavailable ({type(exc).__name__}: {exc})",
                }
        assert self.proc is not None and self.proc.stdin is not None and self.proc.stdout is not None
        try:
            self.proc.stdin.write((json.dumps(req) + "\n").encode())
            await self.proc.stdin.drain()
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout=wait_s)
            if not line:
                raise RuntimeError("sandbox server closed")
            reply = json.loads(line)
            reply.setdefault("output", "")
            return reply
        except Exception as exc:  # restart the server and report to the model
            await self.close()
            try:
                await self.start()
            except Exception:
                await self.close()  # left stopped; the next call starts it again
            return {"output": "", "timed_out": False, "error": f"sandbox restarted ({type(exc).__name__}: {exc})"}

    async def close(self) -> None:
        if self.proc is not None:
            try:
                self.proc.kill()
            except ProcessLookupError:
                pass
            await self.proc.wait()
            self.proc = None


class SandboxPool:
    """A bounded pool of sandbox servers; ``run`` takes one for the duration of a code run."""

    def __init__(
        self,
        num_workers: int = 8,
        timeout_s: float = 30.0,
        out_chars: int = 4000,
        mem_mb: int = 6144,
        extra_imports: Optional[List[str]] = None,
        recursion_limit: int = 0,
        python: Optional[str] = None,
        command_prefix: Optional[List[str]] = None,
    ) -> None:
        self.num_workers = num_workers
        self.timeout_s = timeout_s
        self.out_chars = out_chars
        self.mem_mb = mem_mb
        self.extra_imports = list(extra_imports or [])
        self.recursion_limit = recursion_limit
        self.python = python or sys.executable
        self.command_prefix = list(command_prefix or [])
        self._idle: Optional[asyncio.Queue] = None
        self._servers: List[_Server] = []
        self._init_lock: Optional[asyncio.Lock] = None

    async def _ensure_started(self) -> asyncio.Queue:
        if self._idle is not None:
            return self._idle
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._idle is None:
                idle: asyncio.Queue = asyncio.Queue()
                servers = [_Server(self.python, self.command_prefix) for _ in range(self.num_workers)]
                # A server that fails to start stays stopped; its first call starts it again.
                started = await asyncio.gather(*(s.start() for s in servers), return_exceptions=True)
                for s, result in zip(servers, started):
                    if isinstance(result, BaseException):
                        await s.close()
                self._servers = servers
                for s in servers:
                    idle.put_nowait(s)
                self._idle = idle
        return self._idle

    async def run(self, header: str, codes: List[str], code: str, out_chars: Optional[int] = None) -> Dict[str, Any]:
        """Run ``code`` after the header and the earlier ``codes`` (replayed silently); returns output/timed_out/error."""
        idle = await self._ensure_started()
        server = await idle.get()
        try:
            req: Dict[str, Any] = {
                "id": 0,
                "header": header,
                "codes": codes,
                "code": code,
                "timeout_s": self.timeout_s,
                "out_chars": out_chars or self.out_chars,
                "mem_mb": self.mem_mb,
            }
            if self.extra_imports:
                req["extra_imports"] = self.extra_imports
            if self.recursion_limit:
                req["recursion_limit"] = self.recursion_limit
            return await server.call(req, wait_s=self.timeout_s + 15)
        finally:
            idle.put_nowait(server)

    async def predict_answer(self, header: str, codes: List[str], fn: str) -> Optional[List[List[int]]]:
        """The prediction of sandbox function ``fn`` on the test input as a grid, or None if it failed."""
        reply = await self.run(header, codes, AUTO_PREDICT.format(fn=fn), out_chars=100000)
        if reply.get("error") or reply.get("timed_out"):
            return None
        try:
            grid = json.loads((reply.get("output") or "").strip().splitlines()[-1])
        except (ValueError, IndexError):
            return None
        # The function may return anything (an int, a flat list, ragged rows): only a valid grid is an answer.
        return grid if is_grid(grid) else None

    async def close(self) -> None:
        for s in self._servers:
            await s.close()
        self._servers = []
        self._idle = None
