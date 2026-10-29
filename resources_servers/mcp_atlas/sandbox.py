# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One private MCP-Atlas agent-environment sandbox per rollout.

Each sandbox is the upstream ``ghcr.io/scaleapi/mcp-atlas`` image run through ``nemo_gym.sandbox`` (Apptainer on
Slurm). It serves only the MCP servers its task enables, on its own host port, so stateful servers (filesystem,
memory, git, desktop-commander) never leak state between tasks. Service credentials reach the sandbox through
``SandboxSpec.env``, which the provider writes to a private env file, never onto a command line.
"""

from __future__ import annotations

import asyncio
import contextlib
import json as json_module
import logging
import shutil
import socket
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Optional

from aiohttp import ClientError, ClientTimeout

from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxSpec
from nemo_gym.server_utils import request


logger = logging.getLogger(__name__)

LOG_PATH = "/tmp/mcp-atlas-sandbox.log"
POLL_INTERVAL_S = 2.0
LIST_TOOLS_ATTEMPTS = 5


class SandboxError(RuntimeError):
    """The sandbox could not be started or reached: an infrastructure failure, not a model one."""


@dataclass
class AtlasSandbox:
    """A running sandbox and the names its MCP servers gave the task's tools."""

    base_url: str
    tool_names: dict[str, str]
    missing_tools: list[str]
    sandbox: Optional[AsyncSandbox] = None
    server_task: Optional[asyncio.Task] = None
    port: Optional[int] = None
    scratch_dir: Optional[str] = None
    last_used_at: float = field(default_factory=monotonic)
    # Set once the provider confirmed the stop and the port, scratch directory and capacity slot are free.
    stopped: bool = False
    stop_lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)


def sandbox_tool_names(expected: list[str], listed: set[str], servers: list[str]) -> tuple[dict[str, str], list[str]]:
    """Map dataset tool names (``<server>_<tool>``) to the names the sandbox serves.

    The sandbox drops the server prefix when it serves a single server, so ``calculator_calculate`` is then
    served as ``calculate``.
    """
    names: dict[str, str] = {}
    missing: list[str] = []
    for name in expected:
        if name in listed:
            names[name] = name
            continue
        server = next((s for s in servers if name.startswith(f"{s}_")), None)
        bare = name[len(server) + 1 :] if server is not None else None
        if len(servers) == 1 and bare in listed:
            names[name] = bare
        else:
            missing.append(name)
    return names, missing


async def _http_json(method: str, url: str, timeout_s: float, json: Any = None) -> tuple[int, Any]:
    # One try: a tool call may change state, and the callers poll or report failures themselves.
    response = await request(
        method=method,
        url=url,
        json=json,
        timeout=ClientTimeout(total=timeout_s),
        _max_num_tries=1,
        _resend_after_disconnect=False,
    )
    text = await response.text()
    try:
        return response.status, json_module.loads(text)
    except ValueError:
        return response.status, text


async def call_sandbox_tool(
    box: AtlasSandbox, name: str, arguments: dict[str, Any], *, timeout_s: float
) -> tuple[int, Any]:
    """Call one tool; returns the sandbox's HTTP status and body. Raises on transport failure or timeout.

    Raises SandboxError for a stopped sandbox: its port may already serve another task's sandbox.
    """
    if box.stopped:
        raise SandboxError("the sandbox was stopped")
    box.last_used_at = monotonic()
    payload = {"tool_name": box.tool_names.get(name, name), "tool_args": arguments, "use_cache": False}
    return await _http_json("POST", f"{box.base_url}/call-tool", timeout_s, payload)


class PortAllocator:
    """Lease host ports for sandboxes that share the host network namespace."""

    def __init__(self, port_range: tuple[int, int]) -> None:
        start, end = port_range
        if not (1 <= start <= end <= 65535):
            raise ValueError(f"invalid sandbox port range: {port_range!r}")
        self._range = range(start, end + 1)
        self._leased: set[int] = set()

    @staticmethod
    def _bindable(port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                return False
        return True

    def acquire(self) -> int:
        for port in self._range:
            if port not in self._leased and self._bindable(port):
                self._leased.add(port)
                return port
        raise SandboxError(f"no free sandbox port in {self._range.start}-{self._range.stop - 1}")

    def release(self, port: int) -> None:
        self._leased.discard(port)


class SandboxPool:
    """Start, reach and stop per-rollout sandboxes, at most ``max_concurrency`` at a time."""

    def __init__(
        self,
        *,
        image: str,
        sandbox_provider: Mapping[str, Any],
        sandbox_env: Mapping[str, str],
        sandbox_files: Mapping[str, str],
        port_range: tuple[int, int],
        max_concurrency: int,
        startup_timeout_s: float,
        ttl_s: float,
        scratch_root: Optional[str] = None,
        sandbox_factory: Callable[[Mapping[str, Any], SandboxSpec], AsyncSandbox] = AsyncSandbox,
    ) -> None:
        self.image = image
        self.sandbox_provider = dict(sandbox_provider)
        self.sandbox_env = dict(sandbox_env)
        self.sandbox_files = dict(sandbox_files)
        self.startup_timeout_s = startup_timeout_s
        self.ttl_s = ttl_s
        self.scratch_root = scratch_root
        self._factory = sandbox_factory
        self._ports = PortAllocator(port_range)
        self._capacity = asyncio.Semaphore(max_concurrency)
        self._sandboxes: dict[str, AtlasSandbox] = {}

    def get(self, session_id: str) -> Optional[AtlasSandbox]:
        box = self._sandboxes.get(session_id)
        if box is not None:
            box.last_used_at = monotonic()
        return box

    async def start(self, session_id: str, *, servers: list[str], tools: list[str]) -> AtlasSandbox:
        """Start a sandbox that serves ``servers`` and wait until it lists the task's tools."""
        await self.stop(session_id)
        await self._capacity.acquire()
        box = AtlasSandbox(base_url="", tool_names={}, missing_tools=[])
        try:
            box.port = self._ports.acquire()
            box.base_url = f"http://127.0.0.1:{box.port}"
            box.scratch_dir = tempfile.mkdtemp(prefix="mcp-atlas-", dir=self.scratch_root)
            spec = SandboxSpec(
                image=self.image,
                ttl_s=self.ttl_s,
                env={**self.sandbox_env, "ENABLED_SERVERS": ",".join(servers)},
                files=self.sandbox_files,
                # A private /tmp per sandbox: uv and npm caches and the server log live there.
                provider_options={"binds": [f"{box.scratch_dir}:/tmp"]},
                ports=(box.port,),
            )
            box.sandbox = self._factory(self.sandbox_provider, spec)
            await box.sandbox.start()
            # HOME is set here, not in the env file (providers refuse it there): the image's tools and the
            # sandbox_files configs live under /root, and MCP stdio servers inherit HOME from this process.
            command = (
                "cd /agent-environment && export HOME=/root && exec ./entrypoint.sh .venv/bin/python -m uvicorn "
                f"agent_environment.main:app --host 127.0.0.1 --port {box.port} >{LOG_PATH} 2>&1 </dev/null"
            )
            # The exec stays open for the server's lifetime; stopping the sandbox ends both.
            box.server_task = asyncio.create_task(
                box.sandbox.exec(command, timeout_s=self.ttl_s + self.startup_timeout_s)
            )
            listed = await self._wait_until_ready(box, tools)
            box.tool_names, box.missing_tools = sandbox_tool_names(tools, listed, servers)
            self._sandboxes[session_id] = box
            return box
        except BaseException as exc:
            try:
                await self._release(box)
            except SandboxError as stop_error:
                # Kept, so the next stop of this session (verify, a session close or the reaper) tries again.
                self._sandboxes[session_id] = box
                logger.warning(
                    "MCP-Atlas sandbox of session %s did not stop after its start failed: %s", session_id, stop_error
                )
            if isinstance(exc, SandboxError) or not isinstance(exc, Exception):
                raise
            raise SandboxError(f"sandbox start failed: {type(exc).__name__}: {exc}") from exc

    async def _log_tail(self, box: AtlasSandbox) -> str:
        if box.sandbox is None:
            return ""
        with contextlib.suppress(Exception):
            result = await box.sandbox.exec(f"tail -c 4000 {LOG_PATH}", timeout_s=15)
            return (result.stdout or "") + (result.stderr or "")
        return ""

    async def _wait_until_ready(self, box: AtlasSandbox, tools: list[str]) -> set[str]:
        """Wait for ``/health``, then for ``/list-tools`` to report every tool or stop changing."""
        deadline = monotonic() + self.startup_timeout_s
        while True:
            if box.server_task is not None and box.server_task.done():
                raise SandboxError(f"sandbox server exited during startup: {await self._log_tail(box)}")
            with contextlib.suppress(ClientError, asyncio.TimeoutError, OSError):
                status, body = await _http_json("GET", f"{box.base_url}/health", 10)
                if (
                    status == 200
                    and isinstance(body, dict)
                    and body.get("status") == "health_and_client_connection_ok"
                ):
                    break
            if monotonic() > deadline:
                raise SandboxError(
                    f"sandbox not healthy after {self.startup_timeout_s:g}s: {await self._log_tail(box)}"
                )
            await asyncio.sleep(POLL_INTERVAL_S)
        wanted = set(tools)
        listed: set[str] = set()
        for _attempt in range(LIST_TOOLS_ATTEMPTS):
            try:
                status, body = await _http_json("POST", f"{box.base_url}/list-tools", 120)
            except (ClientError, asyncio.TimeoutError, OSError) as exc:
                raise SandboxError(f"list-tools failed: {type(exc).__name__}: {exc}") from exc
            if status != 200 or not isinstance(body, list):
                raise SandboxError(f"list-tools returned HTTP {status}: {str(body)[:500]}")
            listed = {tool["name"] for tool in body if isinstance(tool, dict) and isinstance(tool.get("name"), str)}
            # The served set is final once every tool is there, or once it lists the bare names of one server.
            if wanted <= listed or not (wanted & listed) and listed:
                break
            await asyncio.sleep(POLL_INTERVAL_S)
        return listed

    async def _release(self, box: AtlasSandbox) -> None:
        """Stop the sandbox, then free its port, scratch directory and capacity slot.

        Raises SandboxError and frees nothing when the provider cannot stop the sandbox, so a later stop retries.
        """
        async with box.stop_lock:
            if box.stopped:
                return
            if box.sandbox is not None:
                try:
                    await box.sandbox.stop()
                except Exception as exc:
                    raise SandboxError(f"sandbox stop failed: {type(exc).__name__}: {exc}") from exc
            box.stopped = True
            if box.server_task is not None:
                box.server_task.cancel()
                with contextlib.suppress(BaseException):
                    await box.server_task
            if box.port is not None:
                self._ports.release(box.port)
            if box.scratch_dir is not None:
                shutil.rmtree(box.scratch_dir, ignore_errors=True)
            self._capacity.release()

    async def stop(self, session_id: str) -> bool:
        """Stop the session's sandbox; False when it has none.

        Raises SandboxError when the sandbox does not stop; the pool then keeps it, so a later stop retries.
        """
        box = self._sandboxes.get(session_id)
        if box is None:
            return False
        await self._release(box)
        if self._sandboxes.get(session_id) is box:
            del self._sandboxes[session_id]
        return True

    async def reap_expired(self) -> int:
        """Stop the sandboxes unused for ``ttl_s``; returns how many stopped. A failed stop is retried next time."""
        cutoff = monotonic() - self.ttl_s
        expired = [sid for sid, box in self._sandboxes.items() if box.last_used_at < cutoff]
        stopped = 0
        for session_id in expired:
            logger.warning("stopping MCP-Atlas sandbox of session %s after %gs without use", session_id, self.ttl_s)
            try:
                stopped += await self.stop(session_id)
            except SandboxError as exc:
                logger.error(
                    "MCP-Atlas sandbox of session %s did not stop; retrying at the next reap: %s", session_id, exc
                )
        return stopped

    async def close(self) -> None:
        """Stop every sandbox. Raises SandboxError naming the sessions whose sandbox did not stop."""
        failed = []
        for session_id in list(self._sandboxes):
            try:
                await self.stop(session_id)
            except SandboxError as exc:
                failed.append(f"{session_id}: {exc}")
        if failed:
            raise SandboxError(f"{len(failed)} sandboxes did not stop: {'; '.join(failed)}"[:4000])
