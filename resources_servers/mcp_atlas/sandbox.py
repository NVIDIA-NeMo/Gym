# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One private MCP-Atlas agent-environment sandbox per rollout.

OpenSandbox reaches the image's loopback HTTP server through lifecycle exec, without exposing MCP ports.
Apptainer development runs use host loopback and private scratch directories. Neither transport establishes
egress isolation: scoring still requires deployment enforcement against peers and Gym control endpoints.
Service credentials reach the sandbox through ``SandboxSpec.env``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json as json_module
import logging
import os
import posixpath
import shlex
import shutil
import socket
import stat
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic
from typing import Any, Optional

from aiohttp import ClientError, ClientTimeout

from nemo_gym.client import request
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, SandboxSpec
from nemo_gym.server_runtime import ResourcesSessionLocks


if __package__:
    from .authentication import is_authentication_failure
    from .credentials import CredentialConfigurationError, task_environment, validate_provider_environment
else:
    from authentication import is_authentication_failure
    from credentials import CredentialConfigurationError, task_environment, validate_provider_environment


logger = logging.getLogger(__name__)

LOG_PATH = "/tmp/mcp-atlas-sandbox.log"
HTTP_HELPER_PATH = "/tmp/nemo-gym-mcp-atlas-http.py"
HTTP_REQUEST_ENV = "MCP_ATLAS_HTTP_REQUEST"
SANDBOX_PORT = 1984
POLL_INTERVAL_S = 2.0
LIST_TOOLS_ATTEMPTS = 5


class SandboxError(RuntimeError):
    """The sandbox could not be started or reached: an infrastructure failure, not a model one."""


def _validate_apptainer_environment(provider: Mapping[str, Any]) -> None:
    """Require clean start/exec environments and reject unmanaged host injection aliases.

    Existing APPTAINERENV_/SINGULARITYENV_ host settings must move into scoped sandbox_env; this checks
    names only. Raw --env/--env-file arguments are rejected; the provider's later SandboxSpec-derived
    --env-file remains the supported injection path.
    """
    block = provider["apptainer"]
    if not isinstance(block, Mapping):
        raise CredentialConfigurationError("MCP-Atlas Apptainer configuration must be a mapping")
    isolation_flags = {"--cleanenv", "-e", "--containall", "-C"}
    assignments = tuple(f"{flag}=" for flag in isolation_flags)
    for section, option in (("create", "extra_start_args"), ("exec", "extra_exec_args")):
        config = block.get(section) or {}
        arguments = config.get(option, []) if isinstance(config, Mapping) else getattr(config, option, [])
        if (
            not isinstance(arguments, list)
            or any(not isinstance(argument, str) for argument in arguments)
            or not isolation_flags.intersection(arguments)
            or any(argument.startswith(assignments) for argument in arguments)
        ):
            raise CredentialConfigurationError(
                f"MCP-Atlas Apptainer {section}.{option} requires --cleanenv/-e or --containall/-C"
            )
        injected = next(
            (
                flag
                for argument in arguments
                for flag in ("--env", "--env-file")
                if argument == flag or argument.startswith(flag + "=")
            ),
            None,
        )
        if injected is not None:
            raise CredentialConfigurationError(
                f"MCP-Atlas Apptainer {section}.{option} must not include {injected}; use scoped sandbox_env"
            )
    aliases = sorted(name for name in os.environ if name.startswith(("APPTAINERENV_", "SINGULARITYENV_")))
    if aliases:
        raise CredentialConfigurationError(
            "MCP-Atlas rejects inherited Apptainer environment aliases: "
            f"{', '.join(aliases)}; move ordinary settings into scoped sandbox_env"
        )


def _remove_scratch_dir(directory: str) -> None:
    root = Path(directory)

    def retry_readonly(function: Callable[..., object], path: str, error: BaseException) -> None:
        if not isinstance(error, PermissionError):
            raise error
        failed = Path(path)
        for entry in (failed.parent, failed):
            if entry.is_relative_to(root):
                mode = entry.lstat().st_mode
                if not stat.S_ISLNK(mode):
                    entry.chmod(stat.S_IMODE(mode) | stat.S_IRWXU)
        if function in (os.open, os.scandir):
            shutil.rmtree(failed, onexc=retry_readonly)
        else:
            function(path)

    shutil.rmtree(root, onexc=retry_readonly)


@dataclass
class AtlasSandbox:
    """A running sandbox and the names its MCP servers gave the task's tools."""

    base_url: str
    tool_names: dict[str, str]
    missing_tools: list[str]
    exec_transport: bool = False
    sandbox: Optional[AsyncSandbox] = None
    server_task: Optional[asyncio.Task] = None
    port: Optional[int] = None
    scratch_dir: Optional[str] = None
    last_used_at: float = field(default_factory=monotonic)
    # Set once the provider confirmed the stop and the port and capacity slot are free.
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
        else:
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


def _check_exec_result(result: SandboxExecResult) -> None:
    if result.error_type == "timeout":
        raise asyncio.TimeoutError("sandbox exec timed out")
    if result.error_type is not None or result.return_code != 0:
        raise SandboxError(f"sandbox exec failed: error_type={result.error_type!r}, return_code={result.return_code}")


async def _sandbox_http_json(
    box: AtlasSandbox, method: str, path: str, timeout_s: float, json: Any = None
) -> tuple[int, Any]:
    if not box.exec_transport:
        return await _http_json(method, f"{box.base_url}{path}", timeout_s, json)
    if box.sandbox is None:
        raise SandboxError("sandbox HTTP exec has no sandbox")
    envelope = json_module.dumps(
        {"method": method, "path": path, "timeout_s": timeout_s, "json": json}, allow_nan=False
    )
    try:
        # Socket deadlines alone reset on each read. Bound the whole exec, including provider overhead,
        # while allowing the provider's cancellation cleanup to finish before reporting the timeout.
        async with asyncio.timeout(timeout_s) as deadline:
            result = await box.sandbox.exec(
                shlex.join(["/agent-environment/.venv/bin/python", HTTP_HELPER_PATH]),
                env={HTTP_REQUEST_ENV: envelope},
                timeout_s=timeout_s + 5,
            )
        if deadline.expired():
            # A provider that consumed cancellation must not turn a late result into a successful call.
            raise asyncio.TimeoutError("sandbox HTTP request timed out")
    except asyncio.TimeoutError:
        raise
    except Exception as exc:
        raise SandboxError(f"sandbox HTTP exec failed: {type(exc).__name__}") from exc
    _check_exec_result(result)
    try:
        response = json_module.loads(result.stdout or "")
    except ValueError as exc:
        raise SandboxError("sandbox HTTP exec returned an invalid response envelope") from exc
    if isinstance(response, dict) and set(response) == {"status", "body"}:
        status = response["status"]
        if isinstance(status, int) and not isinstance(status, bool) and 100 <= status <= 599:
            return status, response["body"]
    if isinstance(response, dict) and set(response) == {"error", "message"} and isinstance(response["message"], str):
        if response["error"] == "timeout":
            raise asyncio.TimeoutError("sandbox HTTP request timed out")
        if response["error"] == "transport":
            raise OSError(f"sandbox HTTP request failed: {response['message']}")
    raise SandboxError("sandbox HTTP exec returned an invalid response envelope")


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
    return await _sandbox_http_json(box, "POST", "/call-tool", timeout_s, payload)


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
        sandbox_metadata: Optional[Mapping[str, Any]] = None,
        sandbox_options: Optional[Mapping[str, Any]] = None,
        sandbox_factory: Callable[[Mapping[str, Any], SandboxSpec], AsyncSandbox] = AsyncSandbox,
    ) -> None:
        self.image = image
        self.sandbox_provider = dict(sandbox_provider)
        if len(self.sandbox_provider) != 1 or next(iter(self.sandbox_provider)) not in {"apptainer", "opensandbox"}:
            raise ValueError("MCP-Atlas requires one resolved opensandbox or apptainer provider")
        validate_provider_environment(self.sandbox_provider)
        self._exec_transport = next(iter(self.sandbox_provider)) == "opensandbox"
        if not self._exec_transport:
            _validate_apptainer_environment(self.sandbox_provider)
        self.sandbox_env = dict(sandbox_env)
        self.sandbox_files = dict(sandbox_files)
        self.sandbox_metadata = dict(sandbox_metadata or {})
        self.sandbox_options = dict(sandbox_options or {})
        if self._exec_transport:
            reserved_paths = {HTTP_HELPER_PATH, *map(str, Path(HTTP_HELPER_PATH).parents)}
            if any(posixpath.normpath("/" + path.lstrip("/")) in reserved_paths for path in self.sandbox_files):
                raise ValueError(f"sandbox_files must not replace the HTTP helper at {HTTP_HELPER_PATH}")
            if HTTP_REQUEST_ENV in self.sandbox_env:
                raise ValueError(f"sandbox_env must not set the reserved {HTTP_REQUEST_ENV}")
            self.sandbox_files[HTTP_HELPER_PATH] = Path(__file__).with_name("sandbox_http.py").read_text()
        elif self.sandbox_options:
            raise ValueError("sandbox_options are supported only for OpenSandbox")
        self.startup_timeout_s = startup_timeout_s
        self.ttl_s = ttl_s
        self.scratch_root = scratch_root
        self._factory = sandbox_factory
        self._ports = PortAllocator(port_range)
        self._capacity = asyncio.Semaphore(max_concurrency)
        self._sandboxes: dict[str, AtlasSandbox] = {}
        self._sessions = ResourcesSessionLocks()

    def get(self, session_id: str) -> Optional[AtlasSandbox]:
        box = self._sandboxes.get(session_id)
        if box is not None:
            box.last_used_at = monotonic()
        return box

    async def start(self, session_id: str, *, servers: list[str], tools: list[str]) -> AtlasSandbox:
        """Start a sandbox that serves ``servers`` and wait until it lists the task's tools."""
        # Reject missing credentials before stopping a prior sandbox or allocating any provider resources.
        if not self._exec_transport:
            _validate_apptainer_environment(self.sandbox_provider)
        environment = task_environment(self.sandbox_env, servers)
        async with self._sessions.locked(session_id):
            await self._stop_locked(session_id)
            await self._capacity.acquire()
            box = AtlasSandbox(base_url="", tool_names={}, missing_tools=[], exec_transport=self._exec_transport)
            self._sandboxes[session_id] = box
            try:
                port = SANDBOX_PORT
                options = self.sandbox_options
                if not self._exec_transport:
                    box.port = self._ports.acquire()
                    port = box.port
                    box.base_url = f"http://127.0.0.1:{port}"
                    box.scratch_dir = tempfile.mkdtemp(prefix="mcp-atlas-", dir=self.scratch_root)
                    options = {"binds": [f"{box.scratch_dir}:/tmp"]}
                spec = SandboxSpec(
                    image=self.image,
                    ttl_s=self.ttl_s,
                    env={**environment, "ENABLED_SERVERS": ",".join(servers)},
                    files=self.sandbox_files,
                    provider_options=options,
                    ports=() if self._exec_transport else (port,),
                    # Override the image's server entrypoint; only the explicit loopback service should start.
                    entrypoint=["/bin/sh", "-c", "exec sleep infinity"] if self._exec_transport else None,
                    metadata=self.sandbox_metadata,
                )
                box.sandbox = self._factory(self.sandbox_provider, spec)
                await box.sandbox.start()
                # HOME is set here, not in the env file (providers refuse it there): the image's tools and the
                # sandbox_files configs live under /root, and MCP stdio servers inherit HOME from this process.
                command = (
                    "cd /agent-environment && export HOME=/root && exec ./entrypoint.sh .venv/bin/python -m uvicorn "
                    f"agent_environment.main:app --host 127.0.0.1 --port {port} >{LOG_PATH} 2>&1 </dev/null"
                )
                if self._exec_transport:
                    result = await box.sandbox.exec(
                        command + " &", timeout_s=self.startup_timeout_s, preserve_background_services=True
                    )
                    _check_exec_result(result)
                else:
                    # The development exec stays open for the server's lifetime; stopping ends both.
                    box.server_task = asyncio.create_task(
                        box.sandbox.exec(command, timeout_s=self.ttl_s + self.startup_timeout_s)
                    )
                listed = await self._wait_until_ready(box, tools, servers)
                box.tool_names, box.missing_tools = sandbox_tool_names(tools, listed, servers)
                return box
            except BaseException as exc:
                try:
                    await self._stop_locked(session_id)
                except SandboxError as stop_error:
                    # Registered before startup, so interrupted cleanup remains owned for the next stop.
                    logger.warning(
                        "MCP-Atlas sandbox of session %s did not stop after its start failed: %s",
                        session_id,
                        stop_error,
                    )
                if isinstance(exc, (SandboxError, CredentialConfigurationError)) or not isinstance(exc, Exception):
                    raise
                raise SandboxError(f"sandbox start failed: {type(exc).__name__}: {exc}") from exc

    async def _log_tail(self, box: AtlasSandbox) -> str:
        if box.sandbox is None:
            return ""
        try:
            result = await box.sandbox.exec(f"tail -c 4000 {LOG_PATH}", timeout_s=15)
            _check_exec_result(result)
            return (result.stdout or "") + (result.stderr or "")
        except Exception as exc:
            logger.warning("could not read the MCP-Atlas sandbox log: %s", exc)
            return f"log unavailable: {type(exc).__name__}"

    async def _wait_until_ready(self, box: AtlasSandbox, tools: list[str], servers: list[str]) -> set[str]:
        """Wait for ``/health``, then for ``/list-tools`` to report every tool or stop changing."""
        deadline = monotonic() + self.startup_timeout_s
        healthy = False
        while not healthy:
            if box.server_task is not None and box.server_task.done():
                try:
                    _check_exec_result(await box.server_task)
                except (SandboxError, asyncio.TimeoutError) as exc:
                    raise SandboxError(
                        f"sandbox server exited during startup: {exc}: {await self._log_tail(box)}"
                    ) from exc
                raise SandboxError(f"sandbox server exited during startup: {await self._log_tail(box)}")
            with contextlib.suppress(ClientError, asyncio.TimeoutError, OSError):
                status, body = await _sandbox_http_json(box, "GET", "/health", 10)
                error = body.get("error", "") if isinstance(body, dict) else ""
                detail = body.get("detail", error) if isinstance(body, dict) else body
                unhealthy = (
                    isinstance(body, dict) and body.get("status") == "health_and_client_connection_health_check_failed"
                )
                if (
                    status >= 400
                    and is_authentication_failure(status, detail if isinstance(detail, str) else "", servers=servers)
                ) or (unhealthy and isinstance(error, str) and is_authentication_failure(500, error, servers=servers)):
                    raise CredentialConfigurationError(
                        "MCP-Atlas authentication failed during startup; check credentials/scopes"
                    )
                healthy = (
                    status == 200
                    and isinstance(body, dict)
                    and body.get("status") == "health_and_client_connection_ok"
                )
            if not healthy and monotonic() > deadline:
                raise SandboxError(
                    f"sandbox not healthy after {self.startup_timeout_s:g}s: {await self._log_tail(box)}"
                )
            if not healthy:
                await asyncio.sleep(POLL_INTERVAL_S)
        wanted = set(tools)
        listed: set[str] = set()
        attempts = 0
        ready = False
        while attempts < LIST_TOOLS_ATTEMPTS and not ready:
            attempts += 1
            try:
                status, body = await _sandbox_http_json(box, "POST", "/list-tools", 120)
            except (ClientError, asyncio.TimeoutError, OSError) as exc:
                raise SandboxError(f"list-tools failed: {type(exc).__name__}: {exc}") from exc
            if status != 200 or not isinstance(body, list):
                detail = body.get("detail", "") if isinstance(body, dict) else body
                if isinstance(detail, str) and detail.startswith("Failed to list tools:"):
                    detail = detail[len("Failed to list tools:") :].lstrip()
                if is_authentication_failure(status, detail if isinstance(detail, str) else "", servers=servers):
                    raise CredentialConfigurationError(
                        "MCP-Atlas authentication failed while listing tools; check credentials/scopes"
                    )
                raise SandboxError(f"list-tools returned HTTP {status}: {str(body)[:500]}")
            listed = {tool["name"] for tool in body if isinstance(tool, dict) and isinstance(tool.get("name"), str)}
            # The served set is final once every tool is there, or once it lists the bare names of one server.
            ready = wanted <= listed or bool(not (wanted & listed) and listed)
            if not ready:
                await asyncio.sleep(POLL_INTERVAL_S)
        return listed

    async def _release(self, box: AtlasSandbox) -> None:
        """Stop the sandbox, then free its port and capacity slot and remove its scratch directory.

        Raises SandboxError and frees nothing when the provider cannot stop the sandbox, so a later stop retries.
        A residual scratch cleanup error is logged after the provider stops; it cannot strand the released leases.
        """
        async with box.stop_lock:
            if box.stopped:
                return
            if box.sandbox is not None:
                try:
                    await box.sandbox.stop()
                except Exception as exc:
                    raise SandboxError(f"sandbox stop failed: {type(exc).__name__}: {exc}") from exc
            try:
                if box.server_task is not None:
                    box.server_task.cancel()
                    results = await asyncio.gather(box.server_task, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            logger.warning("MCP-Atlas sandbox server stopped with an error: %s", result)
                        elif isinstance(result, SandboxExecResult):
                            try:
                                _check_exec_result(result)
                            except (SandboxError, asyncio.TimeoutError) as exc:
                                logger.warning("MCP-Atlas sandbox server stopped with an error: %s", exc)
                if box.scratch_dir is not None:
                    try:
                        _remove_scratch_dir(box.scratch_dir)
                    except OSError as exc:
                        logger.warning(
                            "MCP-Atlas sandbox stopped but scratch directory %s could not be removed: %s",
                            box.scratch_dir,
                            exc,
                        )
            finally:
                if box.port is not None:
                    self._ports.release(box.port)
                self._capacity.release()
                box.stopped = True

    async def stop(self, session_id: str) -> bool:
        """Stop the session's sandbox; False when it has none.

        Raises SandboxError when the sandbox does not stop; the pool then keeps it, so a later stop retries.
        """
        async with self._sessions.locked(session_id):
            return await self._stop_locked(session_id)

    async def _stop_locked(self, session_id: str) -> bool:
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
