# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
NeMo Skills Tools Resources Server.

This resources server provides:
- Integration with nemo_skills ToolManager for tool execution (e.g., DirectPythonTool)
- Verification delegation to math_with_judge
"""

import asyncio
import inspect
import json
import logging
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import httpx
from aiohttp import ClientError, ClientTimeout
from anyio import CancelScope
from anyio.lowlevel import checkpoint_if_cancelled
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from nemo_skills.mcp.servers.python_tool import DirectPythonTool
from nemo_skills.mcp.tool_manager import ToolManager
from nemo_skills.mcp.utils import locate
from pydantic import ConfigDict, Field

from nemo_gym.client import SESSION_ID_KEY, raise_for_status
from nemo_gym.client import request as http_request
from nemo_gym.config import ResourcesServerRef
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSessionLostError,
    SimpleResourcesServer,
    lost_session_verification,
)


logger = logging.getLogger(__name__)

# Like Skills' LocalSandbox.delete_session: 3 attempts, 2 s apart, each bounded by a 10 s timeout.
_DELETE_ATTEMPTS = 3
_DELETE_RETRY_DELAY_SECONDS = 2.0


async def _delete_python_session(url: str, session_id: str) -> None:
    """Delete a sandbox Python session, raising if the sandbox does not acknowledge it."""
    # A transport error and a 5xx reply (a sandbox worker restarting behind its proxy) share one budget of bounded
    # attempts, so an unreachable sandbox cannot hold the session lock for long.
    for attempt in range(1, _DELETE_ATTEMPTS + 1):
        try:
            response = await http_request(
                "DELETE",
                url,
                headers={"X-Session-ID": session_id},
                timeout=ClientTimeout(total=10),
                _max_num_tries=1,
            )
        except (ClientError, TimeoutError):
            if attempt == _DELETE_ATTEMPTS:
                raise
        else:
            async with response:
                # 404: the session is already gone, or the request reached a worker that never held it.
                if 200 <= response.status < 300 or response.status == 404:
                    return
                if response.status < 500 or attempt == _DELETE_ATTEMPTS:
                    await raise_for_status(response)
                    raise RuntimeError(
                        f"Sandbox did not acknowledge deletion of session {session_id}: {response.status}"
                    )
        await asyncio.sleep(_DELETE_RETRY_DELAY_SECONDS)


@dataclass
class _PythonExecution:
    request_id: str
    session_id: Optional[str] = None


class NSToolsToolManager(ToolManager):
    """Keep ownership of Python sessions until the sandbox acknowledges their deletion."""

    def __init__(
        self,
        module_specs: Iterable[str],
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(module_specs=module_specs, overrides=overrides, context=context)
        self._execution: ContextVar[Optional[_PythonExecution]] = ContextVar("ns_tools_execution", default=None)
        self._python_requests: Dict[str, asyncio.Task[Dict[str, Any]]] = {}
        for tool in self._tools.values():
            if isinstance(tool, DirectPythonTool):
                self._track_python_requests(tool)

    def _track_python_requests(self, tool: DirectPythonTool) -> None:
        send_request = tool._sandbox._send_request

        async def tracked_send_request(body: Dict[str, Any], timeout: float) -> Dict[str, Any]:
            execution = self._execution.get()
            if execution is not None and body.get("session_id") is not None:
                execution.session_id = str(body["session_id"])
                tool.requests_to_sessions[execution.request_id] = execution.session_id
            # A DELETE must not overtake this POST and acknowledge 404 before it creates its session.
            # Shield only this request (Skills bounds it by timeout + 5), never the full history replay.
            pending = asyncio.create_task(send_request(body, timeout))
            if execution is not None:
                self._python_requests[execution.request_id] = pending
            try:
                with CancelScope(shield=True):
                    return await asyncio.shield(pending)
            finally:
                # An asyncio Task.cancel() bypasses AnyIO's shield. Keep its still-running POST owned so
                # cleanup (or a later tool call) waits for the response before touching the Python session.
                if execution is not None and pending.done():
                    self._python_requests.pop(execution.request_id, None)
                await checkpoint_if_cancelled()

        tool._sandbox._send_request = tracked_send_request

    async def execute_tool(
        self, raw_name: str, args: Dict[str, Any], extra_args: Optional[Dict[str, Any]] = None
    ) -> object:
        tool, _ = self._resolve(self._raw_to_qualified_map[raw_name])
        request_id = (extra_args or {}).get("request_id")
        if not isinstance(tool, DirectPythonTool) or request_id is None:
            return await super().execute_tool(raw_name, args, extra_args)
        await self._wait_for_python_request(request_id)
        execution = _PythonExecution(request_id=request_id)
        token = self._execution.set(execution)
        try:
            return await super().execute_tool(raw_name, args, extra_args)
        finally:
            # Skills catches ordinary execution errors and overwrites the mapping with None; cancellation skips
            # its assignment entirely. In either case the generated session may still need a DELETE.
            if execution.session_id is not None and tool.requests_to_sessions.get(request_id) is None:
                tool.requests_to_sessions[request_id] = execution.session_id
            self._execution.reset(token)

    async def _wait_for_python_request(self, request_id: str) -> None:
        pending = self._python_requests.get(request_id)
        if pending is None:
            return
        try:
            await asyncio.shield(pending)
        except Exception:
            # A failed response can still have allocated a session; deletion must be attempted afterwards.
            logger.warning("Python request failed before cleanup of %s", request_id, exc_info=True)
        finally:
            if pending.done():
                self._python_requests.pop(request_id, None)

    async def cleanup_request(self, request_id: str) -> None:
        await self._wait_for_python_request(request_id)
        for tool in self._tools.values():
            if not isinstance(tool, DirectPythonTool):
                await tool.cleanup_request(request_id)
                continue
            session_id = tool.requests_to_sessions.get(request_id)
            if session_id is None:
                tool.requests_to_sessions.pop(request_id, None)
                continue

            # Both cleanup_request and LocalSandbox.delete_session in the pinned Skills release suppress errors.
            # Use the sandbox's resolved address and session routing header, and retain both maps on failure.
            sandbox = tool._sandbox
            session_id = str(session_id)
            # The Environment Server closes once. If the DELETE still fails, close fails and the Python session stays
            # owned (a verify still returns its score), so a later close or a checkpoint retire can release it.
            await _delete_python_session(f"http://{sandbox.host}:{sandbox.port}/sessions/{session_id}", session_id)
            sandbox.session_histories.pop(session_id, None)
            tool.requests_to_sessions.pop(request_id, None)


# ============================================================
# Configuration
# ============================================================


class NSToolsConfig(BaseResourcesServerConfig):
    """Config for the NeMo Skills tools resources server."""

    # Default verifier (typically math_with_judge)
    default_verifier: str = "math_with_judge"

    # Map of verifier names to server references
    # At minimum, should include math_with_judge
    verifiers: Dict[str, ResourcesServerRef] = Field(default_factory=dict)

    # NeMo Skills tool modules to load (e.g., "nemo_skills.mcp.servers.python_tool::DirectPythonTool")
    nemo_skills_tools: List[str] = Field(default_factory=list)

    # Per-tool overrides for nemo_skills tools
    nemo_skills_tool_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Sandbox configuration for code execution tools
    sandbox_host: str = "127.0.0.1"
    sandbox_port: str = "6000"

    # Legacy python_tool HTTP server port (only used for pre-main HTTP PythonTool variants)
    python_tool_port: int = 8765

    # Verbose logging for tool execution timing (disabled by default)
    verbose_tool_logging: bool = False

    # When True, skip replaying session history after sandbox worker restarts.
    # The model receives a warning in stderr instead of restored state.
    disable_session_restore: bool = False


# ============================================================
# Run/Verify Request/Response Models
# ============================================================


class NSToolsRunRequest(BaseRunRequest):
    """Run request that allows extra fields from the sample."""

    model_config = ConfigDict(extra="allow")

    # Per-sample verifier selection (optional, falls back to default_verifier)
    verifier_type: Optional[str] = None


class NSToolsVerifyRequest(NSToolsRunRequest, BaseVerifyRequest):
    pass


class NSToolsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    delegated_response: Optional[Dict[str, Any]] = None

    # Timing metrics for tool execution
    total_tool_execution_time_seconds: float = 0.0
    num_tool_calls: int = 0
    avg_tool_call_time_seconds: float = 0.0
    tool_timeout_count: int = 0  # Internal sandbox timeouts (process_status == "timeout")
    tool_request_timeout_count: int = 0  # HTTP/request-level timeouts


# ============================================================
# Resources Server Implementation
# ============================================================


class NSToolsResourcesServer(SimpleResourcesServer):
    config: NSToolsConfig
    tool_manager: Optional[Any] = None
    _tool_name_map: Dict[str, str] = {}  # Maps tool names to qualified names
    _python_tool_process: Optional[subprocess.Popen] = None
    _timing_by_session: Dict[str, list] = {}  # session_id -> list of timing records
    _uses_python_tool_sidecar: bool = False

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Wire shutdown() into the lifespan so a normal server stop reaps the
        # python_tool sidecar and ToolManager. The base runner only starts
        # Uvicorn and never calls it, otherwise leaving the sidecar bound to
        # its fixed port.
        main_app_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan_wrapper(app):
            try:
                async with main_app_lifespan(app) as maybe_state:
                    yield maybe_state
            finally:
                await self.shutdown()

        app.router.lifespan_context = lifespan_wrapper

        # Initialize nemo_skills ToolManager if tools are configured
        if self.config.nemo_skills_tools:
            self._uses_python_tool_sidecar = any(
                self._tool_uses_python_tool_sidecar(tool_spec) for tool_spec in self.config.nemo_skills_tools
            )
            if self._uses_python_tool_sidecar:
                # Legacy HTTP PythonTool variants require a local sidecar process.
                self._start_python_tool_server()
            self._initialize_nemo_skills_tools()

            # Register a catch-all endpoint for tool execution
            # This handles any tool name dynamically
            app.post("/{tool_name}")(self.execute_tool)

        return app

    def _tool_uses_python_tool_sidecar(self, tool_spec: str) -> bool:
        """Detect whether a tool spec refers to the HTTP-backed PythonTool."""
        tool_cls_or_obj = locate(tool_spec)
        tool = tool_cls_or_obj() if inspect.isclass(tool_cls_or_obj) else tool_cls_or_obj
        if tool.__class__.__name__ != "PythonTool":
            return False

        default_config = tool.default_config()
        client_params = default_config.get("client_params", {})
        return "base_url" in client_params

    def _start_python_tool_server(self):
        """Spawn the python_tool HTTP sidecar as a subprocess."""
        logger.info("Starting python_tool HTTP server on port %s", self.config.python_tool_port)

        # Build command with sandbox config
        cmd = [
            sys.executable,
            "-m",
            "nemo_skills.mcp.servers.python_tool",
            "--host",
            "127.0.0.1",
            "--port",
            str(self.config.python_tool_port),
            "--sandbox-host",
            self.config.sandbox_host,
            "--sandbox-port",
            str(self.config.sandbox_port),
        ]
        if self.config.disable_session_restore:
            cmd.append("--disable-session-restore")
        logger.info(f"python_tool command: {' '.join(cmd)}")

        # Don't pipe stdout/stderr so we can see output directly in logs
        self._python_tool_process = subprocess.Popen(cmd)

        # Wait for server to be ready
        self._wait_for_server_ready()
        logger.info(f"python_tool HTTP server started (PID: {self._python_tool_process.pid})")

    def _wait_for_server_ready(self, timeout: float = 30.0, poll_interval: float = 0.5):
        """Wait for the python_tool HTTP sidecar to be ready."""
        url = f"http://127.0.0.1:{self.config.python_tool_port}/mcp"
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check if process died
            if self._python_tool_process.poll() is not None:
                raise RuntimeError(
                    f"python_tool server died during startup (exit code: {self._python_tool_process.returncode}). "
                    f"Check logs above for details."
                )

            try:
                # Try to connect to the server
                with httpx.Client(timeout=2.0) as client:
                    # MCP servers respond to POST on /mcp, but we can check if the port is open
                    # by attempting a connection. The server might return an error, but that's fine.
                    response = client.post(url, json={})
                    # Any response means server is up
                    logger.info(f"python_tool server is ready (status: {response.status_code})")
                    return
            except (httpx.ConnectError, httpx.ConnectTimeout):
                # Server not ready yet
                time.sleep(poll_interval)
            except Exception as e:
                # Other errors might indicate server is up but returned an error - that's ok
                logger.info(f"python_tool server responded with error (server is ready): {e}")
                return

        # Terminate the process if still running
        if self._python_tool_process.poll() is None:
            self._python_tool_process.terminate()
            self._python_tool_process.wait(timeout=5)
        raise TimeoutError(f"python_tool server did not start within {timeout}s. Check logs above for details.")

    def _initialize_nemo_skills_tools(self):
        """Initialize the nemo_skills ToolManager with configured tools."""

        # Reduce verbosity of MCP and httpx loggers (they log every HTTP request at INFO)
        for noisy_logger in [
            "mcp.server.streamable_http_manager",
            "mcp.server.streamable_http",
            "mcp.server.lowlevel.server",
            "mcp.server",
            "mcp.client.streamable_http",
            "httpx",
        ]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

        logger.info(f"Initializing NeMo Skills ToolManager with tools: {self.config.nemo_skills_tools}")

        context = {
            "sandbox": {
                "sandbox_type": "local",
                "host": self.config.sandbox_host,
                "port": self.config.sandbox_port,
                "disable_session_restore": self.config.disable_session_restore,
            }
        }

        overrides = {
            tool_name: dict(tool_config) for tool_name, tool_config in self.config.nemo_skills_tool_overrides.items()
        }
        if self._uses_python_tool_sidecar:
            # Legacy HTTP PythonTool expects an explicit sidecar URL.
            python_tool_url = f"http://127.0.0.1:{self.config.python_tool_port}/mcp"
            overrides.setdefault("PythonTool", {})
            client_params = dict(overrides["PythonTool"].get("client_params", {}))
            client_params["base_url"] = python_tool_url
            overrides["PythonTool"]["client_params"] = client_params

        self.tool_manager = NSToolsToolManager(
            module_specs=self.config.nemo_skills_tools,
            overrides=overrides,
            context=context,
        )

        # Load tools and build name mapping
        async def _load_tools():
            tools = await self.tool_manager.list_all_tools()
            for tool in tools:
                self._tool_name_map[tool["name"]] = tool["name"]
            logger.info(f"Loaded {len(tools)} nemo_skills tools: {list(self._tool_name_map.keys())}")

        asyncio.get_event_loop().run_until_complete(_load_tools())
        logger.info("NeMo Skills ToolManager initialized successfully")

    async def execute_tool(self, tool_name: str, request: Request) -> PlainTextResponse:
        """Run the tool before close can acknowledge releasing its Python session."""
        # Skills records ownership only when execution returns. Close must wait for that record instead of
        # fencing an empty session and relying on a later, unreported best-effort cleanup.
        async with self.session_lock(request.session[SESSION_ID_KEY]):
            return await self._execute_tool(tool_name, request)

    async def _execute_tool(self, tool_name: str, request: Request) -> PlainTextResponse:
        """
        Execute a nemo_skills tool by name.

        Uses the nemo-gym session ID as the request_id for stateful tools.
        Returns the result as plain text for simple_agent compatibility.
        Tracks execution timing and timeout detection per session.
        """
        if not self.tool_manager:
            return PlainTextResponse(json.dumps({"error": "No tools configured"}))

        # Check if tool is in our known tools
        if tool_name not in self._tool_name_map:
            logger.error(f"Unknown tool requested: {tool_name}")
            return PlainTextResponse(json.dumps({"error": f"Unknown tool: {tool_name}"}))

        # The session middleware always sets the session ID, which keys the stateful python session.
        session_id = request.session[SESSION_ID_KEY]
        if self.session_is_closed(session_id) or self.session_is_verified(session_id):
            raise ResourcesSessionLostError(f"ns_tools session {session_id} is closed")

        if session_id not in self._timing_by_session:
            self._timing_by_session[session_id] = []

        start_time = time.perf_counter()
        is_internal_timeout = False
        is_request_timeout = False
        result = None

        try:
            body = await request.json()

            # Execute the tool
            result = await self.tool_manager.execute_tool(
                raw_name=tool_name,
                args=body,
                extra_args={"request_id": session_id},
            )

            # Check for internal sandbox timeout (process_status == "timeout")
            try:
                if isinstance(result, str):
                    result_dict = json.loads(result)
                elif isinstance(result, dict):
                    result_dict = result
                else:
                    result_dict = {}
                is_internal_timeout = result_dict.get("process_status") == "timeout"
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        except (httpx.TimeoutException, TimeoutError) as e:
            is_request_timeout = True
            logger.warning(f"Request timeout executing tool {tool_name}: {e}")
            result = {"error": "Request timeout", "process_status": "timeout"}

        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}: {e}")
            result = {"error": str(e)}

        elapsed = time.perf_counter() - start_time
        self._timing_by_session[session_id].append(
            {
                "tool_name": tool_name,
                "execution_time_seconds": elapsed,
                "is_internal_timeout": is_internal_timeout,
                "is_request_timeout": is_request_timeout,
            }
        )
        if self.config.verbose_tool_logging:
            timeout_info = ""
            if is_internal_timeout:
                timeout_info = " [INTERNAL_TIMEOUT]"
            elif is_request_timeout:
                timeout_info = " [REQUEST_TIMEOUT]"
            logger.info(f"Tool '{tool_name}' executed in {elapsed:.3f}s{timeout_info} (session={session_id[:8]}...)")

        # Return result as plain text to avoid double JSON serialization
        if isinstance(result, str):
            return PlainTextResponse(result)
        return PlainTextResponse(json.dumps(result))

    # --------------------------------------------------------
    # Sessions
    # --------------------------------------------------------

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self._timing_by_session

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        """Skills creates Python state on the first tool call; the base releases a prior cookie run."""

    async def release_session_state(self, session_id: str) -> None:
        """Keep Python ownership and timings until the sandbox acknowledges release."""
        if self.tool_manager is not None:
            await self.tool_manager.cleanup_request(session_id)
        self._timing_by_session.pop(session_id, None)

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    def _aggregate_timing_metrics(self, session_id: Optional[str]) -> Dict[str, float | int]:
        """Aggregate tool execution timing metrics for a session."""
        tool_timings = self._timing_by_session.get(session_id, []) if session_id else []
        if session_id in self._timing_by_session:
            # Drop detailed timings but keep the live Python session visible until release succeeds.
            self._timing_by_session[session_id] = []

        total_tool_time = sum(t["execution_time_seconds"] for t in tool_timings)
        num_tool_calls = len(tool_timings)
        avg_tool_time = total_tool_time / num_tool_calls if num_tool_calls > 0 else 0.0
        tool_timeout_count = sum(1 for t in tool_timings if t.get("is_internal_timeout"))
        tool_request_timeout_count = sum(1 for t in tool_timings if t.get("is_request_timeout"))

        metrics = {
            "total_tool_execution_time_seconds": total_tool_time,
            "num_tool_calls": num_tool_calls,
            "avg_tool_call_time_seconds": avg_tool_time,
            "tool_timeout_count": tool_timeout_count,
            "tool_request_timeout_count": tool_request_timeout_count,
        }
        return metrics

    async def verify(self, request: Request, body: NSToolsVerifyRequest) -> NSToolsVerifyResponse:
        """
        Verify the model's response by delegating to the configured verifier.

        The verifier is selected by:
        1. Per-sample `verifier_type` field (if present)
        2. Config `default_verifier` (fallback)

        Always aggregates and returns tool execution timing metrics for this session.
        Detailed per-call and summary logging is controlled by verbose_tool_logging.
        """
        session_id = request.session.get(SESSION_ID_KEY) if request else None
        if not session_id:
            return await self._verify(request, body, self._aggregate_timing_metrics(None))
        async with self.verifying(session_id, allow_missing=True) as verification:
            if verification.is_lost:
                return NSToolsVerifyResponse(
                    **(body.model_dump() | lost_session_verification(f"ns_tools session {session_id} is closed"))
                )
            metrics = verification.retained
            if not isinstance(metrics, dict):
                metrics = self._aggregate_timing_metrics(session_id)
            # Retain the snapshot even when the delegated verifier fails. A retry must rerun its judge without
            # losing the original timing totals, and a failed release must keep that snapshot owned.
            verification.retain(metrics)
            return await self._verify(request, body, metrics)

    async def _verify(
        self, request: Request, body: NSToolsVerifyRequest, metrics: Dict[str, float | int]
    ) -> NSToolsVerifyResponse:
        session_id = request.session.get(SESSION_ID_KEY) if request else None
        if self.config.verbose_tool_logging:
            logger.info(
                f"Session {session_id[:8] if session_id else 'unknown'}... metrics: "
                f"{metrics['num_tool_calls']} tool calls, total={metrics['total_tool_execution_time_seconds']:.3f}s, "
                f"avg={metrics['avg_tool_call_time_seconds']:.3f}s, "
                f"internal_timeouts={metrics['tool_timeout_count']}, request_timeouts={metrics['tool_request_timeout_count']}"
            )

        # Select verifier
        verifier_type = body.verifier_type or self.config.default_verifier

        if verifier_type not in self.config.verifiers:
            raise ValueError(
                f"Unknown verifier: {verifier_type}. Configure it in 'verifiers' or check 'default_verifier'."
            )

        verifier_ref = self.config.verifiers[verifier_type]

        # Delegate to the verifier
        response = await self.server_client.post(
            server_name=verifier_ref.name,
            url_path="/verify",
            json=body.model_dump(),
        )

        result = await response.json()

        # Hard fail if no reward in response
        if "reward" not in result:
            raise ValueError(f"Verifier did not return 'reward' field. Response: {result}")

        return NSToolsVerifyResponse(
            **{
                **body.model_dump(),
                # A delegated judge outage must stay masked, even if the request carries stale verdict fields.
                "mask_sample": result.get("mask_sample", False),
                "failure_kind": result.get("failure_kind"),
                "failure_reason": result.get("failure_reason"),
            },
            reward=result["reward"],
            delegated_response=result,
            **metrics,
        )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    async def shutdown(self):
        """Cleanup resources on server shutdown."""
        if self.tool_manager:
            await self.tool_manager.shutdown()

        # Terminate the python_tool subprocess if one was started.
        if self._python_tool_process:
            pid: int = self._python_tool_process.pid
            logger.info(f"Terminating python_tool server (PID: {pid})")
            self._python_tool_process.terminate()
            try:
                self._python_tool_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("python_tool server did not terminate gracefully, killing...")
                self._python_tool_process.kill()
                # Reap the child after SIGKILL so it doesn't linger as <defunct>.
                try:
                    self._python_tool_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error(f"python_tool server (PID: {pid}) did not exit after SIGKILL; may leak as a zombie")
            self._python_tool_process = None


if __name__ == "__main__":
    NSToolsResourcesServer.run_webserver()
