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
- Integration with nemo_skills ToolManager for tool execution (e.g., PythonTool)
- Verification delegation to math_with_judge
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from nemo_skills.mcp.tool_manager import ToolManager
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import SESSION_ID_KEY


logger = logging.getLogger(__name__)


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

    # NeMo Skills tool modules to load (e.g., "nemo_skills.mcp.servers.python_tool.PythonTool")
    nemo_skills_tools: List[str] = Field(default_factory=list)

    # Per-tool overrides for nemo_skills tools
    nemo_skills_tool_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Sandbox configuration for code execution tools
    sandbox_host: str = "127.0.0.1"
    sandbox_port: str = "6000"

    # python_tool HTTP server port (spawned automatically)
    python_tool_port: int = 8765

    # Verbose logging for tool execution timing (disabled by default)
    verbose_tool_logging: bool = False

    # Path to write tool call traces (JSONL format). If empty, tracing is disabled.
    # Supports environment variable expansion (e.g., /lustre/.../traces/${SLURM_JOB_ID}/tool_calls.jsonl)
    trace_file_path: str = ""


# ============================================================
# Run/Verify Request/Response Models
# ============================================================


class NSToolsRunRequest(BaseRunRequest):
    """Run request that allows extra fields from the sample."""

    model_config = ConfigDict(extra="allow")

    # Per-sample verifier selection (optional, falls back to default_verifier)
    verifier_type: Optional[str] = None

    # Fields for math_with_judge verifier
    question: Optional[str] = None
    expected_answer: Optional[str] = None


class NSToolsVerifyRequest(NSToolsRunRequest, BaseVerifyRequest):
    pass


class NSToolsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    delegated_response: Optional[Dict[str, Any]] = None

    # Timing metrics for tool execution (auto-logged to wandb)
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
    _timing_by_session: Dict[str, List[dict]] = {}  # session_id -> list of timing records
    _session_call_index: Dict[str, int] = {}  # session_id -> next call index (for trace ordering)
    _trace_file: Optional[Any] = None  # File handle for trace output
    _trace_file_lock: Optional[asyncio.Lock] = None  # Lock for thread-safe writes

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Initialize trace file if configured
        self._initialize_trace_file()

        # Initialize nemo_skills ToolManager if tools are configured
        if self.config.nemo_skills_tools:
            # Start the python_tool HTTP server first
            self._start_python_tool_server()
            self._initialize_nemo_skills_tools()

            # Register a catch-all endpoint for tool execution
            # This handles any tool name dynamically
            app.post("/{tool_name}")(self.execute_tool)

        return app

    def _initialize_trace_file(self):
        """Initialize the trace file for recording tool calls."""
        if not self.config.trace_file_path:
            return

        # Expand environment variables in path
        trace_path = os.path.expandvars(self.config.trace_file_path)
        trace_path = Path(trace_path)

        # Create directory if needed
        trace_path.parent.mkdir(parents=True, exist_ok=True)

        # Open file in append mode
        self._trace_file = open(trace_path, "a", buffering=1)  # Line buffered
        self._trace_file_lock = asyncio.Lock()
        logger.info(f"Trace file initialized: {trace_path}")

    def _start_python_tool_server(self):
        """Spawn python_tool HTTP server as a subprocess."""
        logger.info(f"Starting python_tool HTTP server on port {self.config.python_tool_port}")

        # Write config file for python_tool (compatible with both old and new versions)
        config_content = {
            "sandbox": {
                "sandbox_type": "local",  # "local" sandbox type connects via HTTP to host:port
                "host": self.config.sandbox_host,
                "port": int(self.config.sandbox_port),
            }
        }
        self._python_tool_config_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix="python_tool_config_"
        )
        json.dump(config_content, self._python_tool_config_file)
        self._python_tool_config_file.flush()
        logger.info(f"python_tool config file: {self._python_tool_config_file.name}")
        logger.info(f"python_tool config: {config_content}")

        # Build command - use config file instead of CLI args for compatibility
        # Old python_tool.py only accepts --config, new version accepts both
        cmd = [
            sys.executable,
            "-m",
            "nemo_skills.mcp.servers.python_tool",
            "--config",
            self._python_tool_config_file.name,
        ]
        logger.info(f"python_tool command: {' '.join(cmd)}")

        # Set environment for subprocess
        env = os.environ.copy()
        # Set port via env var as fallback (new python_tool reads PYTHON_TOOL_PORT)
        env["PYTHON_TOOL_PORT"] = str(self.config.python_tool_port)
        env["PYTHON_TOOL_HOST"] = "127.0.0.1"

        # Ensure /nemo_run/code is FIRST in PYTHONPATH so mounted nemo_skills takes precedence
        current_pythonpath = env.get("PYTHONPATH", "")
        if "/nemo_run/code" in current_pythonpath:
            paths = [p for p in current_pythonpath.split(":") if p and p != "/nemo_run/code"]
            env["PYTHONPATH"] = "/nemo_run/code:" + ":".join(paths)
        else:
            env["PYTHONPATH"] = "/nemo_run/code" + (":" + current_pythonpath if current_pythonpath else "")
        logger.info(f"python_tool PYTHONPATH: {env['PYTHONPATH']}")

        # Don't pipe stdout/stderr so we can see output directly in logs
        self._python_tool_process = subprocess.Popen(cmd, env=env)

        # Wait for server to be ready
        self._wait_for_server_ready()
        logger.info(f"python_tool HTTP server started (PID: {self._python_tool_process.pid})")

    def _wait_for_server_ready(self, timeout: float = 30.0, poll_interval: float = 0.5):
        """Wait for the python_tool HTTP server to be ready."""
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
            }
        }

        # Merge in PythonTool URL override to point to our spawned HTTP server
        overrides = dict(self.config.nemo_skills_tool_overrides)
        python_tool_url = f"http://127.0.0.1:{self.config.python_tool_port}/mcp"
        overrides.setdefault("PythonTool", {})
        overrides["PythonTool"]["client_params"] = {"base_url": python_tool_url}

        self.tool_manager = ToolManager(
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
        """
        Execute a nemo_skills tool by name.

        Uses the nemo-gym session ID as the request_id for stateful tools.
        Returns the result as plain text for simple_agent compatibility.
        Tracks execution timing and timeout detection per session.
        Generates trace_id and writes trace records if tracing is enabled.
        """
        if not self.tool_manager:
            return PlainTextResponse(json.dumps({"error": "No tools configured"}))

        # Check if tool is in our known tools
        if tool_name not in self._tool_name_map:
            logger.error(f"Unknown tool requested: {tool_name}")
            return PlainTextResponse(json.dumps({"error": f"Unknown tool: {tool_name}"}))

        # Get session ID for stateful execution
        session_id = request.session.get(SESSION_ID_KEY)
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.warning(f"No session ID found, using fallback: {session_id}")

        # Initialize timing list for this session if needed
        if session_id not in self._timing_by_session:
            self._timing_by_session[session_id] = []

        # Get and increment session call index for trace ordering
        session_call_index = self._session_call_index.get(session_id, 0)
        self._session_call_index[session_id] = session_call_index + 1

        # Generate trace_id if tracing is enabled
        trace_id = None
        if self._trace_file is not None:
            trace_id = uuid.uuid4().hex[:12]  # Short but unique enough

        start_time = time.perf_counter()
        timestamp = datetime.now(timezone.utc).isoformat()
        is_internal_timeout = False
        is_request_timeout = False
        result = None
        body = None

        if trace_id:
            logger.info(f"[TRACE:{trace_id}] layer=ns_tools event=start session={session_id[:8]}... tool={tool_name}")

        try:
            body = await request.json()

            # Execute the tool with trace_id in extra_args
            extra_args = {"request_id": session_id}
            if trace_id:
                extra_args["trace_id"] = trace_id

            result = await self.tool_manager.execute_tool(
                raw_name=tool_name,
                args=body,
                extra_args=extra_args,
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
        elapsed_ms = elapsed * 1000

        # Store timing record
        self._timing_by_session[session_id].append(
            {
                "tool_name": tool_name,
                "execution_time_seconds": elapsed,
                "is_internal_timeout": is_internal_timeout,
                "is_request_timeout": is_request_timeout,
            }
        )

        # Log tool execution
        if trace_id:
            logger.info(f"[TRACE:{trace_id}] layer=ns_tools event=done elapsed_ms={elapsed_ms:.1f}")

        if self.config.verbose_tool_logging:
            timeout_info = ""
            if is_internal_timeout:
                timeout_info = " [INTERNAL_TIMEOUT]"
            elif is_request_timeout:
                timeout_info = " [REQUEST_TIMEOUT]"
            logger.info(f"Tool '{tool_name}' executed in {elapsed:.3f}s{timeout_info} (session={session_id[:8]}...)")

        # Write trace record if tracing is enabled
        if self._trace_file is not None and trace_id:
            trace_record = {
                "trace_id": trace_id,
                "session_id": session_id,
                "session_call_index": session_call_index,
                "timestamp": timestamp,
                "tool_name": tool_name,
                "arguments": body,
                "result": result if isinstance(result, dict) else {"output": result},
                "timings": {
                    "total_ms": elapsed_ms,
                },
                "is_internal_timeout": is_internal_timeout,
                "is_request_timeout": is_request_timeout,
            }
            await self._write_trace_record(trace_record)

        # Return result as plain text to avoid double JSON serialization
        if isinstance(result, str):
            return PlainTextResponse(result)
        return PlainTextResponse(json.dumps(result))

    async def _write_trace_record(self, record: dict):
        """Write a trace record to the trace file (thread-safe)."""
        if self._trace_file is None:
            return

        async with self._trace_file_lock:
            try:
                self._trace_file.write(json.dumps(record) + "\n")
                self._trace_file.flush()
            except Exception as e:
                logger.warning(f"Failed to write trace record: {e}")

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    def _aggregate_timing_metrics(self, session_id: Optional[str]) -> Dict[str, Any]:
        """
        Aggregate tool execution timing metrics for a session.

        Retrieves and cleans up timing data, computes aggregates.
        Returns dict with: total_time, num_calls, avg_time, internal_timeouts, request_timeouts
        """
        # Retrieve and clean up timing for this session (pop to free memory)
        tool_timings = self._timing_by_session.pop(session_id, []) if session_id else []

        # Clean up session call index tracker
        if session_id and session_id in self._session_call_index:
            del self._session_call_index[session_id]

        # Aggregate metrics
        total_tool_time = sum(t["execution_time_seconds"] for t in tool_timings)
        num_tool_calls = len(tool_timings)
        avg_tool_time = total_tool_time / num_tool_calls if num_tool_calls > 0 else 0.0
        tool_timeout_count = sum(1 for t in tool_timings if t.get("is_internal_timeout"))
        tool_request_timeout_count = sum(1 for t in tool_timings if t.get("is_request_timeout"))

        return {
            "total_tool_execution_time_seconds": total_tool_time,
            "num_tool_calls": num_tool_calls,
            "avg_tool_call_time_seconds": avg_tool_time,
            "tool_timeout_count": tool_timeout_count,
            "tool_request_timeout_count": tool_request_timeout_count,
        }

    async def verify(self, request: Request, body: NSToolsVerifyRequest) -> NSToolsVerifyResponse:
        """
        Verify the model's response by delegating to the configured verifier.

        The verifier is selected by:
        1. Per-sample `verifier_type` field (if present)
        2. Config `default_verifier` (fallback)

        Also aggregates and returns tool execution timing metrics for this session.
        """
        # Get session ID to retrieve accumulated timing
        session_id = request.session.get(SESSION_ID_KEY)

        # Aggregate timing metrics
        metrics = self._aggregate_timing_metrics(session_id)

        # Log summary (only if verbose logging enabled)
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
            **body.model_dump(),
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

        # Close trace file
        if self._trace_file:
            logger.info("Closing trace file")
            self._trace_file.close()
            self._trace_file = None

        # Terminate the python_tool subprocess
        if self._python_tool_process:
            logger.info(f"Terminating python_tool server (PID: {self._python_tool_process.pid})")
            self._python_tool_process.terminate()
            try:
                self._python_tool_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("python_tool server did not terminate gracefully, killing...")
                self._python_tool_process.kill()
            self._python_tool_process = None


if __name__ == "__main__":
    NSToolsResourcesServer.run_webserver()
