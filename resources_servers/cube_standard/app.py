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
CubeResourcesServer — bridges any CUBE-compliant benchmark into NeMo Gym.

Architecture:
    NeMo Gym Training Loop
        │  HTTP (OpenAI Responses API)
        ▼
    CubeResourcesServer  (this file)
        │  Python / asyncio.to_thread
        ▼
    CUBE Task  (sync blocking — wrapped in thread)
        │  Python / HTTP / SSH
        ▼
    Tool + Container / VM

One CubeResourcesServer instance wraps one CUBE benchmark. The server is
stateful: POST /seed_session creates a session, POST /step advances it,
POST /verify reads the terminal reward, POST /close tears it down.

Session identity is carried via HTTP cookies (set by the session middleware
from SimpleResourcesServer). No session ID is required in the request body.

Key design decisions (from v3-integration-plan.md):
    - All CUBE sync calls (reset/step/close/setup) run in asyncio.to_thread().
    - Screenshots are written to disk and served as static files at /screenshots/.
      The model server fetches them directly via URL — no base64 in JSON.
    - task-parallel benchmarks share one Benchmark instance; task instances are
      independent. benchmark-parallel benchmarks (e.g. OSWorld) get one Benchmark
      per session (i.e. one VM per rollout).
    - TTL cleanup evicts idle sessions every 60 seconds.
"""

import asyncio
import logging
import shutil
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Type

from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY

from resources_servers.cube_standard.cube_adapters import (
    _action_set_to_function_tool_params,
    _observation_to_nemo_gym_messages,
    _serialize_env_output,
)
from resources_servers.cube_standard.cube_loader import (
    _hydrate_extra_benchmark_config,
    find_benchmark_class,
    pip_install,
    select_task_config,
)
from resources_servers.cube_standard.schemas import (
    CubeCloseRequest,
    CubeCloseResponse,
    CubeSeedSessionRequest,
    CubeSeedSessionResponse,
    CubeStepRequest,
    CubeStepResponse,
    CubeVerifyRequest,
    CubeVerifyResponse,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class CubeResourcesServerConfig(BaseResourcesServerConfig):
    """Configuration for the CubeResourcesServer."""

    cube_id: str = Field(
        description="PyPI package name / entry-point key, e.g. 'miniwob-cube', 'osworld-cube'."
    )
    cube_version: Optional[str] = Field(
        default=None,
        description="Exact version to install. None means 'already installed or latest'.",
    )
    cube_dev_install_url: Optional[str] = Field(
        default=None,
        description="Fallback git/local URL if PyPI install fails, e.g. 'git+https://...'",
    )
    task_timeout_seconds: float = Field(
        default=300.0,
        description="Per-step timeout for CUBE task calls (reset + each step).",
    )
    session_ttl: int = Field(
        default=1800,
        description="Seconds before an idle session is garbage-collected.",
    )
    parallelization_mode: str = Field(
        default="task-parallel",
        description=(
            "Session lifecycle mode. "
            "'task-parallel': one shared Benchmark instance; each session gets its own Task. "
            "Suitable for lightweight benchmarks (MiniWob, WorkArena). "
            "'benchmark-parallel': one Benchmark instance per session (one VM per rollout). "
            "Required for VM-based benchmarks (OSWorld). "
            "Note: BenchmarkMetadata does not yet expose this field, so it cannot be "
            "auto-detected — set it explicitly in your YAML config."
        ),
    )
    max_concurrent_sessions: int = Field(
        default=50,
        description=(
            "Hard cap on simultaneous active sessions. For 'benchmark-parallel' mode "
            "this should match the number of VMs available (often 1)."
        ),
    )
    extra_benchmark_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Passed as kwargs to the Benchmark constructor.",
    )
    screenshot_base_url: Optional[str] = Field(
        default=None,
        description=(
            "Base URL at which this server's /screenshots/ endpoint is reachable by "
            "the model server. E.g. 'http://cube-server:8000'. If None, defaults to "
            "http://localhost:8000. Must be reachable from the model server host."
        ),
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------


@dataclass
class CubeSessionState:
    """
    Per-session mutable state. Stored in the server's _sessions dict, keyed by session_id.

    Not Pydantic — Task and Benchmark are not serializable and are not needed in JSON.
    """

    task: Any  # cube.task.Task (sync, blocking)
    task_config: Any  # cube.task.TaskConfig
    screenshot_dir: Path  # per-session temp dir under _screenshots_root
    last_env_output: Optional[Any] = None  # most recent EnvironmentOutput
    benchmark: Optional[Any] = None  # Benchmark instance (benchmark-parallel only)
    created_at: float = field(default_factory=time.monotonic)
    last_accessed: float = field(default_factory=time.monotonic)
    step_index: int = 0  # incremented per step; used for screenshot filenames


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------


class CubeResourcesServer(SimpleResourcesServer):
    """
    NeMo Gym ResourcesServer that wraps any CUBE-compliant benchmark.

    Registered endpoints:
        POST /seed_session   →  task.reset()
        POST /step           →  task.step(Action(...))
        POST /verify         →  read last_env_output.reward
        POST /close          →  task.close() + cleanup
        POST /aggregate_metrics  (inherited from SimpleResourcesServer)
        GET  /screenshots/<session_id>/<filename>  (StaticFiles)
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: CubeResourcesServerConfig

    # Private state — all declared with PrivateAttr following the NeMo Gym pattern
    # (tavily_search, newton_bench, proof_judge). PrivateAttr is not part of the
    # Pydantic model fields and does not participate in validation/serialization.
    _benchmark_class: Type = PrivateAttr(default=None)
    _parallelization_mode: str = PrivateAttr(default="task-parallel")
    _max_concurrent_tasks: int = PrivateAttr(default=9999)
    _shared_benchmark: Optional[Any] = PrivateAttr(default=None)
    _sessions: Dict[str, CubeSessionState] = PrivateAttr(default_factory=dict)
    _sessions_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _ttl_task: Optional[asyncio.Task] = PrivateAttr(default=None)
    _screenshots_root: Path = PrivateAttr(default=None)  # set in model_post_init
    _extra_config: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        # Create a root directory for all screenshots (survives across sessions).
        # Cannot use PrivateAttr factory because it requires tempfile.mkdtemp().
        self._screenshots_root = Path(tempfile.mkdtemp(prefix="cube_screenshots_"))
        logger.info("Screenshots root: %s", self._screenshots_root)

        # Step 1: Install CUBE package
        pip_install(self.config.cube_id, self.config.cube_version, self.config.cube_dev_install_url)

        # Step 1b: Hydrate extra_benchmark_config — convert any nested dict with '_type'
        # into the corresponding Python object (e.g. VMBackend, ContainerBackend).
        self._extra_config = _hydrate_extra_benchmark_config(self.config.extra_benchmark_config)

        # Step 2: Resolve Benchmark class via entry points
        benchmark_class, err = find_benchmark_class(self.config.cube_id)
        if benchmark_class is None:
            raise RuntimeError(
                f"Failed to resolve Benchmark class for '{self.config.cube_id}': {err}"
            )
        self._benchmark_class = benchmark_class

        # Step 3: Use parallelization_mode from config.
        # Note: BenchmarkMetadata does not expose parallelization_mode as a standard field
        # (as of CUBE standard v0.1), so auto-detection is not possible. Set it explicitly
        # in the YAML config: 'task-parallel' for shared Benchmark, 'benchmark-parallel'
        # for one Benchmark (VM) per session.
        mode = self.config.parallelization_mode
        self._parallelization_mode = mode
        self._max_concurrent_tasks = self.config.max_concurrent_sessions

        # Step 4: For task-parallel, create and set up ONE shared benchmark instance.
        # For benchmark-parallel, each session gets its own Benchmark (e.g. one VM per rollout).
        if mode == "task-parallel":
            shared_benchmark = benchmark_class(**self._extra_config)
            shared_benchmark.setup()
            self._shared_benchmark = shared_benchmark
            meta = getattr(benchmark_class, "benchmark_metadata", None)
            num_tasks = getattr(meta, "num_tasks", "?") if meta else "?"
            logger.info(
                "CubeResourcesServer: task-parallel benchmark '%s' ready. num_tasks=%s",
                self.config.cube_id,
                num_tasks,
            )
        elif mode == "benchmark-parallel":
            logger.info(
                "CubeResourcesServer: benchmark-parallel CUBE '%s'. "
                "Will create one Benchmark per session. max_concurrent_sessions=%d",
                self.config.cube_id,
                self.config.max_concurrent_sessions,
            )
        else:
            raise ValueError(
                f"Invalid parallelization_mode '{mode}'. "
                "Must be 'task-parallel' or 'benchmark-parallel'."
            )

    # -----------------------------------------------------------------------
    # FastAPI app setup
    # -----------------------------------------------------------------------

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()  # registers /seed_session, /verify, /aggregate_metrics

        # Additional endpoints for stateful episode loop
        app.post("/step")(self.step)
        app.post("/close")(self.close)

        # Static file serving for screenshots.
        # The model server fetches screenshot images directly from this endpoint.
        # StaticFiles requires the directory to exist at mount time.
        self._screenshots_root.mkdir(parents=True, exist_ok=True)
        app.mount(
            "/screenshots",
            StaticFiles(directory=str(self._screenshots_root)),
            name="screenshots",
        )

        # Use lifespan context manager (newton_bench pattern) — @app.on_event is deprecated.
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def _lifespan(_app: FastAPI):
            # Startup: launch TTL cleanup background task
            self._ttl_task = asyncio.create_task(self._ttl_cleanup_loop())
            async with parent_lifespan(_app):
                yield
            # Shutdown: cancel TTL task and clean up shared resources
            if self._ttl_task is not None:
                self._ttl_task.cancel()
            if self._shared_benchmark is not None:
                try:
                    self._shared_benchmark.close()
                except Exception as e:
                    logger.warning("shared_benchmark.close() raised during shutdown: %s", e)
            shutil.rmtree(self._screenshots_root, ignore_errors=True)
            logger.info("CubeResourcesServer shutdown complete.")

        app.router.lifespan_context = _lifespan

        return app

    # -----------------------------------------------------------------------
    # Endpoint: POST /seed_session
    # -----------------------------------------------------------------------

    async def seed_session(
        self, request: Request, body: CubeSeedSessionRequest
    ) -> CubeSeedSessionResponse:
        """
        Initialize a new CUBE task session.

        1. Enforce session limit.
        2. Select task config (by task_id or first available).
        3. Create and reset task (in thread — task.reset() is blocking).
        4. Assign a session_id cookie.
        5. Store session state.
        6. Return initial observation + tool list.
        """
        async with self._sessions_lock:
            active = len(self._sessions)
            limit = min(self._max_concurrent_tasks, self.config.max_concurrent_sessions)
            if active >= limit:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"Session limit reached ({active}/{limit}). "
                        f"Benchmark '{self.config.cube_id}' allows max "
                        f"{self._max_concurrent_tasks} concurrent sessions."
                    ),
                )

        # Create task instance
        if self._parallelization_mode == "task-parallel":
            task_config = select_task_config(self._shared_benchmark, body.task_id, body.seed)
            task = await asyncio.wait_for(
                asyncio.to_thread(
                    task_config.make,
                    runtime_context=self._shared_benchmark._runtime_context,
                    container_backend=self._shared_benchmark.container_backend,
                ),
                timeout=self.config.task_timeout_seconds,
            )
            session_benchmark = None
        else:
            # benchmark-parallel: one Benchmark (and one VM) per session.
            # We use this same benchmark for both task config resolution and task creation
            # to avoid launching two VMs (setup() starts the VM for backends like OSWorld).
            session_benchmark = self._benchmark_class(**self._extra_config)
            await asyncio.wait_for(
                asyncio.to_thread(session_benchmark.setup),
                timeout=self.config.task_timeout_seconds,
            )
            task_config = select_task_config(session_benchmark, body.task_id, body.seed)
            task = await asyncio.wait_for(
                asyncio.to_thread(
                    task_config.make,
                    runtime_context=session_benchmark._runtime_context,
                    container_backend=session_benchmark.container_backend,
                ),
                timeout=self.config.task_timeout_seconds,
            )

        # Reset task to get initial observation
        obs, _info = await asyncio.wait_for(
            asyncio.to_thread(task.reset),
            timeout=self.config.task_timeout_seconds,
        )

        # Session ID is assigned by the NeMo Gym session middleware before this handler runs.
        session_id = request.session[SESSION_ID_KEY]

        # Each session gets its own screenshot subdirectory
        screenshot_dir = self._screenshots_root / session_id
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        # Store session
        state = CubeSessionState(
            task=task,
            task_config=task_config,
            benchmark=session_benchmark,
            screenshot_dir=screenshot_dir,
        )
        async with self._sessions_lock:
            self._sessions[session_id] = state

        # Convert CUBE formats to NeMo Gym formats
        obs_messages = _observation_to_nemo_gym_messages(obs)
        tools = _action_set_to_function_tool_params(task.action_set)

        logger.info(
            "Session '%s' started: task_id='%s', mode=%s",
            session_id,
            task_config.task_id,
            self._parallelization_mode,
        )

        return CubeSeedSessionResponse(
            obs=obs_messages,
            tools=tools,
            task_id=task_config.task_id,
        )

    # -----------------------------------------------------------------------
    # Endpoint: POST /step
    # -----------------------------------------------------------------------

    async def step(self, request: Request, body: CubeStepRequest) -> CubeStepResponse:
        """
        Execute one action in the CUBE task.

        1. Look up the session.
        2. Construct a CUBE Action from the tool call components.
        3. Call task.step() in a thread with a timeout.
        4. Serialize the EnvironmentOutput to (output, content_type).
        5. Return CubeStepResponse.
        """
        session_id = request.session[SESSION_ID_KEY]
        state = self._sessions.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        state.last_accessed = time.monotonic()

        # Import here to avoid hard dep before pip_install runs
        from cube.core import Action

        action = Action(
            id=body.call_id,
            name=body.name,
            arguments=body.arguments,
        )

        try:
            env_output = await asyncio.wait_for(
                asyncio.to_thread(state.task.step, action),
                timeout=self.config.task_timeout_seconds,
            )
        except asyncio.TimeoutError:
            state.last_env_output = None
            logger.warning(
                "Session '%s': task.step() timed out after %.1fs.",
                session_id,
                self.config.task_timeout_seconds,
            )
            return CubeStepResponse(
                output=f"Step timed out after {self.config.task_timeout_seconds}s.",
                content_type="text/plain",
                done=True,
                reward=0.0,
                error="timeout",
            )
        except Exception as e:
            logger.exception("Session '%s': task.step() raised an exception.", session_id)
            state.last_env_output = None
            return CubeStepResponse(
                output=f"Error during step: {e}",
                content_type="text/plain",
                done=True,
                reward=0.0,
                error=str(e),
            )

        state.last_env_output = env_output
        state.step_index += 1

        # Run in thread: _serialize_env_output may write PNG bytes to disk,
        # which is blocking I/O that must not block the async event loop.
        output, content_type = await asyncio.to_thread(
            _serialize_env_output,
            env_output=env_output,
            session_id=session_id,
            step_index=state.step_index,
            screenshot_dir=state.screenshot_dir,
            base_url=self._get_screenshot_base_url(),
        )

        return CubeStepResponse(
            output=output,
            content_type=content_type,
            done=env_output.done,
            reward=env_output.reward,
            error=env_output.error.exception_str if env_output.error else None,
        )

    def _get_screenshot_base_url(self) -> str:
        """Return the base URL for screenshot static files."""
        if self.config.screenshot_base_url:
            return self.config.screenshot_base_url.rstrip("/")
        # Fallback: localhost on the port the server is bound to
        port = getattr(self, "_port", 8000)
        return f"http://localhost:{port}"

    # -----------------------------------------------------------------------
    # Endpoint: POST /verify
    # -----------------------------------------------------------------------

    async def verify(
        self, request: Request, body: CubeVerifyRequest
    ) -> CubeVerifyResponse:
        """
        Return the terminal reward for the session.

        Reads last_env_output.reward — no additional evaluate() call needed because
        task.step() calls task.evaluate() internally when done=True.
        """
        session_id = request.session[SESSION_ID_KEY]
        state = self._sessions.get(session_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")

        if state.last_env_output is None:
            reward = 0.0
            reward_info: Dict[str, Any] = {"error": "no completed step found"}
        else:
            reward = state.last_env_output.reward
            reward_info = state.last_env_output.info or {}

        return CubeVerifyResponse(
            **body.model_dump(),
            reward=reward,
            reward_info=reward_info,
        )

    # -----------------------------------------------------------------------
    # Endpoint: POST /close
    # -----------------------------------------------------------------------

    async def close(self, request: Request, body: CubeCloseRequest) -> CubeCloseResponse:
        """
        Close and clean up the session.

        1. Pop the session from the registry (idempotent — returns success if already gone).
        2. Close the task (in thread).
        3. Close the benchmark (in thread, benchmark-parallel only).
        4. Remove the session's screenshot directory.
        """
        session_id = request.session[SESSION_ID_KEY]
        async with self._sessions_lock:
            state = self._sessions.pop(session_id, None)

        if state is None:
            logger.info("Session '%s' not found on /close (already closed?).", session_id)
            return CubeCloseResponse(message="Session not found (already closed?).", success=True)

        await self._cleanup_session(state, session_id)

        logger.info("Session '%s' closed.", session_id)
        return CubeCloseResponse(message="Closed.", success=True)

    async def _cleanup_session(self, state: CubeSessionState, session_id: str) -> None:
        """Close task, benchmark, and screenshot dir for a session state."""
        try:
            await asyncio.to_thread(state.task.close)
        except Exception as e:
            logger.warning("Session '%s': task.close() raised: %s", session_id, e)

        if state.benchmark is not None:
            try:
                await asyncio.to_thread(state.benchmark.close)
            except Exception as e:
                logger.warning("Session '%s': benchmark.close() raised: %s", session_id, e)

        shutil.rmtree(state.screenshot_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # TTL cleanup loop
    # -----------------------------------------------------------------------

    async def _ttl_cleanup_loop(self) -> None:
        """
        Background task that evicts idle sessions every 60 seconds.

        A session is stale if (now - last_accessed) > session_ttl seconds.
        """
        while True:
            await asyncio.sleep(60)
            now = time.monotonic()
            async with self._sessions_lock:
                stale = [
                    sid
                    for sid, state in self._sessions.items()
                    if (now - state.last_accessed) > self.config.session_ttl
                ]
            for sid in stale:
                # Re-check last_accessed inside the lock to avoid TOCTOU: another
                # coroutine may have updated last_accessed between the scan and here.
                async with self._sessions_lock:
                    state = self._sessions.get(sid)
                    if state is not None and (now - state.last_accessed) > self.config.session_ttl:
                        self._sessions.pop(sid)
                    else:
                        state = None  # recently accessed — skip eviction
                if state is not None:
                    logger.info("TTL evicting session '%s'.", sid)
                    await self._cleanup_session(state, sid)


if __name__ == "__main__":
    CubeResourcesServer.run_webserver()
