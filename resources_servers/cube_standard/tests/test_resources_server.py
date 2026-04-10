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
Integration tests for CubeResourcesServer.

Tests that require a real CUBE package (counter-cube) are marked with
@pytest.mark.integration and skip gracefully when counter-cube is not installed.

The mock-based tests verify the HTTP layer without a real CUBE package.
"""

import asyncio
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.server_utils import SESSION_ID_KEY

try:
    import counter_cube  # noqa: F401

    _COUNTER_CUBE_AVAILABLE = True
except ImportError:
    _COUNTER_CUBE_AVAILABLE = False

try:
    import cube  # noqa: F401

    _CUBE_AVAILABLE = True
except ImportError:
    _CUBE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_nemo_response_dict() -> dict:
    """Minimal valid NeMoGymResponse dict for use in verify request bodies."""
    return {
        "id": "resp_test",
        "created_at": 1_700_000_000.0,
        "model": "dummy_model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _make_mock_obs(text: str = "Observation text") -> MagicMock:
    obs = MagicMock()
    obs.to_llm_messages.return_value = [{"role": "user", "content": text}]
    return obs


def _make_mock_env_output(
    text: str = "Step done",
    done: bool = False,
    reward: float = 0.0,
) -> MagicMock:
    obs = _make_mock_obs(text)
    env_out = MagicMock()
    env_out.obs = obs
    env_out.done = done
    env_out.reward = reward
    env_out.error = None
    env_out.info = {}
    return env_out


def _make_mock_task(
    action_names: list[str] | None = None,
    step_response: MagicMock | None = None,
) -> MagicMock:
    """Create a minimal mock CUBE Task."""
    if action_names is None:
        action_names = ["click", "final_step"]

    action_schemas = []
    for name in action_names:
        schema = MagicMock()
        schema.as_dict.return_value = {
            "type": "function",
            "function": {"name": name, "description": f"{name} action", "parameters": {}},
        }
        action_schemas.append(schema)

    task = MagicMock()
    task.action_set = action_schemas
    task.reset.return_value = (_make_mock_obs(), {})
    task.step.return_value = step_response or _make_mock_env_output()
    task.close.return_value = None
    return task


def _make_mock_task_config(task_id: str = "test-task") -> MagicMock:
    config = MagicMock()
    config.task_id = task_id
    config.seed = None
    return config


def _make_mock_benchmark(task: MagicMock, task_config: MagicMock) -> MagicMock:
    """Create a minimal mock CUBE Benchmark (task-parallel)."""
    benchmark = MagicMock()
    benchmark.benchmark_metadata = MagicMock()
    benchmark.benchmark_metadata.parallelization_mode = "task-parallel"
    benchmark.benchmark_metadata.max_concurrent_tasks = 50
    benchmark.benchmark_metadata.num_tasks = 1
    benchmark._runtime_context = MagicMock()
    benchmark.container_backend = MagicMock()
    benchmark.get_task_configs.return_value = [task_config]
    benchmark.setup.return_value = None
    benchmark.close.return_value = None
    task_config.make.return_value = task
    return benchmark


# ---------------------------------------------------------------------------
# Fixture: CubeResourcesServer with full mock CUBE stack
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_cube_server():
    """
    CubeResourcesServer instance backed by mock CUBE objects.

    Patches pip_install, find_benchmark_class so no real package is needed.
    Returns the server instance (not a TestClient — the server is not started).
    """
    from resources_servers.cube_standard.app import CubeResourcesServer, CubeResourcesServerConfig

    task_config = _make_mock_task_config()
    task = _make_mock_task()
    benchmark = _make_mock_benchmark(task, task_config)

    BenchmarkClass = MagicMock(return_value=benchmark)
    BenchmarkClass.benchmark_metadata = benchmark.benchmark_metadata

    with (
        patch("resources_servers.cube_standard.app.pip_install"),
        patch(
            "resources_servers.cube_standard.app.find_benchmark_class",
            return_value=(BenchmarkClass, ""),
        ),
    ):
        config = CubeResourcesServerConfig(
            name="test-cube-server",
            host="0.0.0.0",
            port=8080,
            entrypoint="app.py",
            cube_id="mock-cube",
        )
        from nemo_gym.server_utils import ServerClient

        server = CubeResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    # Expose the mock objects for assertions
    server._mock_task = task
    server._mock_task_config = task_config
    server._mock_benchmark = benchmark
    return server


# ---------------------------------------------------------------------------
# Unit-level tests using mock_cube_server
# ---------------------------------------------------------------------------


class TestCubeResourcesServerMock:
    @pytest.mark.asyncio
    async def test_seed_session_returns_obs_and_tools(self, mock_cube_server):
        server = mock_cube_server

        with patch("resources_servers.cube_standard.app.select_task_config") as mock_select:
            mock_select.return_value = server._mock_task_config

            request = MagicMock()
            request.session = {SESSION_ID_KEY: str(uuid.uuid4())}

            response = await server.seed_session(request, MagicMock(task_id=None, seed=None))

        assert len(response.obs) >= 1
        assert len(response.tools) >= 1
        assert response.task_id == "test-task"

    @pytest.mark.asyncio
    async def test_seed_session_stores_session(self, mock_cube_server):
        server = mock_cube_server

        with patch("resources_servers.cube_standard.app.select_task_config") as mock_select:
            mock_select.return_value = server._mock_task_config

            request = MagicMock()
            request.session = {SESSION_ID_KEY: str(uuid.uuid4())}
            await server.seed_session(request, MagicMock(task_id=None, seed=None))

        assert len(server._sessions) == 1

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _CUBE_AVAILABLE, reason="cube-standard not installed")
    async def test_step_returns_response(self, mock_cube_server):
        server = mock_cube_server
        session_id = str(uuid.uuid4())

        # Manually plant a session
        from resources_servers.cube_standard.app import CubeSessionState

        screenshot_dir = server._screenshots_root / session_id
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        state = CubeSessionState(
            task=server._mock_task,
            task_config=server._mock_task_config,
            screenshot_dir=screenshot_dir,
        )
        server._sessions[session_id] = state

        request = MagicMock()
        request.session = {SESSION_ID_KEY: session_id}

        body = MagicMock()
        body.call_id = "call_001"
        body.name = "click"
        body.arguments = {"x": 10, "y": 20}

        # Action is imported locally inside step() via "from cube.core import Action",
        # so it must be patched at its source location.
        with patch("cube.core.Action") as MockAction:
            mock_action = MagicMock()
            MockAction.return_value = mock_action
            response = await server.step(request, body)

        assert response.content_type in ("text/plain", "image/png")
        assert "done" in response.model_fields or hasattr(response, "done")

    @pytest.mark.asyncio
    async def test_step_404_when_session_missing(self, mock_cube_server):
        from fastapi import HTTPException

        server = mock_cube_server
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "nonexistent-session"}

        body = MagicMock()
        body.call_id = "call_001"
        body.name = "click"
        body.arguments = {}

        with pytest.raises(HTTPException) as exc_info:
            await server.step(request, body)
        assert exc_info.value.status_code == 404

    @pytest.mark.asyncio
    async def test_verify_returns_reward_from_last_env_output(self, mock_cube_server):
        from resources_servers.cube_standard.app import CubeSessionState

        server = mock_cube_server
        session_id = str(uuid.uuid4())

        env_out = _make_mock_env_output(reward=1.0, done=True)
        screenshot_dir = server._screenshots_root / session_id
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        state = CubeSessionState(
            task=server._mock_task,
            task_config=server._mock_task_config,
            screenshot_dir=screenshot_dir,
            last_env_output=env_out,
        )
        server._sessions[session_id] = state

        request = MagicMock()
        request.session = {SESSION_ID_KEY: session_id}

        # Build a minimal CubeVerifyRequest
        from resources_servers.cube_standard.schemas import CubeVerifyRequest
        from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming

        responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input=[])
        mock_response = MagicMock(spec=NeMoGymResponse)
        mock_response.model_dump = lambda mode=None: {}
        body = MagicMock(spec=CubeVerifyRequest)
        body.model_dump = lambda: {
            "responses_create_params": responses_create_params.model_dump(mode="json"),
            "response": _make_nemo_response_dict(),
        }

        response = await server.verify(request, body)
        assert response.reward == 1.0

    @pytest.mark.asyncio
    async def test_verify_returns_zero_reward_when_no_steps(self, mock_cube_server):
        from resources_servers.cube_standard.app import CubeSessionState

        server = mock_cube_server
        session_id = str(uuid.uuid4())

        screenshot_dir = server._screenshots_root / session_id
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        state = CubeSessionState(
            task=server._mock_task,
            task_config=server._mock_task_config,
            screenshot_dir=screenshot_dir,
            last_env_output=None,  # no steps taken
        )
        server._sessions[session_id] = state

        request = MagicMock()
        request.session = {SESSION_ID_KEY: session_id}
        body = MagicMock()
        body.model_dump = lambda: {
            "responses_create_params": {"input": []},
            "response": _make_nemo_response_dict(),
        }

        response = await server.verify(request, body)
        assert response.reward == 0.0
        assert "error" in response.reward_info

    @pytest.mark.asyncio
    async def test_close_cleans_up_session(self, mock_cube_server):
        from resources_servers.cube_standard.app import CubeSessionState

        server = mock_cube_server
        session_id = str(uuid.uuid4())

        screenshot_dir = server._screenshots_root / session_id
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        state = CubeSessionState(
            task=server._mock_task,
            task_config=server._mock_task_config,
            screenshot_dir=screenshot_dir,
        )
        server._sessions[session_id] = state

        request = MagicMock()
        request.session = {SESSION_ID_KEY: session_id}
        body = MagicMock()

        response = await server.close(request, body)

        assert response.success is True
        assert session_id not in server._sessions
        assert not screenshot_dir.exists()

    @pytest.mark.asyncio
    async def test_close_idempotent(self, mock_cube_server):
        """Closing a non-existent session should return success, not error."""
        server = mock_cube_server
        request = MagicMock()
        request.session = {SESSION_ID_KEY: "nonexistent-session-xyz"}
        body = MagicMock()

        response = await server.close(request, body)
        assert response.success is True

    @pytest.mark.asyncio
    async def test_session_limit_enforced(self, mock_cube_server):
        from fastapi import HTTPException
        from resources_servers.cube_standard.app import CubeSessionState

        server = mock_cube_server
        # Fill up sessions to the limit
        for i in range(server.config.max_concurrent_sessions):
            sid = str(uuid.uuid4())
            screenshot_dir = server._screenshots_root / sid
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            server._sessions[sid] = CubeSessionState(
                task=server._mock_task,
                task_config=server._mock_task_config,
                screenshot_dir=screenshot_dir,
            )

        request = MagicMock()
        request.session = {}

        with pytest.raises(HTTPException) as exc_info:
            await server.seed_session(request, MagicMock(task_id=None, seed=None))
        assert exc_info.value.status_code == 503

    @pytest.mark.asyncio
    @pytest.mark.skipif(not _CUBE_AVAILABLE, reason="cube-standard not installed")
    async def test_step_timeout_returns_done_response(self, mock_cube_server):
        from resources_servers.cube_standard.app import CubeSessionState

        server = mock_cube_server
        session_id = str(uuid.uuid4())

        # Make task.step() hang forever
        async def slow_step(*args, **kwargs):
            await asyncio.sleep(9999)

        task = server._mock_task
        task.step.side_effect = None

        screenshot_dir = server._screenshots_root / session_id
        screenshot_dir.mkdir(parents=True, exist_ok=True)

        state = CubeSessionState(
            task=task,
            task_config=server._mock_task_config,
            screenshot_dir=screenshot_dir,
        )
        server._sessions[session_id] = state

        request = MagicMock()
        request.session = {SESSION_ID_KEY: session_id}
        body = MagicMock()
        body.call_id = "c1"
        body.name = "click"
        body.arguments = {}

        # Patch asyncio.wait_for to raise TimeoutError
        with patch("resources_servers.cube_standard.app.asyncio.wait_for") as mock_wf:
            mock_wf.side_effect = asyncio.TimeoutError()
            with patch("cube.core.Action"):
                response = await server.step(request, body)

        assert response.done is True
        assert response.error == "timeout"
        assert response.reward == 0.0

    def test_screenshot_base_url_from_config(self, mock_cube_server):
        mock_cube_server.config.screenshot_base_url = "http://my-server:9000"
        assert mock_cube_server._get_screenshot_base_url() == "http://my-server:9000"

    def test_screenshot_base_url_strips_trailing_slash(self, mock_cube_server):
        mock_cube_server.config.screenshot_base_url = "http://my-server:9000/"
        assert mock_cube_server._get_screenshot_base_url() == "http://my-server:9000"

    def test_screenshot_base_url_fallback_to_localhost(self, mock_cube_server):
        mock_cube_server.config.screenshot_base_url = None
        url = mock_cube_server._get_screenshot_base_url()
        assert url.startswith("http://localhost:")

    def test_screenshots_root_exists(self, mock_cube_server):
        assert mock_cube_server._screenshots_root.exists()
        assert mock_cube_server._screenshots_root.is_dir()


# ---------------------------------------------------------------------------
# Integration tests — require counter-cube
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(not _COUNTER_CUBE_AVAILABLE, reason="counter-cube not installed")
class TestCubeResourcesServerIntegration:
    """
    End-to-end integration tests using the real counter-cube benchmark.

    counter-cube is the canonical CUBE reference implementation:
      - task-parallel mode
      - Text-only observations (no screenshots)
      - Actions: increment, decrement, final_step
    """

    @pytest.fixture(scope="class")
    def counter_server(self):
        """CubeResourcesServer configured for counter-cube."""
        from fastapi.testclient import TestClient

        from resources_servers.cube_standard.app import CubeResourcesServer, CubeResourcesServerConfig

        config = CubeResourcesServerConfig(
            name="test-counter-cube",
            cube_id="counter-cube",
            cube_dev_install_url="github/cube-standard/examples/counter-cube",
            task_timeout_seconds=30.0,
        )
        server = CubeResourcesServer(config=config)
        app = server.setup_webserver()
        client = TestClient(app, raise_server_exceptions=True)
        yield client, server

        # Cleanup
        shutil.rmtree(server._screenshots_root, ignore_errors=True)

    def test_full_episode(self, counter_server):
        client, server = counter_server

        # Seed session
        seed_resp = client.post("/seed_session", json={})
        assert seed_resp.status_code == 200
        seed = seed_resp.json()
        assert "obs" in seed
        assert "tools" in seed
        assert "task_id" in seed
        assert isinstance(seed["obs"], list)
        assert len(seed["obs"]) >= 1
        assert any(t["name"] == "final_step" for t in seed["tools"])

        cookies = seed_resp.cookies

        # Take a step
        step_resp = client.post(
            "/step",
            json={"call_id": "call_001", "name": "increment", "arguments": {}},
            cookies=cookies,
        )
        assert step_resp.status_code == 200
        step = step_resp.json()
        assert "output" in step
        assert "content_type" in step
        assert step["content_type"] == "text/plain"  # counter-cube is text only
        assert "done" in step
        assert "reward" in step

        # Close session
        close_resp = client.post("/close", json={}, cookies=cookies)
        assert close_resp.status_code == 200
        close = close_resp.json()
        assert close["success"] is True

    def test_tools_include_increment_and_final_step(self, counter_server):
        client, server = counter_server

        seed_resp = client.post("/seed_session", json={})
        assert seed_resp.status_code == 200
        tools = seed_resp.json()["tools"]
        tool_names = {t["name"] for t in tools}

        assert "final_step" in tool_names
        # counter-cube should expose at least one domain action
        assert len(tool_names) >= 2

        client.post("/close", json={}, cookies=seed_resp.cookies)

    def test_verify_after_final_step(self, counter_server):
        client, server = counter_server

        seed_resp = client.post("/seed_session", json={})
        cookies = seed_resp.cookies

        # Use final_step immediately to end episode
        step_resp = client.post(
            "/step",
            json={"call_id": "call_001", "name": "final_step", "arguments": {}},
            cookies=cookies,
        )
        assert step_resp.status_code == 200
        step = step_resp.json()
        assert step["done"] is True

        # Verify
        verify_resp = client.post("/verify", json={
            "responses_create_params": {"input": []},
            "response": {"output": [], "id": "r_test", "model": "test", "usage": None},
        }, cookies=cookies)
        assert verify_resp.status_code == 200
        verify = verify_resp.json()
        assert "reward" in verify
        assert isinstance(verify["reward"], float)

        client.post("/close", json={}, cookies=cookies)

    def test_screenshot_cleanup_on_close(self, counter_server):
        """Screenshot dir for the session must not exist after /close."""
        client, server = counter_server

        seed_resp = client.post("/seed_session", json={})
        cookies = seed_resp.cookies

        # Get the session_id from the server state
        session_id = None
        for sid in server._sessions:
            session_id = sid
            break

        assert session_id is not None
        session_dir = server._screenshots_root / session_id

        client.post("/close", json={}, cookies=cookies)

        assert not session_dir.exists()

    def test_multiple_concurrent_sessions(self, counter_server):
        """Two independent sessions must not interfere with each other."""
        client, server = counter_server

        seed_a = client.post("/seed_session", json={})
        seed_b = client.post("/seed_session", json={})
        assert seed_a.status_code == 200
        assert seed_b.status_code == 200

        cookies_a = seed_a.cookies
        cookies_b = seed_b.cookies

        step_a = client.post(
            "/step",
            json={"call_id": "c1", "name": "increment", "arguments": {}},
            cookies=cookies_a,
        )
        step_b = client.post(
            "/step",
            json={"call_id": "c1", "name": "increment", "arguments": {}},
            cookies=cookies_b,
        )
        assert step_a.status_code == 200
        assert step_b.status_code == 200

        client.post("/close", json={}, cookies=cookies_a)
        client.post("/close", json={}, cookies=cookies_b)
