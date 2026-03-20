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
import sys
import types
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from httpx import Cookies

from nemo_gym.server_utils import ServerClient
from resources_servers.openenv.app import (
    OpenEnvResourcesServer,
    OpenEnvResourcesServerConfig,
    _import_class,
)


def _make_test_client(server):
    """Create a TestClient with stateless cookies for session testing."""
    app = server.setup_webserver()
    client = TestClient(app)

    class StatelessCookies(Cookies):
        def extract_cookies(self, response):
            pass

    client._cookies = StatelessCookies(client._cookies)
    return client


def _make_mock_env(reset_obs=None, step_obs=None, step_side_effect=None):
    """Create a mock OpenEnv environment.

    The adapter calls sync env.reset() and env.step() directly to avoid
    extra kwargs that step_async/reset_async inject.
    """
    env = MagicMock()
    if reset_obs is None:
        reset_obs = MagicMock(reward=0.0, done=False)
    env.reset.return_value = reset_obs
    if step_side_effect is not None:
        env.step.side_effect = step_side_effect
    elif step_obs is not None:
        env.step.return_value = step_obs
    env.close = MagicMock()
    return env


def _make_server(env_class="pydantic.BaseModel", action_class="pydantic.BaseModel", is_mcp=False):
    config = OpenEnvResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        env_class=env_class,
        action_class=action_class,
        is_mcp=is_mcp,
    )
    return OpenEnvResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class TestApp:
    def test_sanity(self) -> None:
        server = _make_server()
        assert server.config.env_class == "pydantic.BaseModel"


class TestImportClass:
    def test_import_valid_class(self) -> None:
        cls = _import_class("pydantic.BaseModel")
        from pydantic import BaseModel

        assert cls is BaseModel

    def test_import_invalid_path_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Failed to import"):
            _import_class("nonexistent.module.ClassName")

    def test_import_invalid_attr_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Failed to import"):
            _import_class("pydantic.NonExistentClass")

    def test_import_empty_string_raises(self) -> None:
        with pytest.raises(RuntimeError, match="Failed to import"):
            _import_class("")


class TestSessionManagement:
    def test_seed_session_creates_env_instance(self) -> None:
        """seed_session should instantiate the env class and call reset()."""
        mock_env_instance = _make_mock_env()
        mock_env_class = MagicMock(return_value=mock_env_instance)

        server = _make_server()
        server._env_class = mock_env_class
        client = _make_test_client(server)

        response = client.post("/seed_session", json={})
        assert response.status_code == 200
        mock_env_class.assert_called_once()
        mock_env_instance.reset.assert_called_once()

    def test_seed_session_stores_session_state(self) -> None:
        """After seed_session, the session dict should contain the env."""
        mock_env_instance = _make_mock_env(reset_obs=MagicMock(reward=0.5, done=False))
        mock_env_class = MagicMock(return_value=mock_env_instance)

        server = _make_server()
        server._env_class = mock_env_class
        client = _make_test_client(server)

        response = client.post("/seed_session", json={})
        assert response.status_code == 200
        assert len(server._sessions) == 1
        session = list(server._sessions.values())[0]
        assert session.accumulated_reward == 0.5
        assert session.done is False

    def test_seed_session_with_none_reward(self) -> None:
        """seed_session should default to 0.0 reward when reset() returns None."""
        mock_env_instance = _make_mock_env(reset_obs=MagicMock(reward=None, done=False))
        mock_env_class = MagicMock(return_value=mock_env_instance)

        server = _make_server()
        server._env_class = mock_env_class
        client = _make_test_client(server)

        response = client.post("/seed_session", json={})
        assert response.status_code == 200
        session = list(server._sessions.values())[0]
        assert session.accumulated_reward == 0.0

    def test_seed_session_passes_reset_kwargs(self) -> None:
        """seed_session should forward reset_kwargs from config to env.reset()."""
        mock_env_instance = _make_mock_env()
        mock_env_class = MagicMock(return_value=mock_env_instance)

        config = OpenEnvResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
            env_class="pydantic.BaseModel",
            action_class="pydantic.BaseModel",
            reset_kwargs={"seed": 42, "difficulty": "hard"},
        )
        server = OpenEnvResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        server._env_class = mock_env_class
        client = _make_test_client(server)

        client.post("/seed_session", json={})
        mock_env_instance.reset.assert_called_once_with(seed=42, difficulty="hard")

    def test_multiple_sessions_are_independent(self) -> None:
        """Two different sessions should not share state."""
        obs1 = MagicMock(reward=0.3, done=False)
        obs1.model_dump.return_value = {"reward": 0.3}
        env1 = _make_mock_env(step_obs=obs1)

        obs2 = MagicMock(reward=0.9, done=False)
        obs2.model_dump.return_value = {"reward": 0.9}
        env2 = _make_mock_env(step_obs=obs2)

        mock_env_class = MagicMock(side_effect=[env1, env2])

        server = _make_server(is_mcp=False)
        server._env_class = mock_env_class
        server._action_class = MagicMock()
        client = _make_test_client(server)

        # Session 1
        r1 = client.post("/seed_session", json={})
        cookies1 = r1.cookies
        client.post("/step", json={}, cookies=cookies1)

        # Session 2
        r2 = client.post("/seed_session", json={})
        cookies2 = r2.cookies
        client.post("/step", json={}, cookies=cookies2)

        assert len(server._sessions) == 2
        sessions = list(server._sessions.values())
        rewards = sorted([s.accumulated_reward for s in sessions])
        assert rewards == [pytest.approx(0.3), pytest.approx(0.9)]


class TestNonMCPStepEndpoint:
    def test_step_endpoint_calls_env_step(self) -> None:
        """POST /step should call env.step() with the correct Action type."""
        mock_obs = MagicMock(reward=0.5, done=False)
        mock_obs.model_dump.return_value = {"reward": 0.5, "done": False, "message": "ok"}
        mock_env_instance = _make_mock_env(step_obs=mock_obs)
        mock_env_class = MagicMock(return_value=mock_env_instance)

        server = _make_server(is_mcp=False)
        server._env_class = mock_env_class
        server._action_class = MagicMock()
        client = _make_test_client(server)

        response = client.post("/seed_session", json={})
        cookies = response.cookies

        response = client.post("/step", json={}, cookies=cookies)
        assert response.status_code == 200
        mock_env_instance.step.assert_called_once()

    def test_step_accumulates_reward(self) -> None:
        """Multiple steps should accumulate rewards in session state."""
        rewards = [0.3, 0.5]
        call_count = [0]

        def mock_step(action):
            obs = MagicMock()
            obs.reward = rewards[call_count[0]]
            obs.done = False
            obs.model_dump.return_value = {"reward": obs.reward, "done": False}
            call_count[0] += 1
            return obs

        mock_env_instance = _make_mock_env(step_side_effect=mock_step)
        mock_env_class = MagicMock(return_value=mock_env_instance)

        server = _make_server(is_mcp=False)
        server._env_class = mock_env_class
        server._action_class = MagicMock()
        client = _make_test_client(server)

        response = client.post("/seed_session", json={})
        cookies = response.cookies

        client.post("/step", json={}, cookies=cookies)
        client.post("/step", json={}, cookies=cookies)

        session = list(server._sessions.values())[0]
        assert session.accumulated_reward == pytest.approx(0.8)
        assert session.step_count == 2

    def test_step_with_none_reward(self) -> None:
        """Step with obs.reward=None should not accumulate and not crash."""
        mock_obs = MagicMock(reward=None, done=False)
        mock_obs.model_dump.return_value = {"reward": None, "done": False}
        mock_env_instance = _make_mock_env(step_obs=mock_obs)
        mock_env_class = MagicMock(return_value=mock_env_instance)

        server = _make_server(is_mcp=False)
        server._env_class = mock_env_class
        server._action_class = MagicMock()
        client = _make_test_client(server)

        response = client.post("/seed_session", json={})
        cookies = response.cookies

        response = client.post("/step", json={}, cookies=cookies)
        assert response.status_code == 200
        session = list(server._sessions.values())[0]
        assert session.accumulated_reward == 0.0
        assert session.step_count == 1


def _make_mock_tool(name, properties, required=None):
    """Create a mock MCP tool with the given schema."""
    tool = MagicMock()
    tool.name = name
    tool.input_schema = {
        "type": "object",
        "properties": properties,
        "required": required or list(properties.keys()),
    }
    return tool


def _patch_openenv_mcp_types():
    """Patch openenv MCP types module for tests when openenv is not installed."""
    mock_mcp_types = types.ModuleType("openenv.core.env_server.mcp_types")
    mock_mcp_types.ListToolsAction = MagicMock
    mock_mcp_types.CallToolAction = MagicMock

    # Ensure parent modules exist
    for mod_name in ["openenv", "openenv.core", "openenv.core.env_server"]:
        if mod_name not in sys.modules:
            sys.modules[mod_name] = types.ModuleType(mod_name)
    sys.modules["openenv.core.env_server.mcp_types"] = mock_mcp_types
    return mock_mcp_types


class TestMCPAdapter:
    def test_mcp_tool_discovery_registers_endpoints(self) -> None:
        """MCP environments should have per-tool POST endpoints."""
        _patch_openenv_mcp_types()

        mock_tools = [
            _make_mock_tool("echo_message", {"message": {"type": "string"}}),
            _make_mock_tool("echo_with_length", {"message": {"type": "string"}}),
        ]

        mock_env_class = MagicMock()
        mock_env_instance = MagicMock()
        mock_env_instance.reset.return_value = MagicMock(reward=0.0, done=False)
        mock_list_obs = MagicMock()
        mock_list_obs.tools = mock_tools
        mock_env_instance.step.return_value = mock_list_obs
        mock_env_instance.close = MagicMock()
        mock_env_class.return_value = mock_env_instance

        server = _make_server(is_mcp=True)
        server._env_class = mock_env_class
        app = server.setup_webserver()

        routes = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/echo_message" in routes
        assert "/echo_with_length" in routes

    def test_mcp_tool_call_returns_result(self) -> None:
        """Calling an MCP tool endpoint should return the tool result."""
        _patch_openenv_mcp_types()

        mock_tools = [_make_mock_tool("echo_message", {"message": {"type": "string"}})]

        # Discovery env uses sync methods (called during setup_webserver)
        mock_discovery_env = MagicMock()
        mock_discovery_env.reset.return_value = MagicMock(reward=0.0, done=False)
        mock_list_obs = MagicMock()
        mock_list_obs.tools = mock_tools
        mock_discovery_env.step.return_value = mock_list_obs
        mock_discovery_env.close = MagicMock()

        # Session env uses async methods (called in endpoint handlers)
        mock_tool_obs = MagicMock(reward=0.5, done=False, result="echoed: hello", error=None)
        mock_session_env = _make_mock_env(step_obs=mock_tool_obs)

        mock_env_class = MagicMock(side_effect=[mock_discovery_env, mock_session_env])

        server = _make_server(is_mcp=True)
        server._env_class = mock_env_class
        server._action_class = MagicMock()
        client = _make_test_client(server)

        response = client.post("/seed_session", json={})
        cookies = response.cookies

        response = client.post("/echo_message", json={"message": "hello"}, cookies=cookies)
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "echoed: hello"
        assert data["error"] is None

    def test_mcp_tool_call_no_session_returns_error(self) -> None:
        """MCP tool call without seed_session should return an error."""
        _patch_openenv_mcp_types()

        mock_tools = [_make_mock_tool("echo_message", {"message": {"type": "string"}})]

        mock_discovery_env = MagicMock()
        mock_discovery_env.reset.return_value = MagicMock(reward=0.0, done=False)
        mock_list_obs = MagicMock()
        mock_list_obs.tools = mock_tools
        mock_discovery_env.step.return_value = mock_list_obs
        mock_discovery_env.close = MagicMock()
        mock_env_class = MagicMock(return_value=mock_discovery_env)

        server = _make_server(is_mcp=True)
        server._env_class = mock_env_class
        client = _make_test_client(server)

        response = client.post("/echo_message", json={"message": "hello"})
        data = response.json()
        assert data["error"] is not None
        assert "No active session" in data["error"]

    def test_mcp_tool_call_after_done_returns_error(self) -> None:
        """MCP tool call after done=True should return an error."""
        _patch_openenv_mcp_types()

        mock_tools = [_make_mock_tool("echo_message", {"message": {"type": "string"}})]

        mock_discovery_env = MagicMock()
        mock_discovery_env.reset.return_value = MagicMock(reward=0.0, done=False)
        mock_list_obs = MagicMock()
        mock_list_obs.tools = mock_tools
        mock_discovery_env.step.return_value = mock_list_obs
        mock_discovery_env.close = MagicMock()

        done_obs = MagicMock(reward=1.0, done=True, result="done", error=None)
        mock_session_env = _make_mock_env(step_obs=done_obs)

        mock_env_class = MagicMock(side_effect=[mock_discovery_env, mock_session_env])

        server = _make_server(is_mcp=True)
        server._env_class = mock_env_class
        server._action_class = MagicMock()
        client = _make_test_client(server)

        response = client.post("/seed_session", json={})
        cookies = response.cookies

        client.post("/echo_message", json={"message": "hello"}, cookies=cookies)

        response = client.post("/echo_message", json={"message": "hello"}, cookies=cookies)
        data = response.json()
        assert data["error"] is not None
        assert "ended" in data["error"].lower()

    def test_mcp_tool_call_exception_returns_error(self) -> None:
        """MCP tool call that raises should return a structured error."""
        _patch_openenv_mcp_types()

        mock_tools = [_make_mock_tool("echo_message", {"message": {"type": "string"}})]

        mock_discovery_env = MagicMock()
        mock_discovery_env.reset.return_value = MagicMock(reward=0.0, done=False)
        mock_list_obs = MagicMock()
        mock_list_obs.tools = mock_tools
        mock_discovery_env.step.return_value = mock_list_obs
        mock_discovery_env.close = MagicMock()

        mock_session_env = _make_mock_env(step_side_effect=RuntimeError("MCP tool failed"))

        mock_env_class = MagicMock(side_effect=[mock_discovery_env, mock_session_env])

        server = _make_server(is_mcp=True)
        server._env_class = mock_env_class
        server._action_class = MagicMock()
        client = _make_test_client(server)

        response = client.post("/seed_session", json={})
        cookies = response.cookies

        response = client.post("/echo_message", json={"message": "hello"}, cookies=cookies)
        data = response.json()
        assert data["error"] is not None
        assert "MCP tool failed" in data["error"]


class TestVerify:
    def test_verify_returns_accumulated_reward(self) -> None:
        """verify() should return the sum of per-step rewards."""
        call_count = [0]

        def mock_step(action):
            obs = MagicMock()
            obs.reward = 0.5
            obs.done = call_count[0] == 1  # done on second call
            obs.model_dump.return_value = {"reward": 0.5, "done": obs.done}
            call_count[0] += 1
            return obs

        mock_env_instance = _make_mock_env(step_side_effect=mock_step)
        mock_env_class = MagicMock(return_value=mock_env_instance)

        server = _make_server(is_mcp=False)
        server._env_class = mock_env_class
        server._action_class = MagicMock()
        client = _make_test_client(server)

        # seed
        response = client.post("/seed_session", json={})
        cookies = response.cookies

        # two steps
        client.post("/step", json={}, cookies=cookies)
        client.post("/step", json={}, cookies=cookies)

        # verify
        verify_body = {
            "responses_create_params": {"input": "test"},
            "response": {
                "id": "test",
                "output": [],
                "created_at": 0,
                "model": "test",
                "object": "response",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }
        response = client.post("/verify", json=verify_body, cookies=cookies)
        assert response.status_code == 200
        data = response.json()
        assert data["reward"] == pytest.approx(1.0)

    def test_verify_cleans_up_session(self) -> None:
        """verify() should remove the session and close the env."""
        mock_env_instance = _make_mock_env()
        mock_env_class = MagicMock(return_value=mock_env_instance)

        server = _make_server(is_mcp=False)
        server._env_class = mock_env_class
        client = _make_test_client(server)

        response = client.post("/seed_session", json={})
        cookies = response.cookies
        assert len(server._sessions) == 1

        verify_body = {
            "responses_create_params": {"input": "test"},
            "response": {
                "id": "test",
                "output": [],
                "created_at": 0,
                "model": "test",
                "object": "response",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }
        client.post("/verify", json=verify_body, cookies=cookies)
        assert len(server._sessions) == 0
        mock_env_instance.close.assert_called_once()

    def test_verify_without_session_returns_zero_reward(self) -> None:
        """verify() without a prior seed_session should return reward=0.0."""
        server = _make_server(is_mcp=False)
        client = _make_test_client(server)

        verify_body = {
            "responses_create_params": {"input": "test"},
            "response": {
                "id": "test",
                "output": [],
                "created_at": 0,
                "model": "test",
                "object": "response",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }
        response = client.post("/verify", json=verify_body)
        assert response.status_code == 200
        data = response.json()
        assert data["reward"] == 0.0


class TestSchemaConversion:
    def test_schema_to_pydantic_required_fields(self) -> None:
        """Required fields should not accept None."""
        server = _make_server()
        model = server._schema_to_pydantic(
            "test_tool",
            {
                "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
                "required": ["name", "count"],
            },
        )
        instance = model(name="hello", count=5)
        assert instance.name == "hello"
        assert instance.count == 5

    def test_schema_to_pydantic_optional_fields(self) -> None:
        """Optional fields should default to None."""
        server = _make_server()
        model = server._schema_to_pydantic(
            "test_tool",
            {
                "properties": {"name": {"type": "string"}, "label": {"type": "string"}},
                "required": ["name"],
            },
        )
        instance = model(name="hello")
        assert instance.name == "hello"
        assert instance.label is None

    def test_json_type_to_python_all_types(self) -> None:
        """All JSON schema types should map to correct Python types."""
        assert OpenEnvResourcesServer._json_type_to_python("string") is str
        assert OpenEnvResourcesServer._json_type_to_python("integer") is int
        assert OpenEnvResourcesServer._json_type_to_python("number") is float
        assert OpenEnvResourcesServer._json_type_to_python("boolean") is bool
        assert OpenEnvResourcesServer._json_type_to_python("object") is dict
        assert OpenEnvResourcesServer._json_type_to_python("array") is list

    def test_json_type_to_python_unknown_defaults_to_str(self) -> None:
        """Unknown JSON types should default to str."""
        assert OpenEnvResourcesServer._json_type_to_python("unknown") is str
        assert OpenEnvResourcesServer._json_type_to_python("") is str


class TestErrorHandling:
    def test_step_after_done_returns_error(self) -> None:
        """Tool calls after done=True should return an error, not call env.step()."""
        obs_done = MagicMock(reward=1.0, done=True)
        obs_done.model_dump.return_value = {"reward": 1.0, "done": True}
        mock_env_instance = _make_mock_env(step_obs=obs_done)
        mock_env_class = MagicMock(return_value=mock_env_instance)

        server = _make_server(is_mcp=False)
        server._env_class = mock_env_class
        server._action_class = MagicMock()
        client = _make_test_client(server)

        response = client.post("/seed_session", json={})
        cookies = response.cookies

        # First step sets done=True
        client.post("/step", json={}, cookies=cookies)
        assert mock_env_instance.step.call_count == 1

        # Second step should NOT call env.step
        response = client.post("/step", json={}, cookies=cookies)
        data = response.json()
        assert "error" in data
        assert "ended" in data["error"].lower()
        assert mock_env_instance.step.call_count == 1  # still 1

    def test_step_exception_returns_error(self) -> None:
        """If env.step() raises, the endpoint should catch and return error."""
        mock_env_instance = _make_mock_env(step_side_effect=ValueError("Invalid move: xyz"))
        mock_env_class = MagicMock(return_value=mock_env_instance)

        server = _make_server(is_mcp=False)
        server._env_class = mock_env_class
        server._action_class = MagicMock()
        client = _make_test_client(server)

        response = client.post("/seed_session", json={})
        cookies = response.cookies

        response = client.post("/step", json={}, cookies=cookies)
        data = response.json()
        assert "error" in data
        assert "Invalid move" in data["error"]

    def test_step_no_session_returns_error(self) -> None:
        """Calling /step without seed_session should return an error."""
        server = _make_server(is_mcp=False)
        client = _make_test_client(server)

        response = client.post("/step", json={})
        data = response.json()
        assert "error" in data
        assert "No active session" in data["error"]


class TestIntegrationMCPLifecycle:
    """Full lifecycle test: seed_session -> MCP tool calls -> verify."""

    def test_full_lifecycle_mcp(self) -> None:
        """seed_session -> tool call -> tool call -> verify with mocked MCP env."""
        _patch_openenv_mcp_types()

        mock_tools = [
            _make_mock_tool("echo_message", {"message": {"type": "string"}}),
            _make_mock_tool("echo_with_length", {"message": {"type": "string"}}),
        ]

        # Discovery env uses sync methods (called during setup_webserver)
        mock_discovery_env = MagicMock()
        mock_discovery_env.reset.return_value = MagicMock(reward=0.0, done=False)
        mock_list_obs = MagicMock()
        mock_list_obs.tools = mock_tools
        mock_discovery_env.step.return_value = mock_list_obs
        mock_discovery_env.close = MagicMock()

        # Session env uses async methods (called in endpoint handlers)
        call_count = [0]

        def mock_step(action):
            obs = MagicMock()
            obs.reward = [0.3, 0.7][call_count[0]]
            obs.done = call_count[0] == 1
            obs.result = f"result_{call_count[0]}"
            obs.error = None
            call_count[0] += 1
            return obs

        mock_session_env = _make_mock_env(step_side_effect=mock_step)

        mock_env_class = MagicMock(side_effect=[mock_discovery_env, mock_session_env])

        server = _make_server(is_mcp=True)
        server._env_class = mock_env_class
        server._action_class = MagicMock()
        client = _make_test_client(server)

        # 1. seed_session
        response = client.post("/seed_session", json={})
        assert response.status_code == 200
        cookies = response.cookies

        # 2. first tool call
        response = client.post("/echo_message", json={"message": "hello"}, cookies=cookies)
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "result_0"
        assert data["error"] is None

        # 3. second tool call
        response = client.post("/echo_with_length", json={"message": "test"}, cookies=cookies)
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "result_1"

        # 4. verify
        verify_body = {
            "responses_create_params": {"input": "test"},
            "response": {
                "id": "test",
                "output": [],
                "created_at": 0,
                "model": "test",
                "object": "response",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }
        response = client.post("/verify", json=verify_body, cookies=cookies)
        assert response.status_code == 200
        data = response.json()
        assert data["reward"] == pytest.approx(1.0)  # 0.3 + 0.7
        assert len(server._sessions) == 0
        mock_session_env.close.assert_called_once()


class TestIntegrationNonMCPLifecycle:
    """Full lifecycle test: seed_session -> /step calls -> verify."""

    def test_full_lifecycle_non_mcp(self) -> None:
        """seed_session -> step -> step -> verify with mocked non-MCP env."""
        call_count = [0]

        def mock_step(action):
            obs = MagicMock()
            obs.reward = [0.4, 0.6][call_count[0]]
            obs.done = call_count[0] == 1
            obs.model_dump.return_value = {"reward": obs.reward, "done": obs.done}
            call_count[0] += 1
            return obs

        mock_env_instance = _make_mock_env(step_side_effect=mock_step)
        mock_env_class = MagicMock(return_value=mock_env_instance)

        server = _make_server(is_mcp=False)
        server._env_class = mock_env_class
        server._action_class = MagicMock()
        client = _make_test_client(server)

        # 1. seed
        response = client.post("/seed_session", json={})
        assert response.status_code == 200
        cookies = response.cookies

        # 2. step 1
        response = client.post("/step", json={"move": "e2e4"}, cookies=cookies)
        assert response.status_code == 200

        # 3. step 2
        response = client.post("/step", json={"move": "d2d4"}, cookies=cookies)
        assert response.status_code == 200

        # 4. step after done should error
        response = client.post("/step", json={"move": "c2c4"}, cookies=cookies)
        data = response.json()
        assert "error" in data
        assert "ended" in data["error"].lower()

        # 5. verify
        verify_body = {
            "responses_create_params": {"input": "test"},
            "response": {
                "id": "test",
                "output": [],
                "created_at": 0,
                "model": "test",
                "object": "response",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
        }
        response = client.post("/verify", json=verify_body, cookies=cookies)
        assert response.status_code == 200
        data = response.json()
        assert data["reward"] == pytest.approx(1.0)  # 0.4 + 0.6
        assert len(server._sessions) == 0
        mock_env_instance.close.assert_called_once()
