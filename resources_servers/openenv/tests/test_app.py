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
from http.cookiejar import CookieJar
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import (
    ResourcesCloseSessionRequest,
    ResourcesSeedSessionRequest,
    check_resources_session_contract,
    maybe_auto_expose,
)
from resources_servers.openenv.app import (
    OpenEnvResourcesServer,
    OpenEnvResourcesServerConfig,
    _import_class,
)


def _make_test_client(server):
    """Create a TestClient with stateless cookies for session testing."""
    app = server.setup_webserver()

    class StatelessCookieJar(CookieJar):
        def extract_cookies(self, response: object, request: object) -> None:
            # Sessions must come from explicit request cookies, not earlier responses.
            pass

    client = TestClient(app, cookies=StatelessCookieJar())
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

    def test_mcp_tool_call_no_session_is_lost(self) -> None:
        """MCP tool call without seed_session should answer 404 with the session_lost failure kind."""
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
        assert response.status_code == 404
        assert response.json()["detail"]["failure_kind"] == "session_lost"
        assert "No active OpenEnv session" in response.json()["detail"]["message"]

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

    def test_a_reused_cookie_scores_and_releases_its_next_run(self) -> None:
        """A /run that reuses a verified cookie is scored from its own steps, not the finished run's reward."""
        rewards = iter([1.0, 0.0])

        def mock_step(action):
            obs = MagicMock()
            obs.reward = next(rewards)
            obs.done = False
            obs.model_dump.return_value = {"reward": obs.reward, "done": False}
            return obs

        server = _make_server(is_mcp=False)
        server._env_class = MagicMock(return_value=_make_mock_env(step_side_effect=mock_step))
        server._action_class = MagicMock()
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
        cookies = client.post("/seed_session", json={}).cookies
        client.post("/step", json={}, cookies=cookies)
        assert client.post("/verify", json=verify_body, cookies=cookies).json()["reward"] == pytest.approx(1.0)

        # The next /run on the same cookie keeps its session ID and replaces the finished run.
        assert client.post("/seed_session", json={}, cookies=cookies).status_code == 200
        assert len(server._sessions) == 1
        client.post("/step", json={}, cookies=cookies)
        assert client.post("/verify", json=verify_body, cookies=cookies).json()["reward"] == pytest.approx(0.0)
        assert server._sessions == {}

    def test_verify_without_session_is_masked(self) -> None:
        """verify() without a live session has no reward to report, so the sample is masked."""
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
        assert data["mask_sample"] is True
        assert data["failure_kind"] == "session_lost"


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

    def test_step_no_session_is_lost(self) -> None:
        """Calling /step without seed_session should answer 404 with the session_lost failure kind."""
        server = _make_server(is_mcp=False)
        client = _make_test_client(server)

        response = client.post("/step", json={})
        assert response.status_code == 404
        assert response.json()["detail"]["failure_kind"] == "session_lost"


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


VERIFY_BODY = {
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


def _typed_seed(resources_session_id: str = "resources-session") -> ResourcesSeedSessionRequest:
    # The rows carry no task fields, so an Environment Server's task_data is empty.
    return ResourcesSeedSessionRequest(
        resources_session_id=resources_session_id,
        episode_id=EpisodeId(rollout_id="rollout", attempt=0),
        task_id=TaskId(taskset="openenv", task_id="0"),
        task_data={},
    )


def _typed_close(resources_session_id: str = "resources-session") -> dict:
    return ResourcesCloseSessionRequest(
        resources_session_id=resources_session_id, episode_id=EpisodeId(rollout_id="rollout", attempt=0)
    ).model_dump(mode="json")


def _session_server(**kwargs) -> tuple[OpenEnvResourcesServer, list[MagicMock]]:
    """A non-MCP server whose env class makes a fresh mock env per seed; returns the envs it made."""
    envs: list[MagicMock] = []

    def make_env():
        envs.append(_make_mock_env(**kwargs))
        return envs[-1]

    server = _make_server(is_mcp=False)
    server._env_class = MagicMock(side_effect=make_env)
    server._action_class = MagicMock()
    return server, envs


class TestResourcesSessions:
    def test_resources_session_contract(self) -> None:
        server, envs = _session_server()

        check_resources_session_contract(
            server.setup_webserver(),
            _typed_seed("contract-session"),
            keeps_state=True,
            holds_state=lambda resources_session_id: resources_session_id in server._sessions,
            verify_body=VERIFY_BODY,
        )
        # The seeded session and the one whose seed reply was lost; the repeated seed reused the first env.
        assert len(envs) == 2
        for env in envs:
            env.close.assert_called_once()

    def test_typed_seed_binds_cookie_and_repeated_seed_keeps_env(self) -> None:
        step_obs = MagicMock(reward=0.5, done=False)
        step_obs.model_dump.return_value = {"reward": 0.5, "done": False}
        server, envs = _session_server(step_obs=step_obs)

        with TestClient(server.setup_webserver()) as client:
            assert client.post("/seed_session", json=_typed_seed().model_dump(mode="json")).status_code == 200
            assert client.post("/step", json={}).status_code == 200
            # A seed retried after its reply was lost arrives without the cookie.
            client.cookies.clear()
            assert client.post("/seed_session", json=_typed_seed().model_dump(mode="json")).status_code == 200
            response = client.post("/verify", json=VERIFY_BODY)

        assert len(envs) == 1
        assert response.json()["reward"] == pytest.approx(0.5)
        assert response.json()["mask_sample"] is False

    def test_close_after_verify_succeeds_without_closing_twice(self) -> None:
        server, envs = _session_server()

        with TestClient(server.setup_webserver()) as client:
            client.post("/seed_session", json=_typed_seed().model_dump(mode="json"))
            assert client.post("/verify", json=VERIFY_BODY).status_code == 200
            response = client.post("/close_session", json=_typed_close())

        assert response.status_code == 200
        envs[0].close.assert_called_once()
        assert server._sessions == {}
        assert server._session_episodes == {}

    def test_typed_verify_retry_preserves_reward_until_close(self) -> None:
        server, envs = _session_server(reset_obs=MagicMock(reward=0.5, done=False))

        with TestClient(server.setup_webserver()) as client:
            assert client.post("/seed_session", json=_typed_seed().model_dump(mode="json")).status_code == 200
            for _ in range(2):
                response = client.post("/verify", json=VERIFY_BODY)
                assert response.status_code == 200
                assert response.json()["reward"] == pytest.approx(0.5)
                assert response.json()["mask_sample"] is False
            envs[0].close.assert_called_once()
            assert client.post("/close_session", json=_typed_close()).status_code == 200

        envs[0].close.assert_called_once()
        assert server._sessions == {}

    def test_step_after_typed_verify_cannot_change_a_retried_verify(self) -> None:
        step_obs = MagicMock(reward=1.0, done=False)
        step_obs.model_dump.return_value = {"reward": 1.0, "done": False}
        server, envs = _session_server(reset_obs=MagicMock(reward=0.5, done=False), step_obs=step_obs)

        with TestClient(server.setup_webserver()) as client:
            assert client.post("/seed_session", json=_typed_seed().model_dump(mode="json")).status_code == 200
            assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == pytest.approx(0.5)
            # A tool call still in flight when the Environment Server verified.
            late = client.post("/step", json={})
            retried = client.post("/verify", json=VERIFY_BODY)

        assert late.status_code == 404
        assert late.json()["detail"]["failure_kind"] == "session_lost"
        envs[0].step.assert_not_called()
        assert retried.json()["reward"] == pytest.approx(0.5)

    def test_failed_verify_release_preserves_reward_until_owning_close_retries(self) -> None:
        step_obs = MagicMock(reward=1.0, done=False)
        step_obs.model_dump.return_value = {"reward": 1.0, "done": False}
        server, envs = _session_server(reset_obs=MagicMock(reward=0.5, done=False), step_obs=step_obs)
        with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
            assert client.post("/seed_session", json=_typed_seed().model_dump(mode="json")).status_code == 200
            envs[0].close.side_effect = [RuntimeError("env would not stop"), RuntimeError("still unavailable"), None]
            verified = client.post("/verify", json=VERIFY_BODY)
            assert verified.status_code == 200
            assert verified.json()["reward"] == pytest.approx(0.5)
            assert verified.json()["mask_sample"] is False
            assert "resources-session" in server._sessions
            assert client.post("/seed_session", json=_typed_seed().model_dump(mode="json")).status_code == 409
            assert client.post("/step", json={}).status_code == 404

            wrong_close = _typed_close() | {"episode_id": {"rollout_id": "another", "attempt": 0}}
            assert client.post("/close_session", json=wrong_close).status_code == 409
            assert client.post("/close_session", json=_typed_close()).status_code == 500
            assert "resources-session" in server._sessions
            assert client.post("/close_session", json=_typed_close()).status_code == 200
        assert server._sessions == {}
        assert envs[0].close.call_count == 3
        envs[0].step.assert_not_called()

    def test_failed_env_close_raises_and_a_retried_close_releases(self) -> None:
        server, envs = _session_server()

        with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
            client.post("/seed_session", json=_typed_seed().model_dump(mode="json"))
            envs[0].close.side_effect = RuntimeError("env would not stop")
            response = client.post("/close_session", json=_typed_close())
            assert response.status_code == 500
            # Not recorded as closed: the env is still held and a later seed is not fenced.
            assert "resources-session" in server._sessions
            assert client.post("/seed_session", json=_typed_seed().model_dump(mode="json")).status_code == 200
            assert len(envs) == 1

            envs[0].close.side_effect = None
            assert client.post("/close_session", json=_typed_close()).status_code == 200

        assert server._sessions == {}
        assert envs[0].close.call_count == 2

    def test_tool_calls_after_close_are_lost(self) -> None:
        server, envs = _session_server()

        with TestClient(server.setup_webserver()) as client:
            client.post("/seed_session", json=_typed_seed().model_dump(mode="json"))
            client.post("/close_session", json=_typed_close())
            response = client.post("/step", json={})

        assert response.status_code == 404
        assert response.json()["detail"]["failure_kind"] == "session_lost"
        envs[0].step.assert_not_called()

    def test_failed_reset_closes_the_env(self) -> None:
        server, envs = _session_server()

        def failing_env():
            env = _make_mock_env()
            env.reset.side_effect = RuntimeError("reset failed")
            envs.append(env)
            return env

        server._env_class = MagicMock(side_effect=failing_env)

        with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
            response = client.post("/seed_session", json=_typed_seed().model_dump(mode="json"))

        assert response.status_code == 500
        envs[0].close.assert_called_once()
        assert server._sessions == {}
        assert server._session_episodes == {}

    def test_failed_reset_and_cleanup_remains_owned_until_close(self) -> None:
        server, envs = _session_server()
        env = _make_mock_env()
        env.reset.side_effect = RuntimeError("reset failed")
        env.close.side_effect = [RuntimeError("cleanup failed"), None]
        server._env_class = MagicMock(return_value=env)

        with TestClient(server.setup_webserver()) as client:
            with pytest.raises(RuntimeError, match="reset failed"):
                client.post("/seed_session", json=_typed_seed().model_dump(mode="json"))
            assert server._sessions["resources-session"].env is env
            assert not server.has_session_state("resources-session")
            wrong_close = _typed_close() | {"episode_id": {"rollout_id": "other", "attempt": 0}}
            assert client.post("/close_session", json=wrong_close).status_code == 409
            assert client.post("/close_session", json=_typed_close()).status_code == 200
            assert client.post("/close_session", json=_typed_close()).status_code == 200

        assert env.close.call_count == 2
        server._env_class.assert_called_once()

    def test_seed_retry_releases_failed_reset_before_allocating_again(self) -> None:
        server, envs = _session_server()
        failed_env = _make_mock_env()
        failed_env.reset.side_effect = RuntimeError("reset failed")
        failed_env.close.side_effect = [RuntimeError("cleanup failed"), RuntimeError("still unavailable"), None]
        replacement_env = _make_mock_env()
        server._env_class = MagicMock(side_effect=[failed_env, replacement_env])

        with TestClient(server.setup_webserver()) as client:
            with pytest.raises(RuntimeError, match="reset failed"):
                client.post("/seed_session", json=_typed_seed().model_dump(mode="json"))
            with pytest.raises(RuntimeError, match="still unavailable"):
                client.post("/seed_session", json=_typed_seed().model_dump(mode="json"))
            server._env_class.assert_called_once()
            assert client.post("/seed_session", json=_typed_seed().model_dump(mode="json")).status_code == 200
            assert server._sessions["resources-session"].env is replacement_env
            assert client.post("/close_session", json=_typed_close()).status_code == 200

        assert failed_env.close.call_count == 3
        replacement_env.close.assert_called_once()


@pytest.mark.parametrize("stage", ["active", "verified", "closed"])
def test_cookie_seed_cannot_overwrite_typed_or_closed_cookie(stage: str) -> None:
    server = _session_server(reset_obs=MagicMock(reward=0.5, done=False))[0]
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json=_typed_seed().model_dump(mode="json")).status_code == 200
        if stage == "verified":
            assert client.post("/verify", json=VERIFY_BODY).status_code == 200
        elif stage == "closed":
            assert client.post("/close_session", json=_typed_close()).status_code == 200
        before = dict(server._sessions)
        assert client.post("/seed_session", json={}).status_code == 409
        assert server._sessions == before


@pytest.mark.parametrize("expose_tools_over_mcp", [False, True])
@pytest.mark.parametrize(
    "malformed",
    [
        {"resources_session_id": "resources-session"},
        _typed_seed().model_dump(mode="json", exclude={"resources_session_id"}),
    ],
    ids=["session_id_without_envelope", "envelope_without_session_id"],
)
def test_malformed_typed_seed_does_not_fall_back_to_cookie_seed(malformed: dict, expose_tools_over_mcp: bool) -> None:
    server = _session_server(reset_obs=MagicMock(reward=0.5, done=False))[0]
    server.config.expose_tools_over_mcp = expose_tools_over_mcp
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        assert client.post("/seed_session", json=malformed).status_code == 422
        assert server._sessions == {}
        assert server._session_episodes == {}

        # The refused body did not seed a cookie session, so the complete typed seed of that session succeeds.
        response = client.post("/seed_session", json=_typed_seed().model_dump(mode="json"))
        assert response.status_code == 200
        assert (response.json().get("resources_tools") is not None) is expose_tools_over_mcp
    assert list(server._sessions) == ["resources-session"]


@pytest.mark.parametrize("typed", [False, True])
def test_checkpoint_verify_releases_state_but_retry_survives_until_expiry(typed: bool, monkeypatch) -> None:
    from omegaconf import DictConfig

    clock = [100.0]
    monkeypatch.setattr("nemo_gym.server_runtime.base_resources_server.monotonic", lambda: clock[0])
    server = _session_server(reset_obs=MagicMock(reward=0.5, done=False))[0]
    server.server_client.global_config_dict = DictConfig(
        {"checkpoint": {"enabled": True, "control_auth_token": "test"}}
    )
    prefix = "/ng-rollout/rollout"
    with TestClient(server.setup_webserver()) as client:
        assert (
            client.post(
                prefix + "/seed_session", json=_typed_seed().model_dump(mode="json") if typed else {}
            ).status_code
            == 200
        )
        sid = next(iter(server._sessions))
        assert server._checkpoint.is_open(sid)
        env = server._sessions[sid].env
        first = client.post(prefix + "/verify", json=VERIFY_BODY).json()
        assert not first["mask_sample"]
        assert not server._checkpoint.is_open(sid)
        assert server._sessions == {}
        assert server._session_episodes == {}
        env.close.assert_called_once()
        retry = client.post(prefix + "/verify", json=VERIFY_BODY).json()
        assert retry["reward"] == first["reward"] and retry["mask_sample"] is False
        if typed:
            wrong_owner = _typed_close()
            wrong_owner["episode_id"]["rollout_id"] = "other"
            assert client.post(prefix + "/close_session", json=wrong_owner).status_code == 409
        clock[0] += server.config.session_close_retry_window_seconds + 1
        assert client.post(prefix + "/verify", json=VERIFY_BODY).json()["mask_sample"] is True
        assert not server.session_is_verified(sid)


def test_next_cookie_run_keeps_its_cookie_and_starts_a_new_environment() -> None:
    server = _session_server(reset_obs=MagicMock(reward=0.5, done=False))[0]
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json={}).status_code == 200
        session_id = next(iter(server._sessions))
        assert client.post("/verify", json=VERIFY_BODY).json()["mask_sample"] is False
        assert server._sessions == {}

        # The next /run on the same cookie replaces the finished run's environment under the same ID.
        assert client.post("/seed_session", json={}).status_code == 200
        assert list(server._sessions) == [session_id]
        assert client.post("/verify", json=VERIFY_BODY).json()["mask_sample"] is False


@pytest.mark.parametrize("typed", [False, True])
def test_checkpoint_retire_releases_live_session(typed: bool) -> None:
    from omegaconf import DictConfig

    server = _session_server()[0]
    server.server_client.global_config_dict = DictConfig(
        {"checkpoint": {"enabled": True, "control_auth_token": "test"}}
    )
    with TestClient(server.setup_webserver()) as client:
        assert (
            client.post(
                "/ng-rollout/rollout/seed_session", json=_typed_seed().model_dump(mode="json") if typed else {}
            ).status_code
            == 200
        )
        sid = next(iter(server._sessions))
        handle = server._sessions[sid].env
        client.portal.call(server._checkpoint.retire, EpisodeId(rollout_id="rollout", attempt=0))
        assert server._sessions == {}
        assert server._session_episodes == {}
        handle.close.assert_called_once()


def test_checkpoint_retire_retries_failed_environment_close() -> None:
    from omegaconf import DictConfig

    server, envs = _session_server()
    server.server_client.global_config_dict = DictConfig(
        {"checkpoint": {"enabled": True, "control_auth_token": "test"}}
    )
    with TestClient(server.setup_webserver()) as client:
        client.post("/ng-rollout/rollout/seed_session", json=_typed_seed().model_dump(mode="json"))
        envs[0].close.side_effect = [RuntimeError("close failed"), None]
        with pytest.raises(RuntimeError, match="close failed"):
            client.portal.call(server._checkpoint.retire, EpisodeId(rollout_id="rollout", attempt=0))
        assert "resources-session" in server._sessions
        assert "resources-session" in server._session_episodes
        assert not server.session_is_closed("resources-session")
        client.portal.call(server._checkpoint.retire, EpisodeId(rollout_id="rollout", attempt=0))
        assert server._sessions == {} and server._session_episodes == {}
        assert envs[0].close.call_count == 2


def test_typed_seed_cannot_replace_a_live_cookie_session() -> None:
    server = _session_server()[0]
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json={}).status_code == 200
        sid = next(iter(server._sessions))
        original = server._sessions[sid]
        assert client.post("/seed_session", json=_typed_seed(sid).model_dump(mode="json")).status_code == 409
        assert server._sessions[sid] is original
        assert server._session_episodes == {}
        original.env.close.assert_not_called()


def test_failed_cookie_reseed_keeps_previous_reward_until_initialization_succeeds() -> None:
    server, envs = _session_server()
    with TestClient(server.setup_webserver()) as client:
        assert client.post("/seed_session", json={}).status_code == 200
        [session_id] = server._sessions
        server._sessions[session_id].accumulated_reward = 2.0
        assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 2.0
        assert server.session_is_verified(session_id)
        failed_env = _make_mock_env()
        failed_env.reset.side_effect = RuntimeError("reset unavailable")
        replacement = _make_mock_env()
        server._env_class = MagicMock(side_effect=[failed_env, replacement])

        with pytest.raises(RuntimeError, match="reset unavailable"):
            client.post("/seed_session", json={})
        assert server.session_is_verified(session_id)
        assert server._sessions == {}
        assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 2.0
        failed_env.close.assert_called_once()

        assert client.post("/seed_session", json={}).status_code == 200
        assert list(server._sessions) == [session_id]
        assert not server.session_is_verified(session_id)
        assert client.post("/verify", json=VERIFY_BODY).json()["reward"] == 0.0
        assert server._sessions == {}
        replacement.close.assert_called_once()
        envs[0].close.assert_called_once()


@pytest.mark.parametrize("typed", [False, True])
def test_verify_retry_returns_a_retained_zero_reward(typed: bool) -> None:
    """A reward of 0.0 is evidence, not a missing receipt: the retry is graded, not masked."""
    server, envs = _session_server()  # reset reward 0.0, no steps taken
    with TestClient(server.setup_webserver()) as client:
        body = _typed_seed().model_dump(mode="json") if typed else {}
        assert client.post("/seed_session", json=body).status_code == 200
        first = client.post("/verify", json=VERIFY_BODY).json()
        retry = client.post("/verify", json=VERIFY_BODY).json()

    assert first["reward"] == 0.0 and first["mask_sample"] is False
    assert retry["reward"] == 0.0 and retry["mask_sample"] is False
    envs[0].close.assert_called_once()


def test_partly_reset_cookie_reseed_cannot_serve_tools_or_supply_a_reward() -> None:
    """Reset and cleanup both fail on a cookie reseed: the handle stays owned, but it serves nothing."""
    server, envs = _session_server(reset_obs=MagicMock(reward=0.5, done=False))
    with TestClient(server.setup_webserver(), raise_server_exceptions=False) as client:
        # The first run is still live (never verified), so no finished-run receipt can mask the check under test.
        assert client.post("/seed_session", json={}).status_code == 200

        broken = _make_mock_env()
        broken.reset.side_effect = RuntimeError("reset failed")
        broken.close.side_effect = [RuntimeError("cleanup failed"), None]
        server._env_class = MagicMock(return_value=broken)
        server._action_class = MagicMock()

        assert client.post("/seed_session", json={}).status_code == 500
        [session_id] = server._sessions
        assert server._sessions[session_id].env is broken  # still owned for a retry

        assert client.post("/step", json={}).status_code == 404
        broken.step.assert_not_called()
        assert client.post("/verify", json=VERIFY_BODY).json()["mask_sample"] is True
