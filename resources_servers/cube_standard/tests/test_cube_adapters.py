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
Unit tests for format conversion functions in cube_adapters.py.

These tests use minimal mock objects so that cube-standard does not need
to be installed in the test environment. Integration tests that require
real CUBE packages are in test_resources_server.py.
"""

import base64
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers / minimal mocks
# ---------------------------------------------------------------------------


def _make_action_schema(name: str, description: str = "", parameters: dict | None = None) -> MagicMock:
    """Create a minimal ActionSchema mock whose .as_dict() matches CUBE format."""
    schema = MagicMock()
    schema.as_dict.return_value = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters or {"type": "object", "properties": {}},
        },
    }
    return schema


def _make_env_output(
    obs_messages: list[dict],
    done: bool = False,
    reward: float = 0.0,
    error=None,
    info: dict | None = None,
) -> MagicMock:
    """Create a minimal EnvironmentOutput mock."""
    obs = MagicMock()
    obs.to_llm_messages.return_value = obs_messages
    env_out = MagicMock()
    env_out.obs = obs
    env_out.done = done
    env_out.reward = reward
    env_out.error = error
    env_out.info = info or {}
    return env_out


# ---------------------------------------------------------------------------
# Tests: action_schema_to_function_tool_param
# ---------------------------------------------------------------------------


class TestActionSchemaToFunctionToolParam:
    def test_basic_conversion(self):
        from resources_servers.cube_standard.cube_adapters import action_schema_to_function_tool_param

        schema = _make_action_schema(
            name="click",
            description="Click at coordinates",
            parameters={"type": "object", "properties": {"x": {"type": "integer"}}},
        )
        result = action_schema_to_function_tool_param(schema)

        assert result["type"] == "function"
        assert result["name"] == "click"
        assert result["description"] == "Click at coordinates"
        assert "x" in result["parameters"]["properties"]

    def test_nested_function_key_is_unwrapped(self):
        """FunctionToolParam must be flat — 'function' sub-key must NOT appear."""
        from resources_servers.cube_standard.cube_adapters import action_schema_to_function_tool_param

        schema = _make_action_schema(name="click")
        result = action_schema_to_function_tool_param(schema)

        assert "function" not in result, "FunctionToolParam must be flat, not nested"

    def test_missing_description_defaults_to_empty_string(self):
        from resources_servers.cube_standard.cube_adapters import action_schema_to_function_tool_param

        schema = MagicMock()
        schema.as_dict.return_value = {
            "type": "function",
            "function": {"name": "noop", "parameters": {}},
        }
        result = action_schema_to_function_tool_param(schema)
        assert result["description"] == ""

    def test_stop_action_included(self):
        """STOP_ACTION (final_step) must convert correctly."""
        from resources_servers.cube_standard.cube_adapters import action_schema_to_function_tool_param

        schema = _make_action_schema(
            name="final_step",
            description="Signal that the task is complete.",
        )
        result = action_schema_to_function_tool_param(schema)
        assert result["name"] == "final_step"
        assert result["type"] == "function"


class TestActionSetToFunctionToolParams:
    def test_converts_all_schemas(self):
        from resources_servers.cube_standard.cube_adapters import _action_set_to_function_tool_params

        schemas = [_make_action_schema(f"action_{i}") for i in range(3)]
        results = _action_set_to_function_tool_params(schemas)

        assert len(results) == 3
        assert [r["name"] for r in results] == ["action_0", "action_1", "action_2"]

    def test_empty_action_set(self):
        from resources_servers.cube_standard.cube_adapters import _action_set_to_function_tool_params

        assert _action_set_to_function_tool_params([]) == []


# ---------------------------------------------------------------------------
# Tests: _observation_to_nemo_gym_messages
# ---------------------------------------------------------------------------


class TestObservationToNemoGymMessages:
    def test_string_content_role_user(self):
        from resources_servers.cube_standard.cube_adapters import _observation_to_nemo_gym_messages

        obs = MagicMock()
        obs.to_llm_messages.return_value = [{"role": "user", "content": "Hello"}]
        result = _observation_to_nemo_gym_messages(obs)

        assert len(result) == 1
        assert result[0].role == "user"
        assert result[0].content == "Hello"

    def test_tool_role_remapped_to_user(self):
        """CUBE uses role 'tool' for function call outputs. Must be remapped to 'user'."""
        from resources_servers.cube_standard.cube_adapters import _observation_to_nemo_gym_messages

        obs = MagicMock()
        obs.to_llm_messages.return_value = [
            {"role": "tool", "content": "result", "tool_call_id": "call_abc"}
        ]
        result = _observation_to_nemo_gym_messages(obs)

        assert len(result) == 1
        assert result[0].role == "user"

    def test_unknown_role_remapped_to_user(self):
        from resources_servers.cube_standard.cube_adapters import _observation_to_nemo_gym_messages

        obs = MagicMock()
        obs.to_llm_messages.return_value = [{"role": "environment", "content": "state"}]
        result = _observation_to_nemo_gym_messages(obs)

        assert result[0].role == "user"

    def test_multiple_messages(self):
        from resources_servers.cube_standard.cube_adapters import _observation_to_nemo_gym_messages

        obs = MagicMock()
        obs.to_llm_messages.return_value = [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Click the button."},
        ]
        result = _observation_to_nemo_gym_messages(obs)

        assert len(result) == 2
        assert result[0].role == "system"
        assert result[1].role == "user"

    def test_all_roles_remapped(self):
        """All resulting messages must have an accepted role."""
        from resources_servers.cube_standard.cube_adapters import _observation_to_nemo_gym_messages

        obs = MagicMock()
        obs.to_llm_messages.return_value = [
            {"role": r, "content": "text"} for r in ["tool", "function", "environment", "other"]
        ]
        result = _observation_to_nemo_gym_messages(obs)

        valid_roles = {"user", "assistant", "system", "developer"}
        assert all(m.role in valid_roles for m in result)


# ---------------------------------------------------------------------------
# Tests: _serialize_env_output
# ---------------------------------------------------------------------------


class TestSerializeEnvOutput:
    def test_text_content(self, tmp_path):
        from resources_servers.cube_standard.cube_adapters import _serialize_env_output

        env_out = _make_env_output([{"role": "user", "content": "Clicked successfully."}])
        output, ct = _serialize_env_output(env_out, "sess_1", 1, tmp_path, "http://localhost:8000")

        assert ct == "text/plain"
        assert output == "Clicked successfully."

    def test_empty_text_returns_success(self, tmp_path):
        from resources_servers.cube_standard.cube_adapters import _serialize_env_output

        env_out = _make_env_output([{"role": "user", "content": ""}])
        output, ct = _serialize_env_output(env_out, "sess_1", 1, tmp_path, "http://localhost:8000")

        assert ct == "text/plain"
        assert output == "Success"

    def test_multiple_text_messages_joined(self, tmp_path):
        from resources_servers.cube_standard.cube_adapters import _serialize_env_output

        env_out = _make_env_output([
            {"role": "user", "content": "Line one."},
            {"role": "user", "content": "Line two."},
        ])
        output, ct = _serialize_env_output(env_out, "sess_1", 1, tmp_path, "http://localhost:8000")

        assert ct == "text/plain"
        assert "Line one." in output
        assert "Line two." in output

    def test_image_content_written_to_disk(self, tmp_path):
        from resources_servers.cube_standard.cube_adapters import _serialize_env_output

        # Minimal 1x1 PNG (8 bytes + headers — real enough for the test)
        png_bytes = (
            b"\x89PNG\r\n\x1a\n"  # PNG signature
            b"\x00\x00\x00\rIHDR"  # IHDR chunk size + type
            b"\x00\x00\x00\x01\x00\x00\x00\x01"  # 1x1 px
            b"\x08\x02\x00\x00\x00\x90wS\xde"  # bit depth, color type, etc.
            b"\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"  # IDAT
            b"\x00\x00\x00\x00IEND\xaeB`\x82"  # IEND
        )
        b64 = base64.b64encode(png_bytes).decode()
        data_url = f"data:image/png;base64,{b64}"

        env_out = _make_env_output([
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}]}
        ])

        output, ct = _serialize_env_output(env_out, "sess_1", 1, tmp_path, "http://localhost:8000")

        assert ct == "image/png"
        assert output.startswith("http://localhost:8000/screenshots/")
        assert output.endswith("step_0001.png")
        assert (tmp_path / "step_0001.png").exists()
        assert (tmp_path / "step_0001.png").read_bytes() == png_bytes

    def test_image_url_format(self, tmp_path):
        """URL must be <base_url>/screenshots/<session_id>/<filename>."""
        from resources_servers.cube_standard.cube_adapters import _serialize_env_output

        png_bytes = b"\x89PNG fake"
        b64 = base64.b64encode(png_bytes).decode()
        data_url = f"data:image/png;base64,{b64}"

        env_out = _make_env_output([
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}]}
        ])

        output, _ = _serialize_env_output(
            env_out, "my-session", 42, tmp_path, "http://cube-server:8000"
        )
        assert output == "http://cube-server:8000/screenshots/my-session/step_0042.png"

    def test_error_returns_text_plain(self, tmp_path):
        from resources_servers.cube_standard.cube_adapters import _serialize_env_output

        error = MagicMock()
        error.exception_str = "ValueError: bad argument"
        env_out = _make_env_output([], error=error)

        output, ct = _serialize_env_output(env_out, "sess_1", 1, tmp_path, "http://localhost:8000")

        assert ct == "text/plain"
        assert "Error" in output
        assert "ValueError" in output

    def test_list_content_with_text_block(self, tmp_path):
        from resources_servers.cube_standard.cube_adapters import _serialize_env_output

        env_out = _make_env_output([
            {"role": "user", "content": [{"type": "text", "text": "Step result."}]}
        ])
        output, ct = _serialize_env_output(env_out, "s", 1, tmp_path, "http://localhost:8000")

        assert ct == "text/plain"
        assert output == "Step result."

    def test_step_index_used_in_filename(self, tmp_path):
        from resources_servers.cube_standard.cube_adapters import _serialize_env_output

        png_bytes = b"\x89PNG fake"
        b64 = base64.b64encode(png_bytes).decode()
        data_url = f"data:image/png;base64,{b64}"
        env_out = _make_env_output([
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}]}
        ])

        _serialize_env_output(env_out, "s", 7, tmp_path, "http://localhost:8000")
        assert (tmp_path / "step_0007.png").exists()


# ---------------------------------------------------------------------------
# Tests: build_action
# ---------------------------------------------------------------------------


class TestBuildAction:
    def test_basic_construction(self):
        """build_action wraps cube.core.Action — test with the real import if available."""
        try:
            from cube.core import Action
            from resources_servers.cube_standard.cube_adapters import build_action

            action = build_action("call_abc", "click", {"x": 100, "y": 200})
            assert action.id == "call_abc"
            assert action.name == "click"
            assert action.arguments == {"x": 100, "y": 200}
            assert isinstance(action.arguments, dict)
        except ImportError:
            pytest.skip("cube-standard not installed — skipping build_action real-import test.")

    def test_import_is_deferred(self):
        """cube.core must not be imported at module load time (cube may not be installed)."""
        import importlib
        import sys

        # Ensure cube_adapters can be imported even if cube is not on sys.path
        # by checking the module doesn't reference cube at top level.
        # (This is a structural check, not a runtime check.)
        adapters_src = Path(__file__).parent.parent / "cube_adapters.py"
        src = adapters_src.read_text()
        # The only cube import should be inside a function body
        top_level_cube_imports = [
            line for line in src.splitlines()
            if line.startswith("from cube") or line.startswith("import cube")
        ]
        assert top_level_cube_imports == [], (
            f"cube_adapters.py must not import cube at module top level. "
            f"Found: {top_level_cube_imports}"
        )


# ---------------------------------------------------------------------------
# Tests: extract_reward
# ---------------------------------------------------------------------------


class TestExtractReward:
    def test_returns_reward_and_info(self):
        from resources_servers.cube_standard.cube_adapters import extract_reward

        env_out = MagicMock()
        env_out.reward = 1.0
        env_out.info = {"steps": 5, "reward_breakdown": {"correctness": 1.0}}

        reward, info = extract_reward(env_out)
        assert reward == 1.0
        assert info["steps"] == 5
        assert info["reward_breakdown"]["correctness"] == 1.0

    def test_zero_reward(self):
        from resources_servers.cube_standard.cube_adapters import extract_reward

        env_out = MagicMock()
        env_out.reward = 0.0
        env_out.info = {}

        reward, info = extract_reward(env_out)
        assert reward == 0.0
        assert info == {}
