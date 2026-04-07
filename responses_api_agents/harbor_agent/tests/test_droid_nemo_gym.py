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
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from responses_api_agents.harbor_agent.custom_agents.droid_nemo_gym import (
    DroidNemoGym,
    _rewrite_localhost,
)


def _make_droid(
    model_name: str = "test-model",
    api_base: str = "http://localhost:8000/v1",
    **kwargs: Any,
) -> DroidNemoGym:
    """Create a DroidNemoGym instance with test defaults."""
    with patch.object(DroidNemoGym, "__init__", lambda self, *a, **kw: None):
        agent = DroidNemoGym.__new__(DroidNemoGym)

    agent.model_name = model_name
    agent._api_base = api_base
    agent._api_key = kwargs.get("api_key", "not-needed")
    agent._droid_custom_model_id = kwargs.get("droid_custom_model_id", None)
    agent._droid_version = kwargs.get("droid_version", "0.65.0")
    agent._droid_bundle_path = kwargs.get("droid_bundle_path", None)
    agent._droid_autonomy_mode = kwargs.get("droid_autonomy_mode", "auto-low")
    agent._enable_thinking = kwargs.get("enable_thinking", True)
    agent._max_output_tokens = kwargs.get("max_output_tokens", 32768)
    agent._max_context_limit = kwargs.get("max_context_limit", 131072)

    return agent


class TestDroidSettingsGeneration:
    def test_produces_valid_json(self) -> None:
        agent = _make_droid()
        settings = json.loads(agent._build_settings_json())
        assert isinstance(settings, dict)
        assert "customModels" in settings
        assert "sessionDefaultSettings" in settings

    def test_injects_api_base_and_model(self) -> None:
        agent = _make_droid(api_base="http://10.0.0.5:9000/v1", model_name="Qwen/Qwen3-8B")
        settings = json.loads(agent._build_settings_json())
        assert settings["customModels"][0]["baseUrl"] == "http://10.0.0.5:9000/v1"
        assert settings["customModels"][0]["model"] == "Qwen/Qwen3-8B"
        assert settings["customModels"][0]["id"] == "custom:Qwen/Qwen3-8B"
        assert settings["sessionDefaultSettings"]["model"] == "custom:Qwen/Qwen3-8B"

    def test_custom_model_id_overrides_default(self) -> None:
        agent = _make_droid(
            model_name="nvidia/nvidia/nemotron-3-super-preview",
            droid_custom_model_id="my-custom-id",
        )
        settings = json.loads(agent._build_settings_json())
        assert settings["customModels"][0]["model"] == "nvidia/nvidia/nemotron-3-super-preview"
        assert settings["customModels"][0]["id"] == "custom:my-custom-id"
        assert settings["sessionDefaultSettings"]["model"] == "custom:my-custom-id"

    def test_thinking_and_limits(self) -> None:
        agent = _make_droid(enable_thinking=False, max_output_tokens=16384, max_context_limit=65536)
        settings = json.loads(agent._build_settings_json())
        assert settings["customModels"][0]["enableThinking"] is False
        assert settings["customModels"][0]["maxOutputTokens"] == 16384
        assert settings["customModels"][0]["maxContextLimit"] == 65536

    def test_autonomy_mode(self) -> None:
        agent = _make_droid(droid_autonomy_mode="auto-high")
        settings = json.loads(agent._build_settings_json())
        assert settings["sessionDefaultSettings"]["autonomyMode"] == "auto-high"

    def test_applies_localhost_rewriting(self) -> None:
        agent = _make_droid(api_base="http://localhost:8000/v1")
        settings_json = agent._build_settings_json()
        assert "localhost" not in settings_json or "127.0.0.1" not in settings_json


class TestDroidRunCommands:
    def test_returns_two_exec_inputs(self) -> None:
        agent = _make_droid()
        commands = agent.create_run_agent_commands("Fix the bug")
        assert len(commands) == 2
        assert "settings.json" in commands[0].command
        assert "droid exec" in commands[1].command

    def test_droid_exec_command(self) -> None:
        agent = _make_droid(model_name="my-model")
        commands = agent.create_run_agent_commands("Fix the bug")
        exec_cmd = commands[1].command
        assert "--skip-permissions-unsafe" in exec_cmd
        assert "--output-format stream-json" in exec_cmd
        assert "custom:my-model" in exec_cmd
        assert "DROID_SESSION_TRACE" in exec_cmd

    def test_custom_model_id_in_command(self) -> None:
        agent = _make_droid(
            model_name="nvidia/nvidia/nemotron-3-super-preview",
            droid_custom_model_id="my-custom-id",
        )
        commands = agent.create_run_agent_commands("test")
        assert "custom:my-custom-id" in commands[1].command

    def test_factory_api_key_from_environ(self) -> None:
        agent = _make_droid()
        with patch.dict("os.environ", {"FACTORY_API_KEY": "my-key"}):
            commands = agent.create_run_agent_commands("test")
        assert commands[0].env["FACTORY_API_KEY"] == "my-key"


class TestDroidLocalhostRewriting:
    @patch("responses_api_agents.harbor_agent.custom_agents.droid_nemo_gym._get_host_ip", return_value="192.168.1.100")
    def test_replaces_localhost(self, _mock_ip: Any) -> None:
        assert _rewrite_localhost("http://localhost:8000/v1") == "http://192.168.1.100:8000/v1"

    @patch("responses_api_agents.harbor_agent.custom_agents.droid_nemo_gym._get_host_ip", return_value="10.0.0.5")
    def test_replaces_127_0_0_1(self, _mock_ip: Any) -> None:
        assert _rewrite_localhost("http://127.0.0.1:9000/v1") == "http://10.0.0.5:9000/v1"

    @patch("responses_api_agents.harbor_agent.custom_agents.droid_nemo_gym._get_host_ip", return_value="10.0.0.5")
    def test_leaves_other_addresses_unchanged(self, _mock_ip: Any) -> None:
        assert _rewrite_localhost("http://10.1.2.3:8000/v1") == "http://10.1.2.3:8000/v1"


class TestDroidIntegrationWithApp:
    def test_missing_trajectory_returns_reward_only(self) -> None:
        from responses_api_agents.harbor_agent.utils import HarborAgentUtils

        trial_result = {
            "task_name": "tb2_task",
            "agent_result": {"n_input_tokens": 0, "n_output_tokens": 0, "rollout_details": []},
            "verifier_result": {"rewards": {"reward": 1.0}},
        }
        output_items = HarborAgentUtils.trial_result_to_responses(trial_result, None)
        assert output_items == []
        assert HarborAgentUtils.extract_reward(trial_result["verifier_result"]) == 1.0

    def test_agent_name(self) -> None:
        assert DroidNemoGym.name() == "droid-nemo-gym"

    def test_install_template_path(self) -> None:
        agent = _make_droid()
        template_path = agent._install_agent_template_path
        assert template_path.name == "droid-setup.sh.j2"
        assert template_path.parent == Path(__file__).parent.parent / "custom_agents"
