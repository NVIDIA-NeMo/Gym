# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.harness_agent.app import HarnessAgent, HarnessAgentConfig, stage_and_run_eval


def _config(**kwargs) -> HarnessAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="sbx",
        resources_server=ResourcesServerRef(type="resources_servers", name="rs"),
        model_server=ModelServerRef(type="responses_api_models", name="model"),
        agent_module="responses_api_agents.opencode_agent.app",
        agent_class="OpenCodeAgent",
        agent_config_class="OpenCodeAgentConfig",
        sandbox_provider={"opensandbox": {}},
    )
    base.update(kwargs)
    return HarnessAgentConfig(**base)


def _make_agent(**cfg_kwargs) -> HarnessAgent:
    # skip provider creation and gym tar build (both side effects) during construction
    with (
        patch("responses_api_agents.harness_agent.app.create_provider", return_value=MagicMock()),
        patch.object(HarnessAgent, "_build_gym_tar", return_value=None),
    ):
        return HarnessAgent(config=_config(**cfg_kwargs), server_client=MagicMock(spec=ServerClient))


def test_config_defaults():
    cfg = _config()
    assert cfg.sandbox_image == "python:3.12-slim"
    assert cfg.sandbox_python == "python3"


def test_runner_config_carries_agent_symbols():
    agent = _make_agent(
        agent_module="responses_api_agents.opencode_agent.app",
        agent_class="OpenCodeAgent",
        agent_config_class="OpenCodeAgentConfig",
        sandbox_python="/deps/bin/python3",
    )
    script, runner_config, cmd = agent._runner()
    assert runner_config["agent_module"] == "responses_api_agents.opencode_agent.app"
    assert runner_config["agent_class"] == "OpenCodeAgent"
    assert runner_config["agent_config_class"] == "OpenCodeAgentConfig"
    assert "runner_config.json" in script
    compile(script, "<agent_runner>", "exec")
    assert cmd == "/deps/bin/python3 /work/runner.py"


def test_sandbox_model_url_preserves_remote_hostname_and_port():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client._build_server_base_url = MagicMock(return_value="http://model-host:8000")
    agent.server_client.global_config_dict = MagicMock()
    with (
        patch("responses_api_agents.harness_agent.app.get_first_server_config_dict", return_value={}),
        patch("responses_api_agents.harness_agent.app.socket.gethostbyname") as resolve,
    ):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://model-host:8000"
    resolve.assert_not_called()


def test_sandbox_model_url_prefers_backend_base_url_and_strips_v1():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client.global_config_dict = MagicMock()
    with (
        patch(
            "responses_api_agents.harness_agent.app.get_first_server_config_dict",
            return_value={"base_url": "http://vllm-node:9000/v1"},
        ),
        patch("responses_api_agents.harness_agent.app.socket.gethostbyname") as resolve,
    ):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://vllm-node:9000"
    resolve.assert_not_called()


def test_sandbox_model_url_keeps_loopback_on_dns_failure():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client._build_server_base_url = MagicMock(return_value="http://localhost:8000")
    agent.server_client.global_config_dict = MagicMock()
    with (
        patch("responses_api_agents.harness_agent.app.get_first_server_config_dict", return_value={}),
        patch("responses_api_agents.harness_agent.app.socket.gethostbyname", side_effect=OSError),
    ):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://localhost:8000"


async def test_grading_command_failure_is_not_reward_zero():
    agent = _make_agent()
    agent._provider.exec = AsyncMock(return_value=SandboxExecResult("", "grader crashed", 2))
    agent._provider.download_file = AsyncMock()

    with pytest.raises(RuntimeError, match="eval command failed.*grader crashed"):
        await agent._grade_in_box(MagicMock(), {"eval_command": "false"})
    agent._provider.download_file.assert_not_awaited()


async def test_empty_reward_file_is_not_reward_zero():
    provider = MagicMock()
    provider.exec = AsyncMock(return_value=SandboxExecResult("", "", 0))
    provider.download_file = AsyncMock(side_effect=lambda _, __, path: path.write_text(""))

    with pytest.raises(RuntimeError, match="reward file.*is empty"):
        await stage_and_run_eval(provider, MagicMock(), {}, "true", "/reward", 30)


async def test_setup_failure_closes_sandbox():
    agent = _make_agent(setup_commands=["install deps"])
    handle = MagicMock()
    agent._provider.create = AsyncMock(return_value=handle)
    agent._provider.exec = AsyncMock(
        side_effect=[SandboxExecResult("", "", 0), SandboxExecResult("", "install failed", 2)]
    )
    agent._provider.close = AsyncMock()

    with pytest.raises(RuntimeError, match="setup failed.*install failed"):
        await agent._provision_box("image", {}, "https://model.example")
    agent._provider.close.assert_awaited_once_with(handle)


async def test_download_json_requires_exactly_one_row():
    agent = _make_agent()
    agent._provider.download_file = AsyncMock(side_effect=lambda _, __, path: path.write_text("{}\n{}\n"))

    with pytest.raises(RuntimeError, match="expected one JSON row.*got 2"):
        await agent._download_json(MagicMock(), "/work/rollouts.jsonl")


def test_gym_tar_built_on_init():
    with (
        patch("responses_api_agents.harness_agent.app.create_provider", return_value=MagicMock()),
        patch.object(HarnessAgent, "_build_gym_tar", return_value="/tmp/fake.tar.gz"),
    ):
        agent = HarnessAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        assert agent._gym_tar == "/tmp/fake.tar.gz"


def test_gym_source_prebuilt_path_and_url():
    with (
        patch("responses_api_agents.harness_agent.app.create_provider", return_value=MagicMock()),
        patch.object(HarnessAgent, "_build_gym_tar") as build,
    ):
        prebuilt = HarnessAgent(
            config=_config(gym_source="/tmp/prebuilt.tar.gz"), server_client=MagicMock(spec=ServerClient)
        )
        assert str(prebuilt._gym_tar) == "/tmp/prebuilt.tar.gz"
        assert prebuilt._gym_source_url is None
        remote = HarnessAgent(
            config=_config(gym_source="https://example.com/gym.tar.gz"), server_client=MagicMock(spec=ServerClient)
        )
        assert remote._gym_tar is None
        assert remote._gym_source_url == "https://example.com/gym.tar.gz"
        build.assert_not_called()
