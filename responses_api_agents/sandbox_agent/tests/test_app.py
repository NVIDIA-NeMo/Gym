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

from unittest.mock import MagicMock, patch

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import ServerClient
from responses_api_agents.sandbox_agent.app import SandboxAgent, SandboxAgentConfig


def _config(**kwargs) -> SandboxAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="sbx",
        resources_server=ResourcesServerRef(type="resources_servers", name="rs"),
        sandbox_provider={"opensandbox": {}},
    )
    base.update(kwargs)
    return SandboxAgentConfig(**base)


def _make_agent(**cfg_kwargs) -> SandboxAgent:
    # skip provider creation and gym tar build (both side effects) during construction
    with (
        patch("responses_api_agents.sandbox_agent.app.create_provider", return_value=MagicMock()),
        patch.object(SandboxAgent, "_build_gym_tar", return_value=None),
    ):
        return SandboxAgent(config=_config(**cfg_kwargs), server_client=MagicMock(spec=ServerClient))


def test_config_defaults():
    cfg = _config()
    assert cfg.mode == "agent_only_runner"
    assert cfg.sandbox_image == "python:3.12-slim"
    assert cfg.sandbox_python == "python3"


def test_gym_runner_script_boots_gym_and_collects():
    agent = _make_agent(
        mode="gym_runner",
        nested_config_paths=["a.yaml", "b.yaml"],
        nested_agent_name="nested_math",
        nested_agent_port=12345,
    )
    path, script, cmd = agent._runner()
    assert path == "/work/runner.py"
    assert "+config_paths=[a.yaml,b.yaml]" in script
    assert "http://127.0.0.1:12345" in script
    assert "ng_collect_rollouts" in script
    assert "+agent_name=nested_math" in script
    compile(script, "<runner>", "exec")


def test_gym_runner_skips_gym_tar():
    with (
        patch("responses_api_agents.sandbox_agent.app.create_provider", return_value=MagicMock()),
        patch.object(SandboxAgent, "_build_gym_tar", return_value="/tmp/fake.tar.gz") as tar,
    ):
        nested = SandboxAgent(config=_config(mode="gym_runner"), server_client=MagicMock(spec=ServerClient))
        assert nested._gym_tar is None
        tar.assert_not_called()


def test_runner_substitutes_agent_symbols():
    agent = _make_agent(
        agent_module="responses_api_agents.opencode_agent.app",
        agent_class="OpenCodeAgent",
        agent_config_class="OpenCodeAgentConfig",
        sandbox_python="/deps/bin/python3",
    )
    path, script, cmd = agent._runner()
    assert path == "/work/runner.py"
    assert "responses_api_agents.opencode_agent.app" in script
    assert "OpenCodeAgent" in script
    assert "OpenCodeAgentConfig" in script
    assert cmd == "/deps/bin/python3 /work/runner.py"


def test_sandbox_model_url_rewrites_host_to_ip_and_preserves_port():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client._build_server_base_url = MagicMock(return_value="http://model-host:8000")
    agent.server_client.global_config_dict = MagicMock()
    with (
        patch("responses_api_agents.sandbox_agent.app.get_first_server_config_dict", return_value={}),
        patch("responses_api_agents.sandbox_agent.app.socket.gethostbyname", return_value="10.1.2.3"),
    ):
        url = agent._sandbox_model_url(MagicMock())
    # host rewritten to an IP the sandbox can reach, no /v1 appended, port kept
    assert url == "http://10.1.2.3:8000"


def test_sandbox_model_url_prefers_backend_base_url_and_strips_v1():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client.global_config_dict = MagicMock()
    with (
        patch(
            "responses_api_agents.sandbox_agent.app.get_first_server_config_dict",
            return_value={"base_url": "http://vllm-node:9000/v1"},
        ),
        patch("responses_api_agents.sandbox_agent.app.socket.gethostbyname", return_value="10.1.2.3"),
    ):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://10.1.2.3:9000"


def test_sandbox_model_url_falls_back_to_hostname_on_dns_failure():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client._build_server_base_url = MagicMock(return_value="http://model-host:8000")
    agent.server_client.global_config_dict = MagicMock()
    with (
        patch("responses_api_agents.sandbox_agent.app.get_first_server_config_dict", return_value={}),
        patch("responses_api_agents.sandbox_agent.app.socket.gethostbyname", side_effect=OSError),
    ):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://model-host:8000"


def test_gym_tar_built_on_init():
    with (
        patch("responses_api_agents.sandbox_agent.app.create_provider", return_value=MagicMock()),
        patch.object(SandboxAgent, "_build_gym_tar", return_value="/tmp/fake.tar.gz"),
    ):
        agent = SandboxAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        assert agent._gym_tar == "/tmp/fake.tar.gz"
