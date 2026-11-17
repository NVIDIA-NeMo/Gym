# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf

from environment_servers.harbor.app import HarborEnvironmentServerConfig
from nemo_gym.config.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from responses_api_agents.direct_vision_agent.app import DirectVisionAgentConfig


BENCHMARK_DIR = Path(__file__).resolve().parents[1]


def test_default_uses_native_environment_and_direct_vision() -> None:
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        str(BENCHMARK_DIR / "config.yaml"),
                        "responses_api_agents/direct_vision_agent/configs/direct_vision_agent.yaml",
                        "environment_servers/harbor/configs/harbor.yaml",
                        "responses_api_models/openai_model/configs/openai_model.yaml",
                    ],
                    "policy_base_url": "http://policy/v1",
                    "policy_api_key": "EMPTY",
                    "policy_model_name": "vision-model",
                    "judge_base_url": "https://judge.example/v1",
                    "judge_api_key": "TEST_PLACEHOLDER",
                    "judge_model_name": "gpt-5-mini",
                    "results_dir": "/tmp/vials-config-test",
                    "harbor_environment_server": {
                        "environment_servers": {
                            "harbor": {"agent_server": {"type": "responses_api_agents", "name": "direct_vision_agent"}}
                        }
                    },
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    values = OmegaConf.to_container(config, resolve=True)
    agent = DirectVisionAgentConfig.model_validate(
        {"name": "direct_vision_agent", **values["direct_vision_agent"]["responses_api_agents"]["direct_vision_agent"]}
    )
    environment = HarborEnvironmentServerConfig.model_validate(
        {"name": "harbor", **values["harbor_environment_server"]["environment_servers"]["harbor"]}
    )
    assert agent.model_server.name == "policy_model"
    assert agent.input_path == "/task/input.json"
    assert environment.agent_server.name == "direct_vision_agent"
    assert environment.secrets["VIALS_JUDGE_MODEL"] == "gpt-5-mini"
    assert "harbor_agent_general" not in str(values)
    assert "legacy_agent" not in str(values)
