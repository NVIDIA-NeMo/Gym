# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


def test_opencode_config_has_environment_server() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs/deepswe_external1_opencode.yaml"
    resolved = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                OmegaConf.load(config_path),
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )

    environment = resolved["deepswe_external1_environment_server"]["environment_servers"]["legacy_agent"]
    assert environment["entrypoint"] == "app.py"
    assert environment["agent_server"] == {
        "type": "responses_api_agents",
        "name": "deepswe_external1_opencode_sandboxed_agent",
    }
    agent = resolved[environment["agent_server"]["name"]]["responses_api_agents"]["opencode_sandboxed_agent"]
    assert agent["resources_server"] == {
        "type": "resources_servers",
        "name": "deepswe_external1_opencode_resources_server",
    }
