# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shipped configs, resolved offline by Gym's own parser."""

import yaml
from omegaconf import OmegaConf

from nemo_gym import PARENT_DIR
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict


CONFIGS = PARENT_DIR / "resources_servers" / "vibench" / "configs"


def _server_config(server_name: str, config_path: str = "resources_servers/vibench/configs/vibench.yaml") -> dict:
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.merge(
                GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
                {"config_paths": [config_path]},
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    return OmegaConf.to_container(get_first_server_config_dict(config, server_name), resolve=True)


def test_opencode_keeps_the_command_deny_list_without_web_subagent_or_skill_tools():
    agent = _server_config("vibench_opencode_agent")

    assert agent["resources_server"]["name"] == "vibench_resources_server"
    permission = agent["opencode_config"]["permission"]
    assert [permission[tool] for tool in ("webfetch", "websearch", "task", "skill")] == ["deny"] * 4
    # The denials follow opencode_agent's allow-all, which they override, and keep its command deny-list.
    assert list(permission).index("*") < list(permission).index("webfetch")
    assert permission["bash"]["*git fetch*"] == "deny"
    # OpenCode places a tools map's keys before "*" and the last matching rule wins, so a tools map would undo them.
    assert "tools" not in agent["opencode_config"]
    # A call's prompt plus its output budget must fit the served context.
    assert agent["context_window"] + agent["max_output_tokens"] <= 262144


def test_the_docker_overlay_gives_only_the_model_server_the_host_address():
    docker = yaml.safe_load((CONFIGS / "docker.yaml").read_text())

    assert docker["use_absolute_ip"] is True
    assert docker["absolute_ip_server_types"] == ["responses_api_models"]


def test_the_legacy_agent_config_still_runs_vibench_agent_against_this_server():
    config_path = "resources_servers/vibench/configs/vibench_legacy_agent.yaml"
    agent = _server_config("vibench_opencode_agent", config_path)
    resources = _server_config("vibench_resources_server", config_path)
    environment = _server_config("vibench_opencode_environment_server", config_path)

    assert agent["entrypoint"] == "app.py" and "build_image" in agent
    assert agent["resources_server"]["name"] == "vibench_resources_server"
    # The agent writes tarballs where this server reads them.
    assert resources["artifact_dir"] == agent["artifact_dir"]
    assert "sandbox_provider" not in resources
    assert environment["agent_server"]["name"] == "vibench_opencode_agent"
