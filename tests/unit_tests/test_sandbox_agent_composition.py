# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from omegaconf import OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict


@pytest.mark.parametrize(
    "profile,selected,expected",
    [
        ("benchmarks/terminal_bench_4/hermes_episode.yaml", None, "hermes_agent"),
        ("benchmarks/terminal_bench_4/miniswe_episode.yaml", None, "miniswe_sandboxed_agent"),
        ("benchmarks/swebench/pro/miniswe_episode.yaml", None, "miniswe_sandboxed_agent"),
        ("benchmarks/terminal_bench_4/hermes_episode.yaml", "miniswe_sandboxed_agent", "miniswe_sandboxed_agent"),
        ("benchmarks/terminal_bench_4/miniswe_episode.yaml", "hermes_agent", "hermes_agent"),
        ("benchmarks/swebench/pro/hermes_episode.yaml", "miniswe_sandboxed_agent", "miniswe_sandboxed_agent"),
    ],
)
def test_episode_profiles_and_agent_selector_preserve_environment_bindings(profile, selected, expected):
    paths = [profile]
    if selected:
        paths.append(f"responses_api_agents/{selected}/configs/{selected}.yaml")
    config = GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": paths,
                    "sandbox": {"local": {}},
                    "policy_model": {"responses_api_models": {"dummy_model": {"entrypoint": "app.py"}}},
                }
            ),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )
    environment = get_first_server_config_dict(config, config.environment_server_name)
    agent_name = environment.agent_server.name
    assert list(config[agent_name].responses_api_agents) == [expected]
    agent = get_first_server_config_dict(config, agent_name)
    assert agent.resources_server == environment.resources_server
    assert agent.model_server.name == "policy_model"
    assert agent.entrypoint == ("episode.py" if expected == "miniswe_sandboxed_agent" else "app.py")
    resources = get_first_server_config_dict(config, environment.resources_server.name)
    assert resources.datasets[0].type == "benchmark"
    assert resources.entrypoint == ("episode.py" if "terminal_bench_4" in profile else "app.py")
