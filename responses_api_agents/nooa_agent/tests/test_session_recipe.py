# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from nemo_gym.episode_types import MaterializedTask
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput


ROOT = Path(__file__).resolve().parents[3]


def test_counter_materialized_tasks_have_explicit_identity_and_verifier_state() -> None:
    path = ROOT / "resources_servers/example_session_state_mgmt/data/example_nooa_tasks.jsonl"
    for line in path.read_text().splitlines():
        row = MaterializedTask[SingleAgentTurnTaskInput].model_validate_json(line)
        assert row.task_id.taskset == "nooa-counter"
        assert row.task_input.task_data["expected_count"] > row.task_input.task_data["initial_count"]
        assert len(row.task_input.responses_create_params.tools) == 2


def test_counter_session_recipe_uses_shared_environment() -> None:
    from omegaconf import OmegaConf

    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
    from nemo_gym.config import GlobalConfigDictParser

    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths(
        [str(ROOT / "resources_servers/example_session_state_mgmt/configs/example_session_state_mgmt_nooa.yaml")]
    )
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    environment = SingleAgentTurnEnvironmentServerConfig(
        name="example_session_state_mgmt_environment_server",
        host="localhost",
        port=8000,
        **OmegaConf.to_container(
            config.example_session_state_mgmt_environment_server.environment_servers.single_agent_turn, resolve=True
        ),
    )
    assert environment.agent_server.name == "example_session_state_mgmt_nooa_agent"
    assert environment.resources_server.name == "example_session_state_mgmt_nooa_resources"
    assert environment.resources_tool_transports == ["direct_http"]
    assert environment.default_episode_timeout_seconds == 600
    assert config.environment_server_routes["nooa-counter"] == environment.name
    assert {
        name for name, block in config.items() if isinstance(block, type(config)) and "environment_servers" in block
    } == {environment.name}
    agent = config[environment.agent_server.name].responses_api_agents.nooa_agent
    assert agent.resources_server.name == environment.resources_server.name
    resources = config[environment.resources_server.name].resources_servers.example_session_state_mgmt
    assert resources.datasets[0].type == "example"
