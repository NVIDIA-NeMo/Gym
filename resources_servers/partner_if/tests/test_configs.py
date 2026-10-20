# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The component config and example data, validated as Gym validates them."""

import yaml

from nemo_gym import PARENT_DIR


def test_frozen_history_cannot_dispatch_live_tools():
    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig

    config = yaml.safe_load((PARENT_DIR / "resources_servers/partner_if/configs/partner_if.yaml").read_text())
    agent = config["partner_if_simple_agent"]["responses_api_agents"]["simple_agent"]
    assert agent["execute_tools"] is False
    env = config["partner_if_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    assert env["resources_tool_transports"] == []
    assert env["resources_server"]["name"] == "partner_if"
    assert env["agent_server"]["name"] == "partner_if_simple_agent"
    parsed = SingleAgentTurnEnvironmentServerConfig.model_validate(
        {**env, "name": "partner_if_environment_server", "host": "127.0.0.1", "port": 8080}
    )
    assert parsed.cleanup_timeout_seconds == 180


def test_gym_task_data_tooling_validates_the_same_contract():
    from nemo_gym.task_data import TaskDataValidator, load_task_data_schema
    from resources_servers.partner_if.package import read_jsonl

    root = PARENT_DIR / "resources_servers/partner_if"
    adapter = load_task_data_schema(root)
    validator = TaskDataValidator("partner_if", adapter, "example.jsonl")
    for i, row in enumerate(read_jsonl(root / "data/example.jsonl")):
        validator.validate_row(i, row)
    assert validator.report.clean, validator.report.summary()
