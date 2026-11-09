# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the Prime component against its benchmark pairing configuration."""

from pathlib import Path

import yaml


def test_prime_pairing_validates_its_runtime_and_episode_budgets():
    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
    from resources_servers.partner_if.prime_app import PrimeIFConfig
    from responses_api_agents.prime_agent.child_app import PrimeChildConfig

    config = yaml.safe_load((Path(__file__).resolve().parents[3] / "benchmarks/partner_if/config.yaml").read_text())
    for name, kind, component, schema in [
        ("partner_if_prime", "resources_servers", "partner_if", PrimeIFConfig),
        ("partner_if_prime_agent", "responses_api_agents", "prime_agent", PrimeChildConfig),
        (
            "partner_if_prime_environment_server",
            "environment_servers",
            "single_agent_turn_legacy",
            SingleAgentTurnEnvironmentServerConfig,
        ),
    ]:
        schema.model_validate(config[name][kind][component] | {"name": name, "host": "127.0.0.1", "port": 8123})
    agent = config["partner_if_prime_agent"]["responses_api_agents"]["prime_agent"]
    resources = config["partner_if_prime"]["resources_servers"]["partner_if"]
    assert agent["capture_directory"] == resources["capture_directory"]
    assert (agent["max_turns"], agent["max_output_tokens"], agent["context_window"], agent["temperature"]) == (
        48,
        65536,
        131072,
        0,
    )
