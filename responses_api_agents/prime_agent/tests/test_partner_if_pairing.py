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

    assert "kernel_packages" not in agent
    assert resources["task_python_packages"] == ["setuptools==80.9.0", "pytz==2026.2", "sqlparse==0.5.5"]
    assert set(resources["task_python_package_hashes"]) == set(resources["task_python_packages"])


def test_prime_uses_phase_limits_without_an_outer_episode_deadline():
    from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServerConfig
    from resources_servers.partner_if.prime_app import PrimeIFConfig
    from responses_api_agents.prime_agent.child_app import PrimeChildConfig

    config = yaml.safe_load((Path(__file__).resolve().parents[3] / "benchmarks/partner_if/config.yaml").read_text())
    environment = SingleAgentTurnLegacyEnvironmentServerConfig.model_validate(
        config["partner_if_prime_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
        | {"name": "prime_environment", "host": "127.0.0.1", "port": 8123}
    )
    agent = PrimeChildConfig.model_validate(
        config["partner_if_prime_agent"]["responses_api_agents"]["prime_agent"]
        | {"name": "prime_agent", "host": "127.0.0.1", "port": 8124}
    )
    resources = PrimeIFConfig.model_validate(
        config["partner_if_prime"]["resources_servers"]["partner_if"]
        | {"name": "prime_resources", "host": "127.0.0.1", "port": 8125}
    )
    assert environment.default_episode_timeout_seconds is None
    assert agent.timeout == 900
    assert agent.sandbox_install_timeout_seconds == 1800
    assert resources.setup_timeout_seconds == 1800
    assert agent.session_close_timeout_seconds == 60
    assert environment.cleanup_timeout_seconds == 360


def test_prime_admission_and_agent_capacity_cover_all_dataset_repeats():
    config = yaml.safe_load((Path(__file__).resolve().parents[3] / "benchmarks/partner_if/config.yaml").read_text())
    datasets = config["partner_if_dispatcher"]["responses_api_agents"]["partner_if_dispatcher"]["datasets"]
    repeats = max(dataset["num_repeats"] for dataset in datasets)
    environment = config["partner_if_prime_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    agent = config["partner_if_prime_agent"]["responses_api_agents"]["prime_agent"]
    assert repeats == 3
    # All repeats fit at both layers, even when one evaluation runs longer than the queue timeout.
    assert environment["max_concurrent_episodes"] >= repeats
    assert agent["concurrency"] == environment["max_concurrent_episodes"]
    assert environment["queue_timeout_seconds"] > 0
