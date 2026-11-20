# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The session harness presets run their harness on config.yaml's resources server, rows and datasets, over MCP."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.benchmarks import BenchmarkConfig, _benchmark_config_name
from nemo_gym.config import GlobalConfigDictParser


ROOT = Path(__file__).resolve().parents[3]
BENCHMARK = ROOT / "benchmarks" / "automationbench"
# Preset file stem -> the agent type it runs.
HARNESSES = {
    "opencode": "opencode_agent",
    "openclaw": "openclaw_agent",
    "hermes": "hermes_agent",
    "pi": "pi_agent",
    "codex": "codex_agent",
}


def _merged(path: Path):
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(path)])
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    return config


@pytest.mark.parametrize(("preset", "agent_type"), HARNESSES.items())
def test_a_preset_binds_its_harness_over_mcp_to_the_config_yaml_resources_server(preset: str, agent_type: str) -> None:
    path = BENCHMARK / f"{preset}.yaml"
    instance = f"automationbench_{preset}_agent"
    config = _merged(path)
    baseline = _merged(BENCHMARK / "config.yaml")

    assert _benchmark_config_name(path.relative_to(ROOT / "benchmarks")) == f"automationbench/{preset}"
    benchmark = BenchmarkConfig.from_config_path(path, strict=False)
    assert benchmark is not None and benchmark.agent_name == instance
    # simple_agent is gone, and the environment server runs the harness with the tools granted over MCP.
    assert config.automationbench is None
    environment = config.automationbench_environment_server.environment_servers.single_agent_turn_legacy
    assert (environment.agent_server.name, list(environment.resources_tool_transports)) == (instance, ["mcp"])
    assert environment.resources_server.name == "automationbench_resources_server"
    # The same resources server, now serving its tools over MCP, and the same datasets as config.yaml.
    resources = OmegaConf.to_container(config.automationbench_resources_server.resources_servers.automationbench)
    assert resources.pop("expose_tools_over_mcp") is True
    assert resources == OmegaConf.to_container(
        baseline.automationbench_resources_server.resources_servers.automationbench
    )
    agent = config[instance].responses_api_agents[agent_type]
    assert agent.resources_server.name == "automationbench_resources_server"
    # From the harness's configs/profiles/benchmark.yaml: a sandbox the agent creates, here from the preset's image.
    assert (agent.entrypoint, agent.sandbox_provider) == ("app.py", "sandbox")
    assert (agent.sandbox_config.image, agent.sandbox_config.ready_timeout_s) == ("python:3.12", 1200)
    assert OmegaConf.to_container(agent.datasets) == OmegaConf.to_container(
        baseline.automationbench.responses_api_agents.simple_agent.datasets
    )
    # The harnesses refuse a tool list, an output budget and sampling in the request.
    assert OmegaConf.to_container(config.responses_create_params) == {
        "tools": [],
        "max_output_tokens": None,
        "temperature": None,
        "top_p": None,
    }
    # The sandbox calls the resources server's /mcp by address.
    assert "resources_servers" in config.absolute_ip_server_types
