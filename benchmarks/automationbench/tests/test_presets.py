# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The session harness presets run their harness on config.yaml's resources server and tasks, over MCP."""

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


def _merged(*paths: Path):
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(path) for path in paths])
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
    # config.yaml is not loaded: no simple_agent or its environment server, and no run-level request field such as
    # its sampling, which the harnesses refuse. The preset's own environment server runs the harness with the tools
    # granted over MCP.
    assert not {"automationbench", "automationbench_environment_server", "responses_create_params"} & set(config)
    environment = config[f"automationbench_{preset}_environment_server"].environment_servers.single_agent_turn_legacy
    assert (environment.agent_server.name, list(environment.resources_tool_transports)) == (instance, ["mcp"])
    assert environment.resources_server.name == "automationbench_resources_server"
    # The same resources server, now serving its tools over MCP and the harness-neutral rows without them.
    resources = OmegaConf.to_container(config.automationbench_resources_server.resources_servers.automationbench)
    assert (resources.pop("expose_tools_over_mcp"), resources.pop("rows_carry_tools")) == (True, False)
    assert resources == OmegaConf.to_container(
        baseline.automationbench_resources_server.resources_servers.automationbench
    )
    agent = config[instance].responses_api_agents[agent_type]
    assert agent.resources_server.name == "automationbench_resources_server"
    # From the harness's configs/profiles/benchmark.yaml: a sandbox the agent creates, here from the preset's image.
    assert (agent.entrypoint, agent.sandbox_provider) == ("app.py", "sandbox")
    assert (agent.sandbox_config.image, agent.sandbox_config.ready_timeout_s) == ("python:3.12", 1200)
    # config.yaml's benchmark dataset, but the rows without the tool list and the output budget, which the harnesses
    # refuse in the request.
    (dataset,) = OmegaConf.to_container(agent.datasets)
    (baseline_dataset,) = [
        entry
        for entry in OmegaConf.to_container(baseline.automationbench.responses_api_agents.simple_agent.datasets)
        if entry["type"] == "benchmark"
    ]
    neutral = {
        "jsonl_fpath": "benchmarks/automationbench/data/automationbench-600-harness-neutral.jsonl",
        "prepare_script": "benchmarks/automationbench/prepare_harness_neutral.py",
    }
    assert dataset == baseline_dataset | neutral
    # Episodes are admitted at the harness's rate, before their deadline starts.
    assert environment.max_concurrent_episodes == agent.concurrency == 32
    assert environment.queue_timeout_seconds == 86400
    # The sandbox calls the resources server's /mcp by address.
    assert "resources_servers" in config.absolute_ip_server_types


def test_the_presets_load_together_and_each_environment_server_runs_its_own_harness() -> None:
    config = _merged(*(BENCHMARK / f"{preset}.yaml" for preset in HARNESSES))

    bound = {
        preset: config[
            f"automationbench_{preset}_environment_server"
        ].environment_servers.single_agent_turn_legacy.agent_server.name
        for preset in HARNESSES
    }
    assert bound == {preset: f"automationbench_{preset}_agent" for preset in HARNESSES}
    assert "automationbench_environment_server" not in config


def test_the_presets_leave_another_workloads_run_level_request_fields_as_they_are() -> None:
    other = _merged(BENCHMARK / "config.yaml")
    config = _merged(BENCHMARK / "config.yaml", *(BENCHMARK / f"{preset}.yaml" for preset in HARNESSES))

    assert OmegaConf.to_container(config.responses_create_params) == {"temperature": 1.0, "top_p": 1.0}
    assert config.responses_create_params == other.responses_create_params
