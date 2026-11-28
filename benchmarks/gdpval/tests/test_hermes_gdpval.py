# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.benchmarks import BenchmarkConfig, _benchmark_config_name, _benchmark_config_paths
from nemo_gym.config import GlobalConfigDictParser


ROOT = Path(__file__).resolve().parents[3]


def test_hermes_recipe_swaps_only_the_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TAVILY_API_KEY", "unused")
    monkeypatch.setenv("GDPVAL_SANDBOX_IMAGE", "fixture:1")
    monkeypatch.setenv("PERSIST_DELIVERABLES_DIR", "/tmp/deliverables")
    monkeypatch.setattr("nemo_gym.config.global_config._find_open_port_using_range", lambda **_: 12345)
    recipe = ROOT / "benchmarks/gdpval/hermes.yaml"
    assert recipe in _benchmark_config_paths(recipe.parent)
    assert _benchmark_config_name(recipe.relative_to(ROOT / "benchmarks")) == "gdpval/hermes"
    benchmark = BenchmarkConfig.from_config_path(recipe, strict=False)
    assert benchmark is not None
    assert benchmark.agent_name == "gdpval_hermes_agent"
    assert benchmark.dataset.prompt_config == Path("benchmarks/gdpval/prompts/harness-neutral.yaml")

    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(recipe)])
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    resources = config.gdpval_resources_server.resources_servers.gdpval
    assert resources.allowed_agents == ["hermes_agent"]
    assert resources.expose_tools_over_mcp is True
    assert config.get("gdpval_stirrup_agent") is None
    environment = config.gdpval_environment_server.environment_servers.single_agent_turn_legacy
    assert environment.agent_server.name == "gdpval_hermes_agent"
    assert list(environment.resources_tool_transports) == ["mcp"]
