# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The configs that run GDPVal with a session harness in place of Stirrup (``benchmarks/gdpval/<harness>.yaml``)."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from responses_api_agents.codex_agent.app import CodexAgentConfig
from responses_api_agents.pi_agent.app import PiAgentConfig


_BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "gdpval"
# Each harness config here (<harness>.yaml), with its agent's config class.
_HARNESS_CONFIGS = {"pi": PiAgentConfig, "codex": CodexAgentConfig}


@pytest.fixture(autouse=True)
def _opensandbox_env(monkeypatch):
    # benchmarks/gdpval/config.yaml includes the OpenSandbox provider config, whose api_key has no default.
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-key")


def _resolve(path: Path) -> dict:
    initial = OmegaConf.merge(OmegaConf.load(path), GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT)
    config = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial)
    return OmegaConf.to_container(config, resolve=True)


@pytest.mark.parametrize("harness", _HARNESS_CONFIGS)
def test_harness_replaces_stirrup_and_reaches_the_task_tools_over_mcp(harness):
    path = _BENCHMARK_DIR / f"{harness}.yaml"
    benchmark = BenchmarkConfig.from_config_path(path)
    config = _resolve(path)
    (agent_type,) = config[benchmark.agent_name]["responses_api_agents"]
    agent = config[benchmark.agent_name]["responses_api_agents"][agent_type]
    environment = config["gdpval_environment_server"]["environment_servers"]["single_agent_turn_legacy"]

    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/gdpval/data/gdpval_benchmark_harness_neutral.jsonl")
    assert benchmark.dataset.prompt_config == Path("benchmarks/gdpval/prompts/harness-neutral.yaml")
    assert benchmark.dataset.prepare_script == Path("benchmarks/gdpval/prepare_harness_neutral.py")
    agents = [name for name, value in config.items() if isinstance(value, dict) and "responses_api_agents" in value]
    assert agents == [benchmark.agent_name]
    # gym env start and gym env prefetch read the first key of every top-level mapping.
    assert all(value for value in config.values() if isinstance(value, dict))
    assert agent["timeout"] < environment["default_episode_timeout_seconds"]
    # The agent ignores keys its config class does not define, so a misspelled setting would be dropped silently.
    # Gym reads datasets and description itself.
    assert agent.keys() - {"datasets", "description"} - _HARNESS_CONFIGS[harness].model_fields.keys() == set()
    # Building the config also runs its validators, such as the limits on timeouts and compaction.
    _HARNESS_CONFIGS[harness].model_validate({**agent, "name": benchmark.agent_name})
    # Only the agent, the agents the resources server allows and the tool transport differ from config.yaml, so the
    # judges, scoring, sandbox and multi-stage driver are the ones Stirrup runs with.
    overrides = OmegaConf.to_container(OmegaConf.load(path))
    assert overrides.keys() == {
        "config_paths",
        "gdpval_stirrup_agent",
        "gdpval_resources_server",
        benchmark.agent_name,
        "gdpval_environment_server",
    }
    assert overrides["config_paths"] == ["benchmarks/gdpval/config.yaml"]
    assert overrides["gdpval_resources_server"] == {
        "resources_servers": {"gdpval": {"allowed_agents": [agent_type], "expose_tools_over_mcp": True}}
    }
    assert overrides["gdpval_environment_server"] == {
        "environment_servers": {
            "single_agent_turn_legacy": {
                "agent_server": {"type": "responses_api_agents", "name": benchmark.agent_name},
                "resources_tool_transports": ["mcp"],
            }
        }
    }
