# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The configs that run GDPVal with a session harness in place of Stirrup (``benchmarks/gdpval/<harness>.yaml``)."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig


_BENCHMARK_DIR = Path(__file__).resolve().parents[3] / "benchmarks" / "gdpval"
_HARNESS_CONFIGS = sorted(p for p in _BENCHMARK_DIR.glob("*.yaml") if p.name not in ("config.yaml", "manifest.yaml"))


@pytest.fixture(autouse=True)
def _opensandbox_env(monkeypatch):
    # benchmarks/gdpval/config.yaml includes the OpenSandbox provider config, which requires both.
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "test-key")
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "opensandbox.invalid")


def _resolve(path: Path) -> dict:
    initial = OmegaConf.merge(OmegaConf.load(path), GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT)
    config = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial)
    return OmegaConf.to_container(config, resolve=True)


@pytest.mark.parametrize("path", _HARNESS_CONFIGS, ids=lambda path: path.stem)
def test_harness_replaces_stirrup_and_reaches_the_task_tools_over_mcp(path):
    benchmark = BenchmarkConfig.from_config_path(path)
    config = _resolve(path)
    (agent_type,) = config[benchmark.agent_name]["responses_api_agents"]
    environment = config["gdpval_environment_server"]["environment_servers"]["single_agent_turn_legacy"]
    resources = config["gdpval_resources_server"]["resources_servers"]["gdpval"]

    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/gdpval/data/gdpval_benchmark_harness_neutral.jsonl")
    assert benchmark.dataset.prompt_config == Path("benchmarks/gdpval/prompts/harness-neutral.yaml")
    assert benchmark.dataset.prepare_script == Path("benchmarks/gdpval/prepare_harness_neutral.py")
    assert environment["agent_server"]["name"] == benchmark.agent_name
    assert environment["resources_tool_transports"] == ["mcp"]
    assert resources["expose_tools_over_mcp"] is True
    assert resources["allowed_agents"] == [agent_type]
    # The judges and the multi-stage driver come from config.yaml unchanged.
    assert config["rollout_collection_driver"] == (
        "resources_servers.gdpval.multistage_orchestrator:run_rollout_collection"
    )
    agents = [name for name, value in config.items() if isinstance(value, dict) and "responses_api_agents" in value]
    assert agents == [benchmark.agent_name]
    # gym env start and gym env prefetch read the first key of every top-level mapping.
    assert all(value for value in config.values() if isinstance(value, dict))
