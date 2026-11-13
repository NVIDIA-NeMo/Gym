# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""AgentArena's configs: config.yaml runs each row as an agent_arena Environment Server episode, config_legacy_agent.yaml
keeps the agent's /run behind the legacy_agent relay. Both bind the same agent, rows and scoring settings."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_gym.config import AGENT_SESSION_AGENT_TYPES, GlobalConfigDictParser
from nemo_gym.environment import BenchmarkConfig


_BENCHMARK_DIR = Path(__file__).resolve().parents[1]


def _resolve(path: Path) -> dict:
    initial = {
        "config_paths": [str(path), "responses_api_models/vllm_model/configs/vllm_model.yaml"],
        **{f"{role}_base_url": "http://unused/v1" for role in ("policy", "judge")},
        **{f"{role}_api_key": "dummy" for role in ("policy", "judge")},  # pragma: allowlist secret
        **{f"{role}_model_name": "test" for role in ("policy", "judge")},
    }
    config = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=OmegaConf.create(initial))
    return OmegaConf.to_container(config, resolve=True)


def _environment(config: dict) -> tuple[str, dict]:
    """The one Environment Server in front of the benchmark's agent: a second would make its rows ambiguous."""
    (server,) = [
        (server_type, server)
        for value in config.values()
        if isinstance(value, dict)
        for server_type, server in (value.get("environment_servers") or {}).items()
        if server["agent_server"]["name"] == "agent_arena_benchmark_agent"
    ]
    return server


@pytest.mark.parametrize(
    ("config_name", "server_type", "resources_config"),
    [
        ("config.yaml", "agent_arena", "agentic_turn_simulation.yaml"),
        ("config_legacy_agent.yaml", "legacy_agent", "agentic_turn_simulation_legacy_agent.yaml"),
    ],
)
def test_each_config_binds_the_same_agent_through_its_environment_server(
    config_name: str, server_type: str, resources_config: str
) -> None:
    path = _BENCHMARK_DIR / config_name
    # The judge settings are runtime values (++judge_base_url and so on): the listing tolerates them unset.
    benchmark = BenchmarkConfig.from_config_path(path, strict=False)
    config = _resolve(path)
    assert benchmark.agent_name == "agent_arena_benchmark_agent"
    assert benchmark.dataset.jsonl_fpath == Path("benchmarks/agent_arena/data/agent_arena_benchmark.jsonl")
    assert benchmark.dataset.num_repeats == 4 and benchmark.dataset.prompt_config is None
    assert OmegaConf.load(path).config_paths == [
        f"resources_servers/agentic_turn_simulation/configs/{resources_config}"
    ]

    found_type, server = _environment(config)
    assert found_type == server_type
    assert server["agent_server"]["name"] == "agent_arena_benchmark_agent"
    (agent_type,) = config["agent_arena_benchmark_agent"]["responses_api_agents"]
    assert agent_type == "agentic_turn_simulation_agent" and agent_type in AGENT_SESSION_AGENT_TYPES
    if server_type == "agent_arena":
        # The episode scores with this resources server's settings; gym eval reverify re-scores through it.
        assert server["resources_server"]["name"] == "agentic_turn_simulation_resources_server"
    # The relay's agent scores through the same resources server.
    assert config["agent_arena_benchmark_agent"]["responses_api_agents"][agent_type]["resources_server"]["name"] == (
        "agentic_turn_simulation_resources_server"
    )
