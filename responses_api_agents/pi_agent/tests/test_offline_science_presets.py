# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The HLE and APEX Shortlist Pi presets, composed on profiles/offline_science.yaml as `gym eval` loads them."""

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict
from responses_api_agents.pi_agent.app import PiAgentConfig


IMAGE = "registry.example.invalid/science:1"


def _resolve_config(path: str, monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    monkeypatch.setenv("PI_SANDBOX_IMAGE", IMAGE)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "unused")
    monkeypatch.setenv("TAVILY_API_KEY", "unused")
    monkeypatch.setattr("sys.argv", ["gym"])
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": ["responses_api_models/vllm_model/configs/vllm_model.yaml", path],
                    "policy_base_url": "http://unused.invalid/v1",
                    "policy_api_key": "unused",
                    "policy_model_name": "test-model",
                }
            ),
        )
    )


@pytest.mark.parametrize(
    ("preset", "prefix", "environment", "resources", "network_access", "absolute_ip_server_types"),
    [
        (
            "benchmarks/hle/pi.yaml",
            "hle",
            "hle_equivalence_llm_judge_environment_server",
            "hle_equivalence_llm_judge_resources_server",
            "model_only",
            ["responses_api_models"],
        ),
        (
            "benchmarks/hle/pi_search.yaml",
            "hle",
            "hle_equivalence_llm_judge_environment_server",
            "hle_equivalence_llm_judge_resources_server",
            "model_and_tools",
            ["responses_api_models", "resources_servers"],
        ),
        (
            "benchmarks/apex_shortlist/pi.yaml",
            "apex_shortlist",
            "apex_shortlist_math_with_judge_environment_server",
            "apex_shortlist_math_with_judge_resources_server",
            "model_only",
            ["responses_api_models"],
        ),
    ],
    ids=["hle", "hle_search", "apex_shortlist"],
)
def test_preset_runs_pi_agent_sessions_through_single_agent_turn_legacy(
    preset: str,
    prefix: str,
    environment: str,
    resources: str,
    network_access: str,
    absolute_ip_server_types: list[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _resolve_config(preset, monkeypatch)
    agent_name = f"{prefix}_pi_agent"
    assert OmegaConf.missing_keys(config) == set()
    assert list(config[agent_name].responses_api_agents) == ["pi_agent"]
    settings = get_first_server_config_dict(config, agent_name)
    agent_config = PiAgentConfig.model_validate(OmegaConf.to_container(settings, resolve=True) | {"name": agent_name})
    assert agent_config.resources_server.name == resources
    assert agent_config.network_access == network_access
    assert agent_config.sandbox_config["image"] == IMAGE
    # The seed creates the sandbox and waits up to ready_timeout_s for it; the run budget starts after that.
    ttl_s = agent_config.sandbox_config["ttl_s"]
    assert ttl_s >= agent_config.timeout + agent_config.sandbox_config["ready_timeout_s"]
    # The pinned Pi in the image, with no output cap on its requests.
    assert agent_config.pi_version == "0.85.1"
    assert agent_config.output_token_policy == "remaining_context"
    # Only the server types a sandbox must reach take the host address, and the head server stays on loopback. The
    # judge stays there too, except in hle_search: Tavily is a Resources server, so every Resources server leaves.
    assert list(config.absolute_ip_server_types) == absolute_ip_server_types
    assert config.head_server.host == "127.0.0.1"
    env = get_first_server_config_dict(config, environment)
    assert list(config[environment].environment_servers) == ["single_agent_turn_legacy"]
    assert env.agent_server.name == agent_name
    assert env.resources_server.name == resources
    # Neither the simple agent nor the deprecated relay agent is left to route rows.
    for name in (
        f"{prefix}_equivalence_llm_judge_simple_agent",
        f"{prefix}_math_with_judge_simple_agent",
        f"{prefix}_pi_sandboxed_agent",
    ):
        assert config.get(name) is None
