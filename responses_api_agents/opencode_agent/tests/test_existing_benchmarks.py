# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgentConfig


def _resolve_config(path: str, monkeypatch: pytest.MonkeyPatch) -> DictConfig:
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
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
    ("recipe", "prefix"),
    [
        ("deepswe", "deepswe"),
        ("swebench/verified", "swebench_verified"),
        ("swebench/multilingual", "swebench_multilingual"),
        ("swebench/pro", "swebench_pro"),
        ("terminal_bench_2_1", "terminal_bench_2_1"),
    ],
)
def test_existing_benchmarks_keep_legacy_bindings(recipe: str, prefix: str, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _resolve_config(f"benchmarks/{recipe}/opencode.yaml", monkeypatch)
    agent_name = f"{prefix}_opencode_sandboxed_agent"
    assert f"{prefix}_opencode_agent" not in config
    assert list(config[agent_name].responses_api_agents) == ["opencode_sandboxed_agent"]
    settings = get_first_server_config_dict(config, agent_name)
    agent_config = OpenCodeSandboxedAgentConfig.model_validate(
        OmegaConf.to_container(settings, resolve=True) | {"name": agent_name}
    )
    assert agent_config.resources_server.name == f"{prefix}_opencode_resources_server"
    resources = get_first_server_config_dict(config, agent_config.resources_server.name)
    assert "opencode_sandboxed_agent" in resources.allowed_agents
    assert settings.entrypoint == "app.py"
    assert settings.datasets
    environment = get_first_server_config_dict(config, f"{prefix}_environment_server")
    assert environment.agent_server.name == agent_name


@pytest.mark.parametrize("staged", [False, True])
def test_nemotron_overlays_still_bind_existing_agents(staged: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    suffix = "_with_staged" if staged else ""
    config = _resolve_config(f"benchmarks/nemotron_3.5_super/eval_container_config{suffix}.yaml", monkeypatch)
    for prefix, dataset, repeats in (
        ("swebench_verified", "swebench_verified", 3),
        ("swebench_multilingual", "swebench_multilingual", 3),
        ("swebench_pro", "swebench_pro", 3),
        ("deepswe", "deepswe_v1_1", 5),
    ):
        agent_name = f"{prefix}_opencode_sandboxed_agent"
        assert f"{prefix}_opencode_agent" not in config
        assert list(config[agent_name].responses_api_agents) == ["opencode_sandboxed_agent"]
        settings = get_first_server_config_dict(config, agent_name)
        assert settings.resources_server.name == f"{prefix}_opencode_resources_server"
        assert settings.remote_opencode_binary_path == "/mnt/s3-data/data/bxyu/opencode/opencode-linux-x64"
        if staged:
            assert [(item.name, item.num_repeats) for item in settings.datasets] == [(dataset, repeats)]
        else:
            assert settings.datasets == []
        environment = get_first_server_config_dict(config, f"{prefix}_environment_server")
        assert environment.agent_server.name == agent_name
