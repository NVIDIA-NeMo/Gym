# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict
from nemo_gym.environment import discover_agents
from responses_api_agents.opencode_agent.app import OpenCodeAgentConfig
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgentConfig


def _resolve_config(path: str, monkeypatch: pytest.MonkeyPatch, *, cli_args: tuple[str, ...] = ()) -> DictConfig:
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    monkeypatch.setattr("sys.argv", ["gym", *cli_args])
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=not cli_args,
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


@pytest.mark.parametrize("override", [False, True])
def test_permission_profile_loads_with_cli_override(override: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    cli_args = (
        ("++opencode_agent.responses_api_agents.opencode_agent.opencode_config.permission.edit=deny",)
        if override
        else ()
    )
    config = _resolve_config(
        "responses_api_agents/opencode_agent/configs/opencode_agent.yaml", monkeypatch, cli_args=cli_args
    )
    settings = get_first_server_config_dict(config, "opencode_agent")
    permissions = settings.opencode_config.permission
    assert permissions.edit == ("deny" if override else {"**": "allow"})
    assert permissions["*"] == "allow"
    assert permissions.bash["*"] == "allow"
    for command in (
        "*git submodule add*",
        "*git submodule update*",
        "*git submodule sync*",
        "*git submodule init*",
        "*git archive*--remote*",
        "*git *://*",
        "*git *@*:*",
    ):
        assert permissions.bash[command] == "deny"
    assert settings.opencode_config.tools.webfetch is False
    assert "opencode_version" not in settings


def test_permission_profile_is_not_a_standalone_agent_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    assert list(discover_agents()["opencode_agent"].variants) == ["opencode_agent"]


@pytest.mark.parametrize(
    ("recipe", "prefix"),
    [
        ("deepswe/opencode", "deepswe"),
        ("swebench/verified/opencode", "swebench_verified"),
        ("swebench/multilingual/opencode", "swebench_multilingual"),
        ("swebench/multimodal/opencode", "swebench_multimodal"),
        ("swebench/pro/opencode_sandboxed", "swebench_pro"),
        ("terminal_bench_2_1/opencode", "terminal_bench_2_1"),
    ],
)
def test_existing_benchmarks_keep_legacy_bindings(recipe: str, prefix: str, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _resolve_config(f"benchmarks/{recipe}.yaml", monkeypatch)
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


def _swebench_pro_agent_config(config: DictConfig) -> OpenCodeAgentConfig:
    settings = get_first_server_config_dict(config, "swebench_pro_opencode_agent")
    return OpenCodeAgentConfig.model_validate(
        OmegaConf.to_container(settings, resolve=True) | {"name": "swebench_pro_opencode_agent"}
    )


def test_swebench_pro_runs_opencode_sessions_in_the_resources_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _resolve_config("benchmarks/swebench/pro/opencode.yaml", monkeypatch)
    assert "swebench_pro_opencode_sandboxed_agent" not in config
    assert list(config.swebench_pro_opencode_agent.responses_api_agents) == ["opencode_agent"]
    agent_config = _swebench_pro_agent_config(config)
    assert agent_config.timeout == 10800
    assert agent_config.max_output_tokens == agent_config.context_window == 262144
    # Resources lends its task sandbox; the agent never creates one.
    assert agent_config.sandbox_provider is None
    assert agent_config.opencode_config["permission"]["bash"]["*git fetch*"] == "deny"
    settings = get_first_server_config_dict(config, "swebench_pro_opencode_agent")
    assert [(dataset.name, dataset.num_repeats) for dataset in settings.datasets] == [("swebench_pro", 3)]
    resources = get_first_server_config_dict(config, "swebench_pro_opencode_resources_server")
    assert agent_config.resources_server.name == "swebench_pro_opencode_resources_server"
    assert list(resources.allowed_agents) == ["opencode_agent"]
    # As before, the task sandbox keeps the provider's network settings.
    assert resources.get("task_network_policy") is None
    environment = config.swebench_pro_environment_server.environment_servers
    assert list(environment) == ["single_agent_turn_legacy"]
    assert environment.single_agent_turn_legacy.agent_server.name == "swebench_pro_opencode_agent"
    assert environment.single_agent_turn_legacy.resources_server.name == "swebench_pro_opencode_resources_server"
    # Runs that end on a final model error are graded, as the relay graded every run.
    assert environment.single_agent_turn_legacy.agent_outcome_policy.default_action == "verify"


@pytest.mark.parametrize("staged", [False, True])
def test_nemotron_overlays_bind_swebench_pro_opencode_sessions(staged: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    suffix = "_with_staged" if staged else ""
    config = _resolve_config(f"benchmarks/nemotron_3.5_super/eval_container_config{suffix}.yaml", monkeypatch)
    assert "swebench_pro_opencode_sandboxed_agent" not in config
    # No pre-staged runtime: OpenCode installs its release in each task sandbox.
    assert _swebench_pro_agent_config(config).sandbox_executable_path is None
    settings = get_first_server_config_dict(config, "swebench_pro_opencode_agent")
    assert [(dataset.name, dataset.num_repeats) for dataset in settings.datasets] == (
        [("swebench_pro", 3)] if staged else []
    )
    environment = get_first_server_config_dict(config, "swebench_pro_environment_server")
    assert environment.agent_server.name == "swebench_pro_opencode_agent"
