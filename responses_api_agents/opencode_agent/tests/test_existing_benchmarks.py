# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from nemo_gym.config import GlobalConfigDictParser, GlobalConfigDictParserConfig, get_first_server_config_dict
from nemo_gym.core import EpisodeId
from nemo_gym.data import materialize_task
from nemo_gym.environment import discover_agents
from nemo_gym.server_runtime import SingleAgentTurnRequest
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


def test_multimodal_uses_sessions_and_preserves_images(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _resolve_config("benchmarks/swebench/multimodal/opencode.yaml", monkeypatch)
    assert config.environment_routing_mode == "taskset"
    environment_name = config.environment_server_routes["swebench:multimodal"]
    assert list(config[environment_name].environment_servers) == ["single_agent_turn"]
    environment = SingleAgentTurnEnvironmentServerConfig.model_validate(
        OmegaConf.to_container(get_first_server_config_dict(config, environment_name), resolve=True)
        | {"name": environment_name}
    )
    agent_name = environment.agent_server.name
    assert list(config[agent_name].responses_api_agents) == ["opencode_agent"]
    agent = OpenCodeAgentConfig.model_validate(
        OmegaConf.to_container(get_first_server_config_dict(config, agent_name), resolve=True) | {"name": agent_name}
    )
    assert agent.resources_server is None
    assert agent.num_workers == 1
    assert environment.resources_tool_transports == []
    assert not get_first_server_config_dict(config, agent_name).datasets
    resources_name = environment.resources_server.name
    resources = get_first_server_config_dict(config, resources_name)
    assert list(resources.allowed_agents) == ["opencode_agent"]
    assert resources.num_workers == 1
    assert resources.sandbox_config.ttl_s > environment.default_episode_timeout_seconds
    assert [(item.name, item.taskset, item.num_repeats) for item in resources.datasets] == [
        ("swebench_multimodal", "swebench:multimodal", 3)
    ]
    image = "data:image/png;base64,iVBORw0KGgo="
    content = [
        {"type": "input_text", "text": "Repair the chart."},
        {"type": "input_image", "image_url": image, "detail": "high"},
    ]
    row = {
        "instance_id": "chartjs__Chart.js-1",
        "image_assets": {
            "test_patch": [
                {
                    "path": "test/fixtures/reference.png",
                    "url": "https://example.com/reference.png",
                    "content_base64": "aW1hZ2U=",
                }
            ]
        },
        "responses_create_params": {"input": [{"role": "user", "content": content}]},
    }
    task = materialize_task(row, taskset=resources.datasets[0].taskset)
    request = SingleAgentTurnRequest(episode_id=EpisodeId(rollout_id="multimodal-test", attempt=0), task=task)
    assert request.task.task_input.responses_create_params.input[0].content == content
    assert request.task.task_input.task_data["instance_id"] == row["instance_id"]
    assert request.task.task_input.task_data["image_assets"] == row["image_assets"]


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
