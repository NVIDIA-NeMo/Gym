# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.server_utils import ServerClient


ROOT = Path(__file__).resolve().parents[3]


def test_swe_shared_preparation_preserves_grading_metadata_through_legacy_adapter(tmp_path: Path) -> None:
    from benchmarks.swebench.pro.prepare import prepare, render_prompt

    task = {
        "instance_id": "repo-1",
        "dockerhub_tag": "fixture:1",
        "problem_statement": "Fix the task.",
        "requirements": "Keep the public interface.",
        "interface": "main()",
        "patch": "private ground truth",
    }
    for name in (
        "run_scripts/repo-1/run_script.sh",
        "run_scripts/repo-1/parser.py",
        "dockerfiles/base_dockerfile/repo-1/Dockerfile",
        "dockerfiles/instance_dockerfile/repo-1/Dockerfile",
    ):
        asset = tmp_path / "upstream" / name
        asset.parent.mkdir(parents=True, exist_ok=True)
        asset.write_text("private grading asset")
    output = prepare(
        dataset=[task],
        upstream_root=tmp_path / "upstream",
        output_fpath=tmp_path / "shared.jsonl",
        image_digest_resolver=lambda tag: "sha256:" + "a" * 64,
    )
    original = output.read_bytes()
    row = json.loads(original)
    adapter = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            name="swebench_pro_nooa",
            host="localhost",
            port=8000,
            entrypoint="app.py",
            cleanup_timeout_seconds=180,
            resources_server={"type": "resources_servers", "name": "swebench_pro_nooa_resources_server"},
            agent_server={"type": "responses_api_agents", "name": "swebench_pro_nooa_agent"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    request = adapter._episode_request_from_row(row | {"_ng_task_index": 0, "_ng_rollout_index": 0})
    assert request.task.task_id.taskset == "swebench_pro_nooa_resources_server"
    assert request.task.task_id.task_id == "repo-1"
    assert request.task.task_input.task_data == {k: v for k, v in row.items() if k != "responses_create_params"}
    params = request.task.task_input.responses_create_params
    assert params.input[0].content == render_prompt(task)
    assert "private ground truth" not in params.model_dump_json()
    assert "private grading asset" not in params.model_dump_json()
    for task_source in (adapter.config.name, adapter.config.agent_server.name, adapter.config.resources_server.name):
        routed = adapter._episode_request_from_row(
            row
            | {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "task_source": task_source,
                "agent_ref": {"name": adapter.config.agent_server.name},
            }
        )
        assert routed.task.task_id.taskset == task_source
        assert routed.task.task_input.responses_create_params == params
    with pytest.raises(ValueError, match="does not match configured agent server"):
        adapter._episode_request_from_row(row | {"agent_ref": {"name": "different_agent"}})
    with pytest.raises(ValueError, match="names none of this Environment Server"):
        adapter._episode_request_from_row(row | {"task_source": "different_resources"})
    assert output.read_bytes() == original


def test_swe_recipe_uses_borrowed_sandbox_and_single_agent_turn_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from omegaconf import OmegaConf

    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
    from nemo_gym.cli.env import _configured_servers
    from nemo_gym.global_config import GlobalConfigDictParser
    from nemo_gym.rollout_collection import RolloutCollectionHelper
    from responses_api_agents.nooa_agent.config import NOOAAgentConfig

    for key, value in {
        "POLICY_BASE_URL": "http://model.invalid/v1",
        "POLICY_API_KEY": "fixture",
        "POLICY_MODEL_NAME": "fixture",
        "NOOA_SWE_RUNTIME_WHEELHOUSE": str(tmp_path / "wheelhouse"),
    }.items():
        monkeypatch.setenv(key, value)
    parser = GlobalConfigDictParser()
    paths = [str(ROOT / "benchmarks/swebench/pro/nooa.yaml")]
    _, configs = parser.load_extra_config_paths(paths)
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    environment = SingleAgentTurnEnvironmentServerConfig(
        name="swebench_pro_nooa",
        host="localhost",
        port=8000,
        **OmegaConf.to_container(config.swebench_pro_nooa.environment_servers.single_agent_turn_legacy, resolve=True),
    )
    agent = NOOAAgentConfig(
        name=environment.agent_server.name,
        host="localhost",
        port=8001,
        **OmegaConf.to_container(config[environment.agent_server.name].responses_api_agents.nooa_agent, resolve=True),
    )
    assert agent.nooa.execution_mode == "sandboxed"
    assert agent.num_workers == 1
    assert agent.model_server.name == "policy_model"
    assert agent.max_policy_calls is None
    assert environment.resources_server.name == "swebench_pro_nooa_resources_server"
    assert not environment.resources_tool_transports
    resources = config[environment.resources_server.name].resources_servers.swebench_pro
    assert resources.allowed_agents == ["nooa_agent"]
    assert resources.entrypoint == "app.py"
    assert resources.sandbox_config.provider_options.platform == {"os": "linux", "arch": "amd64"}
    assert resources.task_network_policy.model_server.name == agent.model_server.name
    assert "network_policy" not in resources.sandbox_config.provider_options
    assert agent.runtime_wheelhouse == tmp_path / "wheelhouse"
    assert "sandbox" not in config  # The caller explicitly selects a provider overlay.
    assert resources.datasets[0].jsonl_fpath == "benchmarks/swebench/data/swebench_pro_benchmark.jsonl"
    assert resources.datasets[0].prepare_script == "benchmarks/swebench/pro/prepare.py"
    instances = parser.filter_for_server_instance_configs(config)
    assert {(instance.name, instance.SERVER_TYPE) for instance in instances} == {
        ("swebench_pro_nooa_agent", "responses_api_agents"),
        ("swebench_pro_nooa_resources_server", "resources_servers"),
        ("swebench_pro_nooa", "environment_servers"),
    }
    rows = [{"agent_ref": {"name": agent.name}, "task_source": environment.resources_server.name}]
    RolloutCollectionHelper._validate_agent_pairings(rows, config)
    resources.allowed_agents = ["different_agent"]
    with pytest.raises(ValueError, match="does not accept the agent"):
        RolloutCollectionHelper._validate_agent_pairings(rows, config)
    environments = [instance for instance in instances if instance.SERVER_TYPE == "environment_servers"]
    assert [(instance.name, set(instance.environment_servers)) for instance in environments] == [
        ("swebench_pro_nooa", {"single_agent_turn_legacy"})
    ]
    assert {(server.top_level_path, server.server_type, server.name) for server in _configured_servers(config)} == {
        ("swebench_pro_nooa_resources_server", "resources_servers", "swebench_pro"),
        ("swebench_pro_nooa_agent", "responses_api_agents", "nooa_agent"),
        ("swebench_pro_nooa", "environment_servers", "single_agent_turn_legacy"),
    }


def test_shared_swe_recipe_keeps_default_networking() -> None:
    from omegaconf import OmegaConf

    from nemo_gym.global_config import GlobalConfigDictParser

    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths(
        [str(ROOT / "resources_servers/swebench_pro/configs/swebench_pro.yaml")]
    )
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    resources = config.swebench_pro_resources_server.resources_servers.swebench_pro
    assert resources.get("task_network_policy") is None
    assert "network_policy" not in resources.sandbox_config.provider_options
    assert resources.entrypoint == "app.py"


def test_swe_model_override_drives_the_network_endpoint() -> None:
    from omegaconf import OmegaConf

    from nemo_gym.global_config import GlobalConfigDictParser

    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(ROOT / "benchmarks/swebench/pro/nooa.yaml")])
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    agent = config.swebench_pro_nooa_agent.responses_api_agents.nooa_agent
    agent.model_server.name = "alternate_policy"
    resources = config.swebench_pro_nooa_resources_server.resources_servers.swebench_pro
    assert resources.task_network_policy.model_server.name == "alternate_policy"
    assert resources.task_network_policy.model_server.type == "responses_api_models"


def test_explicit_unrestricted_docker_config_needs_no_opensandbox_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from omegaconf import OmegaConf

    from nemo_gym.global_config import GlobalConfigDictParser
    from responses_api_agents.nooa_agent.config import NOOAAgentConfig

    for name in ["OPENSANDBOX_API_KEY", "OPENSANDBOX_DOMAIN", "NOOA_SWE_RUNTIME_WHEELHOUSE"]:
        monkeypatch.delenv(name, raising=False)
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths(
        [
            str(ROOT / "benchmarks/swebench/pro/nooa.yaml"),
            str(ROOT / "nemo_gym/sandbox/providers/docker/configs/docker.yaml"),
        ]
    )
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    resources = config.swebench_pro_nooa_resources_server.resources_servers.swebench_pro
    resources.task_network_policy = None
    agent = config.swebench_pro_nooa_agent.responses_api_agents.nooa_agent
    agent.runtime_wheelhouse_required = False
    parsed = NOOAAgentConfig(name="agent", host="localhost", port=8001, **OmegaConf.to_container(agent, resolve=True))
    assert parsed.runtime_wheelhouse is None
    assert resources.task_network_policy is None
    assert "docker" in OmegaConf.to_container(config.sandbox, resolve=True)
