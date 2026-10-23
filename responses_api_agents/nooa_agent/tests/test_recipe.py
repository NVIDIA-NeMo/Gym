# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from inspect import signature
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


def test_swe_recipe_uses_borrowed_sandbox_and_single_agent_turn_environment() -> None:
    from omegaconf import OmegaConf

    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
    from nemo_gym.global_config import GlobalConfigDictParser
    from nemo_gym.rollout_collection import RolloutCollectionHelper
    from nemo_gym.sandbox import AsyncSandbox
    from resources_servers.swebench_pro.app import SWEBenchProResourcesServerConfig
    from responses_api_agents.nooa_agent.config import NOOAAgentConfig

    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(ROOT / "benchmarks/swebench/pro/nooa.yaml")])
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
    assert resources.datasets[0].jsonl_fpath == "benchmarks/swebench/data/swebench_pro_benchmark.jsonl"
    assert resources.datasets[0].prepare_script == "benchmarks/swebench/pro/prepare.py"
    assert resources.get("task_network_policy") is None
    resource_config = SWEBenchProResourcesServerConfig(
        name=environment.resources_server.name,
        host="localhost",
        port=8002,
        **OmegaConf.to_container(resources, resolve=True),
    )
    ttl = resource_config.sandbox_config["ttl_s"]
    # Patch extraction and patch-file removal each use AsyncSandbox.exec's default timeout.
    command_timeout = signature(AsyncSandbox.exec).parameters["timeout_s"].default
    assert isinstance(command_timeout, (int, float))
    assert resource_config.verification_stop_timeout is not None
    assert environment.default_episode_timeout_seconds is not None
    # SingleAgentTurn registers Resources, agent, and post-verification Resources retry callbacks.
    final_cleanup_budget = 3 * environment.cleanup_timeout_seconds
    assert ttl > (
        agent.sandbox_runner_timeout_seconds
        + 2 * command_timeout
        + resource_config.verification_stop_timeout
        + final_cleanup_budget
    )
    # The episode deadline bounds the caller's wait for setup and patch download, which has no separate timeout.
    assert ttl > environment.default_episode_timeout_seconds + final_cleanup_budget
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
