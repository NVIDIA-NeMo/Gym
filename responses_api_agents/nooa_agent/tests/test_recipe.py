# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym.episode_types import MaterializedTask
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput


ROOT = Path(__file__).resolve().parents[3]


def test_swe_preparation_preserves_grading_metadata(tmp_path: Path) -> None:
    from benchmarks.swebench.pro.prepare_nooa import prepare_tasks

    source = tmp_path / "source.jsonl"
    task = {
        "instance_id": "repo-1",
        "patch": "private ground truth",
        "run_script": "pytest",
        "responses_create_params": {"input": [{"role": "user", "content": "fix it"}]},
    }
    source.write_text(json.dumps(task) + "\n")
    output = prepare_tasks(source=source)
    row = MaterializedTask[SingleAgentTurnTaskInput].model_validate_json(output.read_text())
    assert row.task_id.task_id == "repo-1"
    assert row.task_input.task_data == {k: v for k, v in task.items() if k != "responses_create_params"}
    assert "private ground truth" not in row.task_input.responses_create_params.model_dump_json()


def test_swe_baseline_reply_budget_preserves_prompt_and_verifier_data(tmp_path: Path) -> None:
    from benchmarks.swebench.pro.prepare_nooa import prepare_tasks

    source = tmp_path / "source.jsonl"
    task = {
        "instance_id": "repo-1",
        "patch": "private ground truth",
        "responses_create_params": {"input": [{"role": "user", "content": "canonical prompt"}]},
    }
    original = json.dumps(task) + "\n"
    source.write_text(original)
    output = prepare_tasks(source=source, output=tmp_path / "baseline.jsonl", max_output_tokens=32768)
    row = MaterializedTask[SingleAgentTurnTaskInput].model_validate_json(output.read_text())
    assert row.task_input.responses_create_params.max_output_tokens == 32768
    assert row.task_input.responses_create_params.input[0].content == "canonical prompt"
    assert row.task_input.task_data == {"instance_id": "repo-1", "patch": "private ground truth"}
    assert source.read_text() == original
    with pytest.raises(ValueError, match="positive integer"):
        prepare_tasks(source=source, max_output_tokens=0)


@pytest.mark.parametrize("recipe", ["nooa.yaml", "nooa_baseline.yaml"])
@pytest.mark.parametrize("benchagent", [False, True])
def test_swe_recipe_uses_borrowed_sandbox_and_single_agent_turn_environment(
    recipe: str, benchagent: bool, monkeypatch, tmp_path: Path
) -> None:
    from omegaconf import OmegaConf

    from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
    from nemo_gym.global_config import GlobalConfigDictParser
    from responses_api_agents.nooa_agent.config import NOOAAgentConfig

    for key, value in {
        "POLICY_BASE_URL": "http://model.invalid/v1",
        "POLICY_API_KEY": "fixture",
        "POLICY_MODEL_NAME": "fixture",
        "NOOA_SWE_BASELINE_RUN_DIR": str(tmp_path),
        "NOOA_SWE_RUNTIME_WHEELHOUSE": str(tmp_path / "wheelhouse"),
    }.items():
        monkeypatch.setenv(key, value)
    parser = GlobalConfigDictParser()
    paths = [str(ROOT / "benchmarks/swebench/pro" / recipe)]
    if benchagent:
        paths.append(str(ROOT / "benchmarks/nooa_baselines/benchagent-swe.yaml"))
    _, configs = parser.load_extra_config_paths(paths)
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    environment = SingleAgentTurnEnvironmentServerConfig(
        name="swebench_pro_nooa",
        host="localhost",
        port=8000,
        **OmegaConf.to_container(config.swebench_pro_nooa.environment_servers.single_agent_turn, resolve=True),
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
    assert resources.task_network_policy.model_server.name == agent.model_server.name
    assert "network_policy" not in resources.sandbox_config.provider_options
    assert agent.runtime_wheelhouse == tmp_path / "wheelhouse"
    assert "opensandbox" in config.sandbox


@pytest.mark.parametrize(
    "recipe,resource_name,resource_type,agent_name",
    [
        (
            "benchmarks/terminal_bench_2_1/nooa.yaml",
            "terminal_bench_2_1_nooa_resources_server",
            "terminal_bench_2_1",
            "terminal_bench_2_1_nooa_agent",
        ),
        ("benchmarks/gdpval/nooa.yaml", "gdpval_nooa_resources_server", "gdpval", "gdpval_nooa_agent"),
        (
            "resources_servers/swebench_pro/configs/swebench_pro.yaml",
            "swebench_pro_resources_server",
            "swebench_pro",
            None,
        ),
    ],
)
def test_network_restriction_and_offline_runtime_do_not_change_other_combinations(
    recipe: str, resource_name: str, resource_type: str, agent_name: str | None
) -> None:
    from omegaconf import OmegaConf

    from nemo_gym.global_config import GlobalConfigDictParser

    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(ROOT / recipe)])
    config = OmegaConf.merge(*configs)
    parser._recursively_swap_keys(config)
    resources = config[resource_name].resources_servers[resource_type]
    assert resources.get("task_network_policy") is None
    assert "network_policy" not in resources.get("sandbox_config", {}).get("provider_options", {})
    if agent_name is not None:
        agent = config[agent_name].responses_api_agents.nooa_agent
        assert agent.get("runtime_wheelhouse") is None
