# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from nemo_gym.benchmarks import BenchmarkConfig
from nemo_gym.cli.eval import PrepareBenchmarkConfig, _prepare_datasets
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.episode_types import MaterializedTask
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput
from nemo_gym.train_data_utils import TrainDataProcessor
from responses_api_agents.nooa_agent.config import NOOAAgentConfig


ROOT = Path(__file__).resolve().parents[3]


def _prepare_fixture(benchmark: str, directory: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    output = directory / "native.jsonl"
    if benchmark == "terminal_bench_2_1":
        from benchmarks.terminal_bench_2_1 import prepare_nooa

        task = directory / "tasks/fixture"
        (task / "tests").mkdir(parents=True)
        (task / "instruction.md").write_text("Fix the task.\n")
        (task / "tests/test.sh").write_text("exit 0\n")
        (task / "task.toml").write_text(
            '[task]\nname = "fixture"\n[environment]\ndocker_image = "fixture:1"\n'
            "[agent]\ntimeout_sec = 900\n[verifier]\ntimeout_sec = 450\n"
        )
        monkeypatch.setattr(prepare_nooa, "TASK_COUNT", 1)
        monkeypatch.setattr(
            prepare_nooa.subprocess, "check_output", Mock(side_effect=[prepare_nooa.TASK_REVISION + "\n", ""])
        )
        return {"repository_path": directory, "output": output}
    source = directory / "source.jsonl"
    if benchmark == "swebench/pro":
        row = {
            "instance_id": "fixture",
            "responses_create_params": {"input": "Fix the task."},
            "patch": "PRIVATE VERIFIER DATA",
            "repo": "fixture/repo",
            "base_commit": "base-revision",
            "problem_statement": "Fix the task.",
            "fail_to_pass": ["test_fix"],
            "pass_to_pass": [],
            "selected_test_files_to_run": ["test_fixture.py"],
            "dockerhub_tag": "fixture:1",
            "run_script": "pytest test_fixture.py",
            "parser_script": "parse-results",
            "evaluator_commit": "evaluator-revision",
            "dataset_revision": "dataset-revision",
        }
    else:
        row = {
            "task_id": "fixture",
            "prompt": "Create a document.",
            "responses_create_params": {"input": []},
            "reference_files": [],
            "reference_file_urls": [],
            "rubric_pretty": "PRIVATE VERIFIER DATA",
        }
    source.write_text(json.dumps(row) + "\n")
    return {"source": source, "output": output}


@pytest.mark.parametrize(
    ("benchmark", "taskset", "environment"),
    [
        ("swebench/pro", "swebench-pro-nooa", "swebench_pro_nooa"),
        ("terminal_bench_2_1", "terminal-bench-2.1-nooa", "terminal_bench_2_1_nooa"),
        ("gdpval", "gdpval-nooa", "gdpval_nooa"),
    ],
)
def test_benchmark_recipe_prepares_collates_and_dispatches_native_rows(
    benchmark: str, taskset: str, environment: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    recipe = ROOT / "benchmarks" / benchmark / "nooa.yaml"
    parser = GlobalConfigDictParser()
    _, configs = parser.load_extra_config_paths([str(recipe)])
    config = OmegaConf.merge(GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT, *configs)
    parser._recursively_swap_keys(config)
    discovered = BenchmarkConfig.from_config_path(recipe, strict=False)
    assert discovered is not None
    assert discovered.dataset.taskset is None
    assert config.environment_server_routes[taskset] == environment
    assert discovered.agent_name == f"{environment}_agent"
    assert {
        name: set(block.environment_servers)
        for name, block in config.items()
        if isinstance(block, type(config)) and "environment_servers" in block
    } == {environment: {"single_agent_turn"}}
    agent_config = NOOAAgentConfig(
        name=discovered.agent_name,
        host="127.0.0.1",
        port=8001,
        **OmegaConf.to_container(config[discovered.agent_name].responses_api_agents.nooa_agent, resolve=True),
    )
    environment_config = config[environment].environment_servers.single_agent_turn
    assert isinstance(agent_config.resources_server, ResourcesServerRef)
    assert agent_config.resources_server.name == environment_config.resources_server.name
    resources_name = agent_config.resources_server.name
    resources_impl = "swebench_pro" if benchmark == "swebench/pro" else benchmark
    resources = config[resources_name].resources_servers[resources_impl]
    assert resources.datasets[0].jsonl_fpath == str(discovered.dataset.jsonl_fpath)
    assert "datasets" not in config[discovered.agent_name].responses_api_agents.nooa_agent

    prepare_args = _prepare_fixture(benchmark, tmp_path, monkeypatch)
    discovered.dataset.jsonl_fpath = prepare_args["output"]
    _prepare_datasets([discovered], PrepareBenchmarkConfig(prepare_script_args=prepare_args))
    original = prepare_args["output"].read_bytes()
    prepared = MaterializedTask[SingleAgentTurnTaskInput].model_validate_json(original)
    assert prepared.task_id.taskset == taskset
    assert "PRIVATE VERIFIER DATA" not in prepared.task_input.responses_create_params.model_dump_json()

    resources.datasets[0].jsonl_fpath = str(prepare_args["output"])
    instances = parser.filter_for_server_instance_configs(config)
    resource_instance = next(instance for instance in instances if instance.name == resources_name)
    agent_instance = next(instance for instance in instances if instance.name == discovered.agent_name)
    assert TrainDataProcessor._owning_resources_server_impl(agent_instance, instances) == resources_impl
    validator = TrainDataProcessor._task_data_validator_for(
        resource_instance, resource_instance.datasets[0], instances
    )
    assert validator is not None
    assert validator.report.server_name == resources_impl
    validator.validate_row(0, json.loads(original))
    assert validator.report.error_rows == 0
    invalid_row = json.loads(original)
    required_field = {"swebench/pro": "repo", "terminal_bench_2_1": "task_name", "gdpval": "task_id"}[benchmark]
    del invalid_row["task_input"]["task_data"][required_field]
    validator.validate_row(1, invalid_row)
    assert validator.report.error_rows == 1
    config.update({"mode": "train_preparation", "output_dirpath": str(tmp_path / "collated")})
    TrainDataProcessor().run(config)
    (loaded,) = RolloutCollectionHelper()._preprocess_rows_from_config(
        RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "collated/benchmark.jsonl"),
            output_jsonl_fpath=str(tmp_path / "unused.jsonl"),
            environment_server_routes={taskset: config.environment_server_routes[taskset]},
        )
    )
    assert loaded["_ng_environment_server"] == environment
    assert loaded["task_id"] == prepared.task_id.model_dump(mode="json")
    assert loaded["task_input"] == prepared.task_input.model_dump(mode="json", exclude_unset=True)
    assert prepare_args["output"].read_bytes() == original
