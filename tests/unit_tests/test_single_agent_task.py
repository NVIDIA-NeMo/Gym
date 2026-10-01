# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from pydantic import JsonValue, ValidationError

from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.global_config import GlobalConfigDictParser
from nemo_gym.rollout_collection import (
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _environment_server_for_config_row,
    _native_episode_request_body,
)
from nemo_gym.server_utils import ServerClient
from nemo_gym.single_agent_task import materialize_single_agent_task
from nemo_gym.single_agent_turn_types import SingleAgentTurnRequest
from nemo_gym.train_data_utils import TrainDataProcessor


@pytest.mark.parametrize(
    "identity, expected",
    [
        ({"task_id": 0, "problem_id": "problem", "instance_id": "instance"}, "0"),
        ({"task_id": None, "problem_id": "problem", "instance_id": "instance"}, "problem"),
        ({"problem_id": None, "instance_id": "instance"}, "instance"),
        ({"_ng_task_index": 0}, "0"),
        ({}, "5"),
    ],
)
def test_identity_precedence_and_prompt_edit_stability(identity: dict[str, JsonValue], expected: str) -> None:
    row = {**identity, "responses_create_params": {"input": "original prompt"}}
    task = materialize_single_agent_task(row, taskset="test", task_index=5)
    assert task.task_id.task_id == expected

    row["responses_create_params"] = {"input": "edited prompt"}
    assert materialize_single_agent_task(row, taskset="test", task_index=5).task_id == task.task_id


def test_conversion_preserves_model_and_verifier_fields_without_mutating_source() -> None:
    row = {
        "task_id": "task",
        "responses_create_params": {"input": "fix it", "temperature": 0.0, "instructions": None},
        "run_script": "verifier\nscript\n",
        "verifier_metadata": {"answer": None, "enabled": False},
        "agent_ref": {"name": "agent"},
        "skills_ref": "skills",
        "task_source": "source",
        "_ng_task_index": 7,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 1,
        "_ng_rollout_id": "rollout",
        "_ng_custom_metadata": "collector-only",
    }
    original = deepcopy(row)
    task = materialize_single_agent_task(row, taskset="test").model_dump(mode="json", exclude_unset=True)

    assert task["task_input"] == {
        "responses_create_params": row["responses_create_params"],
        "task_data": {key: row[key] for key in ("task_id", "run_script", "verifier_metadata")},
    }
    assert row == original


@pytest.mark.parametrize("index", [None, -1, True, "5"])
def test_missing_identity_requires_valid_source_index(index: JsonValue) -> None:
    with pytest.raises(ValueError, match="non-negative task_index"):
        materialize_single_agent_task(
            {"_ng_task_index": index, "responses_create_params": {"input": "task"}}, taskset="test"
        )


def test_requires_identity_when_source_index_is_unavailable() -> None:
    with pytest.raises(ValueError, match="task_index"):
        materialize_single_agent_task({"responses_create_params": {"input": "task"}}, taskset="test")


def test_already_materialized_task_is_not_converted_twice() -> None:
    task = materialize_single_agent_task(
        {"task_id": "task", "responses_create_params": {"input": "task"}}, taskset="test"
    )
    with pytest.raises(ValueError, match="already materialized"):
        materialize_single_agent_task(task.model_dump(mode="json"), taskset="test")


def test_validates_response_parameters() -> None:
    with pytest.raises(ValidationError):
        materialize_single_agent_task({"task_id": "task", "responses_create_params": {"input": 123}}, taskset="test")


def _row():
    return {
        "problem_id": "problem-1",
        "instance_id": "instance-1",
        "responses_create_params": {"input": "fix it", "temperature": 0.4},
        "run_script": "verifier\nscript\n",
        "verifier_metadata": {"answer": "expected"},
        "agent_ref": {"name": "old-agent"},
        "skills_ref": "old-skills",
        "task_source": "old-source",
        "_ng_task_index": 7,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 1,
    }


def test_legacy_adapter_preserves_task_identity_and_data():
    row = _row()
    row.update(task_source="resources", agent_ref={"name": "agent"})
    adapter = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            name="environment",
            host="localhost",
            port=1,
            entrypoint="app.py",
            cleanup_timeout_seconds=10,
            resources_server={"type": "resources_servers", "name": "resources"},
            agent_server={"type": "responses_api_agents", "name": "agent"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    request = adapter._native_request(row)
    assert (request.task.task_id.taskset, request.task.task_id.task_id) == ("resources", "problem-1")
    assert request.task.task_input.task_data == {
        "problem_id": "problem-1",
        "instance_id": "instance-1",
        "run_script": "verifier\nscript\n",
        "verifier_metadata": {"answer": "expected"},
    }
    assert request.episode_id.attempt == 1


@pytest.mark.parametrize("benchmark", [False, True])
def test_collation_routes_declared_taskset_without_rewriting_shared_source(tmp_path, benchmark):
    source = tmp_path / "prepared.jsonl"
    rows = [_row(), {"responses_create_params": {"input": "second"}}]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    original = source.read_bytes()
    dataset = {
        "name": "tasks",
        "type": "benchmark" if benchmark else "example",
        "jsonl_fpath": str(source),
        "num_repeats": 2,
    }
    if benchmark:
        dataset["prepare_script"] = "benchmarks/swebench/pro/prepare.py"
    global_config = OmegaConf.create(
        {
            "native": {
                "resources_servers": {
                    "test": {
                        "entrypoint": "app.py",
                        "domain": "coding",
                        "datasets": [dict(dataset, taskset="swe:test")],
                    }
                }
            },
            "flat": {
                "resources_servers": {"test": {"entrypoint": "app.py", "domain": "coding", "datasets": [dataset]}}
            },
        }
    )
    configs = GlobalConfigDictParser().filter_for_server_instance_configs(global_config)
    assert len(configs) == 2
    with pytest.warns(DeprecationWarning, match="stripped legacy agent_ref"):
        paths = TrainDataProcessor()._collate_samples_single_type(dataset["type"], configs, task_data_validation="off")
    native, flat = ([json.loads(line) for line in path.read_text().splitlines()] for path in paths)
    assert len(native) == len(flat) == 4
    assert native[0] == native[1]
    assert native[2] == native[3]
    assert native[2]["task_id"]["task_id"] == "1"
    assert "task_input" not in flat[0]
    assert flat[0]["task_source"] == "flat"
    assert source.read_bytes() == original
    routing = RolloutCollectionConfig(
        input_jsonl_fpath="unused",
        output_jsonl_fpath="unused",
        environment_routing_mode="taskset",
        environment_server_routes={"swe:test": "swe-environment"},
    )
    assert _environment_server_for_config_row(native[0], routing) == "swe-environment"
    assert native[0]["task_input"]["task_data"]["run_script"] == "verifier\nscript\n"
    # Exercise the preparation and file-loading path used by gym eval run, not only the converter.
    output = tmp_path / "collated"
    global_config.update(
        {
            "mode": "train_preparation" if benchmark else "example_validation",
            "output_dirpath": str(output),
            "should_download": False,
            "environment_server_routes": {"swe:test": "swe-environment"},
            "agent": {"responses_api_agents": {"test": {"entrypoint": "app.py"}}},
            "swe-environment": {
                "environment_servers": {
                    "single_agent_turn": {
                        "entrypoint": "app.py",
                        "agent_server": {"type": "responses_api_agents", "name": "agent"},
                        "resources_server": {"type": "resources_servers", "name": "native"},
                    }
                }
            },
        }
    )
    with pytest.warns(DeprecationWarning, match="stripped legacy agent_ref"):
        TrainDataProcessor().run(global_config)
    routing.input_jsonl_fpath = str(output / f"{dataset['type']}.jsonl")
    routing.environment_routing_mode = "agent"  # One batch can contain both declarations.
    loaded = RolloutCollectionHelper()._preprocess_rows_from_config(routing)
    assert len(loaded) == 8
    assert loaded[0]["_ng_environment_server"] == "swe-environment"
    request = SingleAgentTurnRequest.model_validate(_native_episode_request_body(loaded[0]))
    assert request.task.task_input.task_data["run_script"] == "verifier\nscript\n"
    assert loaded[-1]["task_source"] == "flat"
    assert source.read_bytes() == original


@pytest.mark.parametrize("task_index", [100, 200])
@pytest.mark.parametrize("explicit_rollout_id", [False, True])
def test_collation_preserves_shard_and_retry_identity(tmp_path, task_index, explicit_rollout_id):
    row = {
        "responses_create_params": {"input": "same prompt across shards"},
        "_ng_task_index": task_index,
        "_ng_attempt_index": 2,
        "_ng_rollout_index": 0,
    }
    if explicit_rollout_id:
        row["_ng_rollout_id"] = f"shard-{task_index}"
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps(row) + "\n")
    output = tmp_path / "collated"
    TrainDataProcessor().run(
        OmegaConf.create(
            {
                "resources": {
                    "resources_servers": {
                        "test": {
                            "entrypoint": "app.py",
                            "domain": "coding",
                            "datasets": [
                                {
                                    "name": "tasks",
                                    "type": "example",
                                    "jsonl_fpath": str(source),
                                    "taskset": "tasks",
                                }
                            ],
                        }
                    }
                },
                "mode": "example_validation",
                "output_dirpath": str(output),
                "task_data_validation": "off",
                "environment_server_routes": {"tasks": "environment"},
                "agent": {"responses_api_agents": {"test": {"entrypoint": "app.py"}}},
                "environment": {
                    "environment_servers": {
                        "single_agent_turn": {
                            "entrypoint": "app.py",
                            "agent_server": {"type": "responses_api_agents", "name": "agent"},
                            "resources_server": {"type": "resources_servers", "name": "resources"},
                        }
                    }
                },
            }
        )
    )
    prepared = json.loads((output / "example.jsonl").read_text())
    for key, value in row.items():
        if key.startswith("_ng_"):
            assert prepared[key] == value
            assert key not in prepared["task_input"]["task_data"]
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(output / "example.jsonl"),
        output_jsonl_fpath="unused",
        environment_server_routes={"tasks": "environment"},
    )
    (loaded,) = RolloutCollectionHelper()._preprocess_rows_from_config(config)
    assert loaded["_ng_task_index"] == task_index
    request = SingleAgentTurnRequest.model_validate(_native_episode_request_body(loaded))
    assert request.episode_id.rollout_id == (f"shard-{task_index}" if explicit_rollout_id else f"{task_index}-0")
    assert request.episode_id.attempt == 2


def _resources_server(name, datasets, implementation="test"):
    return {
        name: {
            "resources_servers": {implementation: {"entrypoint": "app.py", "domain": "coding", "datasets": datasets}}
        }
    }


def test_prompt_config_is_applied_before_materialization(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"instance_id": "i1", "question": "What is 2+2?"}) + "\n")
    prompt = tmp_path / "prompt.yaml"
    prompt.write_text("system: Be brief.\nuser: 'Q: {question}'\n")
    dataset = {
        "name": "bench",
        "type": "benchmark",
        "jsonl_fpath": str(source),
        "prepare_script": "prepare.py",
        "prompt_config": str(prompt),
        "taskset": "bench",
    }
    configs = GlobalConfigDictParser().filter_for_server_instance_configs(
        OmegaConf.create(_resources_server("rs", [dataset]))
    )
    (path,) = TrainDataProcessor()._collate_samples_single_type("benchmark", configs, task_data_validation="off")
    row = json.loads(path.read_text())
    assert row["task_id"] == {"taskset": "bench", "task_id": "i1"}
    assert row["task_input"]["responses_create_params"]["input"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Q: What is 2+2?"},
    ]


def test_flat_and_taskset_declarations_share_metrics_sidecar(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"responses_create_params": {"input": "x"}}) + "\n")
    dataset = {"name": "d", "type": "example", "jsonl_fpath": str(source)}
    configs = GlobalConfigDictParser().filter_for_server_instance_configs(
        OmegaConf.create(
            {**_resources_server("flat", [dataset]), **_resources_server("native", [dict(dataset, taskset="t")])}
        )
    )
    TrainDataProcessor().validate_samples_and_aggregate_metrics(configs, overwrite_metrics_conflicts=False)
    assert "taskset" not in json.loads((tmp_path / "source_metrics.json").read_text())
    assert not (tmp_path / "source_metrics_conflict.json").exists()


@pytest.mark.parametrize("taskset", [None, "code"])
def test_collation_rejects_misplaced_fields_before_materialization(tmp_path, taskset):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"responses_create_params": {"input": "code"}, "unit_tests": {"inputs": ["1"], "outputs": ["1"]}})
        + "\n"
    )
    dataset = {"name": "code", "type": "example", "jsonl_fpath": str(source), "taskset": taskset, "num_repeats": 2}
    configs = GlobalConfigDictParser().filter_for_server_instance_configs(
        OmegaConf.create(_resources_server("resources", [dataset], implementation="code_gen"))
    )
    with pytest.raises(ValueError, match=r"wire reads from verifier_metadata.*unit_tests \(1 rows\)"):
        TrainDataProcessor()._collate_samples_single_type("example", configs, task_data_validation="error")
