# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from copy import deepcopy
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

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


def test_converter_preserves_identity_and_verifier_fields_without_mutation():
    row = _row()
    original = deepcopy(row)
    task = materialize_single_agent_task(row, taskset="swe:test")
    assert task.task_id.task_id == "problem-1"  # Legacy adapter precedence, not instance_id first.
    assert task.task_id.taskset == "swe:test"
    assert task.task_input.responses_create_params.temperature == 0.4
    assert task.task_input.task_data == {
        key: row[key] for key in ("problem_id", "instance_id", "run_script", "verifier_metadata")
    }
    assert row == original


def test_fallback_identity_does_not_change_when_prompt_changes():
    row = {"responses_create_params": {"input": "first"}}
    first = materialize_single_agent_task(row, taskset="test", task_index=5)
    row["responses_create_params"]["input"] = "edited"
    assert materialize_single_agent_task(row, taskset="test", task_index=5).task_id == first.task_id
    with pytest.raises(ValueError, match="task_index"):
        materialize_single_agent_task(row, taskset="test")
    with pytest.raises(ValueError, match="already materialized"):
        materialize_single_agent_task(first.model_dump(mode="json"), taskset="test")


def test_legacy_adapter_uses_same_converter():
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
    assert request.task == materialize_single_agent_task(row, taskset="resources")
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
