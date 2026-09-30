# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

from benchmarks.swebench.pro.materialize_single_agent_tasks import materialize_row
from environment_servers.single_agent_turn.app import SingleAgentTurnEnvironmentServerConfig
from environment_servers.single_agent_turn_legacy.app import SingleAgentTurnLegacyEnvironmentServer
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.server_utils import ServerClient


@pytest.mark.parametrize(
    "task_fields, expected_task_id",
    [
        ({"task_id": "task", "problem_id": "problem", "instance_id": "instance"}, "task"),
        ({"problem_id": "problem", "instance_id": "instance"}, "problem"),
        ({"instance_id": "instance"}, "instance"),
        ({}, "7"),
    ],
)
def test_materializer_and_legacy_adapter_use_same_task_contract(
    task_fields: dict[str, str], expected_task_id: str
) -> None:
    row = {
        **task_fields,
        "responses_create_params": {"input": "fix it", "temperature": 0.4},
        "run_script": "verifier\nscript\n",
        "agent_ref": {"name": "agent"},
        "task_source": "resources",
        "skills_ref": "old-skills",
        "_ng_task_index": 7,
        "_ng_rollout_index": 2,
        "_ng_attempt_index": 1,
    }
    adapter = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            name="environment",
            host="localhost",
            port=8000,
            entrypoint="app.py",
            cleanup_timeout_seconds=10,
            resources_server={"type": "resources_servers", "name": "resources"},
            agent_server={"type": "responses_api_agents", "name": "agent"},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    materialized = materialize_row(row, taskset="resources")
    request = adapter._native_request(row)

    assert materialized == request.task.model_dump(mode="json", exclude_unset=True)
    assert materialized["task_id"] == {"taskset": "resources", "task_id": expected_task_id}
    assert materialized["task_input"]["task_data"] == {**task_fields, "run_script": row["run_script"]}
    assert request.episode_id.attempt == 1


def test_materialize_swe_pro_row_separates_identity_input_and_task_data() -> None:
    row = {
        "agent_ref": {"name": "legacy-agent"},
        "instance_id": "instance-1",
        "repo": "owner/repo",
        "responses_create_params": {"input": "fix it"},
        "task_source": "legacy-source",
    }

    materialized = materialize_row(row, taskset="swebench_pro:test")

    assert materialized == {
        "task_id": {
            "taskset": "swebench_pro:test",
            "task_id": "instance-1",
        },
        "task_input": {
            "responses_create_params": {"input": "fix it"},
            "task_data": {
                "instance_id": "instance-1",
                "repo": "owner/repo",
            },
        },
    }


@pytest.mark.parametrize("task_fields", [{"instance_id": "instance-1"}, {}])
def test_materializer_cli_defaults_route_through_hermes_environment(
    tmp_path: Path, task_fields: dict[str, str]
) -> None:
    repo = Path(__file__).parents[2]
    source = tmp_path / "prepared.jsonl"
    target = tmp_path / "tasks.jsonl"
    row = {
        **task_fields,
        "responses_create_params": {"input": "fix it"},
        "run_script": "benchmark-owned verifier script",
        "agent_ref": {"name": "previous-agent"},
        "_ng_task_index": 99,
    }
    second_row = {"responses_create_params": {"input": "second task"}}
    source.write_text(json.dumps(row) + "\n" + json.dumps(second_row) + "\n")
    original_source = source.read_bytes()
    subprocess.run(
        [
            sys.executable,
            str(repo / "benchmarks/swebench/pro/materialize_single_agent_tasks.py"),
            str(source),
            str(target),
        ],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    task, second_task = [json.loads(line) for line in target.read_text().splitlines()]
    assert source.read_bytes() == original_source
    assert task["task_id"] == {"taskset": "swebench_pro", "task_id": task_fields.get("instance_id", "99")}
    assert second_task["task_id"] == {"taskset": "swebench_pro", "task_id": "1"}
    assert task["task_input"] == {
        "responses_create_params": row["responses_create_params"],
        "task_data": {**task_fields, "run_script": row["run_script"]},
    }
    recipe = OmegaConf.load(repo / "benchmarks/swebench/pro/hermes.yaml")
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(target),
        output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
        environment_routing_mode=recipe.environment_routing_mode,
        environment_server_routes=OmegaConf.to_container(recipe.environment_server_routes),
        num_repeats=1,
    )
    planned_rows = RolloutCollectionHelper._preprocess_raw_rows([(0, json.dumps(task), task)], config)
    assert len(planned_rows) == 1
    assert planned_rows[0]["_ng_environment_server"] == "swebench_pro_hermes"
