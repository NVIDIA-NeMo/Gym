# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
from pathlib import Path

from omegaconf import OmegaConf

from benchmarks.swebench.pro.materialize_single_agent_tasks import materialize_row
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


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


def test_materializer_cli_defaults_route_through_hermes_environment(tmp_path: Path) -> None:
    repo = Path(__file__).parents[2]
    source = tmp_path / "prepared.jsonl"
    target = tmp_path / "tasks.jsonl"
    row = {
        "instance_id": "instance-1",
        "responses_create_params": {"input": "fix it"},
        "run_script": "benchmark-owned verifier script",
        "agent_ref": {"name": "previous-agent"},
        "_ng_task_index": 99,
    }
    source.write_text(json.dumps(row) + "\n")
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
    task = json.loads(target.read_text())
    assert task["task_id"] == {"taskset": "swebench_pro", "task_id": "instance-1"}
    assert task["task_input"] == {
        "responses_create_params": row["responses_create_params"],
        "task_data": {"instance_id": "instance-1", "run_script": row["run_script"]},
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
