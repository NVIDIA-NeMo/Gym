# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepared source rows must survive Gym's real typed-task collation."""

import json
from pathlib import Path

from benchmarks.claweval.prepare_sessions import convert
from environment_servers.claweval.app import ClawEvalRequest
from nemo_gym.data import materialize_task


def test_exported_references_materialize_without_a_harness_binding(tmp_path):
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "tasks.jsonl"
    convert(root / "responses_api_agents/claweval_agent/data/example.jsonl", output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == 5
    for index, row in enumerate(rows):
        assert "agent_ref" not in row
        task = materialize_task(row, taskset="claweval", task_index=index)
        request = ClawEvalRequest.model_validate({"episode_id": {"rollout_id": f"row-{index}"}, "task": task})
        assert request.task.task_id.task_id == row["task_id"]
        assert request.task.task_input.prompt == row["prompt"]
