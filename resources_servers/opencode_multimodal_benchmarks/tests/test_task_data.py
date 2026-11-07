# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The same schema validates source rows and collated tasks without exposing labels."""

import pytest
from pydantic import TypeAdapter, ValidationError

from benchmarks.opencode_multimodal_benchmarks.prepare import task_row
from nemo_gym.data import TaskDataValidator
from resources_servers.opencode_multimodal_benchmarks.reporting import benchmark_report
from resources_servers.opencode_multimodal_benchmarks.task_data import TaskData


@pytest.mark.parametrize(
    "benchmark", ["VStarBench", "HRBench8K", "BabyVision", "VTCBench", "ZeroBench", "AgentX_GoalAcc"]
)
def test_flat_legacy_and_materialized_tasks(benchmark):
    metadata = {
        "index": "1",
        "benchmark": benchmark,
        "question": "Visible question",
        "answer": "private answer",
        "options": {"A": "cat"},
        "expected_samples": 1,
    }
    row = task_row(metadata | {"prompt": "Visible question", "images": ["input-0.png"]})
    adapter = TypeAdapter(TaskData)
    assert adapter.validate_python(metadata).answer == "private answer"
    validator = TaskDataValidator("opencode_multimodal_benchmarks", adapter, "synthetic.jsonl")
    validator.validate_row(0, row)
    validator.validate_row(1, {"task_id": {"taskset": "visual_tasks", "task_id": "1"}, "task_input": row})
    assert validator.report.clean, validator.report.summary()
    assert "private answer" not in str(row["responses_create_params"])
    with pytest.raises(ValidationError):
        adapter.validate_python(metadata | {"benchmark": "not-a-benchmark"})
    with pytest.raises(ValidationError):
        adapter.validate_python({key: value for key, value in metadata.items() if key != "answer"})


def test_task_schema_retains_unknown_fields_and_json_encoded_references():
    task = TaskData.model_validate(
        {
            "benchmark": "AgentX_GoalAcc",
            "index": "1",
            "question": "q",
            "answer": "a",
            "GT_json": '{"value":"a"}',
            "custom_provenance": "fixture",
        }
    )
    assert task.GT_json == '{"value":"a"}'
    assert task.model_dump()["custom_provenance"] == "fixture"


def test_optional_categories_do_not_block_overall_reporting():
    rows = [{"hit": 1.0}, {"hit": 0.0}]
    assert benchmark_report("VStarBench", rows) == [{"split": "none", "Overall": 0.5}]
    assert benchmark_report("VTCBench", rows) == [{"Overall": 50.0}]
    assert benchmark_report("BabyVision", rows)[0]["acc"] == 50.0
    with pytest.raises(ValueError, match="cycle_category"):
        benchmark_report("HRBench8K", rows)
