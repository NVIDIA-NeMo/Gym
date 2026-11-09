# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The same schema validates source rows and collated tasks without exposing labels."""

import pytest
from pydantic import TypeAdapter, ValidationError

from benchmarks.opencode_multimodal_benchmarks.prepare import task_row
from nemo_gym.data import TaskDataValidator
from resources_servers.opencode_multimodal_benchmarks.reporting import benchmark_report
from resources_servers.opencode_multimodal_benchmarks.task_data import TaskData


@pytest.mark.parametrize("benchmark", ["VStarBench", "HRBench8K", "VTCBench", "ZeroBench"])
def test_flat_legacy_and_materialized_tasks(benchmark, tmp_path):
    image = tmp_path / "input-0.png"
    image.write_bytes(b"image bytes")
    metadata = {
        "index": "1",
        "benchmark": benchmark,
        "question": "Visible question",
        "answer": "A",
        "options": {"A": "cat"},
        "expected_samples": 1,
        "cycle_category": "1",
    }
    row = task_row(metadata | {"prompt": "Visible question", "images": [str(image)]})
    adapter = TypeAdapter(TaskData)
    assert adapter.validate_python(metadata).answer == "A"
    validator = TaskDataValidator("opencode_multimodal_benchmarks", adapter, "synthetic.jsonl")
    validator.validate_row(0, row)
    validator.validate_row(1, {"task_id": {"taskset": "visual_tasks", "task_id": "1"}, "task_input": row})
    assert validator.report.clean, validator.report.summary()
    assert "private answer" not in str(row["responses_create_params"])
    with pytest.raises(ValidationError):
        adapter.validate_python(metadata | {"benchmark": "not-a-benchmark"})
    with pytest.raises(ValidationError):
        adapter.validate_python({key: value for key, value in metadata.items() if key != "answer"})


def test_optional_categories_do_not_block_overall_reporting():
    rows = [{"hit": 1.0}, {"hit": 0.0}]
    assert benchmark_report("VStarBench", rows) == [{"split": "none", "Overall": 0.5}]
    assert benchmark_report("VTCBench", rows) == [{"Overall": 50.0}]
    with pytest.raises(ValueError, match="cycle_category"):
        benchmark_report("HRBench8K", rows)


@pytest.mark.parametrize("field", ["answer", "expected_samples", "options"])
def test_mcq_wire_metadata_rejects_missing_required_fields(field):
    from resources_servers.opencode_multimodal_benchmarks.tests.test_app import request
    from resources_servers.opencode_multimodal_benchmarks.verifier import VisualVerifyRequest

    value = request("VStarBench", "A").model_dump()
    value["verifier_metadata"].pop(field)
    with pytest.raises(ValidationError):
        VisualVerifyRequest.model_validate(value)
