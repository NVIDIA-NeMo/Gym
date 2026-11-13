# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Web resource servers share one dependency-light, behavior-identical schema."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from nemo_gym.data.task_data import load_task_data_schema, validate_jsonl_rows
from nemo_gym.web.models import WebTask
from resources_servers.visual_browser.task_data import TaskData as VisualTaskData
from resources_servers.webarena_browser.task_data import TaskData as WebArenaTaskData
from resources_servers.webvoyager_judge.models import WebVoyagerJudgeRequest, WebVoyagerStandardVerifyRequest
from resources_servers.webvoyager_judge.task_data import TaskData as JudgeTaskData


REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    "component",
    [
        "resources_servers/visual_browser",
        "resources_servers/webarena_browser",
        "resources_servers/webvoyager_judge",
    ],
)
def test_web_resource_example_rows_match_task_data_schema(component: str) -> None:
    component_dir = REPO_ROOT / component
    adapter = load_task_data_schema(component_dir)
    example = component_dir / "data/example.jsonl"

    assert adapter is not None
    report = validate_jsonl_rows(component_dir.name, adapter, str(example), example.read_text().splitlines())
    assert report.rows > 0
    assert report.clean, report.summary()


def test_web_task_schema_rejects_an_unknown_benchmark() -> None:
    component_dir = REPO_ROOT / "resources_servers/visual_browser"
    adapter = load_task_data_schema(component_dir)
    row = json.loads((component_dir / "data/example.jsonl").read_text().splitlines()[0])
    row["web_task"]["benchmark"] = "not-a-web-benchmark"

    report = validate_jsonl_rows(component_dir.name, adapter, "invalid.jsonl", [json.dumps(row)])

    assert report.error_rows == 1


@pytest.mark.parametrize("field", ["screenshots", "page_urls"])
@pytest.mark.parametrize("count", [0, 200])
def test_judge_evidence_boundary_is_accepted_by_dataset_and_runtime(field: str, count: int) -> None:
    row = json.loads((REPO_ROOT / "resources_servers/webvoyager_judge/data/example.jsonl").read_text().splitlines()[0])
    row[field] = [f"evidence-{index}" for index in range(count)]

    dataset = JudgeTaskData.model_validate(row)
    standard = WebVoyagerStandardVerifyRequest.model_validate(row)
    evidence = WebVoyagerJudgeRequest.model_validate(
        {"task": WebTask.model_validate(row["web_task"]), field: row[field]}
    )

    assert len(getattr(dataset, field)) == count
    assert len(getattr(standard, field)) == count
    assert len(getattr(evidence, field)) == count
    component_dir = REPO_ROOT / "resources_servers/webvoyager_judge"
    adapter = load_task_data_schema(component_dir)
    assert adapter is not None
    report = validate_jsonl_rows(component_dir.name, adapter, "boundary.jsonl", [json.dumps(row)])
    assert report.clean, report.summary()


@pytest.mark.parametrize("field", ["screenshots", "page_urls"])
def test_judge_evidence_overflow_is_rejected_by_dataset_and_runtime(field: str) -> None:
    row = json.loads((REPO_ROOT / "resources_servers/webvoyager_judge/data/example.jsonl").read_text().splitlines()[0])
    row[field] = [f"evidence-{index}" for index in range(201)]

    with pytest.raises(ValidationError, match="List should have at most 200 items"):
        WebVoyagerStandardVerifyRequest.model_validate(row)
    with pytest.raises(ValidationError, match="List should have at most 200 items"):
        WebVoyagerJudgeRequest.model_validate({"task": WebTask.model_validate(row["web_task"]), field: row[field]})
    with pytest.raises(ValidationError, match="List should have at most 200 items"):
        JudgeTaskData.model_validate(row)
    component_dir = REPO_ROOT / "resources_servers/webvoyager_judge"
    adapter = load_task_data_schema(component_dir)
    assert adapter is not None
    report = validate_jsonl_rows(component_dir.name, adapter, "overflow.jsonl", [json.dumps(row)])
    assert report.rows == report.error_rows == 1


@pytest.mark.parametrize("field", ["screenshots", "page_urls"])
def test_judge_evidence_schema_limits_cannot_drift(field: str) -> None:
    limits = [
        model.model_json_schema()["properties"][field].get("maxItems")
        for model in (JudgeTaskData, WebVoyagerStandardVerifyRequest, WebVoyagerJudgeRequest)
    ]

    assert limits == [200, 200, 200]


def test_webarena_reexports_the_visual_browser_schema() -> None:
    assert WebArenaTaskData is VisualTaskData


@pytest.mark.parametrize("server", ["visual_browser", "webarena_browser"])
@pytest.mark.parametrize("benchmark", ["webarena", "visualwebarena", "webvoyager"])
def test_web_schemas_load_and_preserve_task_fields(server: str, benchmark: str) -> None:
    adapter = load_task_data_schema(REPO_ROOT / "resources_servers" / server)
    row = {
        "web_task": {
            "benchmark": benchmark,
            "task_id": 411,
            "intent": "Find the latest update",
            "start_urls": ["http://benchmark.test/"],
            "sites": ["gitlab"],
            "input_images": ["fixture.png"],
            "task_kwargs": {"max_steps": 100},
            "original_metadata": {"reference_task_id": "411"},
            "site_extension": "retained",
        },
        "source_extension": {"recipe": "reference"},
    }
    parsed = adapter.validate_python(row).model_dump()
    assert parsed["source_extension"] == row["source_extension"]
    for key, value in row["web_task"].items():
        assert parsed["web_task"][key] == value
    assert parsed["web_task"]["runtime_profile"] == "visual_browser"
    assert parsed["web_task"]["action_profile"] == "computer_use"
    assert parsed["web_task"]["seed"] == 0
    with pytest.raises(ValidationError):
        adapter.validate_python({"web_task": {"benchmark": benchmark}})
