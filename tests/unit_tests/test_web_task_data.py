# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from nemo_gym.task_data import load_task_data_schema, validate_jsonl_rows


REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "component",
    [
        "resources_servers/browsergym_web",
        "resources_servers/webvoyager_browser",
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
    component_dir = REPO_ROOT / "resources_servers/browsergym_web"
    adapter = load_task_data_schema(component_dir)
    row = json.loads((component_dir / "data/example.jsonl").read_text().splitlines()[0])
    row["web_task"]["benchmark"] = "not-a-web-benchmark"

    report = validate_jsonl_rows(component_dir.name, adapter, "invalid.jsonl", [json.dumps(row)])

    assert report.error_rows == 1
