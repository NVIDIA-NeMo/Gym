# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from unittest.mock import Mock

import pytest

from nemo_gym.standard_analyses.integration import analyze_collection
from tests.unit_tests.standard_analyses.fixtures import gym_rows


def test_collection_exports_without_raw_rollout_upload(tmp_path, monkeypatch):
    exporter = Mock()
    monkeypatch.setattr("nemo_gym.standard_analyses.integration.get_exporters", lambda: [exporter])
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in gym_rows()))
    directory = analyze_collection(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest) == 1
    assert manifest[0]["namespace"] == "efficiency"
    report = json.loads((directory / manifest[0]["directory"] / "results.json").read_text())
    assert len(report["results"]) == 8
    exporter.export_standard_analyses.assert_called_once()
    exporter.export_rollouts.assert_not_called()


def test_collection_separates_benchmark_cohorts(tmp_path, monkeypatch):
    exporter = Mock()
    monkeypatch.setattr("nemo_gym.standard_analyses.integration.get_exporters", lambda: [exporter])
    rows = gym_rows()
    path = tmp_path / "rollouts.jsonl"
    data = []
    for benchmark in ("benchmark_a", "benchmark_b"):
        data.extend({**row, "_ng_environment_server": benchmark} for row in rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in data))
    directory = analyze_collection(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert len(manifest) == 2
    assert len({item["namespace"] for item in manifest}) == 2
    for call in exporter.export_standard_analyses.call_args_list:
        assert call.args[0].results[0].coverage["rollouts_total"] == 12
        assert call.kwargs["namespace"].startswith("efficiency/cohorts/")


def test_empty_collection_records_no_cohorts(tmp_path, monkeypatch):
    monkeypatch.setattr("nemo_gym.standard_analyses.integration.get_exporters", lambda: [])
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    directory = analyze_collection(path)
    assert json.loads((directory / "manifest.json").read_text()) == []


@pytest.mark.parametrize("identity", ["taskset", "task_source"])
def test_collection_separates_datasets_sharing_an_environment(tmp_path, monkeypatch, identity):
    monkeypatch.setattr("nemo_gym.standard_analyses.integration.get_exporters", lambda: [])
    rows = []
    for dataset in ("dataset_a", "dataset_b"):
        for index, row in enumerate(gym_rows()):
            row["_ng_environment_server"] = "shared_environment"
            if identity == "taskset":
                row["task_id"] = {"taskset": dataset, "task_id": str(index)}
                row["_ng_task_index"] = index
            else:
                row["task_source"] = dataset
            rows.append(row)
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    directory = analyze_collection(path)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert {item["cohort"]["benchmark"] for item in manifest} == {"dataset_a", "dataset_b"}
    assert len({item["namespace"] for item in manifest}) == 2
