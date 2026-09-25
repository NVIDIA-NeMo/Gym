# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import stat

import pytest

from benchmarks.gdpval.prepare import prepare, prepare_gdpval_csv


def _source_row(**overrides):
    row = {
        "task_id": "GDP-00001",
        "sector": "Technology",
        "occupation": "Software Engineer",
        "prompt": "Create the requested analysis.\nPreserve this newline. ",
        "reference_files": json.dumps(["reference_files/GDP-00001/input data.csv"]),
        "reference_file_urls": json.dumps(["https://example.test/GDP-00001/input%20data.csv"]),
        "deliverable_files": json.dumps(["answer.docx"]),
        "deliverable_file_urls": json.dumps(["https://example.test/GDP-00001/answer.docx"]),
        "rubric_pretty": "[+2] Produce the requested analysis.",
        "rubric_json": json.dumps([{"criterion": "Produce the requested analysis.", "score": 2}]),
    }
    row.update(overrides)
    return row


def _write_csv(path, rows, *, fieldnames=None):
    fieldnames = fieldnames or list(_source_row())
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_gdpval_csv_emits_policy_safe_gym_rows(tmp_path):
    source = tmp_path / "gdpval_csv.csv"
    output = tmp_path / "gdpval_benchmark.jsonl"
    rows = [
        _source_row(),
        _source_row(
            task_id="GDP-00002",
            prompt="Reference-free task with Unicode: résumé",
            reference_files="[]",
            reference_file_urls="[]",
        ),
    ]
    _write_csv(source, rows)

    assert prepare_gdpval_csv(source, output) == output

    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [record["task_id"] for record in records] == ["GDP-00001", "GDP-00002"]
    assert records[0]["prompt"] == rows[0]["prompt"]
    assert records[1]["prompt"] == rows[1]["prompt"]
    assert records[1]["reference_files"] == []
    assert records[1]["reference_file_urls"] == []
    assert set(records[0]) == {
        "responses_create_params",
        "task_id",
        "sector",
        "occupation",
        "prompt",
        "reference_files",
        "reference_file_urls",
        "rubric_json",
        "rubric_pretty",
    }
    assert records[0]["responses_create_params"] == {"input": []}
    assert "deliverable_files" not in records[0]
    assert "deliverable_file_urls" not in records[0]
    assert "answer.docx" not in json.dumps(records[0])


def test_prepare_source_csv_honors_batch_specific_output_path(tmp_path):
    source = tmp_path / "gdpval_csv.csv"
    output = tmp_path / "gdpval_csv_gdpval_subset_1_20260721.jsonl"
    _write_csv(source, [_source_row()])

    assert prepare(source_csv=source, output_fpath=output) == output
    assert json.loads(output.read_text(encoding="utf-8"))["task_id"] == "GDP-00001"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_prepare_gdpval_csv_rejects_invalid_json(tmp_path):
    source = tmp_path / "gdpval_csv.csv"
    _write_csv(source, [_source_row(rubric_json="not JSON")])

    with pytest.raises(ValueError, match="record 1.*invalid JSON.*rubric_json"):
        prepare_gdpval_csv(source, tmp_path / "output.jsonl")


def test_prepare_gdpval_csv_rejects_missing_columns(tmp_path):
    source = tmp_path / "gdpval_csv.csv"
    fieldnames = [field for field in _source_row() if field != "reference_file_urls"]
    _write_csv(source, [_source_row()], fieldnames=fieldnames)

    with pytest.raises(ValueError, match="missing required columns: reference_file_urls"):
        prepare_gdpval_csv(source, tmp_path / "output.jsonl")


def test_prepare_gdpval_csv_rejects_mismatched_reference_pairs(tmp_path):
    source = tmp_path / "gdpval_csv.csv"
    _write_csv(source, [_source_row(reference_file_urls="[]")])

    with pytest.raises(ValueError, match="1 reference files but 0 reference URLs"):
        prepare_gdpval_csv(source, tmp_path / "output.jsonl")


def test_prepare_gdpval_csv_rejects_mismatched_deliverable_pairs(tmp_path):
    source = tmp_path / "gdpval_csv.csv"
    _write_csv(source, [_source_row(deliverable_file_urls="[]")])

    with pytest.raises(ValueError, match="1 deliverable files but 0 deliverable URLs"):
        prepare_gdpval_csv(source, tmp_path / "output.jsonl")


def test_prepare_gdpval_csv_rejects_duplicate_task_ids(tmp_path):
    source = tmp_path / "gdpval_csv.csv"
    _write_csv(source, [_source_row(), _source_row()])

    with pytest.raises(ValueError, match="record 2.*duplicate task_id 'GDP-00001'"):
        prepare_gdpval_csv(source, tmp_path / "output.jsonl")
