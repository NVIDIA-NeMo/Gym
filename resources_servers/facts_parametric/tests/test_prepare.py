# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import io
import json

import pytest

from benchmarks.facts_parametric import prepare as prepare_module


def _csv(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(prepare_module.COLUMNS))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _pin(monkeypatch, content: bytes, rows: int):
    monkeypatch.setattr(prepare_module, "CSV_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(prepare_module, "EXPECTED_ROWS", rows)


def test_prepare_renders_pinned_rows_with_stable_ids(tmp_path, monkeypatch):
    rows = [
        {"url": "https://en.wikipedia.org/wiki/A", "query": "a question", "answer": "a", "topic": "other"},
        {"url": "https://en.wikipedia.org/wiki/B", "query": "b question", "answer": "b", "topic": "release"},
    ]
    content = _csv(rows)
    source = tmp_path / "FACTS-Parametric-public.csv"
    source.write_bytes(content)
    _pin(monkeypatch, content, 2)
    output = prepare_module.prepare(source_csv=str(source), output_fpath=str(tmp_path / "out.jsonl"))
    written = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in written] == ["facts_parametric_public_0001", "facts_parametric_public_0002"]
    assert written[1] == {
        "id": "facts_parametric_public_0002",
        "question": "b question",
        "expected_answer": "b",
        "source_url": "https://en.wikipedia.org/wiki/B",
        "topic": "release",
        "row_sha256": hashlib.sha256(b"https://en.wikipedia.org/wiki/B\tb question\tb\trelease").hexdigest(),
        "upstream": {
            "dataset": prepare_module.KAGGLE_DATASET,
            "version": prepare_module.KAGGLE_DATASET_VERSION,
            "file": prepare_module.CSV_MEMBER,
            "csv_sha256": hashlib.sha256(content).hexdigest(),
            "license": "Apache 2.0",
        },
    }
    assert "responses_create_params" not in written[0]  # the prompt config supplies the user message


def test_prepare_refuses_hash_mismatch_row_count_and_duplicates(tmp_path, monkeypatch):
    rows = [{"url": "u", "query": "q", "answer": "a", "topic": "t"}]
    content = _csv(rows)
    source = tmp_path / "src.csv"
    source.write_bytes(content)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_module.prepare(source_csv=str(source), output_fpath=str(tmp_path / "out.jsonl"))
    _pin(monkeypatch, content, 2)
    with pytest.raises(ValueError, match="expected 2"):
        prepare_module.prepare(source_csv=str(source), output_fpath=str(tmp_path / "out.jsonl"))
    duplicate = _csv(rows * 2)
    source.write_bytes(duplicate)
    _pin(monkeypatch, duplicate, 2)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_module.prepare(source_csv=str(source), output_fpath=str(tmp_path / "out.jsonl"))
    empty = _csv([{"url": "u", "query": " ", "answer": "a", "topic": "t"}])
    source.write_bytes(empty)
    _pin(monkeypatch, empty, 1)
    with pytest.raises(ValueError, match="empty"):
        prepare_module.prepare(source_csv=str(source), output_fpath=str(tmp_path / "out.jsonl"))


def test_pinned_download_url_targets_the_versioned_public_dataset():
    assert prepare_module.KAGGLE_DOWNLOAD_URL.endswith(
        "kaggle/facts-parametric-public-examples?datasetVersionNumber=2"
    )
    assert prepare_module.EXPECTED_ROWS == 1052
