# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as parquet
import pytest

from benchmarks.gdp_pdf import prepare as prepare_module


def test_extract_rubric_criteria_preserves_metadata() -> None:
    row = {
        "task_id": "task",
        "rubric - 1. criterion": " Reports the total. ",
        "rubric - 1. criterion_type": "explicit",
        "rubric - 2. criterion": None,
    }

    assert prepare_module.extract_rubric_criteria(row) == [
        {"id": "rubric-1", "criterion": "Reports the total.", "criterion_type": "explicit"}
    ]


def test_validate_corpus_totals(monkeypatch) -> None:
    monkeypatch.setattr(prepare_module, "EXPECTED_TASKS", 1)
    monkeypatch.setattr(prepare_module, "EXPECTED_PAGES", 3)
    monkeypatch.setattr(prepare_module, "EXPECTED_CRITERIA", 1)
    monkeypatch.setattr(prepare_module, "EXPECTED_DOMAINS", 1)
    rows = [
        {
            "verifier_metadata": {
                "domain": "Finance",
                "rubric_criteria": [{"id": "rubric-1", "criterion": "States the result."}],
            }
        }
    ]

    prepare_module.validate_corpus_totals(rows, page_count=3)
    with pytest.raises(ValueError, match="corpus totals differ"):
        prepare_module.validate_corpus_totals(rows, page_count=2)


def test_prepare_writes_lightweight_rows(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "pdfs").mkdir(parents=True)
    (source / "pdfs" / "document.pdf").write_bytes(b"pdf")
    parquet.write_table(
        pa.Table.from_pylist(
            [
                {
                    "pdf_path": "pdfs/document.pdf",
                    "prompt": "Analyze this document.",
                    "task_id": "task-1",
                    "task_response_id": "response-1",
                    "worker_id": "worker-1",
                    "domain": "Finance",
                    "rubric - 1. criterion": "States the result.",
                }
            ]
        ),
        source / "data.parquet",
    )
    requested_patterns = []

    def fake_download_source(source_dir, revision, allow_patterns):
        requested_patterns.append(allow_patterns)
        return source

    monkeypatch.setattr(prepare_module, "_download_source", fake_download_source)

    def fake_prepare_document(pdf_path, document_dir, **kwargs):
        document_dir.mkdir(parents=True)
        manifest = document_dir / "manifest.json"
        manifest.write_text('{"page_count": 1, "pages": []}', encoding="utf-8")
        return manifest

    monkeypatch.setattr(prepare_module, "prepare_document", fake_prepare_document)
    output = tmp_path / "data" / "gdp_pdf_benchmark.jsonl"

    result = prepare_module.prepare(
        output_path=output,
        source_dir=source,
        documents_dir=output.parent / "documents",
        revision="revision",
        limit=1,
    )

    assert result == output
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["question"] == "Analyze this document."
    assert "input" not in row["responses_create_params"]
    assert row["verifier_metadata"]["document_manifest"] == "documents/task-1/manifest.json"
    assert row["verifier_metadata"]["rubric_criteria"][0]["criterion"] == "States the result."
    assert requested_patterns == [["data.parquet"], ["pdfs/document.pdf"]]
