# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import csv
import hashlib
import io
import json

import pytest

from benchmarks.facts_grounding_v2 import prepare as prepare_module


def _csv(rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(prepare_module.COLUMNS))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _row(i):
    return {
        "system_instruction": "Answer only from the context.",
        "user_request": f"Question {i}?",
        "context_document": f"Document {i} " * 5,
        "full_prompt": f"Answer only from the context.\n\nQuestion {i}?\n\nDocument {i}",
        "domain": "Medical",
        "type": "Fact Finding",
        "high_level_type": "Q&A",
    }


def _pin(monkeypatch, content, rows):
    monkeypatch.setattr(prepare_module, "CSV_SHA256", hashlib.sha256(content).hexdigest())
    monkeypatch.setattr(prepare_module, "EXPECTED_ROWS", rows)


def test_prepare_bakes_full_prompt_as_the_single_user_message(tmp_path, monkeypatch):
    content = _csv([_row(1), _row(2)])
    source = tmp_path / "examples.csv"
    source.write_bytes(content)
    _pin(monkeypatch, content, 2)
    output = prepare_module.prepare(source_csv=str(source), output_fpath=str(tmp_path / "out.jsonl"))
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert [row["id"] for row in rows] == ["facts_grounding_v2_public_0001", "facts_grounding_v2_public_0002"]
    assert rows[0]["responses_create_params"] == {"input": [{"role": "user", "content": _row(1)["full_prompt"]}]}
    assert rows[0]["context_document_chars"] == len(_row(1)["context_document"])
    assert rows[0]["upstream"]["dataset"] == "deepmind/FACTS-grounding-examples"
    assert rows[0]["upstream"]["version"] == 17
    assert rows[1]["domain"] == "Medical" and rows[1]["high_level_type"] == "Q&A"


def test_prepare_refuses_unpinned_or_malformed_sources(tmp_path, monkeypatch):
    content = _csv([_row(1)])
    source = tmp_path / "examples.csv"
    source.write_bytes(content)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        prepare_module.prepare(source_csv=str(source), output_fpath=str(tmp_path / "out.jsonl"))
    _pin(monkeypatch, content, 3)
    with pytest.raises(ValueError, match="expected 3"):
        prepare_module.prepare(source_csv=str(source), output_fpath=str(tmp_path / "out.jsonl"))
    duplicate = _csv([_row(1), _row(1)])
    source.write_bytes(duplicate)
    _pin(monkeypatch, duplicate, 2)
    with pytest.raises(ValueError, match="duplicate"):
        prepare_module.prepare(source_csv=str(source), output_fpath=str(tmp_path / "out.jsonl"))


def test_pins():
    assert prepare_module.EXPECTED_ROWS == 856 and prepare_module.HF_V1_PUBLIC_ROWS == 860
    assert prepare_module.KAGGLE_DOWNLOAD_URL.endswith("deepmind/FACTS-grounding-examples?datasetVersionNumber=17")
