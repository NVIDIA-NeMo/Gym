# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``benchmarks/birdbench/prepare.py``."""

import json
import sqlite3

import pytest

from benchmarks.birdbench import prepare as birdbench_prepare


class _FakeDbHandle:
    """Stands in for ``build_db_values.DbHandle`` -- schema/BM25 logic is tested separately
    in ``test_build_db_values.py``; here we only care about how ``prepare()`` assembles rows.
    """

    def __init__(self, cur, description_dir, dscp="name_or_col_dscp"):
        del cur, description_dir, dscp

    def sql_context_for_question(self, question: str) -> str:
        del question
        return "#### Tables\n- t:\n"


def _entry(db_id: str = "db1", question: str = "How many students?", evidence=None, sql: str = "SELECT 1") -> dict:
    entry = {"db_id": db_id, "question": question, "SQL": sql, "difficulty": "simple"}
    if evidence is not None:
        entry["evidence"] = evidence
    return entry


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Wires ``prepare()`` up against a fake BIRD dataset directory, returning a helper that
    runs ``prepare(**kwargs)`` given a list of raw BIRD entries and returns the parsed output rows.
    """
    dev_databases_dir = tmp_path / "dev_20240627" / "dev_databases"
    dev_databases_dir.mkdir(parents=True)
    db_dir = dev_databases_dir / "db1"
    db_dir.mkdir()
    sqlite3.connect(str(db_dir / "db1.sqlite")).close()

    data_dir = tmp_path / "data"
    output_fpath = data_dir / "birdbench_benchmark.jsonl"
    monkeypatch.setattr(birdbench_prepare, "DATA_DIR", data_dir)
    monkeypatch.setattr(birdbench_prepare, "OUTPUT_FPATH", output_fpath)
    monkeypatch.setattr(birdbench_prepare, "ensure_bird_sql", lambda: dev_databases_dir)
    monkeypatch.setattr(birdbench_prepare, "DbHandle", _FakeDbHandle)

    def _run(entries, **kwargs):
        (dev_databases_dir.parent / "dev.json").write_text(json.dumps(entries))
        output_path = birdbench_prepare.prepare(**kwargs)
        return [json.loads(line) for line in output_path.read_text().splitlines()]

    return _run


class TestPrepareEvidence:
    def test_default_prepends_evidence(self, prepared):
        rows = prepared([_entry(question="How many students?", evidence="students refers to the students table")])
        assert rows[0]["question"] == "students refers to the students table\nHow many students?"

    def test_no_evidence_field_uses_question_only(self, prepared):
        rows = prepared([_entry(question="How many students?")])
        assert rows[0]["question"] == "How many students?"

    def test_none_evidence_uses_question_only(self, prepared):
        rows = prepared([_entry(question="How many students?", evidence=None)])
        assert rows[0]["question"] == "How many students?"

    def test_empty_string_evidence_uses_question_only(self, prepared):
        rows = prepared([_entry(question="How many students?", evidence="")])
        assert rows[0]["question"] == "How many students?"

    def test_include_evidence_false_ignores_evidence(self, prepared):
        rows = prepared(
            [_entry(question="How many students?", evidence="students refers to the students table")],
            include_evidence=False,
        )
        assert rows[0]["question"] == "How many students?"

    def test_assertion_on_empty_question(self, prepared):
        with pytest.raises(AssertionError):
            prepared([_entry(question="")])
