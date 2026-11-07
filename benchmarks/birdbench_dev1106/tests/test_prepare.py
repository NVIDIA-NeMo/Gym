# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``benchmarks/birdbench_dev1106/prepare.py``."""

import json
import sqlite3

import pytest

from benchmarks.birdbench import prepare as birdbench_prepare
from benchmarks.birdbench_dev1106 import prepare as dev1106_prepare


class _FakeDbHandle:
    """Stands in for ``build_db_values.DbHandle``; schema rendering is tested in ``birdbench``."""

    def __init__(self, cur, description_dir, dscp="name_or_col_dscp"):
        del cur, description_dir, dscp

    def sql_context_for_question(self, question: str) -> str:
        return f"#### Tables\n- t: ({question})\n"


def _row(question_id: int, question: str, evidence: str | None = None) -> dict:
    return {
        "question_id": question_id,
        "db_id": "db1",
        "question": question,
        "evidence": evidence,
        "SQL": f"SELECT {question_id}",
        "difficulty": "moderate",
    }


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    """Runs ``prepare(**kwargs)`` against fake databases and a fake HF split, returning
    ``(output_rows, load_dataset_calls)``.
    """
    dev_databases_dir = tmp_path / "dev_20240627" / "dev_databases"
    (dev_databases_dir / "db1").mkdir(parents=True)
    sqlite3.connect(str(dev_databases_dir / "db1" / "db1.sqlite")).close()

    calls = []

    def _run(hf_rows, **kwargs):
        def _fake_load_dataset(repo_id, *, split, revision):
            calls.append((repo_id, split, revision))
            return hf_rows

        monkeypatch.setattr(dev1106_prepare, "load_dataset", _fake_load_dataset)
        output_path = dev1106_prepare.prepare(**kwargs)
        return [json.loads(line) for line in output_path.read_text().splitlines()], calls

    monkeypatch.setattr(dev1106_prepare, "OUTPUT_FPATH", tmp_path / "data" / "out.jsonl")
    monkeypatch.setattr(dev1106_prepare, "ensure_bird_sql", lambda: dev_databases_dir)
    monkeypatch.setattr(birdbench_prepare, "DbHandle", _FakeDbHandle)
    return _run


def test_loads_pinned_revision(prepared):
    _, calls = prepared([_row(0, "How many?")])
    assert calls == [(dev1106_prepare.HF_REPO_ID, dev1106_prepare.HF_SPLIT, dev1106_prepare.HF_REVISION)]


def test_rows_follow_question_id_order(prepared):
    rows, _ = prepared([_row(1, "Second?"), _row(0, "First?")])
    assert [(row["id"], row["question"], row["gt_sql"]) for row in rows] == [
        (0, "First?", "SELECT 0"),
        (1, "Second?", "SELECT 1"),
    ]


def test_row_fields_match_birdbench_format(prepared):
    rows, _ = prepared([_row(0, "How many?", evidence="many refers to COUNT")])
    assert rows == [
        {
            "question": "many refers to COUNT\nHow many?",
            "gt_sql": "SELECT 0",
            "sql_context": "#### Tables\n- t: (many refers to COUNT\nHow many?)\n",
            "difficulty": "moderate",
            "db_id": "db1",
            "id": 0,
        }
    ]


def test_include_evidence_false_ignores_evidence(prepared):
    rows, _ = prepared([_row(0, "How many?", evidence="many refers to COUNT")], include_evidence=False)
    assert rows[0]["question"] == "How many?"
