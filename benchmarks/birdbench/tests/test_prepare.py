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

    instances = []

    def __init__(
        self, cur, description_dir, dscp="name_or_col_dscp", dialect="sqlite", db_id=None, corpus_limit_per_column=None
    ):
        del cur, description_dir, dscp
        self.corpus_limit_per_column = corpus_limit_per_column
        self.dialect, self.db_id = dialect, db_id
        _FakeDbHandle.instances.append(self)

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
    monkeypatch.setattr(birdbench_prepare, "DATA_DIR", data_dir)
    monkeypatch.setattr(birdbench_prepare, "ensure_bird_sql", lambda: dev_databases_dir)
    monkeypatch.setattr(birdbench_prepare, "DbHandle", _FakeDbHandle)
    _FakeDbHandle.instances.clear()

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


class TestPrepareDialect:
    def test_default_is_sqlite_and_writes_default_output(self, prepared):
        prepared([_entry()])
        assert [(h.dialect, h.db_id) for h in _FakeDbHandle.instances] == [("sqlite", "db1")]

    def test_spark_dialect_is_forwarded_to_the_schema_builder(self, prepared):
        rows = prepared([_entry(sql="SELECT IIF(1, 1, 0)")], dialect="spark")
        assert [(h.dialect, h.db_id) for h in _FakeDbHandle.instances] == [("spark", "db1")]
        assert rows[0]["gt_sql"] == "SELECT IIF(1, 1, 0)"  # gold is never transpiled

    def test_unsupported_dialect_raises(self, prepared):
        with pytest.raises(ValueError, match="Unsupported dialect"):
            prepared([_entry()], dialect="postgres")

    def test_each_dialect_writes_its_own_file_named_by_dialect(self, prepared, tmp_path):
        prepared([_entry()])
        prepared([_entry()], dialect="spark")
        assert sorted(p.name for p in (tmp_path / "data").iterdir()) == [
            "birdbench_spark_benchmark.jsonl",
            "birdbench_sqlite_benchmark.jsonl",
        ]

    def test_output_has_one_row_per_task_never_repeated(self, prepared):
        rows = prepared([_entry(question="q1"), _entry(question="q2")])
        assert [r["id"] for r in rows] == [0, 1]


class TestBuildUberset:
    @staticmethod
    def _train(n):
        return [{"db_id": "db1", "question": f"q{i}", "SQL": f"orig{i}"} for i in range(n)]

    def test_verified_rows_replace_originals_by_position_even_when_question_was_edited(self):
        verified = [
            {"question_id": 1, "db_id": "db1", "question": "q1 fixed", "original_question": "q1", "SQL": "new1"},
            {"question_id": 2, "db_id": "db1", "question": "q2", "original_question": None, "SQL": "new2"},
        ]
        uberset = birdbench_prepare.build_uberset(self._train(4), verified)
        assert [(e["question"], e["SQL"]) for e in uberset] == [
            ("q0", "orig0"),
            ("q3", "orig3"),
            ("q1 fixed", "new1"),
            ("q2", "new2"),
        ]

    def test_mismatching_original_raises(self):
        verified = [{"question_id": 1, "db_id": "db1", "question": "x", "original_question": "other", "SQL": "s"}]
        with pytest.raises(ValueError, match="do not match train.json"):
            birdbench_prepare.build_uberset(self._train(3), verified)


class TestPrepareSplit:
    @pytest.fixture
    def train_run(self, tmp_path, monkeypatch):
        train_dir = tmp_path / "train"
        db_dir = train_dir / "train_databases" / "db1"
        db_dir.mkdir(parents=True)
        sqlite3.connect(str(db_dir / "db1.sqlite")).close()
        entries = [{"db_id": "db1", "question": "q0", "SQL": "SELECT 0", "evidence": "e0"}]
        (train_dir / "train.json").write_text(json.dumps(entries))
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "bird_verified_train.json").write_text("[]")  # cached, so nothing is downloaded
        monkeypatch.setattr(birdbench_prepare, "DATA_DIR", data_dir)
        monkeypatch.setattr(birdbench_prepare, "DbHandle", _FakeDbHandle)
        _FakeDbHandle.instances.clear()
        return lambda **kw: birdbench_prepare.prepare(split="train", train_dir=str(train_dir), **kw)

    def test_train_split_writes_train_file_with_null_difficulty_and_caps_corpus(self, train_run):
        path = train_run()
        assert path.name == "birdbench_train_sqlite.jsonl"
        row = json.loads(path.read_text())
        assert (row["question"], row["difficulty"]) == ("e0\nq0", None)
        assert _FakeDbHandle.instances[0].corpus_limit_per_column == birdbench_prepare.TRAIN_CORPUS_LIMIT_PER_COLUMN

    def test_dev_split_keeps_unbounded_corpus_scan(self, prepared):
        prepared([_entry()])
        assert _FakeDbHandle.instances[0].corpus_limit_per_column is None

    def test_missing_train_databases_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="train_databases"):
            birdbench_prepare.prepare(split="train", train_dir=str(tmp_path))

    def test_unsupported_split_raises(self, prepared):
        with pytest.raises(ValueError, match="Unsupported split"):
            prepared([_entry()], split="test")

    def test_dev_output_name_is_unchanged(self):
        assert birdbench_prepare.output_fpath_for("sqlite").name == "birdbench_sqlite_benchmark.jsonl"
