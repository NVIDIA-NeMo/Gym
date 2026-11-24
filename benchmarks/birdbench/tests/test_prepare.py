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


def _entry(
    db_id: str = "db1",
    question: str = "How many students?",
    evidence=None,
    sql: str = "SELECT 1",
    question_id: int = 0,
) -> dict:
    entry = {"question_id": question_id, "db_id": db_id, "question": question, "SQL": sql, "difficulty": "simple"}
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
            prepared([_entry()], dialect="mysql")

    def test_each_dialect_writes_its_own_file_named_by_dialect(self, prepared, tmp_path):
        prepared([_entry()])
        prepared([_entry()], dialect="spark")
        assert sorted(p.name for p in (tmp_path / "data").iterdir()) == [
            "birdbench_spark_dev.jsonl",
            "birdbench_sqlite_dev.jsonl",
        ]

    def test_output_has_one_row_per_task_never_repeated(self, prepared):
        rows = prepared([_entry(question="q1"), _entry(question="q2")])
        assert [r["id"] for r in rows] == [0, 1]


class TestIsPreparedDataCurrent:
    @staticmethod
    def _write(tmp_path, name, table_heading):
        fpath = tmp_path / name
        row = {
            "id": 0,
            "db_id": "airline",
            "sql_context": f"#### Tables\n{table_heading}\n  - `Code`:\n      data_type: STRING",
        }
        fpath.write_text(json.dumps(row) + "\n\n")
        return fpath

    def test_missing_file_is_not_current(self, tmp_path):
        assert birdbench_prepare.is_prepared_data_current(tmp_path / "birdbench_spark_train.jsonl") is False

    def test_spark_file_with_sanitized_headings_is_current(self, tmp_path):
        fpath = self._write(tmp_path, "birdbench_spark_train.jsonl", "- `airline`.`Air_Carriers`:")
        assert birdbench_prepare.is_prepared_data_current(fpath) is True

    def test_spark_file_prepared_before_table_names_were_sanitized_is_stale(self, tmp_path):
        fpath = self._write(tmp_path, "birdbench_spark_train.jsonl", "- `airline`.`Air Carriers`:")
        assert birdbench_prepare.is_prepared_data_current(fpath) is False

    def test_other_dialects_keep_their_names_and_stay_current(self, tmp_path):
        fpath = self._write(tmp_path, "birdbench_postgres_train.jsonl", '- "airline"."air carriers":')
        assert birdbench_prepare.is_prepared_data_current(fpath) is True
        fpath = self._write(tmp_path, "birdbench_sqlite_train.jsonl", "- Air Carriers:")
        assert birdbench_prepare.is_prepared_data_current(fpath) is True


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
        assert path.name == "birdbench_sqlite_train.jsonl"
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

    def test_output_name_is_dialect_then_split(self):
        assert birdbench_prepare.output_fpath_for("sqlite", "dev").name == "birdbench_sqlite_dev.jsonl"


class TestCli:
    def test_flags_are_forwarded_to_prepare(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(birdbench_prepare, "prepare", lambda **kw: seen.update(kw))
        birdbench_prepare.main(["--dialect", "spark", "--split", "train", "--no-evidence", "--dscp", "name,col_dscp"])
        assert seen == {
            "include_evidence": False,
            "dscp": "name,col_dscp",
            "dialect": "spark",
            "split": "train",
            "train_dir": None,
            "version": "dev_20240627",
            "prompt_format": "default",
        }

    def test_defaults_are_dev_sqlite_with_evidence(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(birdbench_prepare, "prepare", lambda **kw: seen.update(kw))
        birdbench_prepare.main([])
        assert (seen["split"], seen["dialect"], seen["include_evidence"]) == ("dev", "sqlite", True)
        assert (seen["version"], seen["prompt_format"]) == ("dev_20240627", "default")

    def test_version_and_prompt_format_flags(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(birdbench_prepare, "prepare", lambda **kw: seen.update(kw))
        birdbench_prepare.main(["--version", "minidev_20261028", "--prompt-format", "schema_only"])
        assert (seen["version"], seen["prompt_format"]) == ("minidev_20261028", "schema_only")


class _FakeResponse:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def read(self) -> bytes:
        return self.content

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None


@pytest.fixture
def release(monkeypatch):
    """Registers a test release ``dev_test`` served by a fake download; returns the list of downloaded URLs."""
    import hashlib

    def _register(entries, suffix=".json", tasks=None):
        if suffix == ".json":
            content = json.dumps(entries).encode()
        else:
            content = "".join(json.dumps(e) + "\n" for e in entries).encode()
        version = birdbench_prepare.BirdVersion(
            "2099-01-01",
            len(entries) if tasks is None else tasks,
            f"https://example/q{suffix}",
            hashlib.sha256(content).hexdigest(),
        )
        monkeypatch.setitem(birdbench_prepare.VERSIONS, "dev_test", version)
        urls = []
        monkeypatch.setattr(
            birdbench_prepare.urllib.request,
            "urlopen",
            lambda req, timeout: urls.append(req.full_url) or _FakeResponse(content),
        )
        return urls

    return _register


def _q(question_id: int, **kw) -> dict:
    return {**_entry(**kw), "question_id": question_id}


class TestVersions:
    def test_pinned_releases(self):
        versions = birdbench_prepare.VERSIONS
        assert set(versions) == {"dev_20240627", "dev_20251106", "minidev_20250704", "minidev_20261028"}
        assert birdbench_prepare.DEFAULT_VERSION == "dev_20240627" and versions["dev_20240627"].url is None
        for name, v in versions.items():
            assert v.tasks == (1534 if name.startswith("dev") else 500)
            if v.url:
                assert "/resolve/" in v.url and len(v.sha256) == 64

    def test_file_names(self):
        f = birdbench_prepare.output_fpath_for
        assert f("sqlite").name == "birdbench_sqlite_dev.jsonl"
        assert f("sqlite", "dev", "dev_20251106").name == "birdbench_sqlite_dev_20251106.jsonl"
        assert f("postgres", "dev", "minidev_20261028").name == "birdbench_postgres_minidev_20261028.jsonl"
        assert f("sqlite", "dev", "dev_20240627", "schema_only").name == "birdbench_sqlite_dev_schema_only.jsonl"

    def test_named_version_downloads_once_and_tags_rows(self, prepared, release):
        urls = release([_q(7, question="Q7"), _q(9, question="Q9")], suffix=".jsonl")
        rows = prepared([_entry()], version="dev_test")
        assert [(r["question_id"], r["bird_version"], r["id"]) for r in rows] == [
            (7, "dev_test", 0),
            (9, "dev_test", 1),
        ]
        prepared([_entry()], version="dev_test")
        assert len(urls) == 1

    def test_default_version_rows_are_unchanged(self, prepared):
        rows = prepared([_entry()])
        assert set(rows[0]) == {"question", "gt_sql", "sql_context", "difficulty", "db_id", "id"}

    def test_hash_mismatch_is_an_error(self, prepared, release, monkeypatch):
        release([_q(1)])
        v = birdbench_prepare.VERSIONS["dev_test"]
        monkeypatch.setitem(
            birdbench_prepare.VERSIONS, "dev_test", birdbench_prepare.BirdVersion(v.released, 1, v.url, "0" * 64)
        )
        with pytest.raises(RuntimeError, match="pinned SHA-256"):
            prepared([_entry()], version="dev_test")

    def test_wrong_question_count_is_an_error(self, prepared, release):
        release([_q(1)], tasks=2)
        with pytest.raises(RuntimeError, match="has 1 questions, expected 2"):
            prepared([_entry()], version="dev_test")

    @pytest.mark.parametrize(
        "kwargs,message",
        [
            ({"version": "dev_2000"}, "Unsupported version"),
            ({"version": "dev_20251106", "split": "train"}, "can't be used with split"),
            ({"prompt_format": "fancy"}, "Unsupported prompt_format"),
            ({"prompt_format": "schema_only", "dialect": "spark"}, "SQLite only"),
        ],
    )
    def test_bad_arguments(self, prepared, kwargs, message):
        with pytest.raises(ValueError, match=message):
            prepared([_entry()], **kwargs)


class TestSchemaOnly:
    def test_rows_keep_question_and_evidence_apart(self, prepared, tmp_path):
        db = tmp_path / "dev_20240627" / "dev_databases" / "db1"
        conn = sqlite3.connect(str(db / "db1.sqlite"))
        conn.execute("CREATE TABLE t (a INTEGER, b TEXT)")
        conn.commit()
        conn.close()
        (db / "database_description").mkdir()
        (db / "database_description" / "t.csv").write_text(
            "\ufefforiginal_column_name,column_name,column_description,value_description\n"
            "a,a,a,\nb,b,the b column,one of x or y\n"
        )
        rows = prepared([_q(0, evidence="  b means x  "), _q(1, question="Q1")], prompt_format="schema_only")
        assert rows[0]["question"] == "How many students?"
        assert rows[0]["evidence_section"] == "\n\n### Evidence\nb means x"
        assert rows[1]["evidence_section"] == ""
        assert rows[0]["sql_context"] == (
            "BEGIN TRANSACTION;\nCREATE TABLE t (a INTEGER, b TEXT);\nCOMMIT;\n\n"
            "Column descriptions:\nTable t:\n  b: the b column; one of x or y"
        )
        assert (rows[0]["question_id"], rows[0]["bird_version"]) == (0, "dev_20240627")
        no_evidence = prepared([_q(0, evidence="b means x")], prompt_format="schema_only", include_evidence=False)
        assert no_evidence[0]["evidence_section"] == ""

    def test_no_descriptions_leaves_the_dump(self, tmp_path):
        db = tmp_path / "db1"
        db.mkdir()
        sqlite3.connect(str(db / "db1.sqlite")).execute("CREATE TABLE t (a INTEGER);").connection.commit()
        assert birdbench_prepare.schema_only_context(tmp_path, "db1") == (
            "BEGIN TRANSACTION;\nCREATE TABLE t (a INTEGER);\nCOMMIT;"
        )
