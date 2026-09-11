# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``benchmarks/birdbench/build_db_values.py``.

Uses a small in-memory SQLite database instead of the real (~1.4 GB) BIRD dev
database, so these tests don't depend on ``ensure_bird_sql()`` having run.
"""

import importlib.util
import json
import sqlite3

import pytest

from benchmarks.birdbench import build_db_values


_BM25S_INSTALLED = importlib.util.find_spec("bm25s") is not None
_NLTK_INSTALLED = importlib.util.find_spec("nltk") is not None
requires_birdbench_extra = pytest.mark.skipif(
    not (_BM25S_INSTALLED and _NLTK_INSTALLED),
    reason="requires the `birdbench` extra (`uv sync --extra birdbench`)",
)


@pytest.fixture
def conn():
    """A tiny two-table database with a primary key, a foreign key, and a few rows."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        """
        CREATE TABLE students (
            student_id INTEGER PRIMARY KEY,
            name TEXT,
            grade INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE enrollments (
            enrollment_id INTEGER PRIMARY KEY,
            student_id INTEGER,
            course_name TEXT,
            FOREIGN KEY (student_id) REFERENCES students(student_id)
        )
        """
    )
    conn.executemany(
        "INSERT INTO students (student_id, name, grade) VALUES (?, ?, ?)",
        [(1, "Alice", 10), (2, "Bob", 11), (3, "Carol", 10)],
    )
    conn.executemany(
        "INSERT INTO enrollments (enrollment_id, student_id, course_name) VALUES (?, ?, ?)",
        [(1, 1, "Algebra"), (2, 2, "Biology")],
    )
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def cur(conn):
    return conn.cursor()


# ---------------------------------------------------------------------------
# Schema introspection
# ---------------------------------------------------------------------------


class TestSchemaIntrospection:
    def test_table_names(self, cur):
        assert set(build_db_values._table_names(cur)) == {"students", "enrollments"}

    def test_table_names_excludes_sqlite_sequence(self, cur):
        cur.execute("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT)")
        assert "sqlite_sequence" not in build_db_values._table_names(cur)

    def test_column_types(self, cur):
        assert build_db_values._column_types(cur, "students") == {
            "student_id": "INTEGER",
            "name": "TEXT",
            "grade": "INTEGER",
        }

    def test_primary_key_columns(self, cur):
        assert build_db_values._primary_key_columns(cur, "students") == {"student_id"}
        assert build_db_values._primary_key_columns(cur, "enrollments") == {"enrollment_id"}

    def test_all_foreign_keys(self, cur):
        fks = build_db_values._all_foreign_keys(cur, ["students", "enrollments"])
        assert fks == ["enrollments.student_id = students.student_id"]


# ---------------------------------------------------------------------------
# BIRD's column descriptions (database_description/<table>.csv)
# ---------------------------------------------------------------------------


class TestColumnDescriptions:
    def test_missing_csv_returns_empty(self, tmp_path):
        assert build_db_values._column_descriptions(tmp_path, "students", ["name"]) == {}

    def test_default_name_field(self, tmp_path):
        csv_path = tmp_path / "students.csv"
        csv_path.write_text(
            "original_column_name,column_name,column_description\n"
            "student_id,Student ID,The unique id of a student\n"
            "grade,,The grade level of a student\n"
        )
        descriptions = build_db_values._column_descriptions(tmp_path, "students", ["name"])
        assert descriptions == {
            "student_id": "Student ID.",
            # column_name blank and not requested -- no fallback, dropped entirely.
            "grade": "",
        }

    def test_name_and_col_dscp_concatenates_both(self, tmp_path):
        csv_path = tmp_path / "students.csv"
        csv_path.write_text(
            "original_column_name,column_name,column_description\nstudent_id,Student ID,The unique id.\n"
        )
        descriptions = build_db_values._column_descriptions(tmp_path, "students", ["name", "col_dscp"])
        assert descriptions == {"student_id": "Student ID. The unique id."}

    def test_name_or_col_dscp_reproduces_legacy_fallback(self, tmp_path):
        csv_path = tmp_path / "students.csv"
        csv_path.write_text(
            "original_column_name,column_name,column_description\n"
            "student_id,Student ID,The unique id of a student\n"
            "grade,,The grade level of a student\n"
        )
        descriptions = build_db_values._column_descriptions(tmp_path, "students", ["name_or_col_dscp"])
        assert descriptions == {
            "student_id": "Student ID.",
            "grade": "The grade level of a student.",
        }

    def test_blank_original_column_name_skipped(self, tmp_path):
        csv_path = tmp_path / "students.csv"
        csv_path.write_text("original_column_name,column_name,column_description\n,Orphan,Should be skipped\n")
        assert build_db_values._column_descriptions(tmp_path, "students", ["name"]) == {}

    def test_val_dscp_field_with_embedded_newline(self, tmp_path):
        csv_path = tmp_path / "students.csv"
        csv_path.write_text(
            "original_column_name,column_name,column_description,value_description\n"
            'grade,Grade,,"commonsense evidence:\nHigher grade means older student"\n'
        )
        descriptions = build_db_values._column_descriptions(tmp_path, "students", ["val_dscp"])
        assert descriptions == {"grade": "commonsense evidence: Higher grade means older student."}

    def test_clean_text_collapses_whitespace_and_control_chars(self):
        assert build_db_values._clean_text("a\x95 b\n\nc   d") == "a b c d"


class TestParseDscpFields:
    def test_single_field(self):
        assert build_db_values.parse_dscp_fields("name") == ["name"]

    def test_multiple_fields_ordered(self):
        assert build_db_values.parse_dscp_fields("val_dscp,name") == ["val_dscp", "name"]

    def test_whitespace_stripped(self):
        assert build_db_values.parse_dscp_fields(" name , col_dscp ") == ["name", "col_dscp"]

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            build_db_values.parse_dscp_fields("")

    def test_unknown_field_raises(self):
        with pytest.raises(ValueError):
            build_db_values.parse_dscp_fields("name,not_a_real_field")


class TestCombineDescription:
    def test_single_field_period_appended(self):
        assert build_db_values._combine_description(["name"], ("Student ID", "", "")) == "Student ID."

    def test_existing_period_not_doubled(self):
        assert build_db_values._combine_description(["name"], ("Student ID.", "", "")) == "Student ID."

    def test_multiple_fields_joined_in_order(self):
        parts = ("Student ID", "The unique id of a student", "always positive")
        assert (
            build_db_values._combine_description(["name", "col_dscp", "val_dscp"], parts)
            == "Student ID. The unique id of a student. always positive."
        )

    def test_blank_field_dropped(self):
        assert build_db_values._combine_description(["name", "col_dscp"], ("Student ID", "", "")) == "Student ID."

    def test_all_blank_returns_empty_string(self):
        assert build_db_values._combine_description(["name", "col_dscp"], ("", "", "")) == ""

    def test_name_or_col_dscp_prefers_name(self):
        parts = ("Student ID", "The unique id of a student", "")
        assert build_db_values._combine_description(["name_or_col_dscp"], parts) == "Student ID."

    def test_name_or_col_dscp_falls_back_to_description(self):
        parts = ("", "The unique id of a student", "")
        assert build_db_values._combine_description(["name_or_col_dscp"], parts) == "The unique id of a student."


# ---------------------------------------------------------------------------
# Value sources: baseline per-column sample
# ---------------------------------------------------------------------------


class TestSampleTableValues:
    def test_samples_up_to_cap_per_column(self, cur):
        sampled = build_db_values._sample_table_values(cur, ["students"])
        assert len(sampled[("students", "name")]) <= build_db_values._SAMPLE_VALUES_PER_COLUMN

    def test_excludes_null_and_empty_values(self, cur):
        cur.execute("INSERT INTO students (student_id, name, grade) VALUES (4, NULL, 12)")
        cur.execute("INSERT INTO students (student_id, name, grade) VALUES (5, '', 12)")
        sampled = build_db_values._sample_table_values(cur, ["students"])
        assert None not in sampled.get(("students", "name"), [])
        assert "" not in sampled.get(("students", "name"), [])

    def test_column_with_no_values_absent(self, cur):
        cur.execute("CREATE TABLE empty_table (id INTEGER PRIMARY KEY, col TEXT)")
        sampled = build_db_values._sample_table_values(cur, ["empty_table"])
        assert ("empty_table", "col") not in sampled


class TestCollectColumnValues:
    def test_only_short_non_numeric_string_values(self, cur):
        corpus = build_db_values._collect_column_values(cur, ["students", "enrollments"])
        contents = {c["contents"] for c in corpus}
        assert "Alice" in contents
        assert "Algebra" in contents
        # Numeric columns (student_id, grade) are excluded entirely.
        assert not any(c["column"] in {"student_id", "grade"} for c in corpus)

    def test_long_value_excluded(self, cur):
        cur.execute(
            "INSERT INTO students (student_id, name, grade) VALUES (6, ?, 9)",
            ("x" * (build_db_values._VALUE_MAX_LEN + 1),),
        )
        corpus = build_db_values._collect_column_values(cur, ["students"])
        assert all(len(c["contents"]) <= build_db_values._VALUE_MAX_LEN for c in corpus)


# ---------------------------------------------------------------------------
# Pure helpers: numeric check, substring match, scalar rendering
# ---------------------------------------------------------------------------


class TestIsNumber:
    def test_numeric_strings(self):
        assert build_db_values._is_number("42")
        assert build_db_values._is_number("3.14")

    def test_non_numeric_strings(self):
        assert not build_db_values._is_number("Alice")
        assert not build_db_values._is_number("")


class TestSubstringMatchPercentage:
    def test_full_match(self):
        assert build_db_values._substring_match_percentage("Alice", "the name is Alice smith") == 1.0

    def test_no_match(self):
        assert build_db_values._substring_match_percentage("zzz", "the name is Alice smith") == 0.0

    def test_case_insensitive(self):
        assert build_db_values._substring_match_percentage("ALICE", "alice smith") == 1.0

    def test_empty_query(self):
        assert build_db_values._substring_match_percentage("", "anything") == 0.0


class TestRenderScalar:
    def test_plain_string_unquoted(self):
        assert build_db_values._render_scalar("Alice") == "Alice"

    def test_non_string_stringified(self):
        assert build_db_values._render_scalar(42) == "42"

    def test_empty_string_quoted(self):
        assert build_db_values._render_scalar("") == '""'

    def test_embedded_newline_quoted(self):
        assert build_db_values._render_scalar("a\nb") == json.dumps("a\nb")

    def test_surrounding_whitespace_quoted(self):
        assert build_db_values._render_scalar("  a  ") == json.dumps("  a  ")


# ---------------------------------------------------------------------------
# sql_context rendering
# ---------------------------------------------------------------------------


class TestBuildSqlContext:
    def test_renders_tables_columns_and_foreign_keys(self):
        context = build_db_values.build_sql_context(
            table_names=["students"],
            column_types_by_table={"students": {"student_id": "INTEGER", "name": "TEXT"}},
            primary_keys_by_table={"students": {"student_id"}},
            descriptions_by_table={"students": {"name": "The student's name"}},
            values_by_column={("students", "name"): ["Alice", "Bob"]},
            foreign_keys=["enrollments.student_id = students.student_id"],
        )
        assert "#### Tables" in context
        assert "- students:" in context
        assert "data_type: INTEGER (primary key)" in context
        assert "data_type: TEXT" in context
        assert "description: The student's name" in context
        assert "- Alice" in context
        assert "- Bob" in context
        assert "#### Foreign key" in context
        assert "- enrollments.student_id = students.student_id" in context

    def test_no_foreign_keys_omits_section(self):
        context = build_db_values.build_sql_context(
            table_names=["students"],
            column_types_by_table={"students": {"student_id": "INTEGER"}},
            primary_keys_by_table={"students": {"student_id"}},
            descriptions_by_table={},
            values_by_column={},
            foreign_keys=[],
        )
        assert "#### Foreign key" not in context

    def test_missing_description_and_values_render_empty(self):
        context = build_db_values.build_sql_context(
            table_names=["students"],
            column_types_by_table={"students": {"grade": "INTEGER"}},
            primary_keys_by_table={"students": set()},
            descriptions_by_table={},
            values_by_column={},
            foreign_keys=[],
        )
        assert 'description: ""' in context
        assert context.strip().endswith("values:")


# ---------------------------------------------------------------------------
# DbHandle (end-to-end; requires bm25s/nltk)
# ---------------------------------------------------------------------------


@requires_birdbench_extra
class TestDbHandle:
    def test_sql_context_for_question_includes_relevant_value(self, cur, tmp_path):
        handle = build_db_values.DbHandle(cur, description_dir=tmp_path)
        context = handle.sql_context_for_question("What grade is Alice in?")
        assert "#### Tables" in context
        assert "students" in context
        assert "enrollments" in context

    def test_relevant_hits_for_question_empty_corpus_returns_empty(self, tmp_path):
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, val INTEGER)")
        conn.execute("INSERT INTO t (id, val) VALUES (1, 42)")
        conn.commit()
        cur = conn.cursor()
        handle = build_db_values.DbHandle(cur, description_dir=tmp_path)
        assert handle.relevant_hits_for_question("anything") == []
        conn.close()
