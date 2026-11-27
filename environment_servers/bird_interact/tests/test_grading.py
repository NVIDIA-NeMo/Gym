# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The runner on a SQLite database: ``grade`` applying the rules of the ADK's ``test_case_default`` end to end, and
what agent SQL can do to a command.

Each grading test fails when ``grade`` drops one rule: running the submission as written first, the ROUND and DISTINCT
rewrites, and the 10,000-row cap on each side.
"""

import json
import sqlite3
from pathlib import Path

import pytest

from environment_servers.bird_interact.grading import SQLiteDatabase, grade, main


DATA = Path(__file__).resolve().parents[1] / "data"


def _bookshop(tmp_path: Path) -> SQLiteDatabase:
    database = tmp_path / "bookshop.sqlite"
    with sqlite3.connect(database) as connection:
        connection.executescript((DATA / "example_databases/bookshop/bookshop.sql").read_text())
    return SQLiteDatabase(database)


def test_a_submission_that_runs_only_once_rewritten_is_not_executable(tmp_path: Path) -> None:
    # The ADK runs the submission as written first: ROUND with a bad argument list errors there, although the
    # rewritten query, without ROUND, would match the solution.
    outcome = grade(
        _bookshop(tmp_path),
        "SELECT ROUND(list_price, 2, 3) FROM books",
        ["SELECT list_price FROM books"],
        ordered=False,
        timeout_seconds=5,
    )

    assert not outcome.passed
    assert "ROUND" in (outcome.execution_error or "").upper()


def test_round_calls_are_removed_before_the_results_are_compared(tmp_path: Path) -> None:
    outcome = grade(
        _bookshop(tmp_path),
        "SELECT ROUND(list_price / 7, 0) FROM books",
        ["SELECT list_price / 7 FROM books"],
        ordered=False,
        timeout_seconds=5,
    )

    assert outcome.passed


def test_distinct_is_removed_before_the_results_are_compared(tmp_path: Path) -> None:
    outcome = grade(
        _bookshop(tmp_path),
        "SELECT DISTINCT author_id FROM books ORDER BY author_id",
        ["SELECT author_id FROM books ORDER BY author_id"],
        ordered=True,
        timeout_seconds=5,
    )

    assert outcome.passed


def test_each_side_keeps_at_most_10000_rows(tmp_path: Path) -> None:
    rows = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < {n}) SELECT x FROM c"

    outcome = grade(
        _bookshop(tmp_path), rows.format(n=10_005), [rows.format(n=10_000)], ordered=True, timeout_seconds=5
    )

    assert outcome.passed


def test_sql_the_driver_cannot_send_is_the_query_s_error_not_a_failed_command(tmp_path: Path) -> None:
    (query,) = _bookshop(tmp_path).run(["SELECT 'a\x00b'"], timeout_seconds=5)

    assert query.rows is None and query.error


def test_a_preview_shows_each_value_cut_to_the_adk_s_length(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    database = _bookshop(tmp_path)
    request = tmp_path / "execute.json"
    request.write_text(
        json.dumps({"database": str(database.path), "sql": "SELECT printf('%.5000c', 'x'), 1", "timeout_seconds": 5})
    )

    main(["execute", str(request)])

    (row,) = json.loads(capsys.readouterr().out)["preview"]
    assert len(row) < 200 and row.endswith(", 1)")
