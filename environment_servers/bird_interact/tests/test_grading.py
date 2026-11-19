# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``grade`` applies the rules of the ADK's ``test_case_default``, end to end on a database.

Each test fails when ``grade`` drops one rule: running the submission as written first, the ROUND and DISTINCT
rewrites, and the 10,000-row cap on each side.
"""

import sqlite3
from pathlib import Path

from environment_servers.bird_interact.grading import SQLiteDatabase, grade


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
