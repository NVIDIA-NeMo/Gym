# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create the episode database, run queries on it and grade a submission, inside the episode's sandbox.

Grading is an independent implementation of BIRD-Interact's default test case for tasks without custom test cases:
remove comments, ``DISTINCT`` and ``ROUND(...)`` from both queries, run them, round floats to two places, and compare
the rows in order when ``conditions.order`` is set, or as sets otherwise. An empty result on either side fails.

The Environment Server uploads this file to the episode's sandbox and runs ``python3 grading.py <command> <request>``
(see ``main``), so it imports only the standard library.
"""

import json
import re
import shutil
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_DISTINCT = re.compile(r"\bdistinct\b", re.IGNORECASE)
_ROUND_CALL = re.compile(r"\bround\s*\(", re.IGNORECASE)
_FLOAT_PLACES = 2
# Rows returned to the agent by an exploratory query.
MAX_PREVIEW_ROWS = 20


@dataclass(frozen=True)
class QueryResult:
    rows: list[tuple] | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class Grade:
    passed: bool
    # Set when the submitted SQL did not run; the agent sees it in the debug message.
    execution_error: str | None = None


class SolutionFailedError(RuntimeError):
    """The ground-truth SQL did not run: the task data is wrong, not the submission."""


def split_statements(sql: str) -> list[str]:
    """Split a script into complete statements; a semicolon inside a string literal does not split."""
    statements, pending = [], ""
    for piece in sql.split(";"):
        pending += piece + ";"
        if sqlite3.complete_statement(pending):
            if pending.strip(" \t\n;"):
                statements.append(pending.strip())
            pending = ""
    if pending.strip(" \t\n;"):
        statements.append(pending.strip().rstrip(";"))
    return statements


def strip_round_calls(sql: str) -> str:
    """Replace each ``ROUND(x, n)`` with ``x``, nested calls included."""
    while match := _ROUND_CALL.search(sql):
        depth, position, first_argument_end = 1, match.end(), None
        while position < len(sql) and depth:
            character = sql[position]
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
            elif character == "," and depth == 1 and first_argument_end is None:
                first_argument_end = position
            position += 1
        if depth:
            return sql  # Unbalanced parentheses: leave the query for SQLite to reject.
        argument = sql[match.end() : first_argument_end if first_argument_end is not None else position - 1]
        sql = sql[: match.start()] + argument.strip() + sql[position:]
    return sql


def normalize_sql(sql: str) -> str:
    sql = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql))
    return strip_round_calls(_DISTINCT.sub(" ", sql))


def run_query(database: Path, sql: str, *, timeout_seconds: float) -> QueryResult:
    """Run ``sql`` on ``database`` and return the rows of its last statement that returns rows."""
    deadline = time.monotonic() + timeout_seconds
    connection = sqlite3.connect(database)
    # The handler interrupts a running statement once the deadline passes.
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        rows: list[tuple] | None = None
        for statement in split_statements(sql):
            cursor = connection.execute(statement)
            if cursor.description is not None:
                rows = cursor.fetchall()
        connection.commit()
        return QueryResult(rows=rows)
    except sqlite3.OperationalError as error:
        if time.monotonic() > deadline:
            return QueryResult(rows=None, error=f"Query timed out after {timeout_seconds} seconds")
        return QueryResult(rows=None, error=str(error))
    except sqlite3.Error as error:
        return QueryResult(rows=None, error=str(error))
    finally:
        connection.close()


def run_on_copy(database: Path, sql: str, *, timeout_seconds: float) -> QueryResult:
    """Run ``sql`` on a scratch copy, so a statement that writes leaves ``database`` unchanged."""
    with tempfile.TemporaryDirectory(prefix="bird-interact-query-") as scratch:
        copy = Path(scratch) / database.name
        shutil.copyfile(database, copy)
        return run_query(copy, sql, timeout_seconds=timeout_seconds)


def _normalize_value(value: object) -> object:
    return round(value, _FLOAT_PLACES) if isinstance(value, float) else value


def _normalize_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_normalize_value(value) for value in row) for row in rows]


def grade(database: Path, predicted_sql: str, solution_sql: str, *, ordered: bool, timeout_seconds: float) -> Grade:
    """Compare the predicted query's result with the solution's on scratch copies of ``database``."""
    predicted = run_on_copy(database, normalize_sql(predicted_sql), timeout_seconds=timeout_seconds)
    if not predicted.ok:
        return Grade(passed=False, execution_error=predicted.error)
    solution = run_on_copy(database, normalize_sql(solution_sql), timeout_seconds=timeout_seconds)
    if not solution.ok:
        raise SolutionFailedError(f"The ground-truth SQL failed: {solution.error}")
    if not predicted.rows or not solution.rows:
        return Grade(passed=False)
    predicted_rows, solution_rows = _normalize_rows(predicted.rows), _normalize_rows(solution.rows)
    if ordered:
        return Grade(passed=predicted_rows == solution_rows)
    return Grade(passed=set(predicted_rows) == set(solution_rows))


def create_database(source: Path, database: Path, preprocess_sql: list[str]) -> None:
    """Create ``database`` from ``source``, a SQLite file or a SQL script, and apply ``preprocess_sql``."""
    if source.suffix == ".sql":
        with sqlite3.connect(database) as connection:
            connection.executescript(source.read_text())
    else:
        shutil.copyfile(source, database)
    if preprocess_sql:
        with sqlite3.connect(database) as connection:
            for statement in split_statements(";\n".join(preprocess_sql)):
                connection.execute(statement)


def main(argv: list[str]) -> None:
    """Run ``setup``, ``execute`` or ``grade`` with the JSON request in the file ``argv[1]`` and print a JSON result.

    The agent's SQL failing is a result. A non-zero exit means the command itself did not run.
    """
    command, request_path = argv
    request_file = Path(request_path)
    try:
        request = json.loads(request_file.read_text())
    finally:
        # A grade request holds the solution SQL, which must not stay in the sandbox.
        request_file.unlink(missing_ok=True)
    database = Path(request["database"])
    if command == "setup":
        create_database(Path(request["source"]), database, request["preprocess_sql"])
        result: dict[str, object] = {}
    elif command == "execute":
        query = run_on_copy(database, request["sql"], timeout_seconds=request["timeout_seconds"])
        rows = query.rows or []
        result = {
            "error": query.error,
            "row_count": len(rows),
            "preview": [repr(row) for row in rows[:MAX_PREVIEW_ROWS]],
        }
    elif command == "grade":
        try:
            outcome = grade(
                database,
                request["predicted_sql"],
                request["solution_sql"],
                ordered=request["ordered"],
                timeout_seconds=request["timeout_seconds"],
            )
            result = {"passed": outcome.passed, "execution_error": outcome.execution_error}
        except SolutionFailedError as error:
            result = {"solution_error": str(error)}
    else:
        raise SystemExit(f"Unknown command {command!r}; expected setup, execute or grade")
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1:])
