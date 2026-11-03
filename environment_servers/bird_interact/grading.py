# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create the episode database, run queries on it and grade a submission, inside the episode's sandbox.

The database is SQLite or PostgreSQL. A SQLite database is a file that ``setup`` creates from the task's database. A
PostgreSQL database is ``<db>_template`` in the server of the sandbox's image (``benchmarks/bird_interact`` builds it),
which ``setup`` starts; the sandbox is the episode's own, so the database is too.

Grading is an independent implementation of the BIRD-Interact ADK's two graders. The default test case grades every
query phase. A submission runs as the ADK runs it, in order on one database state: the submitted SQL as written, which
shows whether it runs; then the submitted SQL and the solution's, each with its comments, ``DISTINCT`` words and
``ROUND(...)`` calls removed. Their results are compared: values normalized (floats and decimals to two places,
decimals half-up, dates and timestamps to ``YYYY-MM-DD``, JSON values and arrays to sorted JSON), rows in order when
``conditions.order`` is set and as sets otherwise. An empty result on either side fails, and so does a value that
cannot be normalized, such as an array of decimals or dates, as in the ADK.

A management phase (PostgreSQL only) is graded by its own test cases: Python code from the ground truth that inspects
the database after the submission ran (``grade_by_test_cases``).

Nothing that grading or an exploratory query runs stays in the database: SQLite runs it on a scratch copy, PostgreSQL
in a transaction that it rolls back, or on a copy of the database that it drops. Only ``setup`` and a passing phase 1
write the database: as in the ADK, phase 1's SQL is applied once it passes, so phase 2 sees what it changed.

The Environment Server uploads this file to the episode's sandbox and runs ``python3 grading.py <command> <request>``
(see ``main``), so it imports only the standard library, and ``psycopg2`` for PostgreSQL, which the image provides.
"""

import contextlib
import datetime
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Protocol


_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
# A line comment that a line break ends; the break stays.
_LINE_COMMENT = re.compile(r"--.*?(\r\n|\r|\n)")
_BLANK_LINES = re.compile(r"\n\s*\n+")
_ROUND_CALL = re.compile(r"ROUND\s*\(", re.IGNORECASE)
_FLOAT_PLACES = 2
_DECIMAL_STEP = Decimal(1).scaleb(-_FLOAT_PLACES)
# The rows the ADK keeps of a query that starts with SELECT or WITH.
MAX_RESULT_ROWS = 10_000
# Rows returned to the agent by an exploratory query.
MAX_PREVIEW_ROWS = 20
# The copy of the episode's database that a management phase is graded on. The sandbox is the episode's own, so one
# name serves; it is short, because PostgreSQL truncates a name longer than 63 bytes.
TEST_CASE_DATABASE = "bird_interact_test_cases"


@dataclass(frozen=True)
class QueryResult:
    rows: list[tuple] | None
    error: str | None = None
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class Grade:
    passed: bool
    # Set when the submitted SQL did not run; the agent sees it in the debug message.
    execution_error: str | None = None
    timed_out: bool = False


class SolutionFailedError(RuntimeError):
    """The ground-truth SQL did not run: the task data is wrong, not the submission."""


def split_statements(sql: str) -> list[str]:
    """Split a SQLite script into complete statements; a semicolon inside a string literal does not split."""
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


def remove_comments(sql: str) -> str:
    """Drop block comments and the line comments that a line break ends, then blank lines."""
    sql = _LINE_COMMENT.sub(r"\1", _BLOCK_COMMENT.sub("", sql))
    return _BLANK_LINES.sub("\n", sql).strip()


def remove_distinct(sql: str) -> str:
    """Drop each space-separated word that is ``DISTINCT``, in any case; ``COUNT(DISTINCT x)`` keeps its own."""
    return " ".join(word for word in sql.split(" ") if word.lower() != "distinct")


def remove_round(sql: str) -> str:
    """Replace each ``ROUND(x, n)`` with ``x``, nested calls included. Unbalanced parentheses stop the rewrite."""
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
            return sql
        argument = sql[match.end() : first_argument_end if first_argument_end is not None else position - 1]
        sql = sql[: match.start()] + argument.strip() + sql[position:]
    return sql


def normalize_sql(sql: str) -> str:
    return remove_round(remove_distinct(remove_comments(sql)))


def _returns_rows(sql: str) -> bool:
    return sql.strip().lower().startswith(("select", "with"))


def _fetch_rows(cursor: Any, script: str) -> list[tuple] | None:
    """The rows of a PostgreSQL script's last statement, as the ADK keeps them: at most ``MAX_RESULT_ROWS`` when the
    script starts with ``SELECT`` or ``WITH``, all of them otherwise, and none when the statement returns no rows."""
    if _returns_rows(script):
        return cursor.fetchmany(MAX_RESULT_ROWS + 1)[:MAX_RESULT_ROWS]
    return cursor.fetchall() if cursor.description is not None else None


def _round_nested(value: object) -> object:
    if isinstance(value, Decimal):
        return value.quantize(_DECIMAL_STEP, rounding=ROUND_HALF_UP)
    if isinstance(value, float):
        return round(value, _FLOAT_PLACES)
    if isinstance(value, (list, tuple)):
        return type(value)(_round_nested(item) for item in value)
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    return value


def _normalize_value(value: object) -> object:
    """A value as grading compares it. SQLite returns only numbers, strings and bytes; PostgreSQL returns them all."""
    # datetime is a date too: the time of day does not count.
    if isinstance(value, datetime.date):
        return value.strftime("%Y-%m-%d")
    value = _round_nested(value)
    # JSON values and arrays compare as JSON with sorted keys. A decimal or a date inside raises TypeError.
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _normalize_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(_normalize_value(value) for value in row) for row in rows]


def _display_value(value: object) -> object:
    """A value as the agent sees it in a preview: decimals as numbers and dates as ISO strings."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    return value


class Database(Protocol):
    def create(self, request: dict[str, Any]) -> None:
        """Make the episode database ready: ``preprocess_sql`` applied, then ``applied_sql``."""

    def run(self, scripts: Sequence[str], *, timeout_seconds: float, commit: bool = False) -> list[QueryResult]:
        """Run the scripts in order, each as one query, and stop after the first that fails. Each result has the rows
        of its script's last statement. With ``commit`` the changes stay in the database; otherwise none does."""


@dataclass(frozen=True)
class SQLiteDatabase:
    """The episode's SQLite file."""

    path: Path

    def create(self, request: dict[str, Any]) -> None:
        source = Path(request["source"])
        if source.suffix == ".sql":
            with sqlite3.connect(self.path) as connection:
                connection.executescript(source.read_text())
        else:
            shutil.copyfile(source, self.path)
        _apply_setup(self, request)

    def run(self, scripts: Sequence[str], *, timeout_seconds: float, commit: bool = False) -> list[QueryResult]:
        if commit:
            return _run_sqlite(self.path, scripts, timeout_seconds=timeout_seconds)
        with tempfile.TemporaryDirectory(prefix="bird-interact-query-") as scratch:
            copy = Path(scratch) / self.path.name
            shutil.copyfile(self.path, copy)
            return _run_sqlite(copy, scripts, timeout_seconds=timeout_seconds)


def _run_sqlite(database: Path, scripts: Sequence[str], *, timeout_seconds: float) -> list[QueryResult]:
    # Each script has its own deadline, which the handler enforces by interrupting the running statement.
    deadline = [0.0]
    connection = sqlite3.connect(database)
    connection.set_progress_handler(lambda: int(time.monotonic() > deadline[0]), 10_000)
    results: list[QueryResult] = []
    try:
        for script in scripts:
            deadline[0] = time.monotonic() + timeout_seconds
            try:
                rows: list[tuple] | None = None
                for statement in split_statements(script):
                    cursor = connection.execute(statement)
                    rows = cursor.fetchall() if cursor.description is not None else None
                if rows is not None and _returns_rows(script):
                    rows = rows[:MAX_RESULT_ROWS]
                results.append(QueryResult(rows=rows))
            except sqlite3.Error as error:
                timed_out = isinstance(error, sqlite3.OperationalError) and time.monotonic() > deadline[0]
                message = f"Query timed out after {timeout_seconds} seconds" if timed_out else str(error)
                results.append(QueryResult(rows=None, error=message, timed_out=timed_out))
                break
        connection.commit()
        return results
    finally:
        connection.close()


@dataclass(frozen=True)
class PostgresDatabase:
    """A database in the PostgreSQL server of the sandbox's image.

    The image sets ``PGDATA`` and the libpq variables (``PGHOST``, ``PGUSER``) that connections use. Each script is
    sent as one query, as the ADK sends it, so it may hold several statements and dollar-quoted bodies.
    """

    name: str

    def create(self, request: dict[str, Any]) -> None:
        start_postgres()
        _apply_setup(self, request)

    def run(self, scripts: Sequence[str], *, timeout_seconds: float, commit: bool = False) -> list[QueryResult]:
        import psycopg2

        connection = psycopg2.connect(dbname=self.name)
        results: list[QueryResult] = []
        try:
            with connection.cursor() as cursor:
                # Bounds each statement of the transaction.
                cursor.execute("SET LOCAL statement_timeout = %s", (max(1, int(timeout_seconds * 1000)),))
                for script in scripts:
                    try:
                        cursor.execute(script)
                        results.append(QueryResult(rows=_fetch_rows(cursor, script)))
                    except psycopg2.errors.QueryCanceled:
                        message = f"Query timed out after {timeout_seconds} seconds"
                        results.append(QueryResult(rows=None, error=message, timed_out=True))
                        break
                    except psycopg2.Error as error:
                        results.append(QueryResult(rows=None, error=str(error)))
                        break
        finally:
            if commit and all(result.ok for result in results):
                connection.commit()
            else:
                connection.rollback()
            connection.close()
        return results

    @contextlib.contextmanager
    def copy(self, name: str) -> Iterator["PostgresDatabase"]:
        """A copy of this database, named ``name``, that the caller may change. It is dropped on exit."""
        import psycopg2
        from psycopg2 import sql

        drop = sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(name))
        create = sql.SQL("CREATE DATABASE {} TEMPLATE {}").format(sql.Identifier(name), sql.Identifier(self.name))
        connection = psycopg2.connect(dbname="postgres")
        # CREATE DATABASE and DROP DATABASE cannot run in a transaction.
        connection.autocommit = True
        try:
            with connection.cursor() as cursor:
                # A copy that a crashed command left.
                cursor.execute(drop)
                cursor.execute(create)
            try:
                yield PostgresDatabase(name)
            finally:
                with connection.cursor() as cursor:
                    cursor.execute(drop)
        finally:
            connection.close()


class TestCaseHelpers:
    """The names that the ADK gives a test case's code, run on the database that the test cases inspect.

    ``execute_queries`` runs as the ADK's does: each query in its own transaction, committed, under a statement
    timeout; it stops at the first query that fails. The comparison of ``ex_base`` is the default test case's. The
    ``remove_*`` rewrites take and return lists of SQL. The connections that ``execute_queries`` opens, when a test
    case passes none, close with the helpers.
    """

    # Not a pytest test class, although its name starts with "Test".
    __test__ = False

    def __init__(self, *, timeout_seconds: float) -> None:
        self._timeout_ms = max(1, int(timeout_seconds * 1000))
        self._connections: list[Any] = []

    def connect(self, database: str) -> Any:
        import psycopg2

        connection = psycopg2.connect(dbname=database)
        self._connections.append(connection)
        return connection

    def close(self) -> None:
        for connection in self._connections:
            connection.close()

    def execute_queries(
        self, queries: str | Sequence[str], db_name: str, conn: Any = None
    ) -> tuple[list[tuple] | None, str | None, bool]:
        """Return the rows of the last query, an error, and whether a query timed out."""
        import psycopg2

        if isinstance(queries, str):
            queries = [queries]
        if not queries:
            return None, None, False
        if conn is None:
            conn = self.connect(db_name)
        rows = None
        for query in queries:
            if not query or not query.strip():
                continue
            try:
                rows = self._perform(conn, query)
            except psycopg2.errors.QueryCanceled:
                return None, None, True
            except Exception as error:
                return None, str(error), False
        return rows, None, False

    def _perform(self, connection: Any, query: str) -> list[tuple] | None:
        cursor = connection.cursor()
        try:
            cursor.execute("SET statement_timeout = %s", (self._timeout_ms,))
            cursor.execute(query)
            connection.commit()
            return _fetch_rows(cursor, query)
        except Exception:
            connection.rollback()
            raise
        finally:
            cursor.close()

    def ex_base(
        self,
        pred_sqls: Sequence[str],
        sol_sqls: Sequence[str],
        db_name: str,
        conn: Any,
        conditions: dict[str, Any] | None = None,
    ) -> int:
        """1 when the two lists of queries return the same rows, by the default test case's comparison; else 0."""
        if not pred_sqls or not sol_sqls:
            return 0
        predicted, predicted_error, predicted_timed_out = self.execute_queries(pred_sqls, db_name, conn)
        solution, solution_error, solution_timed_out = self.execute_queries(sol_sqls, db_name, conn)
        if predicted_error or predicted_timed_out or solution_error or solution_timed_out:
            return 0
        predicted, solution = _normalize_rows(predicted or []), _normalize_rows(solution or [])
        if not predicted or not solution:
            return 0
        if conditions and conditions.get("order", False):
            return int(predicted == solution)
        return int(set(predicted) == set(solution))

    def namespace(self, pred_query_result: list[tuple] | None) -> dict[str, Any]:
        """The globals of the test cases' code. ``pred_query_result`` holds the rows the submission returned."""
        return {
            "execute_queries": self.execute_queries,
            "ex_base": self.ex_base,
            "remove_distinct": lambda sqls: [remove_distinct(sql) for sql in sqls],
            "remove_comments": lambda sqls: [remove_comments(sql) for sql in sqls],
            "remove_round": lambda sqls: [remove_round(sql) for sql in sqls],
            "pred_query_result": pred_query_result,
        }


def passes_test_cases(
    test_cases: Sequence[str],
    namespace: dict[str, Any],
    pred_sqls: list[str],
    sol_sqls: list[str],
    db_name: str,
    conn: Any,
) -> bool:
    """Run each test case's ``test_case(pred_sqls, sol_sqls, db_name, conn)`` in order, as the ADK runs them.

    A test case that raises anything, an ``AssertionError`` or another error, fails the phase, and the rest do not
    run. As in the ADK, the code runs with ``namespace`` as its globals and its own dictionary as its locals, so its
    functions see the helpers but not the code's other top-level names, such as an import. Code that defines no
    ``test_case`` passes. Whatever the code prints goes to standard error, so the command's JSON result stays alone
    on standard output.
    """
    with contextlib.redirect_stdout(sys.stderr):
        for code in test_cases:
            if not isinstance(code, str):
                continue
            names: dict[str, Any] = {}
            try:
                exec(code, namespace, names)
                test_case = names.get("test_case")
                if callable(test_case):
                    test_case(pred_sqls, sol_sqls, db_name, conn)
            except Exception:
                return False
    return True


def _apply_setup(database: Database, request: dict[str, Any]) -> None:
    """Apply the task's ``preprocess_sql``, then ``applied_sql``: phase 1's passing SQL when a fresh sandbox replaces
    one that failed."""
    scripts = [script for script in (*request["preprocess_sql"], *request.get("applied_sql", [])) if script.strip()]
    if not scripts:
        return
    results = database.run(scripts, timeout_seconds=request["timeout_seconds"], commit=True)
    if not results[-1].ok:
        # The task's own setup failed, and the command fails with it.
        raise RuntimeError(f"Setup SQL failed: {results[-1].error}")


def start_postgres() -> None:
    """Start the image's server, listening on its Unix socket only, unless it already runs."""
    # pg_ctl refuses to run as root, for any command.
    if subprocess.run(["gosu", "postgres", "pg_ctl", "status"], capture_output=True).returncode == 0:
        return
    # The log goes to a file, so the server does not hold this command's output open.
    subprocess.run(
        [
            "gosu",
            "postgres",
            "pg_ctl",
            "start",
            "--wait",
            "--log=/tmp/postgresql.log",
            "--options=-c listen_addresses=",
        ],
        check=True,
        capture_output=True,
    )


def grade(
    database: Database,
    predicted_sql: str,
    solution_sql: Sequence[str],
    *,
    ordered: bool,
    timeout_seconds: float,
    apply_if_passed: bool = False,
) -> Grade:
    """Grade ``predicted_sql`` against the solution's scripts by BIRD-Interact's default test case.

    With ``apply_if_passed``, a passing submission is then applied to ``database``.
    """
    solution = [script for script in map(normalize_sql, solution_sql) if script.strip()]
    results = database.run([predicted_sql, normalize_sql(predicted_sql), *solution], timeout_seconds=timeout_seconds)
    submitted = results[0]
    if not submitted.ok:
        return Grade(passed=False, execution_error=submitted.error, timed_out=submitted.timed_out)
    if len(results) < 2 or not results[1].ok:
        return Grade(passed=False)
    if not results[-1].ok:
        raise SolutionFailedError(f"The ground-truth SQL failed: {results[-1].error}")
    predicted_rows = results[1].rows
    solution_rows = results[-1].rows if solution else None
    if not predicted_rows or not solution_rows:
        return Grade(passed=False)
    try:
        predicted_rows, solution_rows = _normalize_rows(predicted_rows), _normalize_rows(solution_rows)
        passed = predicted_rows == solution_rows if ordered else set(predicted_rows) == set(solution_rows)
    except TypeError:
        # A value that JSON or a set cannot hold; the ADK's comparison fails on it too.
        passed = False
    if passed and apply_if_passed:
        database.run([predicted_sql], timeout_seconds=timeout_seconds, commit=True)
    return Grade(passed=passed)


def grade_by_test_cases(
    database: PostgresDatabase,
    predicted_sql: str,
    solution_sql: Sequence[str],
    test_cases: Sequence[str],
    *,
    timeout_seconds: float,
    apply_if_passed: bool = False,
) -> Grade:
    """Grade a management phase by its test cases, as the ADK does, on a copy of ``database``.

    The submission runs first, and its writes stay, so the test cases see them; then each test case runs (see
    ``passes_test_cases``). The copy is dropped afterwards, so ``database`` changes only when a passing submission is
    applied, with ``apply_if_passed``.
    """
    with database.copy(TEST_CASE_DATABASE) as scratch:
        helpers = TestCaseHelpers(timeout_seconds=timeout_seconds)
        try:
            connection = helpers.connect(scratch.name)
            predicted = [predicted_sql]
            rows, error, timed_out = helpers.execute_queries(predicted, scratch.name, connection)
            if error:
                return Grade(passed=False, execution_error=error)
            if timed_out:
                return Grade(passed=False, timed_out=True)
            namespace = helpers.namespace(rows)
            passed = passes_test_cases(test_cases, namespace, predicted, list(solution_sql), scratch.name, connection)
        finally:
            helpers.close()
    if passed and apply_if_passed:
        database.run([predicted_sql], timeout_seconds=timeout_seconds, commit=True)
    return Grade(passed=passed)


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
    if request.get("engine", "sqlite") == "postgresql":
        database: Database = PostgresDatabase(request["database"])
    else:
        database = SQLiteDatabase(Path(request["database"]))
    if command == "setup":
        database.create(request)
        result: dict[str, object] = {}
    elif command == "execute":
        (query,) = database.run([request["sql"]], timeout_seconds=request["timeout_seconds"])
        rows = query.rows or []
        result = {
            "error": query.error,
            "row_count": len(rows),
            "preview": [repr(tuple(_display_value(value) for value in row)) for row in rows[:MAX_PREVIEW_ROWS]],
        }
    elif command == "grade":
        try:
            if request.get("test_cases"):
                if not isinstance(database, PostgresDatabase):
                    raise SystemExit("Test cases run on PostgreSQL only")
                outcome = grade_by_test_cases(
                    database,
                    request["predicted_sql"],
                    request["solution_sql"],
                    request["test_cases"],
                    timeout_seconds=request["timeout_seconds"],
                    apply_if_passed=request.get("apply_if_passed", False),
                )
            else:
                outcome = grade(
                    database,
                    request["predicted_sql"],
                    request["solution_sql"],
                    ordered=request["ordered"],
                    timeout_seconds=request["timeout_seconds"],
                    apply_if_passed=request.get("apply_if_passed", False),
                )
            result = {
                "passed": outcome.passed,
                "execution_error": outcome.execution_error,
                "timed_out": outcome.timed_out,
            }
        except SolutionFailedError as error:
            result = {"solution_error": str(error)}
    else:
        raise SystemExit(f"Unknown command {command!r}; expected setup, execute or grade")
    print(json.dumps(result))


if __name__ == "__main__":
    main(sys.argv[1:])
