# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution-based evaluation utilities for BIRD text-to-SQL.

Mirrors the official BIRD eval (DAMO-ConvAI): execute both predicted and
ground-truth SQL against the per-db_id SQLite file, compare result sets via
``set(predicted) == set(ground_truth)``.

SQL execution uses ``asyncio.to_thread`` rather than Ray remote tasks.
When this resource server runs alongside a Ray-coordinated multi-node DP
vLLM (``gym env start`` attaches to the same Ray cluster), the vLLM engines
consume the Ray slots and ``@ray.remote`` sqlite tasks sit in the scheduler
queue past the per-query timeout, get cancelled, and every rollout reports
``gold_execution_error``. SQLite queries are fast and self-contained — no
reason to cross process boundaries; run them in the asyncio event loop's
default thread pool under a semaphore for bounded concurrency.
"""

import asyncio
import datetime
import decimal
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional


ResultRow = tuple[Any, ...]
ResultSet = list[ResultRow]


def execute_sqlite(db_path: Path, sql: str) -> Optional[ResultSet]:
    """Execute SQL against a SQLite database file and return all rows.

    Returns ``None`` if execution raises (syntax error, missing table, etc.).
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.text_factory = lambda b: b.decode(errors="ignore")
            cur = conn.cursor()
            cur.execute(sql)
            return cur.fetchall()
    except Exception:
        return None


async def execute_sqlite_async(
    db_path: Path,
    sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
    raise_on_timeout: bool = False,
) -> Optional[ResultSet]:
    """Execute SQL asynchronously in a worker thread, bounded by semaphore.

    Returns ``None`` on timeout (unless ``raise_on_timeout`` is True) or query exception.
    """
    async with semaphore:
        try:
            return await asyncio.wait_for(asyncio.to_thread(execute_sqlite, db_path, sql), timeout=timeout_s)
        except asyncio.TimeoutError:
            if raise_on_timeout:
                raise
            return None


def result_sets_match(gold: ResultSet, pred: ResultSet) -> bool:
    """BIRD's result-set comparison: unordered set equality over tuple rows."""
    try:
        return set(gold) == set(pred)
    except TypeError:
        # Rows may contain unhashable types (bytes, lists) — fall back to sorted lists.
        try:
            return sorted(map(repr, gold)) == sorted(map(repr, pred))
        except Exception:
            return False


def normalize_rows(rows: list[tuple[Any, ...]]) -> ResultSet:
    """Make engine-specific Python types comparable: floats/decimals rounded, dates stringified, bytes decoded."""

    def norm(value: Any) -> Any:
        if isinstance(value, (float, decimal.Decimal)):
            return round(float(value), 6)
        if isinstance(value, (datetime.date, datetime.datetime)):
            return str(value)
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return value

    return [tuple(norm(v) for v in row) for row in rows]


def execute_spark(session: Any, sql: str, job_group: str) -> Optional[ResultSet]:
    """Execute a single read-only query on a Spark session and return all rows.

    Returns ``None`` if execution raises or ``sql`` is not exactly one query (``SELECT``/``WITH``/set operation):
    the Spark warehouse is shared by every rollout, so DDL/DML from a model must never run.
    """
    import sqlglot
    from sqlglot import exp

    try:
        statements = sqlglot.parse(sql, read="spark")
        if len(statements) != 1 or not isinstance(statements[0], exp.Query):
            return None
        session.sparkContext.setJobGroup(job_group, sql[:200], interruptOnCancel=True)
        return normalize_rows([tuple(row) for row in session.sql(sql.strip().rstrip(";")).collect()])
    except Exception:
        return None


async def execute_spark_async(
    session: Any,
    sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
    raise_on_timeout: bool = False,
) -> Optional[ResultSet]:
    """Execute SQL on Spark in a worker thread, bounded by semaphore.

    Returns ``None`` on timeout (unless ``raise_on_timeout`` is True) or query exception. On timeout the Spark job
    group is cancelled; ``asyncio.wait_for`` alone would leave the job running.
    """
    async with semaphore:
        job_group = uuid.uuid4().hex
        try:
            return await asyncio.wait_for(asyncio.to_thread(execute_spark, session, sql, job_group), timeout=timeout_s)
        except asyncio.TimeoutError:
            session.sparkContext.cancelJobGroup(job_group)
            if raise_on_timeout:
                raise
            return None


Runner = Callable[[str], Awaitable[Optional[ResultSet]]]


async def _compare(
    run_gold: Runner, run_pred: Runner, gold_sql: str, pred_sql: str
) -> tuple[bool, Optional[ResultSet], Optional[ResultSet], Optional[str]]:
    """Run gold then pred (runners raise ``asyncio.TimeoutError`` on timeout) and compare."""
    try:
        gold_rows = await run_gold(gold_sql)
    except asyncio.TimeoutError:
        return False, None, None, "gold_sql_timeout"
    if gold_rows is None:
        return False, None, None, "gold_sql_error"

    try:
        pred_rows = await run_pred(pred_sql)
    except asyncio.TimeoutError:
        return False, gold_rows, None, "pred_sql_timeout"
    if pred_rows is None:
        return False, gold_rows, None, "pred_sql_error"

    return result_sets_match(gold_rows, pred_rows), gold_rows, pred_rows, None


async def execute_and_compare(
    db_path: Path,
    gold_sql: str,
    pred_sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
) -> tuple[bool, Optional[ResultSet], Optional[ResultSet], Optional[str]]:
    """Execute both queries on SQLite and compare. Returns (match, gold, pred, error_tag)."""

    async def run(sql: str) -> Optional[ResultSet]:
        return await execute_sqlite_async(db_path, sql, semaphore, timeout_s, raise_on_timeout=True)

    return await _compare(run, run, gold_sql, pred_sql)


async def execute_and_compare_spark(
    db_path: Path,
    session: Any,
    gold_sql: str,
    pred_sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
) -> tuple[bool, Optional[ResultSet], Optional[ResultSet], Optional[str]]:
    """Run the BIRD gold SQL on SQLite and the model's Spark SQL on Spark, then compare. Same return as above.

    No gold transpilation: the gold stays the original SQLite query and only its result set is compared.
    """

    async def run_gold(sql: str) -> Optional[ResultSet]:
        rows = await execute_sqlite_async(db_path, sql, semaphore, timeout_s, raise_on_timeout=True)
        return None if rows is None else normalize_rows(rows)

    async def run_pred(sql: str) -> Optional[ResultSet]:
        return await execute_spark_async(session, sql, semaphore, timeout_s, raise_on_timeout=True)

    return await _compare(run_gold, run_pred, gold_sql, pred_sql)
