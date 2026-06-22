# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution-based evaluation utilities for BIRD text-to-SQL.

Mirrors the official BIRD eval (DAMO-ConvAI): execute both predicted and
ground-truth SQL against the per-db_id SQLite file, compare result sets via
``set(predicted) == set(ground_truth)``.

SQL execution uses ``asyncio.to_thread`` rather than Ray remote tasks.
When this resource server runs alongside a Ray-coordinated multi-node DP
vLLM (Gym's ``ng_run`` attaches to the same Ray cluster), the vLLM engines
consume the Ray slots and ``@ray.remote`` sqlite tasks sit in the scheduler
queue past the per-query timeout, get cancelled, and every rollout reports
``gold_execution_error``. SQLite queries are fast and self-contained — no
reason to cross process boundaries; run them in the asyncio event loop's
default thread pool under a semaphore for bounded concurrency.
"""

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Optional


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
) -> Optional[ResultSet]:
    """Execute SQL asynchronously in a worker thread, bounded by semaphore.

    Returns ``None`` on timeout or query exception.
    """
    async with semaphore:
        try:
            return await asyncio.wait_for(asyncio.to_thread(execute_sqlite, db_path, sql), timeout=timeout_s)
        except asyncio.TimeoutError:
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


async def execute_and_compare(
    db_path: Path,
    gold_sql: str,
    pred_sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
) -> tuple[bool, Optional[ResultSet], Optional[ResultSet], Optional[str]]:
    """Execute both queries and compare. Returns (match, gold, pred, error_tag)."""
    gold_rows = await execute_sqlite_async(db_path, gold_sql, semaphore, timeout_s)
    if gold_rows is None:
        return False, None, None, "gold_sql_error"

    pred_rows = await execute_sqlite_async(db_path, pred_sql, semaphore, timeout_s)
    if pred_rows is None:
        return False, gold_rows, None, "pred_sql_error"

    return result_sets_match(gold_rows, pred_rows), gold_rows, pred_rows, None
