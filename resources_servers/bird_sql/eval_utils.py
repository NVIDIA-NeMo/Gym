# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution-based evaluation utilities for BIRD text-to-SQL.

Mirrors the official BIRD eval (DAMO-ConvAI): execute both predicted and
ground-truth SQL against the per-db_id SQLite file, compare result sets via
``set(predicted) == set(ground_truth)``.
"""

import asyncio
import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional

import ray


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


@ray.remote(
    num_cpus=1,
    scheduling_strategy="SPREAD",
    runtime_env={"py_executable": sys.executable},
)
def execute_sqlite_remote(*args, **kwargs):
    return execute_sqlite(*args, **kwargs)


async def execute_sqlite_async(
    db_path: Path,
    sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
) -> Optional[ResultSet]:
    """Execute SQL asynchronously via Ray remote, bounded by semaphore."""
    async with semaphore:
        task = execute_sqlite_remote.remote(db_path, sql)
        fut: asyncio.Future = asyncio.wrap_future(task.future())

        _, in_progress = await asyncio.wait([fut], timeout=timeout_s)

        if in_progress:
            ray.cancel(task)
            return None
        return ray.get(task)


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
