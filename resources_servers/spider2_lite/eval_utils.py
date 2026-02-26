# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution-based evaluation utilities for Spider 2.0-Lite."""

import asyncio
import math
import sqlite3
from pathlib import Path
from typing import Any


ResultRow = tuple[Any, ...]
ResultSet = list[ResultRow]


def _normalize(v: Any) -> Any:
    """Normalize None/float-nan to 0."""
    if v is None:
        return 0
    if isinstance(v, float) and math.isnan(v):
        return 0
    return v


def _coerce(v: Any) -> Any:
    """Coerce string values to int or float if possible.

    Gold result rows loaded from pre-computed CSVs via JSON are all strings.
    sqlite3 returns Python native types. Coerce so comparisons work correctly.
    """
    if not isinstance(v, str):
        return v
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def execute_sqlite(db_path: Path, sql: str, timeout_s: float = 30.0) -> ResultSet:
    """Execute SQL against a SQLite database file and return all rows.

    Copies db to in-memory connection before querying (matches official eval).
    """
    conn = sqlite3.connect(str(db_path))
    mem = sqlite3.connect(":memory:")
    try:
        conn.backup(mem)
    finally:
        conn.close()
    try:
        cur = mem.cursor()
        cur.execute(sql)
        return cur.fetchall()
    finally:
        mem.close()


async def execute_sqlite_async(
    db_path: Path,
    sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
) -> ResultSet:
    """Execute SQL asynchronously via thread executor, bounded by semaphore."""
    loop = asyncio.get_running_loop()
    async with semaphore:
        return await asyncio.wait_for(
            loop.run_in_executor(None, execute_sqlite, db_path, sql, timeout_s),
            timeout=timeout_s,
        )


def _col_vector(rows: ResultSet, col_idx: int) -> list[Any]:
    return [_normalize(_coerce(row[col_idx])) for row in rows]


def _vectors_match(v1: list, v2: list, abs_tol: float = 1e-2, ignore_order: bool = False) -> bool:
    if len(v1) != len(v2):
        return False
    a, b = list(v1), list(v2)
    if ignore_order:
        key = lambda x: (x is None, str(x), isinstance(x, (int, float)))
        a = sorted(a, key=key)
        b = sorted(b, key=key)
    for x, y in zip(a, b):
        if x is None and y is None:
            continue
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if not math.isclose(float(x), float(y), abs_tol=abs_tol):
                return False
        elif x != y:
            return False
    return True


def compare_result_sets(
    gold: ResultSet,
    pred: ResultSet,
    condition_cols: list[int] | None = None,
    ignore_order: bool = True,
    abs_tol: float = 1e-2,
) -> bool:
    """Compare two result sets using column-vector matching (mirrors official eval).

    Each gold column vector must appear somewhere in the predicted column vectors.
    Extra columns in pred are allowed.
    """
    if not gold and not pred:
        return True
    if not gold or not pred:
        return False

    num_gold_cols = len(gold[0])
    num_pred_cols = len(pred[0])

    if condition_cols:
        valid_cols = [i for i in condition_cols if i < num_gold_cols]
        cols = valid_cols if valid_cols else list(range(num_gold_cols))
    else:
        cols = list(range(num_gold_cols))

    gold_vecs = [_col_vector(gold, i) for i in cols]
    pred_vecs = [_col_vector(pred, j) for j in range(num_pred_cols)]

    for gv in gold_vecs:
        if not any(_vectors_match(gv, pv, abs_tol=abs_tol, ignore_order=ignore_order) for pv in pred_vecs):
            return False
    return True


def compare_multi_result_sets(
    gold_sets: list[ResultSet],
    pred: ResultSet,
    multi_condition_cols: list[list[int]] | list[int] | None = None,
    ignore_order: bool = True,
    abs_tol: float = 1e-2,
) -> bool:
    """Compare pred against multiple gold result sets. Returns True if any match."""
    if not gold_sets:
        return False

    n = len(gold_sets)
    if not multi_condition_cols:
        cols_per_set: list[list[int]] = [[] for _ in range(n)]
    elif not all(isinstance(c, list) for c in multi_condition_cols):
        cols_per_set = [list(multi_condition_cols)] * n
    else:
        cols_per_set = [list(c) if isinstance(c, list) else [] for c in multi_condition_cols]

    for gold, cols in zip(gold_sets, cols_per_set):
        if compare_result_sets(gold, pred, condition_cols=cols, ignore_order=ignore_order, abs_tol=abs_tol):
            return True
    return False


async def execute_and_compare(
    db_path: Path,
    gold_sql: str,
    pred_sql: str,
    semaphore: asyncio.Semaphore,
    condition_cols: list | None = None,
    ignore_order: bool = True,
    timeout_s: float = 30.0,
) -> tuple[bool, ResultSet | None, ResultSet | None, str | None]:
    """Execute both queries and compare. Returns (match, gold_rows, pred_rows, error_msg)."""
    try:
        gold_rows = await execute_sqlite_async(db_path, gold_sql, semaphore, timeout_s)
    except Exception as e:
        return False, None, None, f"gold_sql_error: {e}"

    try:
        pred_rows = await execute_sqlite_async(db_path, pred_sql, semaphore, timeout_s)
    except Exception as e:
        return False, gold_rows, None, f"pred_sql_error: {e}"

    match = compare_multi_result_sets(
        [gold_rows], pred_rows, multi_condition_cols=condition_cols, ignore_order=ignore_order
    )
    return match, gold_rows, pred_rows, None
