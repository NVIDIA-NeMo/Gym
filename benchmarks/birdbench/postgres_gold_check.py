# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit for the PostgreSQL variant of BIRD: does a correct PostgreSQL answer grade as correct?

The ``postgres`` dialect of the ``bird_sql`` server runs the model's PostgreSQL on PostgreSQL and the untouched gold
on SQLite. As a proxy for a correct PostgreSQL answer, this script transpiles each gold query from SQLite with
``sqlglot`` (test input only; nothing is written back to the dataset) and grades it with the server's own
``execute_and_compare_postgres``.

Each gold is graded twice:

- ``raw``: ``sqlglot.transpile`` output as is.
- ``fixed``: the same, plus three mechanical fixes a PostgreSQL author would make anyway: identifiers are spelled as
  loaded (tables in BIRD's case, columns in lower case) and quoted (PostgreSQL folds unquoted names to lower case,
  SQLite ignores case); a double-quoted name that is not in the catalog becomes a string literal (SQLite's
  fallback); ``REAL`` becomes ``DOUBLE PRECISION`` (SQLite's ``REAL`` is 8 bytes, PostgreSQL's 4).

Writes ``results.jsonl`` (statuses: match / mismatch / pg_error / pg_timeout / transpile_error / sqlite_gold_error,
with the PostgreSQL error message) and prints match rates. Needs a PostgreSQL server loaded by
``setup_bird_postgres`` plus ``psycopg``, ``sqlglot``, ``pandas``.
Run: ``python -m benchmarks.birdbench.postgres_gold_check <out_dir> --dsn <reader dsn> [--admin-dsn <dsn>] [--limit N]``.
"""

import argparse
import asyncio
import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import psycopg
import sqlglot
from psycopg import sql as pgsql
from sqlglot import exp

from resources_servers.bird_sql.eval_utils import PostgresPool, execute_and_compare_postgres
from resources_servers.bird_sql.setup_bird_postgres import (
    check_bird_postgres,
    ensure_bird_postgres,
    postgres_column_name,
)
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


def catalog_names(sqlite_path: Path) -> Dict[str, Dict[str, str]]:
    """``{"table"|"column": {lower-case name: name as loaded into PostgreSQL}}`` for one database.

    Tables and columns are kept apart: a table and a column can share a name in different case (``tags`` vs
    ``posts.Tags`` in codebase_community).
    """
    conn = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        tables = [
            t
            for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'")
        ]
        columns = {}
        for t in tables:
            columns.update(
                {row[1].lower(): postgres_column_name(row[1]) for row in conn.execute(f"PRAGMA table_info('{t}')")}
            )
        return {"table": {t.lower(): t for t in tables}, "column": columns}
    finally:
        conn.close()


def fix_for_postgres(tree: exp.Expression, names: Dict[str, Dict[str, str]]) -> Dict[str, int]:
    """Apply the three mechanical fixes (see module docstring) in place; return how often each fired."""
    counts = Counter()
    aliases = {
        (a.alias if isinstance(a, exp.Alias) else a.name).lower() for a in tree.find_all(exp.Alias, exp.TableAlias)
    }
    for ident in list(tree.find_all(exp.Identifier)):
        kind = "table" if isinstance(ident.parent, exp.Table) and ident.arg_key == "this" else "column"
        canonical = names[kind].get(ident.name.lower())
        if canonical is not None:
            # Without the fix PostgreSQL would look up a different name: quoted with another case, or unquoted
            # (folded to lower case) while the catalog name is not all lower case.
            counts["case"] += ident.name != canonical if ident.quoted else canonical != canonical.lower()
            ident.set("this", canonical)
            ident.set("quoted", True)
        elif ident.quoted and isinstance(ident.parent, exp.Column) and ident.name.lower() not in aliases:
            column = ident.parent
            if column.table == "":
                column.replace(exp.Literal.string(ident.name))
                counts["dq_string"] += 1
    for dtype in list(tree.find_all(exp.DataType)):
        if dtype.this == exp.DataType.Type.FLOAT:
            dtype.replace(exp.DataType.build("DOUBLE PRECISION"))
            counts["real"] += 1
    return dict(counts)


async def _pg_error(dsn: str, db_id: str, query: str) -> str:
    """The PostgreSQL error for ``query`` (the grader itself only reports that one happened)."""
    try:
        async with await psycopg.AsyncConnection.connect(dsn) as conn:
            await conn.set_read_only(True)
            await conn.execute(pgsql.SQL("SET search_path = {}").format(pgsql.Identifier(db_id)))
            await conn.execute("SET statement_timeout = '30s'")
            await conn.execute(query, prepare=True)
        return ""
    except psycopg.Error as e:
        return f"{e.sqlstate}: {str(e).strip().splitlines()[0]}"[:300]


async def _grade(
    entry: Dict[str, Any], query: str, pool: PostgresPool, db_path: Path, sem: asyncio.Semaphore
) -> Dict[str, Any]:
    match, gold, pred, err = await execute_and_compare_postgres(
        db_path, pool, entry["db_id"], entry["SQL"], query, sem, timeout_s=30.0
    )
    result: Dict[str, Any] = {}
    if err == "gold_sql_error" or err == "gold_sql_timeout":
        return {**result, "status": "sqlite_gold_error"}
    if err == "pred_sql_timeout":
        return {**result, "status": "pg_timeout"}
    if err == "pred_sql_error":
        return {**result, "status": "pg_error", "error": await _pg_error(pool.dsn, entry["db_id"], query)}
    if match:
        return {**result, "status": "match"}
    return {
        **result,
        "status": "mismatch",
        "n_gold": len(set(gold)),
        "n_pg": len(set(pred)),
        "gold_sample": repr(sorted(map(repr, set(gold)))[:3])[:300],
        "pg_sample": repr(sorted(map(repr, set(pred)))[:3])[:300],
    }


async def _check_one(
    entry: Dict[str, Any], pool: PostgresPool, dev_databases_dir: Path, names: Dict[str, str], sem: asyncio.Semaphore
) -> Dict[str, Any]:
    result = {"id": entry["id"], "db_id": entry["db_id"], "difficulty": entry["difficulty"], "gold": entry["SQL"]}
    db_path = dev_databases_dir / entry["db_id"] / f"{entry['db_id']}.sqlite"
    try:
        tree = sqlglot.parse_one(entry["SQL"], read="sqlite")
        result["pg_sql_raw"] = tree.sql(dialect="postgres")
        result["fixes"] = fix_for_postgres(tree, names)
        result["pg_sql"] = tree.sql(dialect="postgres")
    except Exception as e:
        return {**result, "status": "transpile_error", "status_raw": "transpile_error", "error": str(e)[:200]}
    raw = await _grade(entry, result["pg_sql_raw"], pool, db_path, sem)
    fixed = await _grade(entry, result["pg_sql"], pool, db_path, sem)
    return {**result, **fixed, "status_raw": raw["status"], "error_raw": raw.get("error")}


async def _main(args: argparse.Namespace) -> None:
    dev_databases_dir = ensure_bird_sql()
    if args.admin_dsn:
        ensure_bird_postgres(dev_databases_dir, args.admin_dsn, args.dsn)
    else:
        check_bird_postgres(dev_databases_dir, args.dsn)
    entries = json.load(open(dev_databases_dir.parent / "dev.json"))
    for i, entry in enumerate(entries):
        entry["id"] = i
    entries = entries[: args.limit]
    names = {
        db_id: catalog_names(dev_databases_dir / db_id / f"{db_id}.sqlite") for db_id in {e["db_id"] for e in entries}
    }

    sem = asyncio.Semaphore(args.concurrency)
    pool = PostgresPool(args.dsn, args.concurrency)
    start = time.time()
    try:
        results = await asyncio.gather(
            *(_check_one(e, pool, dev_databases_dir, names[e["db_id"]], sem) for e in entries)
        )
    finally:
        await pool.close()
    elapsed = time.time() - start

    with open(args.out_dir / "results.jsonl", "w") as f:
        for r in sorted(results, key=lambda r: r["id"]):
            f.write(json.dumps(r) + "\n")
    for key in ("status_raw", "status"):
        counts = Counter(r[key] for r in results)
        print(f"TOTAL[{key}]", len(results), dict(counts), f"match rate {counts['match'] / len(results):.1%}")
    fixes = Counter(k for r in results for k, n in r.get("fixes", {}).items() if n)
    print("golds needing each fix:", dict(fixes))
    print(f"{2 * len(results)} grades (raw + fixed) in {elapsed:.1f}s at concurrency {args.concurrency}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--dsn", required=True, help="Read-only DSN the grader uses")
    parser.add_argument("--admin-dsn", help="Load missing db_ids first (see setup_bird_postgres)")
    parser.add_argument("--limit", type=int, help="Only check the first N dev questions")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
