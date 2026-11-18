# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit for the Snowflake variant of BIRD: does a correct Snowflake answer grade as correct on the local stand-in?

The ``snowflake`` dialect of the ``bird_sql`` server runs the model's Snowflake SQL on a local stand-in (fakesnow on
DuckDB, see ``setup_bird_snowflake``) and the untouched gold on SQLite. As a proxy for a correct Snowflake answer, this
script transpiles each gold query from SQLite with ``sqlglot`` (test input only; nothing is written back to the
dataset) and grades it with the server's own ``execute_and_compare_snowflake``.

Each gold is graded twice:

- ``raw``: ``sqlglot.transpile`` output as is.
- ``fixed``: the same, plus two mechanical fixes a Snowflake author would make anyway: identifiers get the catalog's
  case and are quoted (Snowflake folds unquoted names to upper case, SQLite ignores case); a double-quoted name that is
  not in the catalog becomes a string literal (SQLite's fallback).

When the server's static check (``snowflake_rejection``) refuses a query, the query is also run on the stand-in
without it, to show what the check changed (``*_unguarded``). Writes ``results.jsonl`` (statuses: match / mismatch /
rejected / sf_error / sf_timeout / transpile_error / sqlite_gold_error, with the stand-in's error message) and prints
match rates.

Every fixed gold that doesn't match gets a ``failure_class`` (see ``FAILURE_CLASSES``), and
``failures_by_class.json`` lists the question ids of each class, as ``prepare`` numbers them (the ``id`` of
``data/birdbench_snowflake_<split>.jsonl``). These are the questions whose correct answer the stand-in can't confirm:
leave them out when the stand-in validates generated SQL. ``--split train`` checks the train uberset instead of dev.

Needs ``fakesnow``, ``snowflake-connector-python<5``, ``sqlglot`` and ``pandas``, which the ``bird_sql`` server's
environment has: run it with ``resources_servers/bird_sql/.venv/bin/python``.
Run: ``python -m benchmarks.birdbench.snowflake_gold_check <out_dir> [--split dev|train] [--train-dir DIR]
[--db-dir DIR] [--limit N]``.
"""

import argparse
import asyncio
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp

from benchmarks.birdbench.prepare import databases_dir_for, load_entries
from resources_servers.bird_sql.eval_utils import (
    execute_and_compare_snowflake,
    normalize_rows,
    result_sets_match,
    snowflake_rejection,
)
from resources_servers.bird_sql.setup_bird_snowflake import LocalSnowflake, ensure_bird_snowflake


_TIMEOUT_S = 30.0

# Why a fixed gold doesn't match, and whether real Snowflake would likely accept the query. "Guess" means it hasn't been
# checked on an account. ``stand_in`` classes are the stand-in's own errors: a correct Snowflake answer fails there.
FAILURE_CLASSES = {
    "bare_group_by_column": "a column outside GROUP BY; Snowflake errors too",
    "string_arithmetic": "arithmetic on two strings, e.g. TO_CHAR(..) - TO_CHAR(..); stand_in: Snowflake casts (guess)",
    "type_mismatch": "another operator or function type mismatch; unknown on Snowflake",
    "text_number_comparison": "text compared with a number; stand_in: Snowflake casts row by row (guess)",
    "text_to_number_cast": "non-numeric text cast to a number; Snowflake errors too (guess)",
    "scalar_subquery_rows": "a scalar subquery returns several rows; Snowflake errors too",
    "stand_in_parser_error": "stand_in: fakesnow or sqlglot fails to parse its own translation",
    "other_error": "another stand-in error; read its message",
    "float_precision": "floats agree to 4 decimals but not to 6 (summation order); likely on Snowflake too",
    "limit_ties_or_order": "the gold has LIMIT: ties, NULL placement or no ORDER BY; nondeterministic on Snowflake too",
    "like_case": "LIKE is case-sensitive on Snowflake, not on SQLite; Snowflake differs too",
    "division": "/ returns decimals where SQLite truncates; Snowflake differs too",
    "other_mismatch": "another result difference; read the samples",
    "refused": "refused by the server's static check (a SQLite-only function, unparsable); Snowflake refuses too",
    "stand_in_timeout": "stand_in: timed out",
    "transpile_error": "sqlglot can't translate the gold",
    "gold_fails_on_sqlite": "the gold itself fails or times out on SQLite",
}
# Stand-in error messages by class, checked in order.
_ERROR_PATTERNS = (
    ("bare_group_by_column", r"must appear in the GROUP BY"),
    ("string_arithmetic", r"'-\(VARCHAR, VARCHAR\)'"),
    ("type_mismatch", r"No function matches"),
    ("text_number_comparison", r"Cannot (compare|mix) values"),
    ("text_to_number_cast", r"Could not convert string"),
    ("scalar_subquery_rows", r"More than one row returned"),
    ("stand_in_parser_error", r"Parser Error"),
)


def error_class(error: str) -> str:
    """The ``FAILURE_CLASSES`` key for a stand-in error message."""
    return next((name for name, pattern in _ERROR_PATTERNS if re.search(pattern, error)), "other_error")


def mismatch_class(gold_sql: str, gold: List[tuple], pred: List[tuple]) -> str:
    """The likely ``FAILURE_CLASSES`` key for a result mismatch, from the rows and the SQLite gold's text."""

    def rounded(rows: List[tuple]) -> List[tuple]:
        return [tuple(round(v, 4) if isinstance(v, float) else v for v in row) for row in rows]

    if result_sets_match(rounded(gold), rounded(pred)):
        return "float_precision"
    sql = gold_sql.upper()
    if re.search(r"\bLIMIT\b", sql):
        return "limit_ties_or_order"
    if re.search(r"\bLIKE\b", sql):
        return "like_case"
    if "/" in sql:
        return "division"
    return "other_mismatch"


def catalog_names(tables: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    """``{"table"|"column": {lower-case name: catalog spelling}}`` for one schema of the stand-in.

    Tables and columns are kept apart: a table and a column can share a name in different case (``tags`` vs
    ``posts.Tags`` in codebase_community).
    """
    return {
        "table": {t.lower(): t for t in tables},
        "column": {c.lower(): c for columns in tables.values() for c in columns},
    }


def fix_for_snowflake(tree: exp.Expression, names: Dict[str, Dict[str, str]]) -> Dict[str, int]:
    """Apply the two mechanical fixes (see module docstring) in place; return how often each fired."""
    counts = Counter()
    aliases = {
        (a.alias if isinstance(a, exp.Alias) else a.name).lower() for a in tree.find_all(exp.Alias, exp.TableAlias)
    }
    for ident in list(tree.find_all(exp.Identifier)):
        # A column's qualifier names a table (``superhero.id``) unless it is an alias of the query.
        qualifier = isinstance(ident.parent, exp.Column) and ident.arg_key == "table"
        if qualifier and ident.name.lower() in aliases:
            continue
        is_table = (isinstance(ident.parent, exp.Table) and ident.arg_key == "this") or qualifier
        canonical = names["table" if is_table else "column"].get(ident.name.lower())
        if canonical is not None:
            # Without the fix Snowflake would look up a different name: quoted with another case, or unquoted
            # (folded to upper case) while the catalog name is not all upper case.
            counts["case"] += ident.name != canonical if ident.quoted else canonical != canonical.upper()
            ident.set("this", canonical)
            ident.set("quoted", True)
        elif ident.quoted and isinstance(ident.parent, exp.Column) and ident.name.lower() not in aliases:
            column = ident.parent
            if column.table == "":
                column.replace(exp.Literal.string(ident.name))
                counts["dq_string"] += 1
    return dict(counts)


def _run_unguarded(sf: LocalSnowflake, db_id: str, query: str, connections: list) -> Tuple[Optional[list], str]:
    """Run ``query`` on the stand-in without ``snowflake_rejection``; return ``(rows, "")`` or ``(None, error)``."""
    conn = sf.connect(db_id)
    connections.append(conn)
    try:
        cur = conn.cursor()
        cur.execute(query.strip().rstrip(";"))
        return normalize_rows(sf.fetch_all(cur)), ""
    except Exception as e:
        return None, f"{type(e).__name__}: {str(e).strip().splitlines()[0]}"[:300]
    finally:
        conn.close()


async def _unguarded(sf: LocalSnowflake, db_id: str, query: str, sem: asyncio.Semaphore) -> Tuple[Optional[list], str]:
    async with sem:
        connections: list = []
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_run_unguarded, sf, db_id, query, connections), timeout=_TIMEOUT_S
            )
        except asyncio.TimeoutError:
            for conn in connections:
                sf.interrupt(conn)
            return None, "timeout"


async def _grade(
    entry: Dict[str, Any], query: str, sf: LocalSnowflake, db_path: Path, sem: asyncio.Semaphore
) -> Dict[str, Any]:
    db_id = entry["db_id"]
    match, gold, pred, err = await execute_and_compare_snowflake(
        db_path, sf, db_id, entry["SQL"], query, sem, timeout_s=_TIMEOUT_S
    )
    if err == "gold_sql_error" or err == "gold_sql_timeout":
        return {"status": "sqlite_gold_error", "failure_class": "gold_fails_on_sqlite"}
    if err == "pred_sql_timeout":
        return {"status": "sf_timeout", "failure_class": "stand_in_timeout"}
    if err == "pred_sql_error":
        reason = snowflake_rejection(query, db_id, sf.tables[db_id])
        rows, error = await _unguarded(sf, db_id, query, sem)
        if reason is None:
            return {"status": "sf_error", "failure_class": error_class(error), "error": error}
        unguarded = "error" if rows is None else ("match" if result_sets_match(gold, rows) else "mismatch")
        return {
            "status": "rejected",
            "failure_class": "refused",
            "reason": reason,
            "unguarded": unguarded,
            "error": error,
        }
    if match:
        return {"status": "match"}
    return {
        "status": "mismatch",
        "failure_class": mismatch_class(entry["SQL"], gold, pred),
        "n_gold": len(set(gold)),
        "n_sf": len(set(pred)),
        "gold_sample": repr(sorted(map(repr, set(gold)))[:3])[:300],
        "sf_sample": repr(sorted(map(repr, set(pred)))[:3])[:300],
    }


async def _check_one(
    entry: Dict[str, Any], sf: LocalSnowflake, databases_dir: Path, sem: asyncio.Semaphore
) -> Dict[str, Any]:
    result = {"id": entry["id"], "db_id": entry["db_id"], "difficulty": entry.get("difficulty"), "gold": entry["SQL"]}
    db_path = databases_dir / entry["db_id"] / f"{entry['db_id']}.sqlite"
    try:
        tree = sqlglot.parse_one(entry["SQL"], read="sqlite")
        result["sf_sql_raw"] = tree.sql(dialect="snowflake")
        result["fixes"] = fix_for_snowflake(tree, catalog_names(sf.tables[entry["db_id"]]))
        result["sf_sql"] = tree.sql(dialect="snowflake")
    except Exception as e:
        return {
            **result,
            "status": "transpile_error",
            "failure_class": "transpile_error",
            "status_raw": "transpile_error",
            "error": str(e)[:200],
        }
    raw = await _grade(entry, result["sf_sql_raw"], sf, db_path, sem)
    fixed = await _grade(entry, result["sf_sql"], sf, db_path, sem)
    return {**result, **fixed, **{f"{k}_raw": v for k, v in raw.items()}}


async def _main(args: argparse.Namespace) -> None:
    databases_dir = databases_dir_for(args.split, args.train_dir)
    db_dir = args.db_dir or Path(
        f"resources_servers/bird_sql/.bird_sql/snowflake{'' if args.split == 'dev' else '_train'}"
    )
    sf = ensure_bird_snowflake(databases_dir, db_dir)
    entries = load_entries(args.split, databases_dir)
    for i, entry in enumerate(entries):
        entry["id"] = i  # as prepare numbers the rows
    entries = entries[: args.limit]

    sem = asyncio.Semaphore(args.concurrency)
    start = time.time()
    results = await asyncio.gather(*(_check_one(e, sf, databases_dir, sem) for e in entries))
    elapsed = time.time() - start

    with open(args.out_dir / "results.jsonl", "w") as f:
        for r in sorted(results, key=lambda r: r["id"]):
            f.write(json.dumps(r) + "\n")
    for suffix in ("_raw", ""):
        counts = Counter(r[f"status{suffix}"] for r in results)
        print(f"TOTAL[status{suffix}]", len(results), dict(counts), f"match rate {counts['match'] / len(results):.1%}")
        rejected = [r for r in results if r[f"status{suffix}"] == "rejected"]
        print(
            f"  rejected{suffix} by reason:",
            dict(Counter(r[f"reason{suffix}"].split(":")[0] for r in rejected)),
            "| unguarded outcome:",
            dict(Counter(r[f"unguarded{suffix}"] for r in rejected)),
        )
    fixes = Counter(k for r in results for k, n in r.get("fixes", {}).items() if n)
    print("golds needing each fix:", dict(fixes))
    by_class: Dict[str, List[int]] = defaultdict(list)
    for r in sorted(results, key=lambda r: r["id"]):
        if r["status"] != "match":
            by_class[r["failure_class"]].append(r["id"])
    by_class = dict(sorted(by_class.items(), key=lambda kv: -len(kv[1])))
    with open(args.out_dir / "failures_by_class.json", "w") as f:
        json.dump(by_class, f, indent=1)
    print(f"{sum(map(len, by_class.values()))} {args.split} questions the stand-in can't confirm, by class:")
    for name, ids in by_class.items():
        print(f"  {len(ids):5d}  {name}: {FAILURE_CLASSES[name]}")
        print(f"         ids: {ids}")
    print(f"{2 * len(results)} grades (raw + fixed) in {elapsed:.1f}s at concurrency {args.concurrency}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--split", choices=("dev", "train"), default="dev")
    parser.add_argument("--train-dir", help="--split train: an extracted BIRD train/ directory (default: download)")
    parser.add_argument(
        "--db-dir",
        type=Path,
        help="Where the stand-in's BIRD.db is (built there if missing); the server's snowflake_db_dir. Default: "
        "resources_servers/bird_sql/.bird_sql/snowflake, or snowflake_train for train, as eval.sh sets it",
    )
    parser.add_argument("--limit", type=int, help="Only check the first N questions")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
