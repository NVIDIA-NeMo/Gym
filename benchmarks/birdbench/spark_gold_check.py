# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit for the Spark SQL variant of BIRD: how SQLite-specific are the gold queries?

The ``spark`` dialect of the ``bird_sql`` server runs the model's Spark SQL on Spark and the untouched gold on
SQLite, so no transpilation is needed at evaluation time. This script transpiles each gold query anyway, as a proxy
for a correct Spark answer, to estimate how many tasks depend on SQLite-only behavior (bare ``GROUP BY`` columns,
integer division, ``IIF``/``strftime``, ...) where a correct Spark query can legitimately return a different result.

Loads every BIRD dev SQLite database into a local Spark warehouse (``setup_bird_spark.ensure_bird_spark``),
transpiles each gold query from SQLite to Spark SQL with ``sqlglot``, runs the original on SQLite and the
transpiled query on Spark, and compares the result sets with BIRD's unordered set equality. Writes
``results.jsonl`` (one status per task: match / mismatch / spark_error / transpile_error /
sqlite_gold_error) and prints the overall match rate and Spark query latency.

Needs a JDK 17 and ``pyspark``, ``sqlglot``, ``pandas``, ``pyarrow`` (see benchmarks/birdbench/README.md).
Run: ``python -m benchmarks.birdbench.spark_gold_check <out_dir> [--limit N]``.
"""

import argparse
import json
import re
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict

import sqlglot

from resources_servers.bird_sql.eval_utils import normalize_rows, result_sets_match
from resources_servers.bird_sql.setup_bird_spark import ensure_bird_spark
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


def _run_sqlite(dev_databases_dir: Path, db_id: str, sql: str) -> list:
    conn = sqlite3.connect(str(dev_databases_dir / db_id / f"{db_id}.sqlite"))
    conn.text_factory = lambda b: b.decode(errors="ignore")
    try:
        return conn.execute(sql).fetchall()
    finally:
        conn.close()


def _check_one(entry: Dict[str, Any], dev_databases_dir: Path, sessions: Dict[str, Any]) -> Dict[str, Any]:
    result = {"id": entry["id"], "db_id": entry["db_id"], "difficulty": entry["difficulty"], "gold": entry["SQL"]}
    try:
        result["spark_sql"] = sqlglot.transpile(entry["SQL"], read="sqlite", write="spark")[0]
    except Exception as e:
        return {**result, "status": "transpile_error", "error": str(e)[:200]}
    try:
        gold = normalize_rows(_run_sqlite(dev_databases_dir, entry["db_id"], entry["SQL"]))
    except Exception as e:
        return {**result, "status": "sqlite_gold_error", "error": str(e)[:200]}
    start = time.time()
    try:
        rows = normalize_rows([tuple(r) for r in sessions[entry["db_id"]].sql(result["spark_sql"]).collect()])
    except Exception as e:
        return {**result, "status": "spark_error", "error": re.sub(r"\s+", " ", str(e))[:300]}
    result["spark_s"] = round(time.time() - start, 2)
    if result_sets_match(gold, rows):
        return {**result, "status": "match"}
    return {
        **result,
        "status": "mismatch",
        "n_gold": len(set(gold)),
        "n_spark": len(set(rows)),
        "gold_sample": repr(sorted(map(repr, set(gold)))[:3])[:200],
        "spark_sample": repr(sorted(map(repr, set(rows)))[:3])[:200],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--limit", type=int, help="Only check the first N dev questions")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dev_databases_dir = ensure_bird_sql()
    entries = json.load(open(dev_databases_dir.parent / "dev.json"))
    for i, entry in enumerate(entries):
        entry["id"] = i
    entries = entries[: args.limit]

    sessions = ensure_bird_spark(dev_databases_dir, args.out_dir / "warehouse")

    by_db = defaultdict(list)
    for entry in entries:
        by_db[entry["db_id"]].append(entry)

    results = []
    for db_id, db_entries in by_db.items():
        with ThreadPoolExecutor(4) as pool:
            db_results = list(pool.map(lambda e: _check_one(e, dev_databases_dir, sessions), db_entries))
        results += db_results
        print(db_id, dict(Counter(r["status"] for r in db_results)), flush=True)

    with open(args.out_dir / "results.jsonl", "w") as f:
        for r in sorted(results, key=lambda r: r["id"]):
            f.write(json.dumps(r) + "\n")
    counts = Counter(r["status"] for r in results)
    print("TOTAL", len(results), dict(counts), f"match rate {counts['match'] / len(results):.1%}")
    latencies = sorted(r["spark_s"] for r in results if "spark_s" in r)
    if latencies:
        print(
            f"spark query latency median {latencies[len(latencies) // 2]}s p95 {latencies[int(len(latencies) * 0.95)]}s"
        )


if __name__ == "__main__":
    main()
