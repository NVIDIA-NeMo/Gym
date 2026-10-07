# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Feasibility check for a Spark SQL variant of BIRD: do transpiled gold queries agree with SQLite?

Loads every BIRD dev SQLite database into a local Spark warehouse (numeric columns are typed; a column
whose data does not convert cleanly falls back to string and is logged to ``type_fallbacks.json``),
transpiles each gold query from SQLite to Spark SQL with ``sqlglot``, runs the original on SQLite and the
transpiled query on Spark, and compares the result sets with BIRD's unordered set equality. Writes
``results.jsonl`` (one status per task: match / mismatch / spark_error / transpile_error /
sqlite_gold_error) and prints the overall match rate and Spark query latency.

Needs a JDK 17 and ``pyspark``, ``sqlglot``, ``pandas``, ``pyarrow``, ``setuptools`` (see
benchmarks/birdbench/README.md). Run: ``python -m benchmarks.birdbench.spark_gold_check <out_dir> [--limit N]``.
"""

import argparse
import datetime
import decimal
import json
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import sqlglot
from pyspark.sql import SparkSession

from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


DBS = ensure_bird_sql()  # <base>/dev_20240627/dev_databases
DEV = DBS.parent


def spark_session(out: Path) -> SparkSession:
    return (
        SparkSession.builder.master("local[4]")
        .config("spark.sql.warehouse.dir", str(out / "wh"))
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.memory", "4g")
        .getOrCreate()
    )


def load_db(spark, db_id: str, report: list) -> None:
    conn = sqlite3.connect(str(DBS / db_id / f"{db_id}.sqlite"))
    conn.text_factory = lambda b: b.decode(errors="ignore")
    spark.sql(f"DROP DATABASE IF EXISTS `{db_id}` CASCADE")
    spark.sql(f"CREATE DATABASE `{db_id}`")
    tables = [
        r[0] for r in conn.execute("select name from sqlite_master where type='table' and name!='sqlite_sequence'")
    ]
    for t in tables:
        decl = {r[1]: (r[2] or "").upper() for r in conn.execute(f"pragma table_info('{t}')")}
        pdf = pd.read_sql_query(f'select * from "{t}"', conn)
        for col, ty in decl.items():
            s = pdf[col]
            if "INT" in ty:
                targets = ["int", "float"]
            elif any(k in ty for k in ("REAL", "FLOAT", "DOUB", "NUM", "DEC")):
                targets = ["float"]
            else:
                targets = []
            used = "string"
            for tg in targets:
                try:
                    conv = pd.to_numeric(s, errors="raise")
                    if tg == "int":
                        if conv.isna().any() or (conv % 1 != 0).any():
                            # nullable int ok if only NaN; non-integral -> next target
                            if not conv.isna().any():
                                continue
                            conv = conv.astype("Int64") if ((conv.dropna() % 1) == 0).all() else None
                            if conv is None:
                                continue
                        else:
                            conv = conv.astype("int64")
                    else:
                        conv = conv.astype("float64")
                    pdf[col] = conv
                    used = tg
                    break
                except (ValueError, TypeError):
                    continue
            if targets and used == "string":
                report.append({"db": db_id, "table": t, "col": col, "declared": ty, "fallback": "string"})
                pdf[col] = s.map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v))
            elif not targets:
                pdf[col] = s.map(lambda v: None if v is None or (isinstance(v, float) and v != v) else str(v))
        sdf = spark.createDataFrame(pdf)
        sdf.write.mode("overwrite").saveAsTable(f"`{db_id}`.`{t}`")
    print(f"loaded {db_id}: {len(tables)} tables", flush=True)


def norm(v):
    if isinstance(v, (float, decimal.Decimal)):
        return round(float(v), 6)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return str(v)
    if isinstance(v, bytes):
        return v.decode(errors="ignore")
    return v


def norm_rows(rows):
    return {tuple(norm(x) for x in r) for r in rows}


def run_sqlite(db_id, sql):
    conn = sqlite3.connect(str(DBS / db_id / f"{db_id}.sqlite"))
    conn.text_factory = lambda b: b.decode(errors="ignore")
    return conn.execute(sql).fetchall()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--skip-load", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    entries = json.load(open(DEV / "dev.json"))
    for i, e in enumerate(entries):
        e["id"] = i
    if a.limit:
        entries = entries[: a.limit]
    db_ids = sorted({e["db_id"] for e in entries})

    spark = spark_session(out)
    spark.sparkContext.setLogLevel("ERROR")

    if not a.skip_load:
        t0 = time.time()
        report = []
        for d in db_ids:
            load_db(spark, d, report)
        json.dump(report, open(out / "type_fallbacks.json", "w"), indent=1)
        print(f"load done in {time.time() - t0:.0f}s; {len(report)} typed columns fell back to string", flush=True)

    by_db = defaultdict(list)
    for e in entries:
        by_db[e["db_id"]].append(e)

    results = []

    def one(e):
        r = {"id": e["id"], "db_id": e["db_id"], "difficulty": e["difficulty"], "gold": e["SQL"]}
        try:
            r["spark_sql"] = sqlglot.transpile(e["SQL"], read="sqlite", write="spark")[0]
        except Exception as ex:
            r.update(status="transpile_error", error=str(ex)[:200])
            return r
        try:
            gold = norm_rows(run_sqlite(e["db_id"], e["SQL"]))
        except Exception as ex:
            r.update(status="sqlite_gold_error", error=str(ex)[:200])
            return r
        t0 = time.time()
        try:
            rows = norm_rows([tuple(x) for x in spark.sql(r["spark_sql"]).collect()])
        except Exception as ex:
            r.update(status="spark_error", error=re.sub(r"\s+", " ", str(ex))[:300])
            return r
        r["spark_s"] = round(time.time() - t0, 2)
        if rows == gold:
            r["status"] = "match"
        else:
            r["status"] = "mismatch"
            r["n_gold"], r["n_spark"] = len(gold), len(rows)
            r["gold_sample"] = repr(sorted(map(repr, gold))[:3])[:200]
            r["spark_sample"] = repr(sorted(map(repr, rows))[:3])[:200]
        return r

    for d in db_ids:
        spark.sql(f"USE `{d}`")
        with ThreadPoolExecutor(4) as ex:
            res = list(ex.map(one, by_db[d]))
        results += res
        print(d, Counter(r["status"] for r in res), flush=True)

    with open(out / "results.jsonl", "w") as f:
        for r in sorted(results, key=lambda r: r["id"]):
            f.write(json.dumps(r) + "\n")
    c = Counter(r["status"] for r in results)
    print("TOTAL", len(results), dict(c), f"match rate {c['match'] / len(results):.1%}")
    ts = sorted(r["spark_s"] for r in results if "spark_s" in r)
    if ts:
        print(f"spark query latency median {ts[len(ts) // 2]}s p95 {ts[int(len(ts) * 0.95)]}s")


if __name__ == "__main__":
    sys.exit(main())
