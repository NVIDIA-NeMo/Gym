#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert Spider 2.0-Lite source data into NeMo Gym JSONL format.

Requires the xlang-ai/Spider2 reference repo:
    git clone https://github.com/xlang-ai/Spider2.git

Usage:
    python prepare_dataset.py --spider2-dir /path/to/Spider2/spider2-lite --output-dir .

Outputs:
    example.jsonl              5 tasks using gold_sql mode (for smoke testing)
    spider2_lite_sqlite_validation.jsonl  135 tasks using gold_result mode (for full eval)
"""

import argparse
import csv
import json
import re
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "generate a SQLite query that answers the question. Return only the SQL query "
    "inside a ```sql``` code block."
)

EXAMPLE_IDS = ["local022", "local023", "local038", "local039", "local017"]


def parse_value(s: str):
    """Convert CSV string to int, float, or leave as str."""
    try:
        v = int(s)
        return v
    except ValueError:
        pass
    try:
        v = float(s)
        return v
    except ValueError:
        pass
    return s


def load_tasks(spider2_dir: Path) -> dict:
    tasks = {}
    with open(spider2_dir / "spider2-lite.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["instance_id"].startswith("local"):
                tasks[d["instance_id"]] = {"db_id": d["db"], "question": d["question"]}
    return tasks


def load_eval_metadata(spider2_dir: Path) -> dict:
    meta = {}
    with open(spider2_dir / "evaluation_suite/gold/spider2lite_eval.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["instance_id"].startswith("local"):
                meta[d["instance_id"]] = {
                    "ignore_order": d["ignore_order"],
                    "condition_cols": d.get("condition_cols", []),
                }
    return meta


def _resolve_db_dir(spider2_dir: Path, db_id: str) -> Path:
    db_base = spider2_dir / "resource/databases/sqlite"
    direct = db_base / db_id
    if direct.exists():
        return direct
    normalized = db_id.lower().replace("-", "_")
    for d in db_base.iterdir():
        if d.is_dir() and d.name.lower().replace("-", "_") == normalized:
            return d
    raise FileNotFoundError(f"No database directory found for {db_id} in {db_base}")


def load_schema(spider2_dir: Path, db_id: str, sqlite_dir: Path | None = None) -> str:
    if sqlite_dir:
        db_path = sqlite_dir / f"{db_id}.sqlite"
        if db_path.exists():
            return _schema_from_sqlite(db_path)
    db_dir = _resolve_db_dir(spider2_dir, db_id)
    ddl_path = db_dir / "DDL.csv"
    ddls = []
    with open(ddl_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ddls.append(row["DDL"])
    return "\n\n".join(ddls)


def _schema_from_sqlite(db_path: Path) -> str:
    import sqlite3

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL ORDER BY name"
    ).fetchall()
    conn.close()
    return "\n\n".join(r[0] for r in rows)


def load_gold_sql(spider2_dir: Path, instance_id: str) -> str | None:
    sql_path = spider2_dir / f"evaluation_suite/gold/sql/{instance_id}.sql"
    if sql_path.exists():
        return sql_path.read_text().strip()
    return None


def load_gold_result(spider2_dir: Path, instance_id: str) -> list[list[list]]:
    exec_dir = spider2_dir / "evaluation_suite/gold/exec_result"
    result_sets = []
    # Try {id}_a.csv, {id}_b.csv, ... pattern first
    suffix = ord("a")
    while True:
        csv_path = exec_dir / f"{instance_id}_{chr(suffix)}.csv"
        if not csv_path.exists():
            break
        result_sets.append(_read_result_csv(csv_path))
        suffix += 1
    # Fall back to {id}.csv (single result set, no suffix)
    if not result_sets:
        csv_path = exec_dir / f"{instance_id}.csv"
        if csv_path.exists():
            result_sets.append(_read_result_csv(csv_path))
    return result_sets


def _read_result_csv(csv_path: Path) -> list[list]:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return [[parse_value(cell) for cell in row] for row in reader]


def build_row(instance_id: str, task: dict, meta: dict, schema: str, gold_sql=None, gold_result=None) -> dict:
    user_content = f"<DATABASE_SCHEMA>\n{schema}\n</DATABASE_SCHEMA>\n\n<QUESTION>\n{task['question']}\n</QUESTION>"
    row = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ]
        },
        "instance_id": instance_id,
        "db_id": task["db_id"],
        "question": task["question"],
    }
    if gold_sql is not None:
        row["gold_sql"] = gold_sql
    if gold_result is not None:
        row["gold_result"] = gold_result
    row["ignore_order"] = meta["ignore_order"]
    row["condition_cols"] = meta.get("condition_cols", [])
    return row


def main():
    parser = argparse.ArgumentParser(description="Convert Spider 2.0-Lite to NeMo Gym JSONL")
    parser.add_argument("--spider2-dir", required=True, help="Path to Spider2/spider2-lite directory")
    parser.add_argument(
        "--sqlite-dir", default=None, help="Path to downloaded SQLite databases (for schema extraction)"
    )
    parser.add_argument("--output-dir", default=".", help="Output directory for JSONL files")
    args = parser.parse_args()

    spider2_dir = Path(args.spider2_dir)
    sqlite_dir = Path(args.sqlite_dir) if args.sqlite_dir else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(spider2_dir)
    meta = load_eval_metadata(spider2_dir)
    schema_cache = {}

    print(f"Loaded {len(tasks)} local tasks, {len(meta)} eval metadata entries")

    # example.jsonl: 5 tasks with gold_sql (uses DDL.csv schemas)
    example_rows = []
    for iid in EXAMPLE_IDS:
        task = tasks[iid]
        if task["db_id"] not in schema_cache:
            schema_cache[task["db_id"]] = load_schema(spider2_dir, task["db_id"], sqlite_dir=None)
        gold_sql = load_gold_sql(spider2_dir, iid)
        assert gold_sql is not None, f"Example task {iid} must have gold SQL"
        row = build_row(iid, task, meta[iid], schema_cache[task["db_id"]], gold_sql=gold_sql)
        example_rows.append(row)

    example_path = output_dir / "example.jsonl"
    with open(example_path, "w") as f:
        for row in example_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(example_rows)} examples to {example_path}")

    # validation.jsonl: all 135 tasks with gold_result (uses SQLite schemas if available)
    schema_cache = {}
    sorted_ids = sorted(tasks.keys(), key=lambda x: int(re.search(r"\d+", x).group()))
    validation_rows = []
    for iid in sorted_ids:
        task = tasks[iid]
        if task["db_id"] not in schema_cache:
            schema_cache[task["db_id"]] = load_schema(spider2_dir, task["db_id"], sqlite_dir)
        gold_result = load_gold_result(spider2_dir, iid)
        assert gold_result, f"Task {iid} must have gold result CSVs"
        gold_sql = load_gold_sql(spider2_dir, iid)
        row = build_row(iid, task, meta[iid], schema_cache[task["db_id"]], gold_sql=gold_sql, gold_result=gold_result)
        validation_rows.append(row)

    validation_path = output_dir / "spider2_lite_sqlite_validation.jsonl"
    with open(validation_path, "w") as f:
        for row in validation_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(validation_rows)} validation tasks to {validation_path}")


if __name__ == "__main__":
    main()
