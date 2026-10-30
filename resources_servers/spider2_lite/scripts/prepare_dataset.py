#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert Spider 2.0-Lite source data into NeMo Gym JSONL format.

Requires the xlang-ai/Spider2 reference repo:
    git clone https://github.com/xlang-ai/Spider2.git

Usage:
    python prepare_dataset.py --spider2-dir /path/to/Spider2/spider2-lite --output-dir .

Outputs:
    spider2_lite_sqlite_validation.jsonl       135 SQLite tasks using gold_result mode
    spider2_lite_bigquery_validation.jsonl     all bq/ga tasks (gold_result, engine=bigquery)
    spider2_lite_combined_validation.jsonl     SQLite + BigQuery tasks concatenated
"""

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from subprocess import run


def _raise_csv_field_limit() -> None:
    """Raise csv.field_size_limit so BigQuery DDL cells fit (sys.maxsize fails on 32-bit)."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 2


_raise_csv_field_limit()


SYSTEM_PROMPT_SQLITE = (
    "You are a SQL expert. Given a database schema and a natural language question, "
    "generate a SQLite query that answers the question. Return only the SQL query "
    "inside a ```sql``` code block."
)

SYSTEM_PROMPT_BIGQUERY = (
    "You are a SQL expert. Given a BigQuery database schema and a natural language question, "
    "generate a BigQuery Standard SQL query that answers the question. Fully qualify tables as "
    "`project.dataset.table`. Return only the SQL query inside a ```sql``` code block."
)

# BigQuery instance_id prefixes used by Spider 2.0-Lite.
BIGQUERY_INSTANCE_PREFIXES = ("bq", "ga")


def parse_value(s: str):
    """Convert CSV string to int, float, or leave as str. Non-finite floats become None."""
    try:
        v = int(s)
        return v
    except ValueError:
        pass
    try:
        v = float(s)
    except ValueError:
        return s
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def load_tasks(spider2_dir: Path, prefix: str | tuple[str, ...] = "local") -> dict:
    """Load tasks from ``spider2-lite.jsonl`` filtered by ``instance_id`` prefix."""
    tasks: dict = {}
    with open(spider2_dir / "spider2-lite.jsonl") as f:
        for line in f:
            d = json.loads(line)
            if d["instance_id"].startswith(prefix):
                tasks[d["instance_id"]] = {"db_id": d["db"], "question": d["question"]}
    return tasks


def load_eval_metadata(spider2_dir: Path, prefix: str | tuple[str, ...] = "local") -> dict:
    meta: dict = {}
    with open(spider2_dir / "evaluation_suite/gold/spider2lite_eval.jsonl") as f:
        for line in f:
            d = json.loads(line)
            iid = d["instance_id"]
            if iid.startswith(prefix):
                meta[iid] = {
                    "ignore_order": d["ignore_order"],
                    "condition_cols": d.get("condition_cols", []),
                }
    return meta


def _resolve_db_dir(spider2_dir: Path, db_id: str, engine: str = "sqlite") -> Path:
    """Resolve ``db_id`` under ``resource/databases/{engine}``, with case/separator-insensitive fallback."""
    db_base = spider2_dir / "resource" / "databases" / engine
    direct = db_base / db_id
    if direct.exists():
        return direct
    normalized = db_id.lower().replace("-", "_").replace(".", "_")
    for d in db_base.iterdir():
        if d.is_dir() and d.name.lower().replace("-", "_").replace(".", "_") == normalized:
            return d
    raise FileNotFoundError(f"No {engine} database directory for {db_id!r} in {db_base}")


def load_schema(spider2_dir: Path, db_id: str, engine: str = "sqlite", sqlite_dir: Path | None = None) -> str:
    """Concatenate every ``DDL.csv`` under the ``db_id`` directory for ``engine``.

    For ``engine="sqlite"``, if ``sqlite_dir`` holds a live ``{db_id}.sqlite``,
    the schema is read from that database instead. Accepts both ``DDL`` and
    ``ddl`` column headers.
    """
    if engine == "sqlite" and sqlite_dir:
        db_path = sqlite_dir / f"{db_id}.sqlite"
        if db_path.exists():
            return _schema_from_sqlite(db_path)

    db_dir = _resolve_db_dir(spider2_dir, db_id, engine)
    ddls: list[str] = []
    for ddl_path in sorted(db_dir.rglob("DDL.csv")):
        with open(ddl_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ddl = row.get("DDL") or row.get("ddl")
                if ddl:
                    ddls.append(ddl)
    if not ddls:
        raise FileNotFoundError(f"No DDL rows found under {db_dir}")
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
    """Load every acceptable gold result for ``instance_id``.

    A bare ``{id}.csv`` is the only result when it exists. Otherwise every
    ``{id}_[a-z].csv`` is kept, including when ``_a`` is missing or a letter
    in the middle is missing. This matches Spider 2.0-Lite ``resolve_gold_paths``.
    """
    exec_dir = spider2_dir / "evaluation_suite/gold/exec_result"
    single = exec_dir / f"{instance_id}.csv"
    if single.is_file():
        return [_read_result_csv(single)]
    if not exec_dir.is_dir():
        return []
    pattern = re.compile(rf"^{re.escape(instance_id)}_[a-z]\.csv$")
    paths = sorted(path for path in exec_dir.iterdir() if path.is_file() and pattern.match(path.name))
    return [_read_result_csv(path) for path in paths]


def _read_result_csv(csv_path: Path) -> list[list]:
    with open(csv_path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        return [[parse_value(cell) for cell in row] for row in reader]


def build_row(
    instance_id: str,
    task: dict,
    meta: dict,
    schema: str,
    gold_sql=None,
    gold_result=None,
    engine: str = "sqlite",
) -> dict:
    system_prompt = SYSTEM_PROMPT_BIGQUERY if engine == "bigquery" else SYSTEM_PROMPT_SQLITE
    user_content = f"<DATABASE_SCHEMA>\n{schema}\n</DATABASE_SCHEMA>\n\n<QUESTION>\n{task['question']}\n</QUESTION>"
    row = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]
        },
        "instance_id": instance_id,
        "db_id": task["db_id"],
        "question": task["question"],
        "engine": engine,
    }
    if gold_sql is not None:
        row["gold_sql"] = gold_sql
    if gold_result is not None:
        row["gold_result"] = gold_result
    row["ignore_order"] = meta["ignore_order"]
    row["condition_cols"] = meta.get("condition_cols", [])
    return row


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    # ``allow_nan=False`` fails loudly on NaN/Inf leaks since strict downstream readers reject them.
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def _sort_instance_ids(ids: list[str]) -> list[str]:
    def _key(iid: str) -> tuple[str, int]:
        m = re.search(r"\d+", iid)
        prefix = re.match(r"[A-Za-z]+", iid)
        return (prefix.group() if prefix else iid, int(m.group()) if m else 0)

    return sorted(ids, key=_key)


def _build_sqlite_rows(
    spider2_dir: Path,
    tasks: dict,
    meta: dict,
    sqlite_dir: Path | None,
) -> list[dict]:
    """Build SQLite validation rows. Asserts gold_result present for every task."""
    schema_cache: dict = {}
    rows: list[dict] = []
    for iid in _sort_instance_ids(list(tasks.keys())):
        task = tasks[iid]
        if task["db_id"] not in schema_cache:
            schema_cache[task["db_id"]] = load_schema(spider2_dir, task["db_id"], sqlite_dir=sqlite_dir)
        gold_result = load_gold_result(spider2_dir, iid)
        assert gold_result, f"Task {iid} must have gold result CSVs"
        gold_sql = load_gold_sql(spider2_dir, iid)
        rows.append(
            build_row(
                iid,
                task,
                meta[iid],
                schema_cache[task["db_id"]],
                gold_sql=gold_sql,
                gold_result=gold_result,
                engine="sqlite",
            )
        )
    return rows


def _build_bigquery_rows(
    spider2_dir: Path,
    tasks: dict,
    meta: dict,
) -> tuple[list[dict], list[tuple[str, str]]]:
    """Build BigQuery validation rows, returning ``(rows, skipped)`` where skipped lists ``(iid, reason)``."""
    schema_cache: dict = {}
    rows: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for iid in _sort_instance_ids(list(tasks.keys())):
        task = tasks[iid]
        gold_result = load_gold_result(spider2_dir, iid)
        if not gold_result:
            skipped.append((iid, "no gold_result CSV"))
            continue
        if iid not in meta:
            skipped.append((iid, "no eval metadata"))
            continue
        if task["db_id"] not in schema_cache:
            try:
                schema_cache[task["db_id"]] = load_schema(spider2_dir, task["db_id"], "bigquery")
            except FileNotFoundError as exc:
                skipped.append((iid, str(exc)))
                continue
        gold_sql = load_gold_sql(spider2_dir, iid)
        rows.append(
            build_row(
                iid,
                task,
                meta[iid],
                schema_cache[task["db_id"]],
                gold_sql=gold_sql,
                gold_result=gold_result,
                engine="bigquery",
            )
        )
    return rows, skipped


def _main(args: argparse.Namespace):
    spider2_dir = Path(args.spider2_dir)
    sqlite_dir = Path(args.sqlite_dir) if args.sqlite_dir else None
    output_dir = Path(args.output_dir)
    engine = getattr(args, "engine", "all")
    output_dir.mkdir(parents=True, exist_ok=True)

    sqlite_rows: list[dict] = []
    bq_rows: list[dict] = []

    if engine in ("sqlite", "all"):
        sqlite_tasks = load_tasks(spider2_dir)
        sqlite_meta = load_eval_metadata(spider2_dir)
        print(f"Loaded {len(sqlite_tasks)} local tasks, {len(sqlite_meta)} eval metadata entries")
        sqlite_rows = _build_sqlite_rows(spider2_dir, sqlite_tasks, sqlite_meta, sqlite_dir)
        sqlite_path = output_dir / "spider2_lite_sqlite_validation.jsonl"
        _write_jsonl(sqlite_path, sqlite_rows)
        print(f"Wrote {len(sqlite_rows)} validation tasks to {sqlite_path}")

    if engine in ("bigquery", "all"):
        bq_tasks = load_tasks(spider2_dir, BIGQUERY_INSTANCE_PREFIXES)
        bq_meta = load_eval_metadata(spider2_dir, BIGQUERY_INSTANCE_PREFIXES)
        print(f"Loaded {len(bq_tasks)} BigQuery (bq/ga) tasks, {len(bq_meta)} eval metadata entries")
        bq_rows, bq_skipped = _build_bigquery_rows(spider2_dir, bq_tasks, bq_meta)
        bq_path = output_dir / "spider2_lite_bigquery_validation.jsonl"
        _write_jsonl(bq_path, bq_rows)
        print(f"Wrote {len(bq_rows)} BigQuery tasks to {bq_path} ({len(bq_skipped)} skipped)")

    if engine == "all":
        combined_path = output_dir / "spider2_lite_combined_validation.jsonl"
        _write_jsonl(combined_path, sqlite_rows + bq_rows)
        print(f"Wrote {len(sqlite_rows) + len(bq_rows)} combined tasks to {combined_path}")


def clone_spider2_repo(parent_dir: str):
    if (Path(parent_dir) / "Spider2").exists():
        print("Skipping git clone as the repository is already cloned!")
        return

    run(
        ["git", "clone", "https://github.com/xlang-ai/Spider2.git"],
        cwd=parent_dir,
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Convert Spider 2.0-Lite to NeMo Gym JSONL")
    parser.add_argument("--spider2-dir", required=True, help="Path to Spider2/spider2-lite directory")
    parser.add_argument(
        "--sqlite-dir", default=None, help="Path to downloaded SQLite databases (for schema extraction)"
    )
    parser.add_argument("--output-dir", default=".", help="Output directory for JSONL files")
    parser.add_argument(
        "--engine",
        choices=["sqlite", "bigquery", "all"],
        default="all",
        help="Which engine(s) to emit. 'all' also writes the combined JSONL (default: all).",
    )
    args = parser.parse_args()

    _main(args)


if __name__ == "__main__":
    main()
