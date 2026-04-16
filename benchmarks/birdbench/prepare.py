# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare BIRD benchmark data for NeMo Gym.

Replicates `nemo_skills/dataset/birdbench/prepare.py` so Skills and Gym see
identical rows:

    per-row output schema: {question, gt_sql, sql_context, difficulty, db_id, id}

The schema dump (``sql_context``) is produced via ``sqlite3.Connection.iterdump()``
with the same INSERT-chain truncation Skills applies: at most 10 consecutive
``INSERT`` statements are kept per run, and long ``INSERT ... VALUES (...), ...``
chains are collapsed after 10 tuples.

We call ``ensure_bird_sql()`` to reuse the same cached download as the
``bird_sql`` resource server, avoiding duplicate ~1.4 GB downloads.
"""

import glob
import json
import os
import re
import sqlite3
from pathlib import Path

from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "birdbench_benchmark.jsonl"


def _read_tables_info(dev_databases_dir: Path) -> dict[str, str]:
    """Dump each BIRD database's schema + (truncated) inserts to a string.

    Port of `read_tables_file` from `nemo_skills/dataset/birdbench/prepare.py`.
    """
    tables_info: dict[str, str] = {}
    db_dirs = glob.glob("*", root_dir=str(dev_databases_dir))

    for db_dir in sorted(db_dirs):
        sqlite_file = dev_databases_dir / db_dir / f"{db_dir}.sqlite"
        if not sqlite_file.exists():
            continue

        print(f"Reading database info from: {db_dir}")
        table_info = ""
        with sqlite3.connect(str(sqlite_file)) as con:
            con.text_factory = lambda b: b.decode(errors="ignore")
            for line in con.iterdump():
                if line[:6] == "INSERT":
                    line = line.replace("\n", " ")
                line = re.sub(r" +", " ", line)
                table_info += line + "\n"

        # Truncate any long consecutive INSERT chains (keep 10 max).
        insert_chain = r"((INSERT.*$\n){10})((INSERT.*\n)*)"
        table_info = re.sub(insert_chain, r"\1\n...\n", table_info, flags=re.MULTILINE)

        # Collapse ``INSERT INTO * VALUES (...), (...), ...`` chains >10 tuples.
        many_values = r"(?:VALUES )(((\([^)]*)\)[,;]\s*)){10}(.*)(?:;)"
        table_info = re.sub(many_values, r"...", table_info, flags=re.MULTILINE)

        tables_info[db_dir] = table_info

    return tables_info


def _format_entries(dev_json_path: Path, tables_info: dict[str, str], out_fpath: Path) -> int:
    with open(dev_json_path, "r") as f_in:
        entries = json.load(f_in)

    count = 0
    with open(out_fpath, "w") as f_out:
        for i, entry in enumerate(entries):
            row = {
                "question": entry["question"],
                "gt_sql": entry["SQL"],
                "sql_context": tables_info[entry["db_id"]],
                "difficulty": entry["difficulty"],
                "db_id": entry["db_id"],
                "id": i,
            }
            f_out.write(json.dumps(row) + "\n")
            count += 1
    return count


def prepare() -> Path:
    """Download BIRD dev, produce birdbench_benchmark.jsonl. Returns the output path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dev_databases_dir = ensure_bird_sql()
    # dev_databases_dir == <base>/dev_20240627/dev_databases → dev.json is one level up.
    dev_dir = dev_databases_dir.parent
    dev_json_path = dev_dir / "dev.json"
    if not dev_json_path.exists():
        raise RuntimeError(f"Expected BIRD dev.json at {dev_json_path}")

    print("Building per-db schema dumps...")
    tables_info = _read_tables_info(dev_databases_dir)
    print(f"Collected schema dumps for {len(tables_info)} databases.")

    count = _format_entries(dev_json_path, tables_info, OUTPUT_FPATH)
    print(f"Wrote {count} BIRD dev entries to {OUTPUT_FPATH}")

    # Keep the downloaded BIRD archive cache in place so the bird_sql resource
    # server can use the same files at runtime.
    _ = os.environ  # mark used, no env changes required.
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
