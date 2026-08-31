# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare BIRD benchmark data with a compact per-column YAML schema in ``sql_context``.

Second iteration of ``prepare_with_bm25_yaml.py``'s per-column YAML schema, adding primary
key and foreign key information and compacting each column into a single string (rather than
a nested ``data_type``/``description``/``sampled_values`` mapping) to avoid repeating those
keys on every column:

    - column_name: "data_type, (example: [value1, value2]), is primary key, description"

"is primary key" is present only for actual primary-key columns (from SQLite's own
``PRAGMA table_info``). Foreign keys are listed once per database, after every table, as
``table.from_col = ref_table.to_col`` (from ``PRAGMA foreign_key_list``). ``description`` is
BIRD's column description (``database_description/<table>.csv``, same source as
``prepare_with_bm25_yaml.py``) with BIRD's ``value_description`` appended when non-empty.

Produces the same row schema as ``prepare.py``/``prepare_with_bm25.py`` (``question, gt_sql,
sql_context, difficulty, db_id, id``) -- this is purely an alternate ``sql_context``
representation, kept alongside (not replacing) the other two.

Requires ``bm25s`` (``pip install bm25s``), unlike plain ``prepare.py``.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.birdbench.build_db_values import DbHandle
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "birdbench_benchmark_bm25_yaml_v2.jsonl"


def prepare() -> Path:
    """Download BIRD dev, produce ``birdbench_benchmark_bm25_yaml_v2.jsonl``. Returns the output path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dev_databases_dir = ensure_bird_sql()
    # dev_databases_dir == <base>/dev_20240627/dev_databases → dev.json is one level up.
    dev_json_path = dev_databases_dir.parent / "dev.json"
    if not dev_json_path.exists():
        raise RuntimeError(f"Expected BIRD dev.json at {dev_json_path}")

    with open(dev_json_path) as f:
        entries: List[Dict[str, Any]] = json.load(f)

    print("Building per-db schemas (types, PK/FK, descriptions, BM25 indexes)...")
    db_ids = sorted({entry["db_id"] for entry in entries})
    db_handles: Dict[str, DbHandle] = {}
    for db_id in db_ids:
        print(f"Indexing {db_id} ...")
        db_path = dev_databases_dir / db_id / f"{db_id}.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda b: b.decode(errors="ignore")
        db_handles[db_id] = DbHandle(conn.cursor(), dev_databases_dir / db_id / "database_description")

    count = 0
    with open(OUTPUT_FPATH, "w") as f_out:
        for i, entry in enumerate(entries):
            db_id = entry["db_id"]
            # Matches prepare.py: evidence carries the literal-value hints (e.g. "triple
            # type bonds refers to bond_type = '#'") that the model needs to disambiguate.
            question = entry["evidence"] + "\n" + entry["question"]
            row = {
                "question": question,
                "gt_sql": entry["SQL"],
                "sql_context": db_handles[db_id].yaml_v2_context_for_question(question),
                "difficulty": entry["difficulty"],
                "db_id": db_id,
                "id": i,
            }
            f_out.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} BIRD dev entries (compact YAML schema) to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
