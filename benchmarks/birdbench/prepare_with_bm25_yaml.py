# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare BIRD benchmark data with a per-column YAML schema in ``sql_context``.

Alternate rendering of the same retrieval this benchmark already does elsewhere
(``prepare_with_bm25.py``'s baseline sample + per-question BM25 hits, both from
``build_db_values.py``): instead of dumping selected rows as ``INSERT`` statements inside a
raw SQL schema, this represents the schema as YAML, one entry per table, each listing its
columns' data type (from SQLite's own ``PRAGMA table_info``), a human-readable description
(from BIRD's ``database_description/<table>.csv``, aligned to real column names via
``original_column_name``), and sampled values.

Produces the same row schema as ``prepare.py``/``prepare_with_bm25.py`` (``question, gt_sql,
sql_context, difficulty, db_id, id``) -- this is purely an alternate ``sql_context``
representation, kept alongside (not replacing) the ``INSERT``-based one.

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
OUTPUT_FPATH = DATA_DIR / "birdbench_benchmark_bm25_yaml.jsonl"


def prepare() -> Path:
    """Download BIRD dev, produce ``birdbench_benchmark_bm25_yaml.jsonl``. Returns the output path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dev_databases_dir = ensure_bird_sql()
    # dev_databases_dir == <base>/dev_20240627/dev_databases → dev.json is one level up.
    dev_json_path = dev_databases_dir.parent / "dev.json"
    if not dev_json_path.exists():
        raise RuntimeError(f"Expected BIRD dev.json at {dev_json_path}")

    with open(dev_json_path) as f:
        entries: List[Dict[str, Any]] = json.load(f)

    print("Building per-db schemas (types, descriptions, BM25 indexes)...")
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
                "sql_context": db_handles[db_id].yaml_context_for_question(question),
                "difficulty": entry["difficulty"],
                "db_id": db_id,
                "id": i,
            }
            f_out.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} BIRD dev entries (YAML schema) to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
