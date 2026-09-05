# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare BIRD benchmark data.

Per-row output schema: ``{question, gt_sql, sql_context, difficulty, db_id, id}``.

``sql_context`` is a per-table, per-column schema listing each column's data type (from
SQLite's own ``PRAGMA table_info``), a human-readable description (from BIRD's
``database_description/<table>.csv``, aligned to real column names via
``original_column_name``), and example values, followed by a "#### Foreign key" section
(from ``PRAGMA foreign_key_list``). Example values combine a baseline per-column sample
(independent of the question) with per-question BM25 hits (via ``bm25s`` -- pure Python,
no JVM) against the question text, so the values shown are ones actually likely to be
relevant. See ``build_db_values.py`` for the full retrieval and rendering design.

Calls ``ensure_bird_sql()`` so the download cache is shared with the ``bird_sql`` resource
server (avoids a duplicate ~1.4 GB download).

Requires ``bm25s`` and ``nltk`` (``pip install bm25s nltk``), unlike a plain schema dump.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.birdbench.build_db_values import DbHandle
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "birdbench_benchmark.jsonl"


def prepare() -> Path:
    """Download BIRD dev, produce ``birdbench_benchmark.jsonl``. Returns the output path."""
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
            # Evidence carries the literal-value hints (e.g. "triple type bonds refers to
            # bond_type = '#'") that the model needs to disambiguate.
            question = entry["evidence"] + "\n" + entry["question"]
            row = {
                "question": question,
                "gt_sql": entry["SQL"],
                "sql_context": db_handles[db_id].sql_context_for_question(question),
                "difficulty": entry["difficulty"],
                "db_id": db_id,
                "id": i,
            }
            f_out.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} BIRD dev entries to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
