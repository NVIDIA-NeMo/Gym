# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare BIRD benchmark data with BM25-selected example rows in ``sql_context``.

Unlike ``prepare.py``, which shows whichever rows happen to come first in
``sqlite3.Connection.iterdump()`` (truncated to 10 consecutive ``INSERT``s per table),
every row shown here was chosen because it's "interesting" for the question: either it
was part of a baseline per-column sample (``build_db_values.py``'s ``_sample_table_values``,
independent of the question) or it substring-matched the question well via BM25
(``build_db_values.py``'s ``_relevant_hits_for_question``). See ``build_db_values.py`` for
the full retrieval design and why it's ``bm25s``-based rather than pyserini/Java-based.

Produces the same row schema as ``prepare.py`` (``question, gt_sql, sql_context, difficulty,
db_id, id``).

Requires ``bm25s`` (``pip install bm25s``), unlike plain ``prepare.py``.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.birdbench.build_db_values import build_db_values, build_sql_context
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "birdbench_benchmark_bm25.jsonl"


def prepare() -> Path:
    """Download BIRD dev, produce ``birdbench_benchmark_bm25.jsonl``. Returns the output path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    dev_databases_dir = ensure_bird_sql()
    # dev_databases_dir == <base>/dev_20240627/dev_databases → dev.json is one level up.
    dev_json_path = dev_databases_dir.parent / "dev.json"
    if not dev_json_path.exists():
        raise RuntimeError(f"Expected BIRD dev.json at {dev_json_path}")

    with open(dev_json_path) as f:
        entries: List[Dict[str, Any]] = json.load(f)

    print("Building per-db schemas and selecting per-question example rows...")
    create_statements_by_db, per_question_rows = build_db_values(dev_databases_dir, dev_json_path)
    print(f"Selected example rows for {len(per_question_rows)} questions.")

    count = 0
    with open(OUTPUT_FPATH, "w") as f_out:
        for i, (entry, question_row) in enumerate(zip(entries, per_question_rows)):
            assert question_row["id"] == i, "dev.json and build_db_values() must enumerate in the same order"

            db_id = entry["db_id"]
            row = {
                # Matches prepare.py: evidence carries the literal-value hints (e.g. "triple
                # type bonds refers to bond_type = '#'") that the model needs to disambiguate.
                "question": entry["evidence"] + "\n" + entry["question"],
                "gt_sql": entry["SQL"],
                "sql_context": build_sql_context(create_statements_by_db[db_id], question_row["insert_statements"]),
                "difficulty": entry["difficulty"],
                "db_id": db_id,
                "id": i,
            }
            f_out.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} BIRD dev entries (schema + BM25-selected rows) to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
