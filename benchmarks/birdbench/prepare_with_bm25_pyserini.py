# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare BIRD benchmark data using pyserini-retrieved example rows in ``sql_context``.

Same output as ``prepare_with_bm25.py``, but sourced from a precomputed ``extract_values.py``
dump (``data/db_values_pyserini.json``) instead of running retrieval itself -- so this script
only needs SQLite, never pyserini. Row selection and rendering reuse
``build_db_values.py``'s ``_select_example_rows``/``build_sql_context`` verbatim, so both
engines produce ``sql_context`` in exactly the same format; only how the "interesting"
(table, column, value) pairs were found differs (``bm25s`` vs. real pyserini/Lucene).

Usage::

    # On a machine with pyserini (e.g. remote):
    bash build_index.sh
    python -m benchmarks.birdbench.extract_values --db-content-index-path <path from build_index.sh>
    # copy data/db_values_pyserini.json to this machine, then:
    python -m benchmarks.birdbench.prepare_with_bm25_pyserini
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from benchmarks.birdbench.build_db_values import (
    _column_names,
    _create_table_statements,
    _select_example_rows,
    _table_names,
    build_sql_context,
)
from benchmarks.birdbench.extract_values import OUTPUT_FPATH as VALUES_FPATH
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


BENCHMARK_DIR = Path(__file__).parent
DATA_DIR = BENCHMARK_DIR / "data"
OUTPUT_FPATH = DATA_DIR / "birdbench_benchmark_bm25_pyserini.jsonl"


def prepare(values_fpath: Path = VALUES_FPATH) -> Path:
    """Reads ``extract_values.py``'s dump + BIRD dev, produces the pyserini-variant jsonl."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not values_fpath.exists():
        raise RuntimeError(f"Expected {values_fpath} -- run extract_values.py on a machine with pyserini first.")
    with open(values_fpath) as f:
        values_dump = json.load(f)

    dev_databases_dir = ensure_bird_sql()
    dev_json_path = dev_databases_dir.parent / "dev.json"
    if not dev_json_path.exists():
        raise RuntimeError(f"Expected BIRD dev.json at {dev_json_path}")
    with open(dev_json_path) as f:
        entries: List[Dict[str, Any]] = json.load(f)

    db_ids = sorted({entry["db_id"] for entry in entries})
    cursors, table_names_by_db, column_names_by_db = {}, {}, {}
    create_statements_by_db, sampled_values_by_db = {}, {}
    for db_id in db_ids:
        db_path = dev_databases_dir / db_id / f"{db_id}.sqlite"
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda b: b.decode(errors="ignore")
        cur = conn.cursor()
        cursors[db_id] = cur

        table_names_by_db[db_id] = _table_names(cur)
        column_names_by_db[db_id] = {t: _column_names(cur, t) for t in table_names_by_db[db_id]}
        create_statements_by_db[db_id] = _create_table_statements(cur, table_names_by_db[db_id])
        sampled_values_by_db[db_id] = {
            (rec["table"], rec["column"]): rec["values"] for rec in values_dump["sampled_values_by_db"][db_id]
        }

    relevant_hits_by_id = {row["id"]: row["relevant_hits"] for row in values_dump["rows"]}

    count = 0
    with open(OUTPUT_FPATH, "w") as f_out:
        for i, entry in enumerate(entries):
            db_id = entry["db_id"]
            statements_by_table = _select_example_rows(
                cursors[db_id],
                table_names_by_db[db_id],
                column_names_by_db[db_id],
                sampled_values_by_db[db_id],
                relevant_hits_by_id[i],
            )
            insert_statements = [s for statements in statements_by_table.values() for s in statements]

            row = {
                # Matches prepare.py: evidence carries the literal-value hints (e.g. "triple
                # type bonds refers to bond_type = '#'") that the model needs to disambiguate.
                "question": entry["evidence"] + "\n" + entry["question"],
                "gt_sql": entry["SQL"],
                "sql_context": build_sql_context(create_statements_by_db[db_id], insert_statements),
                "difficulty": entry["difficulty"],
                "db_id": db_id,
                "id": i,
            }
            f_out.write(json.dumps(row) + "\n")
            count += 1

    print(f"Wrote {count} BIRD dev entries (schema + pyserini-selected rows) to {OUTPUT_FPATH}")
    return OUTPUT_FPATH


if __name__ == "__main__":
    prepare()
