# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare BIRD benchmark data.

Per-row output schema: ``{question, gt_sql, sql_context, difficulty, db_id, id}``.

The schema dump (``sql_context``) is produced via a stripped-down reimplementation of
``sqlite3.Connection.iterdump()`` (see ``_iterdump_no_fk_check``) that skips the
``PRAGMA foreign_key_check`` probe -- some BIRD dev databases declare foreign keys
without a matching unique index/PK, which makes that probe itself raise
``OperationalError: foreign key mismatch``. INSERT-chain truncation is then applied:
at most 10 consecutive ``INSERT`` statements are kept per run, and long
``INSERT ... VALUES (...), ...`` chains are collapsed after 10 tuples.

Calls ``ensure_bird_sql()`` so the download cache is shared with the
``bird_sql`` resource server (avoids a duplicate ~1.4 GB download).
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


def _iterdump_no_fk_check(con: sqlite3.Connection):
    """Like ``sqlite3.Connection.iterdump()``, minus the leading ``PRAGMA foreign_key_check``.

    Some BIRD dev databases (e.g. ``european_football_2.sqlite``) declare a foreign key
    whose parent columns aren't backed by a unique index/PK. ``iterdump()`` runs
    ``PRAGMA foreign_key_check`` up front to decide whether to emit
    ``PRAGMA foreign_keys=OFF;``, and SQLite raises ``OperationalError: foreign key
    mismatch`` on that PRAGMA for such schemas -- before any dumping happens. We don't
    need FK-violation detection for a schema/data text dump, so reimplement the dump
    without that probe.
    """
    from sqlite3.dump import _quote_name, _quote_value

    cu = con.cursor()
    cu.row_factory = None
    yield "BEGIN TRANSACTION;"

    q = """
        SELECT "name", "type", "sql"
        FROM "sqlite_master"
            WHERE "sql" NOT NULL AND "type" == 'table'
            ORDER BY "name"
        """
    schema_res = cu.execute(q)
    sqlite_sequence = []
    writeable_schema = False
    for table_name, _type, sql in schema_res.fetchall():
        if table_name == "sqlite_sequence":
            rows = cu.execute('SELECT * FROM "sqlite_sequence";')
            sqlite_sequence = ["DELETE FROM \"sqlite_sequence\""]
            sqlite_sequence += [
                f"INSERT INTO \"sqlite_sequence\" VALUES({_quote_value(table_name)},{seq_value})"
                for table_name, seq_value in rows.fetchall()
            ]
            continue
        elif table_name == "sqlite_stat1":
            yield 'ANALYZE "sqlite_master";'
        elif table_name.startswith("sqlite_"):
            continue
        elif sql.startswith("CREATE VIRTUAL TABLE"):
            if not writeable_schema:
                writeable_schema = True
                yield "PRAGMA writable_schema=ON;"
            yield (
                "INSERT INTO sqlite_master(type,name,tbl_name,rootpage,sql)"
                "VALUES('table',{0},{0},0,{1});".format(_quote_value(table_name), _quote_value(sql))
            )
        else:
            yield f"{sql};"

        table_name_ident = _quote_name(table_name)
        res = cu.execute(f"PRAGMA table_info({table_name_ident})")
        column_names = [str(table_info[1]) for table_info in res.fetchall()]
        q = "SELECT 'INSERT INTO {0} VALUES('{1}')' FROM {0};".format(
            table_name_ident,
            "','".join(f"||quote({_quote_name(col)})||" for col in column_names),
        )
        query_res = cu.execute(q)
        for row in query_res:
            yield f"{row[0]};"

    q = """
        SELECT "name", "type", "sql"
        FROM "sqlite_master"
            WHERE "sql" NOT NULL AND "type" IN ('index', 'trigger', 'view')
        """
    schema_res = cu.execute(q)
    for _name, _type, sql in schema_res.fetchall():
        yield f"{sql};"

    if writeable_schema:
        yield "PRAGMA writable_schema=OFF;"

    for row in sqlite_sequence:
        yield f"{row};"

    yield "COMMIT;"


def _read_tables_info(dev_databases_dir: Path) -> dict[str, str]:
    """Dump each BIRD database's schema + (truncated) inserts to a string."""
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
            for line in _iterdump_no_fk_check(con):
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
