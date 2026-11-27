# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a PostgreSQL copy of the BIRD SQLite databases.

Every ``db_id`` becomes a schema of that name in one PostgreSQL database, so the grader needs a single connection
target (one pool) and selects the ``db_id`` per query with ``search_path``; a database per ``db_id`` would need a
connection pool per database (11 for dev, 69 for train). Column types follow ``setup_bird_spark`` exactly (same
``_typed_tables``): numeric when the data converts cleanly, else ``TEXT``; date/datetime columns stay ``TEXT``.
``TEXT`` columns use the ``"C"`` collation so ordering and comparison are bytewise, as in SQLite, rather than the
server's locale. Tables keep BIRD's names; column names are lower-cased (``postgres_column_name``), so an unquoted
column name finds its column, as any spelling does in SQLite. The loader quotes every identifier (some have spaces).

A ``db_id`` is loaded in one transaction and recorded, with its string-fallback columns and the layout version, in
``_bird_meta.loaded``. An already-recorded ``db_id`` is skipped; one recorded with an older layout is an error until
it is dropped and loaded again. The loader also creates the read-only login role the grader connects as
(the user and password of ``reader_dsn``): ``SELECT`` on the BIRD schemas only, read-only transactions by default.
Requires ``psycopg`` and ``pandas``.
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict

from resources_servers.bird_sql.setup_bird_spark import _typed_tables


logger = logging.getLogger(__name__)

_META_SCHEMA = "_bird_meta"
_PG_TYPE_BY_DTYPE = {"Int64": "BIGINT", "float64": "DOUBLE PRECISION"}
_TEXT_TYPE = 'TEXT COLLATE "C"'
# Version of the schema layout, recorded per db_id in _bird_meta.loaded. Layout 2 lower-cases column names; loads made
# before the version was recorded kept BIRD's column names and read as NULL.
_LAYOUT = 2


def postgres_column_name(column: str) -> str:
    """The name ``load_sqlite_db`` gives a BIRD column: lower case, which an unquoted name folds to in PostgreSQL.

    SQLite column names are case-insensitive, so two columns of one table can't map to the same name.
    """
    return column.lower()


def postgres_column_types(sqlite_path: Path) -> Dict[str, Dict[str, str]]:
    """``{table: {column: PostgreSQL type}}`` as ``load_sqlite_db`` stores them (collation not shown).

    Keyed by BIRD's column names; the loaded column is ``postgres_column_name(column)``.
    """
    return {
        table: {column: _PG_TYPE_BY_DTYPE.get(str(dtype), "TEXT") for column, dtype in df.dtypes.items()}
        for table, df, _fallbacks in _typed_tables(sqlite_path)
    }


def _to_pg(value: object) -> object:
    """``None`` for any null (``pd.NA``, NaN); strings lose NUL bytes, which PostgreSQL ``TEXT`` cannot hold."""
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        return value.replace("\x00", "")
    return value


def load_sqlite_db(conn: psycopg.Connection, db_id: str, sqlite_path: Path) -> List[Dict[str, str]]:
    """(Re)create schema ``db_id`` from ``sqlite_path`` in one transaction and record it as loaded.

    Returns the string-fallback records (see ``setup_bird_spark._typed_tables``).
    """
    schema = sql.Identifier(db_id)
    all_fallbacks: List[Dict[str, str]] = []
    with conn.transaction():
        conn.execute(sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(schema))
        conn.execute(sql.SQL("CREATE SCHEMA {}").format(schema))
        for table, df, fallbacks in _typed_tables(sqlite_path):
            table_ref = sql.Identifier(db_id, table)
            columns = [sql.Identifier(postgres_column_name(c)) for c in df.columns]
            column_defs = [
                sql.SQL("{} {}").format(col, sql.SQL(_PG_TYPE_BY_DTYPE.get(str(dtype), _TEXT_TYPE)))
                for col, dtype in zip(columns, df.dtypes)
            ]
            conn.execute(sql.SQL("CREATE TABLE {} ({})").format(table_ref, sql.SQL(", ").join(column_defs)))
            copy_sql = sql.SQL("COPY {} ({}) FROM STDIN").format(table_ref, sql.SQL(", ").join(columns))
            values_by_column = [df[c].tolist() for c in df.columns]
            with conn.cursor().copy(copy_sql) as copy:
                for row in zip(*values_by_column):
                    copy.write_row([_to_pg(v) for v in row])
            # Planner statistics now: autovacuum reaches a new table only a minute or more later, and until then a
            # grade can get a plan that runs past the timeout.
            conn.execute(sql.SQL("ANALYZE {}").format(table_ref))
            all_fallbacks += fallbacks
        conn.execute(
            sql.SQL("INSERT INTO {}.loaded (db_id, fallbacks, layout) VALUES (%s, %s, %s)").format(
                sql.Identifier(_META_SCHEMA)
            ),
            (db_id, json.dumps(all_fallbacks), _LAYOUT),
        )
    return all_fallbacks


def _ensure_reader_role(conn: psycopg.Connection, reader_dsn: str, db_ids: List[str]) -> None:
    """Create the grader's login role (from ``reader_dsn``) with ``SELECT`` on ``db_ids`` only."""
    params = conninfo_to_dict(reader_dsn)
    user, password = params.get("user"), params.get("password")
    admin_user = conn.execute("SELECT current_user").fetchone()[0]
    if not user or user == admin_user:
        logger.warning("BIRD Postgres: grader DSN uses the loader's role %r; no read-only role is enforced", user)
        return
    role = sql.Identifier(user)
    with conn.transaction():
        if conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (user,)).fetchone() is None:
            conn.execute(sql.SQL("CREATE ROLE {} LOGIN").format(role))
        if password:
            conn.execute(sql.SQL("ALTER ROLE {} PASSWORD {}").format(role, sql.Literal(password)))
        conn.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(role))
        for schema in (_META_SCHEMA, *db_ids):
            conn.execute(sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(sql.Identifier(schema), role))
            conn.execute(sql.SQL("GRANT SELECT ON ALL TABLES IN SCHEMA {} TO {}").format(sql.Identifier(schema), role))


def _loaded_layouts(conn: psycopg.Connection) -> Dict[str, Optional[int]]:
    """``{db_id: layout}`` for every recorded ``db_id``.

    Read through ``to_jsonb`` so a meta table from before the layout column existed reads as NULL instead of failing.
    """
    query = sql.SQL("SELECT db_id, (to_jsonb(l) ->> 'layout')::int FROM {}.loaded AS l").format(
        sql.Identifier(_META_SCHEMA)
    )
    return dict(conn.execute(query).fetchall())


def _check_layout(loaded: Dict[str, Optional[int]], db_ids: List[str]) -> None:
    """Raise if any of ``db_ids`` was loaded with another layout: the grader would look up names it doesn't have."""
    stale = sorted(db_id for db_id in db_ids if db_id in loaded and loaded[db_id] != _LAYOUT)
    if stale:
        raise RuntimeError(
            f"BIRD db_ids loaded into PostgreSQL with an older layout (this code needs layout {_LAYOUT}, lower-case "
            f"column names): {stale}. Drop their schemas and their rows in {_META_SCHEMA}.loaded, or use a new "
            "database, then load again"
        )


def check_bird_postgres(databases_dir: Path, reader_dsn: str) -> List[str]:
    """Raise unless every ``db_id`` under ``databases_dir`` is recorded as loaded, with this layout. Returns the db_ids.

    For a grader that only reads: loading needs an admin DSN (see ``ensure_bird_postgres``).
    """
    db_ids = [p.parent.name for p in sorted(databases_dir.glob("*/[!.]*.sqlite"))]
    with psycopg.connect(reader_dsn) as conn:
        loaded = _loaded_layouts(conn)
    missing = sorted(set(db_ids) - set(loaded))
    if missing:
        raise RuntimeError(f"BIRD db_ids not loaded into PostgreSQL: {missing}; set postgres_admin_dsn to load them")
    _check_layout(loaded, db_ids)
    return db_ids


def ensure_bird_postgres(databases_dir: Path, admin_dsn: str, reader_dsn: str) -> List[str]:
    """Load every ``<db_id>/<db_id>.sqlite`` under ``databases_dir`` into PostgreSQL (idempotent). Returns the db_ids.

    ``admin_dsn`` must be able to create schemas and roles; the grader connects with ``reader_dsn``.
    """
    db_ids = []
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(_META_SCHEMA)))
        conn.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {}.loaded (db_id TEXT PRIMARY KEY, fallbacks JSONB, loaded_at TIMESTAMPTZ DEFAULT now())"
            ).format(sql.Identifier(_META_SCHEMA))
        )
        conn.execute(
            sql.SQL("ALTER TABLE {}.loaded ADD COLUMN IF NOT EXISTS layout INT").format(sql.Identifier(_META_SCHEMA))
        )
        loaded = _loaded_layouts(conn)
        sqlite_paths = sorted(databases_dir.glob("*/[!.]*.sqlite"))
        _check_layout(loaded, [p.parent.name for p in sqlite_paths])
        for sqlite_path in sqlite_paths:
            db_id = sqlite_path.parent.name
            db_ids.append(db_id)
            if db_id in loaded:
                continue
            logger.info("Loading %s into PostgreSQL ...", db_id)
            for record in load_sqlite_db(conn, db_id, sqlite_path):
                logger.warning("BIRD Postgres load: stored as text: %s", record)
        _ensure_reader_role(conn, reader_dsn, db_ids)
    return db_ids
