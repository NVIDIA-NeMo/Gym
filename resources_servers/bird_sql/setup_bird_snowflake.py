# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Snowflake copy of the BIRD SQLite databases that the ``snowflake`` dialect grades on.

Two engines sit behind one interface (``tables``, ``connect``, ``fetch_all``, ``interrupt``), chosen by the server's
``snowflake_engine``:

- ``fakesnow`` (the default; no Snowflake account): ``fakesnow`` transpiles each Snowflake statement with ``sqlglot``
  and runs it on DuckDB. ``ensure_bird_snowflake`` builds ``<db_dir>/BIRD.db`` on first start and reopens it read-only,
  so a model's DDL or DML fails instead of changing the data other rollouts read.
- ``snowflake``: a real account, through ``snowflake-connector-python`` with a named connection from the connector's
  own ``connections.toml``. ``check_bird_snowflake_account`` expects the data already loaded in the same layout;
  ``load_bird_snowflake_account`` loads it, once, through an admin connection. Calibrated against the stand-in on a
  real account (see the README).

Every ``db_id`` becomes a schema of that name in one database, ``BIRD``. Column types follow ``setup_bird_spark``
exactly (same ``_typed_tables``) and are shown to the model as Snowflake types: ``NUMBER(38,0)``, ``FLOAT`` or
``VARCHAR``; date/datetime columns stay ``VARCHAR``. Identifiers keep BIRD's original names (mixed case, spaces) and are
created quoted, so on Snowflake they are case-sensitive.

A ``db_id`` is loaded in one transaction and recorded, with its string-fallback columns and the layout version, in
``_bird_meta.loaded``; an already-recorded ``db_id`` is skipped, and one recorded with another layout is an error.
``snowflake_column_types`` (used by ``prepare``) needs only ``pandas``.
"""

import contextlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Dict, FrozenSet, List, Optional

from resources_servers.bird_sql.setup_bird_spark import _typed_tables


logger = logging.getLogger(__name__)

SNOWFLAKE_DATABASE = "BIRD"
_DB_FILE = f"{SNOWFLAKE_DATABASE}.db"
_META_SCHEMA = "_bird_meta"
# DuckDB storage type and the Snowflake type shown to the model, by pandas dtype from ``_typed_tables``.
_TYPES_BY_DTYPE = {"Int64": ("BIGINT", "NUMBER(38,0)"), "float64": ("DOUBLE", "FLOAT")}
_TEXT_TYPES = ("VARCHAR", "VARCHAR")
# Version of the schema layout, recorded per db_id in _bird_meta.loaded. Layout 1: BIRD's original names, quoted.
_LAYOUT = 1
# Comment on every schema load_bird_snowflake_account creates in an account: the only schemas it may replace.
_LOADER_COMMENT = "BIRD db_id loaded by Gym bird_sql (load_bird_snowflake_account)"

Catalog = Dict[str, Dict[str, FrozenSet[str]]]


def snowflake_column_types(sqlite_path: Path) -> Dict[str, Dict[str, str]]:
    """``{table: {column: Snowflake type}}`` of the columns ``load_sqlite_db`` creates (in DuckDB, ``_TYPES_BY_DTYPE``)."""
    return {
        table: {column: _TYPES_BY_DTYPE.get(str(dtype), _TEXT_TYPES)[1] for column, dtype in df.dtypes.items()}
        for table, df, _fallbacks in _typed_tables(sqlite_path)
    }


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def load_sqlite_db(duck: Any, db_id: str, sqlite_path: Path) -> List[Dict[str, str]]:
    """(Re)create schema ``db_id`` of ``BIRD`` from ``sqlite_path`` in one transaction and record it as loaded.

    Returns the string-fallback records (see ``setup_bird_spark._typed_tables``).
    """
    schema = f"{SNOWFLAKE_DATABASE}.{_quote(db_id)}"
    all_fallbacks: List[Dict[str, str]] = []
    duck.execute("BEGIN TRANSACTION")
    try:
        duck.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        duck.execute(f"CREATE SCHEMA {schema}")
        for table, df, fallbacks in _typed_tables(sqlite_path):
            types = [_TYPES_BY_DTYPE.get(str(dtype), _TEXT_TYPES)[0] for dtype in df.dtypes]
            column_defs = ", ".join(f"{_quote(c)} {t}" for c, t in zip(df.columns, types))
            casts = ", ".join(f"CAST({_quote(c)} AS {t})" for c, t in zip(df.columns, types))
            duck.execute(f"CREATE TABLE {schema}.{_quote(table)} ({column_defs})")
            duck.register("_bird_df", df)
            try:
                duck.execute(f"INSERT INTO {schema}.{_quote(table)} SELECT {casts} FROM _bird_df")
            finally:
                duck.unregister("_bird_df")
            all_fallbacks += fallbacks
        duck.execute(
            f"INSERT INTO {SNOWFLAKE_DATABASE}.{_META_SCHEMA}.loaded (db_id, fallbacks, layout) VALUES (?, ?, ?)",
            [db_id, json.dumps(all_fallbacks), _LAYOUT],
        )
        duck.execute("COMMIT")
    except BaseException:
        duck.execute("ROLLBACK")
        raise
    return all_fallbacks


def _check_layout(loaded: Dict[str, Optional[int]], db_ids: List[str], where: str) -> None:
    """Raise if any of ``db_ids`` was loaded with another layout: the grader would look up names it doesn't have."""
    stale = sorted(db_id for db_id in db_ids if db_id in loaded and loaded[db_id] != _LAYOUT)
    if stale:
        raise RuntimeError(
            f"BIRD db_ids loaded into {where} with another layout (this code needs layout {_LAYOUT}): {stale}. "
            "Delete them and load again"
        )


@dataclass(frozen=True)
class LocalSnowflake:
    """The ``fakesnow`` engine: a fakesnow instance over a read-only ``BIRD.db``, plus its catalog.

    ``tables`` is ``{db_id: {table: columns}}`` with exact (case-sensitive) names, as Snowflake would resolve them.
    """

    fake_snow: Any
    tables: Catalog

    def connect(self, db_id: str) -> Any:
        """A new connection (its own DuckDB cursor, so safe to use from another thread) with ``db_id`` selected."""
        return self.fake_snow.connect(database=SNOWFLAKE_DATABASE, schema=db_id)

    def fetch_all(self, cur: Any, max_rows: Optional[int] = None) -> Optional[List[tuple]]:
        """All rows of ``cur``'s result, by position (``None`` if there are more than ``max_rows``, without converting
        them); the result then leaves fakesnow's cache.

        fakesnow's ``fetchall()`` builds a dict per row, which keeps only one of two same-named columns
        (``SELECT T1.name, T2.name``); its Arrow table, private as ``_arrow_table``, keeps both. fakesnow also keeps
        every completed query's result in ``results_cache``, shared by all connections and never freed (it serves
        ``RESULT_SCAN``), so memory would grow with every graded query: this query's own entry is removed once read,
        and other queries' entries are left alone.
        """
        try:
            if max_rows is not None and cur._arrow_table.num_rows > max_rows:
                return None
            return list(zip(*(column.to_pylist() for column in cur._arrow_table.columns)))
        finally:
            self.fake_snow.results_cache.pop(cur.sfqid, None)

    @staticmethod
    def interrupt(conn: Any) -> None:
        """Stop the query running on ``conn``, if any. fakesnow exposes the DuckDB connection only as ``_duck_conn``."""
        import duckdb

        with contextlib.suppress(duckdb.Error):  # already closed: the query finished first
            conn._duck_conn.interrupt()


@dataclass(frozen=True)
class AccountSnowflake:
    """The ``snowflake`` engine: a Snowflake account, one connection per query.

    ``connection_name`` names a connection in the connector's ``connections.toml`` (``None``: its default), so no
    credential goes through this server's config. The server enforces ``timeout_s`` (``STATEMENT_TIMEOUT_IN_SECONDS``).
    """

    connection_name: Optional[str]
    timeout_s: float
    tables: Catalog
    # The server stops a query at STATEMENT_TIMEOUT_IN_SECONDS; the client's timer waits this much longer, so a timeout
    # is the server's verdict rather than a race with it.
    timeout_buffer_s: ClassVar[float] = 10.0

    def connect(self, db_id: str) -> Any:
        """A new connection with ``db_id`` selected and only the connection's own role active.

        A user's default secondary roles are often ``ALL``: the session would then hold every role the user has and
        could write through any of them, whatever read-only role the connection names. They are turned off here.
        """
        import snowflake.connector

        kwargs = {"connection_name": self.connection_name} if self.connection_name else {}
        conn = snowflake.connector.connect(
            **kwargs,
            database=SNOWFLAKE_DATABASE,
            schema=_quote(db_id),
            session_parameters={"STATEMENT_TIMEOUT_IN_SECONDS": max(1, round(self.timeout_s))},
        )
        try:
            conn.cursor().execute("USE SECONDARY ROLES NONE")
        except BaseException:
            conn.close()
            raise
        return conn

    @staticmethod
    def fetch_all(cur: Any, max_rows: Optional[int] = None) -> Optional[List[tuple]]:
        """All rows, as tuples by position; ``None``, without downloading them, if there are more than ``max_rows``."""
        if max_rows is not None and (cur.rowcount or 0) > max_rows:
            return None
        return cur.fetchall()

    @staticmethod
    def interrupt(conn: Any) -> None:
        """Nothing to do: the server stops the query at ``STATEMENT_TIMEOUT_IN_SECONDS``."""


def _catalog(rows: List[tuple], db_ids: List[str]) -> Catalog:
    tables: Dict[str, Dict[str, set]] = {db_id: {} for db_id in db_ids}
    for schema, table, column in rows:
        if schema in tables:
            tables[schema].setdefault(table, set()).add(column)
    return {db_id: {t: frozenset(cs) for t, cs in by_table.items()} for db_id, by_table in tables.items()}


def _db_ids(databases_dir: Path) -> List[str]:
    return [p.parent.name for p in sorted(databases_dir.glob("*/[!.]*.sqlite"))]


def _loaded_layouts(db_file: Path) -> Dict[str, Optional[int]]:
    """``{db_id: layout}`` recorded in ``db_file``; a ``db_id`` recorded without a layout reads as ``None``."""
    import duckdb

    if not db_file.exists():
        return {}
    with duckdb.connect(str(db_file), read_only=True) as duck:
        try:
            rows = duck.execute(f"SELECT db_id, to_json(l) ->> 'layout' FROM {_META_SCHEMA}.loaded AS l").fetchall()
        except duckdb.CatalogException:
            return {}
    return {db_id: None if layout is None else int(layout) for db_id, layout in rows}


def ensure_bird_snowflake(databases_dir: Path, db_dir: Path) -> LocalSnowflake:
    """Load every ``<db_id>/<db_id>.sqlite`` under ``databases_dir`` into ``<db_dir>/BIRD.db`` (idempotent).

    Returns the read-only ``fakesnow`` engine over it. Queries run single-threaded inside DuckDB: a parallel float
    ``SUM`` adds in a different order on every run, so the same answer could pass or fail at random (as in PostgreSQL).
    """
    from fakesnow.instance import FakeSnow

    db_dir.mkdir(parents=True, exist_ok=True)
    db_file = db_dir / _DB_FILE
    sqlite_paths = sorted(databases_dir.glob("*/[!.]*.sqlite"))
    db_ids = [p.parent.name for p in sqlite_paths]
    loaded = _loaded_layouts(db_file)
    _check_layout(loaded, db_ids, str(db_file))
    if set(db_ids) - set(loaded):
        # fakesnow creates BIRD.db with the information-schema views and macros its transpiled SQL relies on.
        builder = FakeSnow(db_path=db_dir, create_schema_on_connect=False)
        try:
            duck = builder.connect(database=SNOWFLAKE_DATABASE)._duck_conn
            duck.execute(f"CREATE SCHEMA IF NOT EXISTS {SNOWFLAKE_DATABASE}.{_META_SCHEMA}")
            duck.execute(
                f"CREATE TABLE IF NOT EXISTS {SNOWFLAKE_DATABASE}.{_META_SCHEMA}.loaded (db_id VARCHAR PRIMARY KEY, "
                "fallbacks VARCHAR, layout INTEGER, loaded_at TIMESTAMP DEFAULT current_timestamp)"
            )
            for sqlite_path, db_id in zip(sqlite_paths, db_ids):
                if db_id in loaded:
                    continue
                logger.info("Loading %s into the local Snowflake stand-in ...", db_id)
                for record in load_sqlite_db(duck, db_id, sqlite_path):
                    logger.warning("BIRD Snowflake load: stored as string: %s", record)
            duck.execute("CHECKPOINT")
        finally:
            builder.duck_conn.close()

    # Attached read-only before fakesnow's first connect, so fakesnow uses it instead of attaching it writable.
    fake_snow = FakeSnow(db_path=db_dir, create_database_on_connect=False, create_schema_on_connect=False)
    fake_snow.duck_conn.execute(f"ATTACH '{db_file}' AS {SNOWFLAKE_DATABASE} (READ_ONLY)")
    fake_snow.duck_conn.execute("SET threads = 1")
    # No file or network access for model SQL: DuckDB's table functions (read_csv, sqlite_scan, ...) would otherwise read
    # any file the server can, including the gold SQLite databases. Locked, so a query can't turn it back on.
    fake_snow.duck_conn.execute("SET enable_external_access = false")
    fake_snow.duck_conn.execute("SET lock_configuration = true")
    rows = fake_snow.duck_conn.execute(
        "SELECT schema_name, table_name, column_name FROM duckdb_columns() WHERE database_name = ?",
        [SNOWFLAKE_DATABASE],
    ).fetchall()
    return LocalSnowflake(fake_snow=fake_snow, tables=_catalog(rows, db_ids))


def check_bird_snowflake_account(
    databases_dir: Path, connection_name: Optional[str], timeout_s: float
) -> AccountSnowflake:
    """Raise unless every ``db_id`` under ``databases_dir`` is loaded, with this layout, in the account's ``BIRD``.

    Returns the ``snowflake`` engine with the account's catalog. ``load_bird_snowflake_account`` loads an account.
    """
    db_ids = _db_ids(databases_dir)
    engine = AccountSnowflake(connection_name=connection_name, timeout_s=timeout_s, tables={})
    with contextlib.closing(engine.connect(_META_SCHEMA)) as conn:
        cur = conn.cursor()
        cur.execute(f"SELECT db_id, layout FROM {SNOWFLAKE_DATABASE}.{_quote(_META_SCHEMA)}.loaded")
        loaded = dict(cur.fetchall())
        missing = sorted(set(db_ids) - set(loaded))
        if missing:
            raise RuntimeError(f"BIRD db_ids not loaded into the Snowflake account: {missing}")
        _check_layout(loaded, db_ids, "the Snowflake account")
        cur.execute(
            f"SELECT table_schema, table_name, column_name FROM {SNOWFLAKE_DATABASE}.information_schema.columns"
        )
        tables = _catalog(cur.fetchall(), db_ids)
    # The catalog lists only what the connection's role may read. A db_id with no visible table (a missing grant, a
    # dropped schema) would refuse every model query on it as an unknown table and score it 0.
    unreadable = sorted(db_id for db_id in db_ids if not tables[db_id])
    if unreadable:
        raise RuntimeError(
            f"BIRD db_ids whose tables the Snowflake connection's role can't see: {unreadable}. Grant it SELECT on "
            "them (load_bird_snowflake_account's grant_to does), or load them again"
        )
    return AccountSnowflake(connection_name=connection_name, timeout_s=timeout_s, tables=tables)


def _load_account_db(conn: Any, db_id: str, sqlite_path: Path) -> List[Dict[str, str]]:
    """(Re)create schema ``db_id`` of the account's ``BIRD`` from ``sqlite_path`` and record it as loaded.

    Only a schema this loader created (marked by ``_LOADER_COMMENT``, for example one left by a failed load) is
    replaced; any other schema of that name raises, since on a shared account it may be someone else's. Snowflake
    commits DDL on its own, so unlike ``load_sqlite_db`` this can't be one transaction: the ``loaded`` row is written
    last, after every table's row count is checked, and the schema is dropped if anything fails.
    """
    from snowflake.connector.pandas_tools import write_pandas

    cur = conn.cursor()
    schema = f"{SNOWFLAKE_DATABASE}.{_quote(db_id)}"
    all_fallbacks: List[Dict[str, str]] = []
    cur.execute(f"SHOW SCHEMAS LIKE %s IN DATABASE {SNOWFLAKE_DATABASE}", (db_id,))
    names = [d[0].lower() for d in cur.description]
    comments = [row[names.index("comment")] for row in cur.fetchall() if row[names.index("name")] == db_id]
    if comments and comments[0] != _LOADER_COMMENT:
        raise RuntimeError(f"Schema {schema} already exists and was not created by this loader; drop it first")
    cur.execute(f"CREATE {'OR REPLACE ' if comments else ''}SCHEMA {schema} COMMENT = %s", (_LOADER_COMMENT,))
    try:
        for table, df, fallbacks in _typed_tables(sqlite_path):
            types = [_TYPES_BY_DTYPE.get(str(dtype), _TEXT_TYPES)[1] for dtype in df.dtypes]
            column_defs = ", ".join(f"{_quote(c)} {t}" for c, t in zip(df.columns, types))
            cur.execute(f"CREATE TABLE {schema}.{_quote(table)} ({column_defs})")
            if len(df):
                write_pandas(conn, df, table, database=SNOWFLAKE_DATABASE, schema=db_id, chunk_size=200_000)
            (count,) = cur.execute(f"SELECT COUNT(*) FROM {schema}.{_quote(table)}").fetchone()
            if count != len(df):
                raise RuntimeError(f"{db_id}.{table}: {count} rows loaded into Snowflake, {len(df)} in SQLite")
            all_fallbacks += fallbacks
        cur.execute(
            f"INSERT INTO {SNOWFLAKE_DATABASE}.{_quote(_META_SCHEMA)}.loaded (db_id, fallbacks, layout) "
            "VALUES (%s, %s, %s)",
            (db_id, json.dumps(all_fallbacks), _LAYOUT),
        )
    except BaseException:
        cur.execute(f"DROP SCHEMA IF EXISTS {schema}")
        raise
    return all_fallbacks


def load_bird_snowflake_account(databases_dir: Path, connection_name: Optional[str], grant_to: Optional[str]) -> None:
    """Load every ``db_id`` under ``databases_dir`` that the account's ``BIRD`` doesn't have yet, in this layout.

    ``connection_name`` is an admin connection in the connector's ``connections.toml``, with a warehouse and a role that
    may create schemas in the existing database ``BIRD``. Unlike the grader's connections, it keeps the user's
    secondary roles, on purpose: loading may need the privileges of any of them. ``grant_to`` names the grader's
    read-only role: it gets USAGE on ``BIRD`` and on every ``db_id``'s schema and SELECT on their tables, each
    ``db_id`` right after it loads (so a later failure leaves the loaded ones readable) and on every run, since a
    reload replaces the tables and FUTURE grants need MANAGE GRANTS. Run it once before serving with
    ``snowflake_engine: snowflake``.
    """
    import snowflake.connector

    if grant_to is not None and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_$]*", grant_to):
        raise ValueError(f"grant_to must be a plain role name, got {grant_to!r}")
    sqlite_paths = sorted(databases_dir.glob("*/[!.]*.sqlite"))
    db_ids = [p.parent.name for p in sqlite_paths]
    meta = f"{SNOWFLAKE_DATABASE}.{_quote(_META_SCHEMA)}"
    kwargs = {"connection_name": connection_name} if connection_name else {}
    with contextlib.closing(snowflake.connector.connect(**kwargs, database=SNOWFLAKE_DATABASE)) as conn:
        cur = conn.cursor()

        def grant(schema: str) -> None:
            if grant_to is not None:
                cur.execute(f"GRANT USAGE ON SCHEMA {SNOWFLAKE_DATABASE}.{_quote(schema)} TO ROLE {grant_to}")
                cur.execute(
                    f"GRANT SELECT ON ALL TABLES IN SCHEMA {SNOWFLAKE_DATABASE}.{_quote(schema)} TO ROLE {grant_to}"
                )

        cur.execute(f"CREATE SCHEMA IF NOT EXISTS {meta}")
        cur.execute(
            f"CREATE TABLE IF NOT EXISTS {meta}.loaded (db_id VARCHAR PRIMARY KEY, fallbacks VARCHAR, layout INTEGER, "
            "loaded_at TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP())"
        )
        loaded = dict(cur.execute(f"SELECT db_id, layout FROM {meta}.loaded").fetchall())
        _check_layout(loaded, db_ids, "the Snowflake account")
        if grant_to is not None:
            cur.execute(f"GRANT USAGE ON DATABASE {SNOWFLAKE_DATABASE} TO ROLE {grant_to}")
        grant(_META_SCHEMA)
        for sqlite_path, db_id in zip(sqlite_paths, db_ids):
            if db_id not in loaded:
                logger.info("Loading %s into the Snowflake account ...", db_id)
                for record in _load_account_db(conn, db_id, sqlite_path):
                    logger.warning("BIRD Snowflake load: stored as string: %s", record)
            grant(db_id)
