# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a local Spark SQL copy of the BIRD dev SQLite databases.

Each SQLite table is written to Parquet (``<warehouse>/<db_id>/<table>/data.parquet``) and registered as an
external Spark table in a database named after the ``db_id``. Numeric columns are typed; a column whose data does
not convert cleanly (SQLite is dynamically typed) falls back to string. Date/datetime columns stay strings, as
they are in SQLite. Requires a JDK plus ``pyspark``, ``pandas`` and ``pyarrow`` (see benchmarks/birdbench/README.md).
"""

import logging
import sqlite3
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd


logger = logging.getLogger(__name__)

_LOADED_MARKER = ".loaded"
_FLOAT_TYPE_HINTS = ("REAL", "FLOA", "DOUB", "NUM", "DEC")


def _is_null(value: object) -> bool:
    return value is None or (isinstance(value, float) and value != value)


def _coerce_column(series: pd.Series, declared_type: str) -> Optional[pd.Series]:
    """Return ``series`` as a numeric column per its declared SQLite type, or ``None`` if it is not numeric/typed."""
    declared_type = declared_type.upper()
    is_int = "INT" in declared_type
    if not is_int and not any(hint in declared_type for hint in _FLOAT_TYPE_HINTS):
        return None
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().sum() != series.isna().sum():
        return None  # some non-null value is not a number
    if is_int and (numeric.dropna() % 1 == 0).all():
        return numeric.astype("Int64")
    return numeric.astype("float64")


def _typed_tables(sqlite_path: Path) -> Iterator[Tuple[str, pd.DataFrame, List[Dict[str, str]]]]:
    """Yield ``(table, typed DataFrame, string-fallback records)`` for every table of ``sqlite_path``.

    A fallback record is produced for each column that had a numeric declared type but non-numeric data and so
    is stored as string.
    """
    conn = sqlite3.connect(str(sqlite_path))
    conn.text_factory = lambda b: b.decode(errors="ignore")
    try:
        tables = [
            name
            for (name,) in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence'"
            )
        ]
        for table in tables:
            fallbacks: List[Dict[str, str]] = []
            declared = {row[1]: row[2] or "" for row in conn.execute(f"PRAGMA table_info('{table}')")}
            df = pd.read_sql_query(f'SELECT * FROM "{table}"', conn)
            for column, declared_type in declared.items():
                typed = _coerce_column(df[column], declared_type)
                if typed is not None:
                    df[column] = typed
                    continue
                if any(hint in declared_type.upper() for hint in ("INT", *_FLOAT_TYPE_HINTS)):
                    fallbacks.append({"db": sqlite_path.stem, "table": table, "column": column, "type": declared_type})
                df[column] = df[column].map(lambda v: None if _is_null(v) else str(v)).astype(object)
            yield table, df, fallbacks
    finally:
        conn.close()


def sqlite_to_parquet(sqlite_path: Path, out_dir: Path) -> List[Dict[str, str]]:
    """Write every table of ``sqlite_path`` to ``out_dir/<table>/data.parquet``.

    Returns the string-fallback records (see ``_typed_tables``).
    """
    all_fallbacks: List[Dict[str, str]] = []
    for table, df, fallbacks in _typed_tables(sqlite_path):
        table_dir = out_dir / table
        table_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(table_dir / "data.parquet", index=False)
        all_fallbacks += fallbacks
    return all_fallbacks


def spark_column_types(sqlite_path: Path) -> Dict[str, Dict[str, str]]:
    """``{table: {column: Spark type}}`` exactly as ``sqlite_to_parquet`` would store them.

    Used to show the model the real column types, including string fallbacks for dirty numeric columns.
    """
    spark_type_by_dtype = {"Int64": "BIGINT", "float64": "DOUBLE"}
    return {
        table: {column: spark_type_by_dtype.get(str(dtype), "STRING") for column, dtype in df.dtypes.items()}
        for table, df, _fallbacks in _typed_tables(sqlite_path)
    }


def spark_session(warehouse_dir: Path):
    """Local-mode SparkSession. Bound to loopback so it does not stall on unreachable LAN/VPN addresses."""
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[4]")
        .appName("bird_sql")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.sql.warehouse.dir", str(warehouse_dir / "spark-warehouse"))
        .config("spark.sql.shuffle.partitions", "4")
        .config(
            "spark.sql.ansi.enabled", "true"
        )  # as in Spark 4 / Databricks: invalid casts, divide-by-zero error out
        .config("spark.driver.memory", "2g")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def ensure_bird_spark(dev_databases_dir: Path, warehouse_dir: Path) -> Dict[str, object]:
    """Load BIRD dev into Spark (idempotent) and return ``{db_id: SparkSession}``.

    Parquet files are written once; the Spark catalog is in-memory, so the tables are re-registered on every
    call. Each ``db_id`` gets its own ``newSession()`` with that database selected, so concurrent queries against
    different databases never fight over the session's current database.
    """
    spark = spark_session(warehouse_dir)
    spark.sparkContext.setLogLevel("ERROR")
    sessions: Dict[str, object] = {}
    for sqlite_path in sorted(dev_databases_dir.glob("*/[!.]*.sqlite")):
        db_id = sqlite_path.parent.name
        db_dir = warehouse_dir / db_id
        if not (db_dir / _LOADED_MARKER).exists():
            logger.info("Loading %s into Spark ...", db_id)
            fallbacks = sqlite_to_parquet(sqlite_path, db_dir)
            for record in fallbacks:
                logger.warning("BIRD Spark load: stored as string: %s", record)
            (db_dir / _LOADED_MARKER).touch()
        spark.sql(f"CREATE DATABASE IF NOT EXISTS `{db_id}`")
        for table_dir in sorted(p for p in db_dir.iterdir() if p.is_dir()):
            spark.sql(
                f"CREATE TABLE IF NOT EXISTS `{db_id}`.`{table_dir.name}` USING parquet LOCATION '{table_dir.resolve()}'"
            )
        session = spark.newSession()
        session.sql(f"USE `{db_id}`")
        sessions[db_id] = session
    return sessions
