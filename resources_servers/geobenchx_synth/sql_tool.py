# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A read-only spatial SQL tool over the task catalog.

The tables come from prepare/build_sql.py: GeoParquet with WKB geometry in EPSG:4326 and
EPSG:3857, line lengths and Cropland Data Layer classes precomputed as the GIS tools
measure them. Queries then need no reprojection, so the dialect is the subset of PostGIS
that DuckDB Spatial also runs, and the same query works on PostgreSQL/PostGIS.

Each state is loaded into a DuckDB file once. A session gets its own in-memory database
with only the offered states attached read-only, then external access and configuration
are locked, so a query cannot read other files, write anywhere, attach databases or load
extensions. Each query runs under a time limit and results are truncated to a few rows.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any


SQL_TOOL_NAME = "run_sql"
RASTER_KINDS = ("cdl",)

SQL_TOOL_DESCRIPTION = """Run one read-only SQL SELECT on the task's spatial database and return the result.
The dialect is PostgreSQL with PostGIS functions, limited to what DuckDB Spatial also runs:
ST_Intersects, ST_Within, ST_Contains, ST_Covers, ST_Touches, ST_DWithin, ST_Distance,
ST_Buffer, ST_Area, ST_Centroid, ST_Envelope, ST_Intersection, ST_X, ST_Y, count/sum/avg/min/max,
joins, subqueries, CTEs and window functions. There is no ST_Transform, ST_Union aggregate, ILIKE,
round, UNION or information_schema. Column names are lowercase.

Geometry is stored as WKB: wrap a geometry column in ST_GeomFromWKB(...) before using it.
Each vector table has two geometry columns:
- geom: EPSG:4326 longitude and latitude, for spatial relationships such as intersects or within.
- geom_3857: EPSG:3857 meters, for buffers and distances, as the GIS tools measure them.
Never combine geom and geom_3857 in one expression.
Line tables have length_km, each feature's length in its UTM zone as the GIS tools measure it; sum it
instead of measuring geometry. Building tables have cdl_2020 to cdl_2023, the USDA Cropland Data
Layer class code at each building's centroid.

See a table's columns with SELECT * FROM <table> LIMIT 1.
At most {max_rows} rows are returned; aggregate in SQL instead of listing features.

Tables:
{tables}"""


def table_name(entry: dict[str, Any]) -> str:
    return re.sub(r"[^a-z0-9]+", "_", Path(entry["file"]).stem.lower()).strip("_")


def state_slug(state: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", state.lower()).strip("_")


def sql_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [e for e in entries if e["kind"] not in RASTER_KINDS]


def sql_tool_definition(offered: list[dict[str, Any]], max_rows: int = 50) -> dict[str, Any]:
    tables = "\n".join(f"- {table_name(e)}: {e['name']}" for e in sorted(sql_entries(offered), key=table_name))
    return {
        "type": "function",
        "name": SQL_TOOL_NAME,
        "description": SQL_TOOL_DESCRIPTION.format(max_rows=max_rows, tables=tables),
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "A single SELECT statement."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    }


def _format_value(value: Any) -> str:
    if value is None:
        return "NULL"
    text = str(value)
    return text if len(text) <= 80 else text[:77] + "..."


def _format_table(header: list[str], rows: list[tuple], max_rows: int) -> str:
    lines = [" | ".join(header)] + [" | ".join(_format_value(v) for v in row) for row in rows[:max_rows]]
    if len(rows) > max_rows:
        lines.append(f"(first {max_rows} rows shown; aggregate to see all)")
    elif not rows:
        lines.append("(0 rows)")
    return "\n".join(lines)


def _error(error: Exception) -> str:
    return f"Error: {str(error)[:500]}"


class StateDatabases:
    """The SQL tables of each state, from the build_sql.py output; for duckdb, one DuckDB file per state."""

    def __init__(self, sql_data_path: str, db_dir: str, extension_dir: str | None):
        self.sql_data_path = Path(sql_data_path)
        self.db_dir = Path(db_dir)
        self.extension_dir = extension_dir
        manifest = json.loads((self.sql_data_path / "catalog_sql.json").read_text())
        self.by_state: dict[str, list[dict[str, Any]]] = {}
        for entry in manifest:
            self.by_state.setdefault(entry["state"], []).append(entry)
        self._locks = {state: threading.Lock() for state in self.by_state}

    def parquet(self, entry: dict[str, Any]) -> Path:
        return self.sql_data_path / entry["parquet"]

    def connect(self, path: str = ":memory:"):
        import duckdb

        config = {"autoinstall_known_extensions": False, "autoload_known_extensions": False}
        if self.extension_dir:
            config["extension_directory"] = self.extension_dir
        connection = duckdb.connect(path, config=config)
        try:
            connection.execute("LOAD spatial")
        except duckdb.Error:
            connection.execute("INSTALL spatial")
            connection.execute("LOAD spatial")
        # GLOBAL, so that the cursor each query runs on inherits it.
        connection.execute("SET GLOBAL geometry_always_xy = true")
        return connection

    def path(self, state: str) -> Path:
        entries = self.by_state[state]
        stamp = json.dumps([(e["parquet"], self.parquet(e).stat().st_size) for e in entries], sort_keys=True)
        digest = hashlib.sha256(stamp.encode()).hexdigest()[:12]
        path = self.db_dir / f"{state_slug(state)}-{digest}.duckdb"
        with self._locks[state]:
            if not path.exists():
                self._build(entries, path)
        return path

    def _build(self, entries: list[dict[str, Any]], path: Path) -> None:
        self.db_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(f".partial{os.getpid()}")
        partial.unlink(missing_ok=True)
        connection = self.connect(str(partial))
        try:
            # Keep WKB as BLOB, as PostGIS bytea sees it, so queries decode it with ST_GeomFromWKB.
            connection.execute("SET enable_geoparquet_conversion = false")
            for entry in entries:
                source = str(self.parquet(entry)).replace("'", "''")
                connection.execute(f"CREATE TABLE \"{entry['table']}\" AS SELECT * FROM read_parquet('{source}')")
        finally:
            connection.close()
        partial.rename(path)


class DuckdbSession:
    def __init__(self, databases: StateDatabases, states: list[str], timeout_s: float, max_rows: int):
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self.lock = threading.Lock()
        self.connection = databases.connect()
        for state in states:
            path = str(databases.path(state)).replace("'", "''")
            schema = state_slug(state)
            self.connection.execute(f"ATTACH '{path}' AS {schema} (READ_ONLY)")
            # Views in the session's own database, because each query runs on a fresh cursor,
            # which does not inherit a search_path.
            for entry in databases.by_state[state]:
                name = entry["table"]
                self.connection.execute(f'CREATE VIEW "{name}" AS SELECT * FROM {schema}."{name}"')
        self.connection.execute("SET memory_limit = '1GB'")
        self.connection.execute("SET threads = 2")
        self.connection.execute("SET enable_external_access = false")
        self.connection.execute("SET lock_configuration = true")

    def run(self, query: str) -> str:
        import duckdb

        with self.lock:
            try:
                statements = self.connection.extract_statements(query)
            except duckdb.Error as error:
                return _error(error)
            if len(statements) != 1 or statements[0].type != duckdb.StatementType.SELECT:
                return "Error: run_sql accepts exactly one SELECT statement"
            cursor = self.connection.cursor()
            timer = threading.Timer(self.timeout_s, cursor.interrupt)
            timer.start()
            try:
                relation = cursor.sql(query)
                columns = []
                for column, kind in zip(relation.columns, relation.types):
                    quoted = '"' + column.replace('"', '""') + '"'
                    if str(kind).startswith("GEOMETRY"):
                        columns.append(f"ST_AsText({quoted}) AS {quoted}")
                    elif str(kind) == "BLOB":
                        wkt = f"ST_AsText(TRY(ST_GeomFromWKB({quoted})))"
                        columns.append(f"COALESCE({wkt}, CAST({quoted} AS VARCHAR)) AS {quoted}")
                    else:
                        columns.append(quoted)
                rows = relation.project(", ".join(columns)).limit(self.max_rows + 1).fetchall()
                header = list(relation.columns)
            except duckdb.InterruptException:
                return f"Error: query exceeded the {self.timeout_s:g} s time limit"
            except duckdb.Error as error:
                return _error(error)
            finally:
                timer.cancel()
                cursor.close()
        return _format_table(header, rows, self.max_rows)

    def close(self) -> None:
        self.connection.close()


class SqlSandbox:
    """Per-session SQL databases, opened on a session's first query."""

    def __init__(
        self,
        databases: StateDatabases,
        timeout_s: float = 60.0,
        max_rows: int = 50,
        max_sessions: int = 256,
    ):
        self.databases = databases
        self.timeout_s = timeout_s
        self.max_rows = max_rows
        self.max_sessions = max_sessions
        self._states: dict[str, list[str]] = {}
        self._sessions: OrderedDict[str, DuckdbSession] = OrderedDict()
        self._lock = threading.Lock()

    def _open(self, states: list[str]) -> DuckdbSession:
        return DuckdbSession(self.databases, states, self.timeout_s, self.max_rows)

    def reset(self, session_id: str, states: list[str]) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            self._states[session_id] = [s for s in states if s in self.databases.by_state]
        if session:
            session.close()

    def run(self, session_id: str, query: str) -> str:
        with self._lock:
            session = self._sessions.get(session_id)
            states = self._states.get(session_id)
            if session:
                self._sessions.move_to_end(session_id)
        if session is None:
            if not states:
                return "Error: no spatial database is available for this task"
            session = self._open(states)
            with self._lock:
                self._sessions[session_id] = session
                evicted = []
                while len(self._sessions) > self.max_sessions:
                    old_id, old = self._sessions.popitem(last=False)
                    self._states.pop(old_id, None)
                    evicted.append(old)
            for old in evicted:
                old.close()
        return session.run(query)

    def close(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for session in sessions:
            session.close()
