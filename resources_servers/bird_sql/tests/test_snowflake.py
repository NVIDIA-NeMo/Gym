# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Snowflake dialect tests, on the local stand-in (fakesnow on DuckDB); no Snowflake account or server is needed.

They run when ``fakesnow`` (with ``snowflake-connector-python<5``) and ``pandas`` are installed, as they are in this
server's venv, and load one small schema (``bird_pytest_db``) into a temporary ``BIRD.db``. The ``snowflake_rejection``
tests need only ``sqlglot``. The ``snowflake`` engine (a real account) is tested only for how it is selected.
"""

import asyncio
import importlib.util
import sqlite3
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import VERIFIER_ERROR
from resources_servers.bird_sql.app import BirdSqlResourcesServer, BirdSqlResourcesServerConfig, FailureCode
from resources_servers.bird_sql.tests.test_app import _make_verify_request


DB_ID = "bird_pytest_db"
_HAS_SQLGLOT = importlib.util.find_spec("sqlglot") is not None
_HAS_DEPS = _HAS_SQLGLOT and all(importlib.util.find_spec(m) for m in ("fakesnow", "duckdb", "pandas"))

requires_fakesnow = pytest.mark.skipif(not _HAS_DEPS, reason="needs fakesnow, snowflake-connector-python<5 and pandas")
requires_sqlglot = pytest.mark.skipif(not _HAS_SQLGLOT, reason="needs sqlglot")
requires_pandas = pytest.mark.skipif(importlib.util.find_spec("pandas") is None, reason="needs pandas")


@requires_pandas
@pytest.mark.asyncio
async def test_verify_snowflake_dialect_routes_to_snowflake(tmp_path, monkeypatch):
    # Gold runs on the SQLite file, so verify needs it to exist; the mocked executors never open it.
    (tmp_path / "TestDB").mkdir()
    (tmp_path / "TestDB" / "TestDB.sqlite").touch()
    monkeypatch.setattr("resources_servers.bird_sql.app.ensure_bird_sql", lambda _path: tmp_path)
    loaded = []
    monkeypatch.setattr(
        "resources_servers.bird_sql.setup_bird_snowflake.ensure_bird_snowflake",
        lambda *a: loaded.append(a) or "local-snowflake",
    )
    seen = {}

    async def fake_snowflake(**kw):
        seen.update(kw)
        return True, [(1,)], [(1,)], None

    async def fail_sqlite(**_kw):
        raise AssertionError("sqlite path must not run for dialect=snowflake")

    monkeypatch.setattr("resources_servers.bird_sql.app.execute_and_compare_snowflake", fake_snowflake)
    monkeypatch.setattr("resources_servers.bird_sql.app.execute_and_compare", fail_sqlite)
    config = BirdSqlResourcesServerConfig(
        host="127.0.0.1",
        port=20099,
        entrypoint="",
        bird_sql_dir=str(tmp_path),
        dialect="snowflake",
        snowflake_db_dir=str(tmp_path / "sf"),
    )
    server = BirdSqlResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    resp = await server.verify(_make_verify_request("```sql\nSELECT x FROM t\n```", gt_sql="SELECT x FROM t"))
    assert resp.reward == 1.0
    assert loaded == [(tmp_path, tmp_path / "sf")]
    assert (seen["snowflake"], seen["db_id"]) == ("local-snowflake", "TestDB")
    assert seen["db_path"] == tmp_path / "TestDB" / "TestDB.sqlite"  # gold runs on SQLite
    assert seen["gold_sql"] == "SELECT x FROM t"  # untouched, never transpiled


@requires_pandas
def test_snowflake_engine_selects_a_real_account(tmp_path, monkeypatch):
    # Only the selection is tested: there is no account to run the queries on.
    monkeypatch.setattr("resources_servers.bird_sql.app.ensure_bird_sql", lambda _path: tmp_path)
    checked = []
    monkeypatch.setattr(
        "resources_servers.bird_sql.setup_bird_snowflake.check_bird_snowflake_account",
        lambda *a: checked.append(a) or "account",
    )
    monkeypatch.setattr(
        "resources_servers.bird_sql.setup_bird_snowflake.ensure_bird_snowflake",
        lambda *_a: pytest.fail("the stand-in must not be built for snowflake_engine=snowflake"),
    )
    config = BirdSqlResourcesServerConfig(
        host="127.0.0.1",
        port=20099,
        entrypoint="",
        bird_sql_dir=str(tmp_path),
        dialect="snowflake",
        snowflake_engine="snowflake",
        snowflake_connection_name="bird",
        sql_execution_timeout_s=12.0,
    )
    server = BirdSqlResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    assert checked == [(tmp_path, "bird", 12.0)]
    assert server._snowflake == "account"


class _FakeAccountConnection:
    """Records what the account engine runs on a new connection; ``fail``: every statement raises."""

    def __init__(self, fail):
        self.statements, self.closed, self.fail = [], False, fail

    def cursor(self):
        return self

    def execute(self, sql):
        if self.fail:
            raise RuntimeError("network")
        self.statements.append(sql)

    def close(self):
        self.closed = True


@requires_fakesnow
@pytest.mark.parametrize("fail", [False, True])
def test_account_engine_turns_off_secondary_roles(monkeypatch, fail):
    # With the user's default secondary roles (often ALL), a session on a read-only connection could write through the
    # user's other roles, so each connection turns them off. A connection where that fails is closed, not returned.
    import snowflake.connector

    from resources_servers.bird_sql.setup_bird_snowflake import AccountSnowflake

    conn, seen = _FakeAccountConnection(fail), {}
    monkeypatch.setattr(snowflake.connector, "connect", lambda **kw: seen.update(kw) or conn)
    engine = AccountSnowflake(connection_name="bird_ro", timeout_s=12.4, tables={})
    if fail:
        with pytest.raises(RuntimeError):
            engine.connect(DB_ID)
        assert conn.closed
    else:
        assert engine.connect(DB_ID) is conn
        assert (conn.statements, conn.closed) == (["USE SECONDARY ROLES NONE"], False)
    assert seen == {
        "connection_name": "bird_ro",
        "database": "BIRD",
        "schema": f'"{DB_ID}"',
        "session_parameters": {"STATEMENT_TIMEOUT_IN_SECONDS": 12},
    }


class _EngineRaising:
    """Stands in for a Snowflake engine whose query raises ``error``."""

    tables = {DB_ID: {"t": frozenset({"x"})}}

    def __init__(self, error):
        self.error = error

    def connect(self, db_id):
        return self

    def cursor(self):
        return self

    def execute(self, sql):
        raise self.error

    def close(self):
        pass


def _snowflake_error(cls_name, sqlstate):
    import snowflake.connector.errors as errors

    return getattr(errors, cls_name)(msg="message", sqlstate=sqlstate)


@requires_fakesnow
@pytest.mark.parametrize(
    ("make_error", "outcome"),
    [
        (lambda: _snowflake_error("ProgrammingError", "42000"), None),  # a syntax error
        (lambda: _snowflake_error("ProgrammingError", "22007"), None),  # a value that does not convert
        (lambda: OverflowError("date value out of range"), None),  # 'infinity'::date has no Python date
        (lambda: _snowflake_error("DatabaseError", "08003"), "raises"),  # the connection is closed
        (lambda: __import__("duckdb").IOException("disk"), "raises"),  # the engine cannot read its data
        (lambda: _snowflake_error("ProgrammingError", "57014"), "timeout"),  # the server's statement timeout
        # Without a SQLSTATE the connector says "n/a": a failed request (250003) or interface error is the engine's,
        # anything else still the query's.
        (lambda: _snowflake_error("OperationalError", None), "raises"),
        (lambda: _snowflake_error("InterfaceError", None), "raises"),
        (lambda: _snowflake_error("ProgrammingError", None), None),
    ],
)
def test_errors_are_split_into_query_faults_and_engine_failures(make_error, outcome):
    from resources_servers.bird_sql.eval_utils import execute_snowflake

    engine = _EngineRaising(make_error())
    if outcome is None:
        assert execute_snowflake(engine, DB_ID, 'SELECT "x" FROM "t"', []) is None
    else:
        with pytest.raises(TimeoutError if outcome == "timeout" else type(engine.error)):
            execute_snowflake(engine, DB_ID, 'SELECT "x" FROM "t"', [])


class _SlowConnectEngine:
    """A Snowflake engine whose ``connect`` takes ``connect_s``; counts executed queries and overlapping workers."""

    tables = {DB_ID: {"t": frozenset({"x"})}}

    def __init__(self, connect_s):
        self.connect_s, self.executed, self.active, self.most_active = connect_s, 0, 0, 0
        self.lock = threading.Lock()

    def connect(self, db_id):
        with self.lock:
            self.active += 1
            self.most_active = max(self.most_active, self.active)
        time.sleep(self.connect_s)
        return self

    def cursor(self):
        return self

    def execute(self, sql):
        self.executed += 1

    def close(self):
        with self.lock:
            self.active -= 1

    def fetch_all(self, cur):
        return [(1,)]

    def interrupt(self, conn):
        pass


@requires_fakesnow
@pytest.mark.asyncio
async def test_a_timeout_while_connecting_holds_the_slot_and_runs_nothing():
    # The timeout can fire while connect() is still running. The worker must not run the query once connected, and
    # the semaphore slot stays held until the worker stops, so no more queries run at once than the semaphore allows.
    from resources_servers.bird_sql.eval_utils import execute_snowflake_async

    engine, sem = _SlowConnectEngine(connect_s=0.5), asyncio.Semaphore(1)
    sql = 'SELECT "x" FROM "t"'
    results = await asyncio.gather(
        *(execute_snowflake_async(engine, DB_ID, sql, sem, timeout_s=0.1) for _ in range(3))
    )
    assert results == [None, None, None]
    assert (engine.executed, engine.most_active, engine.active) == (0, 1, 0)


@requires_fakesnow
def test_gold_check_names_why_a_gold_fails():
    # The data generators leave these question ids out, so every non-match must get a known class.
    from benchmarks.birdbench.snowflake_gold_check import _ERROR_PATTERNS, FAILURE_CLASSES, error_class, mismatch_class

    group_by = 'Binder Error: column "x" must appear in the GROUP BY clause or must be part of an aggregate function'
    strings = "Binder Error: No function matches the given name and argument types '-(VARCHAR, VARCHAR)'"
    assert error_class(group_by) == "bare_group_by_column"
    assert error_class(strings) == "string_arithmetic"
    assert error_class("Binder Error: No function matches the given name and argument types") == "type_mismatch"
    assert error_class("something new") == "other_error"
    assert mismatch_class("SELECT SUM(r) FROM t", [(1.000001,)], [(1.000002,)]) == "float_precision"
    assert mismatch_class("SELECT x FROM t ORDER BY y LIMIT 1", [(1,)], [(2,)]) == "limit_ties_or_order"
    assert mismatch_class("SELECT x FROM t WHERE s LIKE 'a%'", [(1,)], [(2,)]) == "like_case"
    assert mismatch_class("SELECT x / y FROM t", [(1,)], [(2,)]) == "division"
    assert mismatch_class("SELECT x FROM t", [(1,)], [(2,)]) == "other_mismatch"
    assert {name for name, _ in _ERROR_PATTERNS} <= set(FAILURE_CLASSES)


_TABLES = {
    "t": frozenset({"x"}),
    "Mixed Case": frozenset({"Col Name", "ID", "n"}),
    "u": frozenset({"X"}),  # "X" in another case than t's "x"
}


@requires_sqlglot
@pytest.mark.parametrize(
    "sql, reason",
    [
        ('SELECT "Col Name" FROM "Mixed Case"', None),
        ('SELECT "Col Name" FROM "bird_pytest_db"."Mixed Case"', None),
        ('SELECT m."Col Name" FROM bird."bird_pytest_db"."Mixed Case" AS m', None),
        ('SELECT id, "ID" FROM "Mixed Case"', None),  # unquoted id means ID
        ('SELECT COUNT(*) AS cnt FROM "Mixed Case" ORDER BY CNT', None),
        ('WITH c AS (SELECT "n" AS k FROM "Mixed Case") SELECT k FROM c', None),
        ('SELECT "n", ROW_NUMBER() OVER (ORDER BY "n") AS rn FROM "Mixed Case" QUALIFY rn = 1', None),
        ("SELECT f.value FROM \"t\", LATERAL FLATTEN(input => PARSE_JSON('[1]')) f", None),
        ("SELECT IFF(1 > 0, 1, 2)", None),
        ('SELECT "Col Name" FROM "Mixed Case" LIMIT 1 OFFSET 1', None),
        ('SELECT x FROM "t"', "unknown_column:X"),  # created quoted as lower-case "x"
        ('SELECT "col name" FROM "Mixed Case"', "unknown_column:col name"),
        ('SELECT "x" FROM t', "unknown_table:T"),
        ('SELECT m."n" FROM "Mixed Case" AS "m" WHERE M."n" > 0', "unknown_qualifier:M"),
        ('SELECT COUNT(*) AS "Cnt" FROM "t" ORDER BY cnt', "unknown_column:CNT"),
        ('SELECT "x" FROM "other_db"."t"', "unknown_schema:other_db"),
        ('SELECT "x" FROM other."bird_pytest_db"."t"', "unknown_database:OTHER"),
        ("SELECT IIF(1 > 0, 1, 2)", "sqlite_function:IIF"),
        ("SELECT STRFTIME('%Y', '2020-01-01')", "sqlite_function:STRFTIME"),
        ('SELECT "x" FROM "t" LIMIT 1, 2', "limit_offset_comma"),
        ("SELECT MAX(1, 2)", "scalar_max_min"),
        ('DELETE FROM "t"', "not_a_single_query"),
        ('SELECT "x" FROM "t"; SELECT 1', "not_a_single_query"),
        ("SELECT (", "unparsable"),
        ('SELECT COUNT(*) FROM "t" JOIN "Mixed Case" USING ("n")', None),
        ('SELECT COUNT(*) FROM "t" JOIN "Mixed Case" USING (n)', "unknown_column:N"),  # USING names fold too
        # A USING name a CTE outputs is fine (Snowflake runs this).
        (
            'WITH a AS (SELECT "x" AS k FROM "t"), b AS (SELECT "n" AS k FROM "Mixed Case") SELECT k FROM a JOIN b USING (k)',
            None,
        ),
        # A comment after the last semicolon is no second statement (Snowflake runs these); a second statement is.
        ('SELECT "x" FROM "t"; -- the x values', None),
        ('SELECT "x" FROM "t"; /* note */', None),
        ('SELECT "x" FROM "t"; -- why\nDELETE FROM "t"', "not_a_single_query"),
        ('SELECT "n" AS N2 FROM "Mixed Case" WHERE n2 > 0', None),  # a select alias, unqualified
        ('SELECT m."n" AS N2 FROM "Mixed Case" m WHERE m.n2 > 0', "unknown_column:M.N2"),  # but not table-qualified
        ('SELECT s.k FROM (SELECT "n" AS k FROM "Mixed Case") AS s', None),  # a derived table's column is fine
        ("WITH total(x) AS (SELECT 1) SELECT x FROM total", None),  # a CTE named TOTAL is not the function
        # Snowflake refuses a BOOLEAN argument to SUM/AVG (DuckDB counts the true rows); a number in between is fine.
        ('SELECT SUM("x" = 1) FROM "t"', "boolean_aggregate:SUM"),
        ('SELECT AVG(NOT ("x" > 1)) FROM "t"', "boolean_aggregate:AVG"),
        ('SELECT SUM(DISTINCT ("x" > 1 AND "x" < 5)) FROM "t"', "boolean_aggregate:SUM"),
        ('SELECT SUM(IFF("x" = 1, 1, 0)), SUM(CAST("x" = 1 AS INT)), COUNT_IF("x" = 1) FROM "t"', None),
        ('SELECT SUM(IFF("x" = 1, TRUE, FALSE)) FROM "t"', "boolean_aggregate:SUM"),  # still a BOOLEAN
        ('SELECT SUM(COALESCE("x" = 1, FALSE)) FROM "t"', "boolean_aggregate:SUM"),
        ('SELECT AVG(CASE WHEN "x" = 1 THEN TRUE ELSE FALSE END) FROM "t"', "boolean_aggregate:AVG"),
        ('SELECT SUM(CASE WHEN "x" = 1 THEN 1 ELSE 0 END), SUM(COALESCE("x", 0)) FROM "t"', None),
        ('SELECT AVG(DISTINCT) FROM "t"', "unparsable"),
        # Snowflake has no aggregate FILTER clause and none of DuckDB's own functions; its FILTER(array, lambda),
        # DIV0 and LISTAGG are fine, and so is a CTE that happens to share a DuckDB function's name.
        ('SELECT COUNT(*) FILTER (WHERE "x" > 1) * 100.0 / COUNT(*) FROM "t"', "aggregate_filter"),
        ('SELECT DIVIDE(SUM("x"), COUNT(*)) FROM "t"', "duckdb_function:DIVIDE"),
        ('SELECT string_agg("x", \',\') FROM "t"', "duckdb_function:STRING_AGG"),
        ("SELECT FILTER([1, 2, 3], a -> a > 1), DIV0(1, 0), LISTAGG('a', ',')", None),
        ('WITH divide AS (SELECT "x" FROM "t") SELECT "x" FROM divide', None),
        # An alias doesn't make a wrong-case column inside its own definition valid; used elsewhere, it's fine.
        ('SELECT MAX(n) AS n FROM "Mixed Case"', "unknown_column:N"),
        ('SELECT MAX("n") AS n FROM "Mixed Case"', None),
        ('SELECT "n" AS k, k + 1 AS k1 FROM "Mixed Case" WHERE k > 1 ORDER BY k', None),
        # A column belongs to its own table, qualified or not; a correlated subquery may use the outer table's.
        ('SELECT COUNT(*) FROM "t" WHERE "X" > 0', "unknown_column:X"),
        ('SELECT COUNT(*) FROM "t" AS a JOIN "u" AS b ON a."X" = b."X"', "unknown_column:A.X"),
        ('SELECT COUNT(*) FROM "t" AS a JOIN "u" AS b ON a."x" = b."X"', None),
        ('SELECT COUNT(*) FROM "t" WHERE EXISTS (SELECT 1 FROM "u" WHERE "X" = "x")', None),
        # A bare function in FROM is DuckDB's; Snowflake's table functions are wrapped in TABLE() or LATERAL.
        ("SELECT COUNT(*) FROM range(10)", "table_function:RANGE"),
        ("SELECT * FROM query('SELECT 1')", "table_function:QUERY"),
        ("SELECT COUNT(*) FROM TABLE(GENERATOR(ROWCOUNT => 3))", None),
        # VALUES outputs COLUMN1, COLUMN2, ...
        ("SELECT column1, column2 FROM VALUES (1, 'a'), (2, 'b')", None),
        # SELECT DISTINCT can't ORDER BY a column it doesn't select; a selected column, alias or position can, and so
        # can an expression that is selected or built from selected columns.
        ('SELECT DISTINCT "n" FROM "Mixed Case" ORDER BY "ID"', "distinct_order_by:ID"),
        ('SELECT DISTINCT "n" AS k FROM "Mixed Case" ORDER BY k, "n" DESC, 1', None),
        ('SELECT DISTINCT * FROM "Mixed Case" ORDER BY "ID"', None),
        ('SELECT "n" FROM "Mixed Case" ORDER BY "ID"', None),  # without DISTINCT
        ('SELECT DISTINCT "n" + 1 AS k FROM "Mixed Case" ORDER BY "n"', "distinct_order_by:n"),
        ('SELECT DISTINCT "n" + 1 AS k FROM "Mixed Case" ORDER BY "n" + 1, k', None),
        ('SELECT DISTINCT "n", "ID" FROM "Mixed Case" ORDER BY "ID" - 1', None),
        # The same column name from another table is not the selected column.
        (
            'SELECT DISTINCT a."n" FROM "Mixed Case" AS a JOIN "Mixed Case" AS b ON a."ID" = b."ID" ORDER BY b."n"',
            "distinct_order_by:n",
        ),
        ('SELECT DISTINCT a."n" FROM "Mixed Case" AS a JOIN "Mixed Case" AS b ON a."ID" = b."ID" ORDER BY "n"', None),
        ('SELECT DISTINCT "n" FROM "Mixed Case" AS a ORDER BY a."n"', None),
    ],
)
def test_snowflake_rejection(sql, reason):
    from resources_servers.bird_sql.eval_utils import snowflake_rejection

    assert snowflake_rejection(sql, DB_ID, _TABLES) == reason


@requires_sqlglot
def test_snowflake_rejection_never_raises_on_a_cut_off_query():
    # A crash here would mask the answer as a verifier error instead of scoring it 0.
    from resources_servers.bird_sql.eval_utils import snowflake_rejection

    queries = [
        'SELECT AVG(DISTINCT "x") FROM "t" WHERE "x" > 1 GROUP BY "x" ORDER BY SUM(IFF("x" = 1, TRUE, FALSE))',
        'SELECT DISTINCT a."n" FROM "Mixed Case" AS a JOIN "t" USING ("x") ORDER BY a."ID" LIMIT 1',
    ]
    for query in queries:
        words = query.split(" ")
        for k in range(1, len(words) + 1):
            snowflake_rejection(" ".join(words[:k]), DB_ID, _TABLES)


class _SnowflakeEnv:
    dev_dir: Path
    db_dir: Path
    local: object


@pytest.fixture(scope="module")
def snowflake_env(tmp_path_factory):
    from resources_servers.bird_sql.setup_bird_snowflake import ensure_bird_snowflake

    dev = tmp_path_factory.mktemp("dev_databases")
    (dev / DB_ID).mkdir()
    conn = sqlite3.connect(str(dev / DB_ID / f"{DB_ID}.sqlite"))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    conn.execute('CREATE TABLE "Mixed Case" ("Col Name" TEXT, n INTEGER, r REAL, dirty INTEGER, d DATE)')
    conn.executemany(
        'INSERT INTO "Mixed Case" VALUES (?, ?, ?, ?, ?)',
        [("a", 1, 1.5, 1, "2020-01-01"), ("B", None, 2.5, "oops", "2020-01-02")],
    )
    conn.execute("CREATE TABLE big (x INTEGER)")
    conn.executemany("INSERT INTO big VALUES (?)", [(i,) for i in range(3000)])
    conn.commit()
    conn.close()
    env = _SnowflakeEnv()
    env.dev_dir = dev
    env.db_dir = tmp_path_factory.mktemp("snowflake")
    env.local = ensure_bird_snowflake(dev, env.db_dir)
    yield env


@requires_fakesnow
class TestLoader:
    def test_types_fallback_and_catalog(self, snowflake_env):
        import duckdb

        from resources_servers.bird_sql.setup_bird_snowflake import snowflake_column_types

        with duckdb.connect(str(snowflake_env.db_dir / "BIRD.db"), read_only=True) as duck:
            types = dict(
                duck.execute(
                    "SELECT column_name, data_type FROM duckdb_columns() WHERE schema_name = ? AND table_name = ?",
                    [DB_ID, "Mixed Case"],
                ).fetchall()
            )
            fallbacks = duck.execute("SELECT fallbacks FROM _bird_meta.loaded WHERE db_id = ?", [DB_ID]).fetchone()[0]
        assert types == {"Col Name": "VARCHAR", "n": "BIGINT", "r": "DOUBLE", "dirty": "VARCHAR", "d": "VARCHAR"}
        assert '"column": "dirty"' in fallbacks  # INTEGER column with non-numeric data falls back to string
        sqlite_path = snowflake_env.dev_dir / DB_ID / f"{DB_ID}.sqlite"
        assert snowflake_column_types(sqlite_path)["Mixed Case"] == {
            "Col Name": "VARCHAR",
            "n": "NUMBER(38,0)",
            "r": "FLOAT",
            "dirty": "VARCHAR",
            "d": "VARCHAR",
        }
        assert snowflake_env.local.tables[DB_ID]["Mixed Case"] == {"Col Name", "n", "r", "dirty", "d"}

    def test_second_load_is_a_no_op(self, snowflake_env):
        from resources_servers.bird_sql.setup_bird_snowflake import ensure_bird_snowflake

        before = (snowflake_env.db_dir / "BIRD.db").stat().st_mtime_ns
        assert DB_ID in ensure_bird_snowflake(snowflake_env.dev_dir, snowflake_env.db_dir).tables
        assert (snowflake_env.db_dir / "BIRD.db").stat().st_mtime_ns == before

    @pytest.mark.parametrize("layout", ["2", "NULL"])
    def test_a_load_with_another_layout_must_be_rebuilt(self, snowflake_env, tmp_path, layout):
        # NULL: a BIRD.db built before the layout was recorded.
        import shutil

        import duckdb

        from resources_servers.bird_sql.setup_bird_snowflake import ensure_bird_snowflake

        shutil.copy(snowflake_env.db_dir / "BIRD.db", tmp_path / "BIRD.db")
        with duckdb.connect(str(tmp_path / "BIRD.db")) as duck:
            duck.execute(f"UPDATE _bird_meta.loaded SET layout = {layout}")
        with pytest.raises(RuntimeError, match=f"another layout.*{DB_ID}"):
            ensure_bird_snowflake(snowflake_env.dev_dir, tmp_path)


@requires_fakesnow
def test_stand_in_frees_each_result_once_read(snowflake_env):
    # fakesnow caches every completed query's result (for RESULT_SCAN) and never frees it. The grader reads each result
    # once, so it drops that entry and memory stays flat; another connection's unread result is left alone.
    from resources_servers.bird_sql.eval_utils import execute_snowflake

    local = snowflake_env.local
    cache = local.fake_snow.results_cache
    before = set(cache)
    pending = local.connect(DB_ID)
    try:
        pending_cur = pending.cursor()
        pending_cur.execute('SELECT "x" FROM "t"')
        for _ in range(5):
            assert execute_snowflake(local, DB_ID, 'SELECT COUNT(*) FROM "big"', []) == [(3000,)]
        assert set(cache) - before == {pending_cur.sfqid}
        assert sorted(local.fetch_all(pending_cur)) == [(1,), (2,), (3,)]
        assert set(cache) == before
    finally:
        pending.close()


class _FakeAdminConnection:
    """An admin connection for the account loader: records statements, answers the loaded-db_ids query from ``loaded``,
    SHOW SCHEMAS from ``schemas`` ({name: comment}) and each COUNT(*) from what ``write_pandas`` wrote (one row short
    for ``short_table``)."""

    def __init__(self, loaded=None, short_table=None, schemas=None):
        self.statements, self.written, self.closed = [], {}, False
        self.loaded, self.short_table, self.schemas, self._rows = loaded or {}, short_table, schemas or {}, []
        self.description = []

    def cursor(self):
        return self

    def execute(self, sql, params=None):
        self.statements.append(sql)
        if sql.startswith("SELECT db_id, layout"):
            self._rows = list(self.loaded.items())
        elif sql.startswith("SHOW SCHEMAS LIKE"):
            self.description = [("created_on",), ("name",), ("comment",)]
            self._rows = [(None, name, comment) for name, comment in self.schemas.items() if name == params[0]]
        elif sql.startswith("SELECT COUNT(*)"):
            table = sql.rsplit(".", 1)[1].strip('"')
            self._rows = [(self.written.get(table, 0) - (table == self.short_table),)]
        return self

    def fetchall(self):
        return self._rows

    def fetchone(self):
        return self._rows[0]

    def close(self):
        self.closed = True


@requires_fakesnow
class TestAccountLoader:
    @staticmethod
    def _load(env, monkeypatch, conn, grant_to="BIRD_RO"):
        import snowflake.connector
        import snowflake.connector.pandas_tools

        from resources_servers.bird_sql.setup_bird_snowflake import load_bird_snowflake_account

        def write_pandas(c, df, table, database, schema, chunk_size):
            assert (c, database, schema) == (conn, "BIRD", DB_ID)
            c.written[table] = len(df)

        monkeypatch.setattr(snowflake.connector, "connect", lambda **kw: conn)
        monkeypatch.setattr(snowflake.connector.pandas_tools, "write_pandas", write_pandas)
        load_bird_snowflake_account(env.dev_dir, "bird_admin", grant_to)

    def test_loads_the_layout_records_it_last_and_grants_the_reader(self, snowflake_env, monkeypatch):
        conn = _FakeAdminConnection()
        self._load(snowflake_env, monkeypatch, conn)
        s = conn.statements
        create = s.index(f'CREATE SCHEMA BIRD."{DB_ID}" COMMENT = %s')
        assert s[:create].count("GRANT USAGE ON DATABASE BIRD TO ROLE BIRD_RO") == 1
        assert 'GRANT SELECT ON ALL TABLES IN SCHEMA BIRD."_bird_meta" TO ROLE BIRD_RO' in s[:create]
        mixed = f'CREATE TABLE BIRD."{DB_ID}"."Mixed Case" '
        assert mixed + '("Col Name" VARCHAR, "n" NUMBER(38,0), "r" FLOAT, "dirty" VARCHAR, "d" VARCHAR)' in s
        assert conn.written == {"t": 3, "Mixed Case": 2, "big": 3000}
        record = next(i for i, sql in enumerate(s) if sql.startswith('INSERT INTO BIRD."_bird_meta".loaded'))
        assert all(not sql.startswith(("CREATE TABLE BIRD.", "SELECT COUNT")) for sql in s[record:])
        # The db_id is readable as soon as it's recorded, before any later db_id loads.
        assert s[record + 1 :] == [
            f'GRANT USAGE ON SCHEMA BIRD."{DB_ID}" TO ROLE BIRD_RO',
            f'GRANT SELECT ON ALL TABLES IN SCHEMA BIRD."{DB_ID}" TO ROLE BIRD_RO',
        ]
        assert conn.closed

    def test_a_loaded_db_id_is_skipped_but_still_granted(self, snowflake_env, monkeypatch):
        conn = _FakeAdminConnection(loaded={DB_ID: 1})
        self._load(snowflake_env, monkeypatch, conn)
        assert conn.written == {}
        assert not any(
            sql.startswith(("CREATE SCHEMA BIRD.", "CREATE OR REPLACE", "INSERT")) for sql in conn.statements
        )
        assert f'GRANT SELECT ON ALL TABLES IN SCHEMA BIRD."{DB_ID}" TO ROLE BIRD_RO' in conn.statements

    def test_a_short_table_drops_the_schema_and_records_nothing(self, snowflake_env, monkeypatch):
        conn = _FakeAdminConnection(short_table="Mixed Case")
        with pytest.raises(RuntimeError, match="Mixed Case: 1 rows loaded into Snowflake, 2 in SQLite"):
            self._load(snowflake_env, monkeypatch, conn)
        assert conn.statements[-1] == f'DROP SCHEMA IF EXISTS BIRD."{DB_ID}"'
        assert not any(sql.startswith("INSERT") for sql in conn.statements)

    def test_a_failed_load_leaves_the_earlier_db_ids_readable(self, tmp_path, monkeypatch):
        import snowflake.connector
        import snowflake.connector.pandas_tools

        from resources_servers.bird_sql.setup_bird_snowflake import load_bird_snowflake_account

        for db_id in ("a_db", "b_db"):
            (tmp_path / db_id).mkdir()
            with sqlite3.connect(str(tmp_path / db_id / f"{db_id}.sqlite")) as sqlite_conn:
                sqlite_conn.execute(f"CREATE TABLE {db_id}_t (x INTEGER)")
                sqlite_conn.execute(f"INSERT INTO {db_id}_t VALUES (1)")
        conn = _FakeAdminConnection(short_table="b_db_t")
        monkeypatch.setattr(snowflake.connector, "connect", lambda **kw: conn)
        monkeypatch.setattr(
            snowflake.connector.pandas_tools,
            "write_pandas",
            lambda c, df, table, **kw: c.written.update({table: len(df)}),
        )
        with pytest.raises(RuntimeError, match="b_db_t: 0 rows loaded"):
            load_bird_snowflake_account(tmp_path, "bird_admin", "BIRD_RO")
        assert 'GRANT SELECT ON ALL TABLES IN SCHEMA BIRD."a_db" TO ROLE BIRD_RO' in conn.statements
        assert not any('BIRD."b_db" TO ROLE' in sql for sql in conn.statements)

    def test_replaces_only_a_schema_it_created(self, snowflake_env, monkeypatch):
        # On a shared account a same-named schema may be someone else's: only the loader's own (a failed load's) goes.
        from resources_servers.bird_sql.setup_bird_snowflake import _LOADER_COMMENT

        theirs = _FakeAdminConnection(schemas={DB_ID: "someone else's"})
        with pytest.raises(RuntimeError, match="was not created by this loader; drop it first"):
            self._load(snowflake_env, monkeypatch, theirs)
        assert not any(f'SCHEMA BIRD."{DB_ID}"' in sql and sql.startswith("CREATE") for sql in theirs.statements)
        assert theirs.written == {}
        ours = _FakeAdminConnection(schemas={DB_ID: _LOADER_COMMENT})
        self._load(snowflake_env, monkeypatch, ours)
        assert f'CREATE OR REPLACE SCHEMA BIRD."{DB_ID}" COMMENT = %s' in ours.statements
        assert ours.written == {"t": 3, "Mixed Case": 2, "big": 3000}

    @pytest.mark.parametrize(
        ("conn", "grant_to", "error"),
        [
            (_FakeAdminConnection(loaded={DB_ID: 2}), "BIRD_RO", "another layout"),
            (_FakeAdminConnection(), "BIRD_RO; DROP DATABASE BIRD", "plain role name"),
        ],
    )
    def test_refuses_another_layout_and_an_odd_role(self, snowflake_env, monkeypatch, conn, grant_to, error):
        with pytest.raises((RuntimeError, ValueError), match=error):
            self._load(snowflake_env, monkeypatch, conn, grant_to)
        assert conn.written == {}


class _FakeGraderConnection(_FakeAdminConnection):
    """The grader's connection at startup: ``loaded`` as recorded, and ``columns`` as its role can see them."""

    def __init__(self, loaded, columns):
        super().__init__(loaded=loaded)
        self.columns = columns

    def execute(self, sql, params=None):
        super().execute(sql, params)
        if sql.startswith("SELECT table_schema"):
            self._rows = self.columns
        return self


@requires_fakesnow
@pytest.mark.parametrize("visible", [True, False])
def test_account_check_refuses_a_db_id_its_role_cannot_read(snowflake_env, monkeypatch, visible):
    # Without the grant the catalog is empty for that db_id, and every model query on it would be refused and score 0.
    import snowflake.connector

    from resources_servers.bird_sql.setup_bird_snowflake import check_bird_snowflake_account

    columns = [(DB_ID, "t", "x")] if visible else [("other_db", "t", "x")]
    conn = _FakeGraderConnection(loaded={DB_ID: 1}, columns=columns)
    monkeypatch.setattr(snowflake.connector, "connect", lambda **kw: conn)
    if visible:
        assert check_bird_snowflake_account(snowflake_env.dev_dir, "bird_ro", 30).tables == {DB_ID: {"t": {"x"}}}
    else:
        with pytest.raises(RuntimeError, match=f"can't see: \\['{DB_ID}'\\]"):
            check_bird_snowflake_account(snowflake_env.dev_dir, "bird_ro", 30)


@requires_fakesnow
class TestExecuteAndCompareSnowflake:
    @staticmethod
    async def _compare(env, gold, pred, timeout_s=30.0):
        from resources_servers.bird_sql.eval_utils import execute_and_compare_snowflake

        db_path = env.dev_dir / DB_ID / f"{DB_ID}.sqlite"
        return await execute_and_compare_snowflake(
            db_path, env.local, DB_ID, gold, pred, asyncio.Semaphore(2), timeout_s=timeout_s
        )

    @pytest.mark.asyncio
    async def test_match_ignores_row_order(self, snowflake_env):
        match, gold, pred, err = await self._compare(
            snowflake_env, "SELECT x FROM t ORDER BY x", 'SELECT "x" FROM "t" ORDER BY "x" DESC'
        )
        assert (match, err) == (True, None)
        assert sorted(gold) == sorted(pred) == [(1,), (2,), (3,)]

    @pytest.mark.asyncio
    async def test_gold_stays_sqlite_and_pred_is_snowflake(self, snowflake_env):
        # IIF() is SQLite-only and IFF() Snowflake-only; the gold is not transpiled, so each side runs its own dialect.
        match, _gold, _pred, err = await self._compare(
            snowflake_env, "SELECT IIF(x > 1, 1, 0) FROM t", 'SELECT IFF("x" > 1, 1, 0) FROM "t"'
        )
        assert (match, err) == (True, None)

    @pytest.mark.asyncio
    async def test_unquoted_or_miscased_identifiers_are_errors_although_duckdb_would_run_them(self, snowflake_env):
        gold = 'SELECT "Col Name" FROM "mixed case"'  # SQLite ignores identifier case
        quoted = await self._compare(snowflake_env, gold, 'SELECT "Col Name" FROM "Mixed Case"')
        lowered = await self._compare(snowflake_env, gold, 'SELECT "col name" FROM "Mixed Case"')
        unquoted = await self._compare(snowflake_env, "SELECT x FROM t", "SELECT x FROM t")
        assert (quoted[0], quoted[3]) == (True, None)
        assert lowered[3] == unquoted[3] == "pred_sql_error"

    @pytest.mark.asyncio
    async def test_same_named_columns_are_both_kept(self, snowflake_env):
        match, _gold, pred, err = await self._compare(
            snowflake_env,
            "SELECT a.x, b.x FROM t a JOIN t b ON b.x = a.x + 1",
            'SELECT a."x", b."x" FROM "t" a JOIN "t" b ON b."x" = a."x" + 1',
        )
        assert (match, err) == (True, None)
        assert sorted(pred) == [(1, 2), (2, 3)]

    @pytest.mark.asyncio
    async def test_like_is_case_sensitive_and_ilike_is_not(self, snowflake_env):
        gold = 'SELECT "Col Name" FROM "Mixed Case" WHERE "Col Name" LIKE \'b\''  # SQLite LIKE ignores case
        like = await self._compare(
            snowflake_env, gold, 'SELECT "Col Name" FROM "Mixed Case" WHERE "Col Name" LIKE \'b\''
        )
        ilike = await self._compare(
            snowflake_env, gold, 'SELECT "Col Name" FROM "Mixed Case" WHERE "Col Name" ILIKE \'b\''
        )
        assert (like[0], like[2], like[3]) == (False, [], None)
        assert (ilike[0], ilike[3]) == (True, None)

    @pytest.mark.asyncio
    async def test_mismatch(self, snowflake_env):
        match, _gold, pred, err = await self._compare(
            snowflake_env, "SELECT x FROM t", 'SELECT "x" FROM "t" WHERE "x" > 1'
        )
        assert (match, err) == (False, None)
        assert sorted(pred) == [(2,), (3,)]

    @pytest.mark.asyncio
    async def test_trailing_semicolon_is_accepted(self, snowflake_env):
        match, _g, _p, err = await self._compare(snowflake_env, "SELECT x FROM t", 'SELECT "x" FROM "t";')
        assert (match, err) == (True, None)

    @pytest.mark.asyncio
    async def test_pred_error(self, snowflake_env):
        match, gold, pred, err = await self._compare(snowflake_env, "SELECT x FROM t", 'SELECT "x" + FROM "t"')
        assert (match, pred, err) == (False, None, "pred_sql_error")
        assert gold is not None

    @pytest.mark.asyncio
    async def test_a_result_python_cannot_hold_is_a_pred_error(self, snowflake_env):
        # The stand-in returns 'infinity', which has no Python date: the query's fault, not the grader's.
        match, _gold, pred, err = await self._compare(snowflake_env, "SELECT 1", "SELECT 'infinity'::date")
        assert (match, pred, err) == (False, None, "pred_sql_error")

    @pytest.mark.asyncio
    async def test_gold_error(self, snowflake_env):
        assert await self._compare(snowflake_env, "SELEC", 'SELECT "x" FROM "t"') == (
            False,
            None,
            None,
            "gold_sql_error",
        )

    @pytest.mark.asyncio
    async def test_slow_query_is_a_pred_timeout_and_is_interrupted(self, snowflake_env):
        slow = 'SELECT COUNT(*) FROM "big" a, "big" b, "big" c WHERE MOD(a."x" * b."x" + c."x", 7) = 3'
        match, _gold, pred, err = await self._compare(snowflake_env, "SELECT 1", slow, timeout_s=0.5)
        assert (match, pred, err) == (False, None, "pred_sql_timeout")
        match, *_ = await self._compare(snowflake_env, "SELECT x FROM t", 'SELECT "x" FROM "t"')
        assert match is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "pred",
        [
            'DROP TABLE "t"',
            'INSERT INTO "t" VALUES (9)',
            'SELECT 1; DROP TABLE "t"',
            'CREATE TABLE "u" (a INT)',
            'UPDATE "t" SET "x" = 0',
        ],
    )
    async def test_non_query_statements_are_rejected_and_do_not_modify_data(self, snowflake_env, pred):
        _match, _gold, pred_rows, err = await self._compare(snowflake_env, "SELECT x FROM t", pred)
        assert (pred_rows, err) == (None, "pred_sql_error")
        match, *_ = await self._compare(snowflake_env, "SELECT x FROM t", 'SELECT "x" FROM "t"')
        assert match is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reader", ["read_csv('{csv}')", "sqlite_scan('{gold}', 't')"])
    async def test_model_sql_cannot_read_files(self, snowflake_env, tmp_path, reader):
        # DuckDB's table functions pass the static check; external access is off, so they fail on the stand-in. One
        # file is any readable CSV, the other the gold SQLite database itself.
        csv = tmp_path / "hosts.csv"
        csv.write_text("a,b\n1,2\n3,4\n")
        gold = snowflake_env.dev_dir / DB_ID / f"{DB_ID}.sqlite"
        sql = f"SELECT COUNT(*) FROM {reader.format(csv=csv, gold=gold)}"
        _match, _gold, pred_rows, err = await self._compare(snowflake_env, "SELECT COUNT(*) FROM t", sql)
        assert (pred_rows, err) == (None, "pred_sql_error")

    def test_the_stand_in_itself_is_read_only(self, snowflake_env):
        # The static check refuses DML first; the read-only attach is the second line.
        cur = snowflake_env.local.connect(DB_ID).cursor()
        with pytest.raises(Exception, match="read-only"):
            cur.execute('DELETE FROM "t"')

    @pytest.mark.asyncio
    async def test_verify_end_to_end(self, snowflake_env, tmp_path, monkeypatch):
        monkeypatch.setattr("resources_servers.bird_sql.app.ensure_bird_sql", lambda _path: snowflake_env.dev_dir)
        config = BirdSqlResourcesServerConfig(
            host="127.0.0.1",
            port=20099,
            entrypoint="",
            bird_sql_dir=str(tmp_path),
            dialect="snowflake",
            snowflake_db_dir=str(snowflake_env.db_dir),
            sql_execution_timeout_s=0.5,
        )
        server = BirdSqlResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        sqlite_gold = "SELECT IIF(x > 1, 1, 0) FROM t"  # SQLite-only syntax; never transpiled
        answer = '```sql\nSELECT IFF("x" > 1, 1, 0) FROM "t"\n```'
        good = await server.verify(_make_verify_request(answer, gt_sql=sqlite_gold, db_id=DB_ID))
        bad = await server.verify(
            _make_verify_request('```sql\nSELECT MAX("x") FROM "t"\n```', gt_sql=sqlite_gold, db_id=DB_ID)
        )
        refused = await server.verify(
            _make_verify_request("```sql\nSELECT IIF(x > 1, 1, 0) FROM t\n```", gt_sql=sqlite_gold, db_id=DB_ID)
        )
        slow = await server.verify(
            _make_verify_request(
                '```sql\nSELECT COUNT(*) FROM "big" a, "big" b, "big" c WHERE MOD(a."x" * b."x" + c."x", 7) = 3\n```',
                gt_sql=sqlite_gold,
                db_id=DB_ID,
            )
        )
        assert (good.reward, good.failure_reason) == (1.0, FailureCode.NONE)
        assert (bad.reward, bad.failure_reason) == (0.0, FailureCode.RESULT_MISMATCH)
        assert (refused.reward, refused.failure_reason) == (0.0, FailureCode.EXECUTION_ERROR)
        assert slow.failure_reason == FailureCode.EXECUTION_TIMEOUT

    @pytest.mark.asyncio
    async def test_a_broken_stand_in_is_not_scored_as_a_pred_error(self, snowflake_env, tmp_path, monkeypatch):
        # A grader failure must not look like the model's mistake: it raises, and verify() masks the row as a
        # verifier error (unknown_error) instead of scoring it. Here the stand-in hands out a closed connection.
        from resources_servers.bird_sql.setup_bird_snowflake import LocalSnowflake

        monkeypatch.setattr("resources_servers.bird_sql.app.ensure_bird_sql", lambda _path: snowflake_env.dev_dir)
        config = BirdSqlResourcesServerConfig(
            host="127.0.0.1",
            port=20099,
            entrypoint="",
            bird_sql_dir=str(tmp_path),
            dialect="snowflake",
            snowflake_db_dir=str(snowflake_env.db_dir),
        )
        server = BirdSqlResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        connect = LocalSnowflake.connect

        def closed(self, db_id):
            conn = connect(self, db_id)
            conn.close()
            return conn

        monkeypatch.setattr(LocalSnowflake, "connect", closed)
        resp = await server.verify(
            _make_verify_request('```sql\nSELECT "x" FROM "t"\n```', gt_sql="SELECT x FROM t", db_id=DB_ID)
        )
        assert (resp.reward, resp.failure_reason) == (0.0, FailureCode.UNKNOWN_ERROR)
        assert (resp.mask_sample, resp.failure_kind) == (True, VERIFIER_ERROR)
