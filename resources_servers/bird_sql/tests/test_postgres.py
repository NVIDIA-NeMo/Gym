# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PostgreSQL dialect tests.

Tests that need a server run only when ``BIRD_PG_TEST_ADMIN_DSN`` points at a reachable PostgreSQL server whose role
can create schemas and roles, e.g. a ``postgres:16`` container. They load one small schema (``bird_pytest_db``) and a
``bird_pytest_reader`` role, and drop the schema afterwards.
"""

import asyncio
import contextlib
import importlib.util
import os
import sqlite3
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import VERIFIER_ERROR
from resources_servers.bird_sql.app import BirdSqlResourcesServer, BirdSqlResourcesServerConfig, FailureCode
from resources_servers.bird_sql.tests.test_app import _make_verify_request


DB_ID = "bird_pytest_db"
ADMIN_DSN = os.environ.get("BIRD_PG_TEST_ADMIN_DSN", "")
READER_DSN = os.environ.get(
    "BIRD_PG_TEST_DSN", "postgresql://bird_pytest_reader:bird_pytest_reader@127.0.0.1:5432/bird"
)
_HAS_DEPS = all(importlib.util.find_spec(m) for m in ("psycopg", "psycopg_pool", "pandas"))


def _postgres_reachable() -> bool:
    if not (_HAS_DEPS and ADMIN_DSN):
        return False
    import psycopg

    try:
        psycopg.connect(ADMIN_DSN, connect_timeout=2).close()
        return True
    except psycopg.Error:
        return False


requires_postgres = pytest.mark.skipif(
    not _postgres_reachable(),
    reason="PostgreSQL dialect tests need psycopg, psycopg_pool, pandas and a server at BIRD_PG_TEST_ADMIN_DSN",
)
requires_psycopg = pytest.mark.skipif(not _HAS_DEPS, reason="needs psycopg, psycopg_pool and pandas")


@requires_psycopg
@pytest.mark.asyncio
async def test_verify_postgres_dialect_routes_to_postgres(tmp_path, monkeypatch):
    # Gold runs on the SQLite file, so verify needs it to exist; the mocked executors never open it.
    (tmp_path / "TestDB").mkdir()
    (tmp_path / "TestDB" / "TestDB.sqlite").touch()
    monkeypatch.setattr("resources_servers.bird_sql.app.ensure_bird_sql", lambda _path: tmp_path)
    checked = []
    monkeypatch.setattr(
        "resources_servers.bird_sql.setup_bird_postgres.check_bird_postgres", lambda *a: checked.append(a)
    )
    monkeypatch.setattr(
        "resources_servers.bird_sql.setup_bird_postgres.ensure_bird_postgres",
        lambda *_a: pytest.fail("no admin DSN configured: startup must only check, not load"),
    )
    seen = {}

    async def fake_postgres(**kw):
        seen.update(kw)
        return True, [(1,)], [(1,)], None

    async def fail_sqlite(**_kw):
        raise AssertionError("sqlite path must not run for dialect=postgres")

    monkeypatch.setattr("resources_servers.bird_sql.app.execute_and_compare_postgres", fake_postgres)
    monkeypatch.setattr("resources_servers.bird_sql.app.execute_and_compare", fail_sqlite)
    config = BirdSqlResourcesServerConfig(
        host="127.0.0.1",
        port=20099,
        entrypoint="",
        bird_sql_dir=str(tmp_path),
        dialect="postgres",
        postgres_dsn="dsn",
        max_concurrency=7,
    )
    server = BirdSqlResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    resp = await server.verify(_make_verify_request("```sql\nSELECT x FROM t\n```", gt_sql="SELECT x FROM t"))
    assert resp.reward == 1.0
    assert checked == [(tmp_path, "dsn")]
    # One pool per server, sized to max_concurrency so a grade holding the semaphore always finds a connection.
    assert (seen["pool"].dsn, seen["pool"].size, seen["db_id"]) == ("dsn", 7, "TestDB")
    assert seen["db_path"] == tmp_path / "TestDB" / "TestDB.sqlite"  # gold runs on SQLite
    assert seen["gold_sql"] == "SELECT x FROM t"  # untouched, never transpiled


@requires_psycopg
@pytest.mark.asyncio
async def test_unreachable_server_is_not_scored_as_a_pred_error(tmp_path, monkeypatch):
    # A grader outage must not look like the model's mistake: it raises, and verify() masks the row as a
    # verifier error (unknown_error) instead of scoring it.
    (tmp_path / "TestDB").mkdir()
    (tmp_path / "TestDB" / "TestDB.sqlite").touch()
    monkeypatch.setattr("resources_servers.bird_sql.app.ensure_bird_sql", lambda _path: tmp_path)
    monkeypatch.setattr("resources_servers.bird_sql.setup_bird_postgres.check_bird_postgres", lambda *_a: None)
    monkeypatch.setattr("resources_servers.bird_sql.eval_utils.execute_sqlite", lambda *_a: [(1,)])
    config = BirdSqlResourcesServerConfig(
        host="127.0.0.1",
        port=20099,
        entrypoint="",
        bird_sql_dir=str(tmp_path),
        dialect="postgres",
        postgres_dsn="postgresql://nobody@127.0.0.1:1/none?connect_timeout=2",
    )
    server = BirdSqlResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    resp = await server.verify(_make_verify_request("```sql\nSELECT 1\n```", gt_sql="SELECT 1"))
    assert (resp.reward, resp.failure_reason) == (0.0, FailureCode.UNKNOWN_ERROR)
    assert (resp.mask_sample, resp.failure_kind) == (True, VERIFIER_ERROR)


@requires_psycopg
def test_missing_psycopg_pool_fails_at_startup_not_on_every_grade(tmp_path, monkeypatch):
    monkeypatch.setattr("resources_servers.bird_sql.app.ensure_bird_sql", lambda _path: tmp_path)
    monkeypatch.setattr("resources_servers.bird_sql.setup_bird_postgres.check_bird_postgres", lambda *_a: None)
    monkeypatch.setitem(sys.modules, "psycopg_pool", None)  # makes `import psycopg_pool` raise ImportError
    config = BirdSqlResourcesServerConfig(
        host="127.0.0.1", port=20099, entrypoint="", bird_sql_dir=str(tmp_path), dialect="postgres", postgres_dsn="dsn"
    )
    with pytest.raises(ImportError):
        BirdSqlResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class _ConnRaising:
    """Stands in for a pooled connection whose query raises ``error``."""

    connection = None  # psycopg.sql takes the encoding from it
    closed = False

    def __init__(self, error):
        self.error = error

    def transaction(self, force_rollback):
        return contextlib.nullcontext()

    async def execute(self, *_args, **_kwargs):
        raise self.error


@requires_psycopg
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "is_query_fault"),
    [
        ("DataError", True),  # a result psycopg cannot convert, e.g. 'infinity'::date
        ("OperationalError", False),  # the connection was lost
        ("InterfaceError", False),  # the connection is closed
        ("errors.StatementTooComplex", True),  # a server error (SQLSTATE 54001) the query caused
        ("errors.AdminShutdown", False),  # a server error (SQLSTATE 57P01) the query did not cause
    ],
)
async def test_errors_are_split_into_query_faults_and_grader_failures(error, is_query_fault):
    # psycopg raises the first three itself, without a SQLSTATE; the server sends the last two with one.
    import operator

    import psycopg

    from resources_servers.bird_sql.eval_utils import _run_postgres

    conn = _ConnRaising(operator.attrgetter(error)(psycopg)("message"))
    if is_query_fault:
        assert await _run_postgres(conn, DB_ID, "SELECT 1", 1.0) is None
    else:
        with pytest.raises(type(conn.error)):
            await _run_postgres(conn, DB_ID, "SELECT 1", 1.0)


class _PostgresEnv:
    dev_dir: Path


def _drop_test_schema() -> None:
    import psycopg

    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{DB_ID}" CASCADE')
        if conn.execute("SELECT to_regclass('_bird_meta.loaded')").fetchone()[0] is not None:
            conn.execute("DELETE FROM _bird_meta.loaded WHERE db_id = %s", (DB_ID,))


@pytest.fixture(scope="module")
def postgres_env(tmp_path_factory):
    from resources_servers.bird_sql.setup_bird_postgres import ensure_bird_postgres

    dev = tmp_path_factory.mktemp("dev_databases")
    (dev / DB_ID).mkdir()
    conn = sqlite3.connect(str(dev / DB_ID / f"{DB_ID}.sqlite"))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    conn.execute('CREATE TABLE "Mixed Case" ("Col Name" TEXT, N INTEGER, r REAL, dirty INTEGER, d DATE)')
    conn.executemany(
        'INSERT INTO "Mixed Case" VALUES (?, ?, ?, ?, ?)',
        [("a", 1, 1.5, 1, "2020-01-01"), ("B", None, 2.5, "oops", "2020-01-02")],
    )
    conn.commit()
    conn.close()
    _drop_test_schema()
    ensure_bird_postgres(dev, ADMIN_DSN, READER_DSN)
    env = _PostgresEnv()
    env.dev_dir = dev
    yield env
    _drop_test_schema()


@requires_postgres
class TestLoader:
    def test_types_fallback_and_collation(self, postgres_env):
        import psycopg

        with psycopg.connect(ADMIN_DSN) as conn:
            types = dict(
                conn.execute(
                    "SELECT column_name, data_type || coalesce(' ' || collation_name, '') FROM information_schema.columns"
                    " WHERE table_schema = %s AND table_name = 'Mixed Case'",
                    (DB_ID,),
                ).fetchall()
            )
            fallbacks = conn.execute("SELECT fallbacks FROM _bird_meta.loaded WHERE db_id = %s", (DB_ID,)).fetchone()[
                0
            ]
            rows = conn.execute(f'SELECT dirty, d FROM "{DB_ID}"."Mixed Case" ORDER BY d').fetchall()
        assert types == {  # column names lower-cased, the table name kept
            "col name": "text C",
            "n": "bigint",
            "r": "double precision",
            "dirty": "text C",  # INTEGER column with non-numeric data falls back to text
            "d": "text C",  # dates stay text, as in SQLite
        }
        assert fallbacks == [{"db": DB_ID, "table": "Mixed Case", "column": "dirty", "type": "INTEGER"}]
        assert rows == [("1", "2020-01-01"), ("oops", "2020-01-02")]

    def test_second_load_is_a_no_op(self, postgres_env):
        import psycopg

        from resources_servers.bird_sql.setup_bird_postgres import ensure_bird_postgres

        query = "SELECT loaded_at FROM _bird_meta.loaded WHERE db_id = %s"
        with psycopg.connect(ADMIN_DSN) as conn:
            before = conn.execute(query, (DB_ID,)).fetchone()
        assert DB_ID in ensure_bird_postgres(postgres_env.dev_dir, ADMIN_DSN, READER_DSN)
        with psycopg.connect(ADMIN_DSN) as conn:
            assert conn.execute(query, (DB_ID,)).fetchone() == before

    def test_tables_have_planner_statistics_after_the_load(self, postgres_env):
        import psycopg

        with psycopg.connect(ADMIN_DSN) as conn:
            never_analyzed = conn.execute(
                "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace"
                " WHERE n.nspname = %s AND c.relkind = 'r' AND c.reltuples < 0",
                (DB_ID,),
            ).fetchall()
        assert never_analyzed == []

    def test_reader_role_cannot_write_even_outside_the_grader(self, postgres_env):
        import psycopg

        with psycopg.connect(READER_DSN, autocommit=True) as conn:
            with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                conn.execute(f'INSERT INTO "{DB_ID}".t VALUES (9)')
            conn.execute("SET default_transaction_read_only = off")
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(f'INSERT INTO "{DB_ID}".t VALUES (9)')

    def test_a_load_with_an_older_layout_must_be_reloaded(self, postgres_env):
        import psycopg

        from resources_servers.bird_sql.setup_bird_postgres import (
            _LAYOUT,
            check_bird_postgres,
            ensure_bird_postgres,
        )

        set_layout = "UPDATE _bird_meta.loaded SET layout = %s WHERE db_id = %s"
        with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
            conn.execute(set_layout, (None, DB_ID))  # as recorded before layouts had a version
        try:
            with pytest.raises(RuntimeError, match=f"older layout.*{DB_ID}"):
                check_bird_postgres(postgres_env.dev_dir, READER_DSN)
            with pytest.raises(RuntimeError, match=f"older layout.*{DB_ID}"):
                ensure_bird_postgres(postgres_env.dev_dir, ADMIN_DSN, READER_DSN)
        finally:
            with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
                conn.execute(set_layout, (_LAYOUT, DB_ID))
        assert check_bird_postgres(postgres_env.dev_dir, READER_DSN) == [DB_ID]

    def test_check_reports_missing_db_ids(self, postgres_env, tmp_path):
        from resources_servers.bird_sql.setup_bird_postgres import check_bird_postgres

        assert check_bird_postgres(postgres_env.dev_dir, READER_DSN) == [DB_ID]
        (tmp_path / "not_loaded").mkdir()
        (tmp_path / "not_loaded" / "not_loaded.sqlite").touch()
        with pytest.raises(RuntimeError, match="not_loaded"):
            check_bird_postgres(tmp_path, READER_DSN)


@contextlib.asynccontextmanager
async def _reader_pool(size):
    from resources_servers.bird_sql.eval_utils import PostgresPool

    pool = PostgresPool(READER_DSN, size)
    try:
        yield pool
    finally:
        await pool.close()


@requires_postgres
class TestExecuteAndComparePostgres:
    @staticmethod
    async def _compare(env, gold, pred, timeout_s=30.0):
        from resources_servers.bird_sql.eval_utils import execute_and_compare_postgres

        db_path = env.dev_dir / DB_ID / f"{DB_ID}.sqlite"
        async with _reader_pool(2) as pool:
            return await execute_and_compare_postgres(
                db_path, pool, DB_ID, gold, pred, asyncio.Semaphore(2), timeout_s=timeout_s
            )

    @pytest.mark.asyncio
    async def test_match_ignores_row_order(self, postgres_env):
        match, gold, pred, err = await self._compare(
            postgres_env, "SELECT x FROM t ORDER BY x", "SELECT x FROM t ORDER BY x DESC"
        )
        assert (match, err) == (True, None)
        assert sorted(gold) == sorted(pred) == [(1,), (2,), (3,)]

    @pytest.mark.asyncio
    async def test_gold_stays_sqlite_and_pred_is_postgres(self, postgres_env):
        # IIF() is SQLite-only; the gold is not transpiled, so each side runs its own dialect.
        match, _gold, _pred, err = await self._compare(
            postgres_env, "SELECT IIF(x > 1, 1, 0) FROM t", "SELECT CASE WHEN x > 1 THEN 1 ELSE 0 END FROM t"
        )
        assert (match, err) == (True, None)

    @pytest.mark.asyncio
    async def test_numeric_and_float_noise_are_normalized(self, postgres_env):
        # PostgreSQL's AVG(bigint) is NUMERIC (a Decimal); SQLite's AVG is a float.
        match, _gold, _pred, err = await self._compare(
            postgres_env, "SELECT AVG(x) * 0.1 FROM t", "SELECT AVG(x) / 10 FROM t"
        )
        assert (match, err) == (True, None)

    @pytest.mark.asyncio
    async def test_column_names_are_lower_case_and_table_names_keep_their_case(self, postgres_env):
        gold = 'SELECT "Col Name", N FROM "mixed case"'  # SQLite ignores identifier case
        for pred in ('SELECT "col name", n FROM "Mixed Case"', 'SELECT "col name", N FROM "Mixed Case"'):
            match, _gold, _pred, err = await self._compare(postgres_env, gold, pred)  # unquoted N folds to n
            assert (match, err) == (True, None)
        for pred in ('SELECT "Col Name", n FROM "Mixed Case"', 'SELECT "col name", n FROM "mixed case"'):
            assert (await self._compare(postgres_env, gold, pred))[3] == "pred_sql_error"

    @pytest.mark.asyncio
    async def test_text_sorts_bytewise_as_in_sqlite(self, postgres_env):
        # With the server's en_US collation 'a' sorts before 'B'; SQLite and the "C" collation put 'B' first.
        match, _gold, pred, err = await self._compare(
            postgres_env,
            'SELECT "Col Name" FROM "Mixed Case" ORDER BY 1 LIMIT 1',
            'SELECT "col name" FROM "Mixed Case" ORDER BY 1 LIMIT 1',
        )
        assert (match, pred, err) == (True, [("B",)], None)

    @pytest.mark.asyncio
    async def test_mismatch(self, postgres_env):
        match, _gold, pred, err = await self._compare(postgres_env, "SELECT x FROM t", "SELECT x FROM t WHERE x > 1")
        assert (match, err) == (False, None)
        assert sorted(pred) == [(2,), (3,)]

    @pytest.mark.asyncio
    async def test_trailing_semicolon_is_accepted(self, postgres_env):
        match, _g, _p, err = await self._compare(postgres_env, "SELECT x FROM t", "SELECT x FROM t;")
        assert (match, err) == (True, None)

    @pytest.mark.asyncio
    async def test_pred_error(self, postgres_env):
        match, gold, pred, err = await self._compare(postgres_env, "SELECT x FROM t", "SELECT nope FROM t")
        assert (match, pred, err) == (False, None, "pred_sql_error")
        assert gold is not None

    @pytest.mark.asyncio
    async def test_a_result_psycopg_cannot_convert_is_a_pred_error(self, postgres_env):
        # The server returns 'infinity', but psycopg has no Python date for it: a DataError with no SQLSTATE.
        match, _gold, pred, err = await self._compare(postgres_env, "SELECT 1", "SELECT 'infinity'::date")
        assert (match, pred, err) == (False, None, "pred_sql_error")

    @pytest.mark.asyncio
    async def test_gold_error(self, postgres_env):
        assert await self._compare(postgres_env, "SELEC", "SELECT x FROM t") == (False, None, None, "gold_sql_error")

    @pytest.mark.asyncio
    async def test_statement_timeout_is_a_pred_timeout(self, postgres_env):
        match, _gold, pred, err = await self._compare(
            postgres_env, "SELECT 1", "SELECT 1 FROM pg_sleep(5)", timeout_s=0.5
        )
        assert (match, pred, err) == (False, None, "pred_sql_timeout")

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "pred",
        [
            "DROP TABLE t",
            "INSERT INTO t VALUES (9)",
            "SELECT 1; DROP TABLE t",
            "COMMIT; DROP TABLE t",
            "SET x.y = 1",
        ],
    )
    async def test_non_query_statements_are_rejected_and_do_not_modify_data(self, postgres_env, pred):
        _match, _gold, pred_rows, err = await self._compare(postgres_env, "SELECT x FROM t", pred)
        assert (pred_rows, err) == (None, "pred_sql_error")
        match, *_ = await self._compare(postgres_env, "SELECT x FROM t", "SELECT x FROM t")
        assert match is True

    @pytest.mark.asyncio
    async def test_verify_end_to_end(self, postgres_env, tmp_path, monkeypatch):
        monkeypatch.setattr("resources_servers.bird_sql.app.ensure_bird_sql", lambda _path: postgres_env.dev_dir)
        config = BirdSqlResourcesServerConfig(
            host="127.0.0.1",
            port=20099,
            entrypoint="",
            bird_sql_dir=str(tmp_path),
            dialect="postgres",
            postgres_dsn=READER_DSN,
            sql_execution_timeout_s=1.0,
        )
        server = BirdSqlResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        sqlite_gold = "SELECT IIF(x > 1, 1, 0) FROM t"  # SQLite-only syntax; never transpiled
        answer = "```sql\nSELECT CASE WHEN x > 1 THEN 1 ELSE 0 END FROM t\n```"
        good = await server.verify(_make_verify_request(answer, gt_sql=sqlite_gold, db_id=DB_ID))
        bad = await server.verify(
            _make_verify_request("```sql\nSELECT max(x) FROM t\n```", gt_sql=sqlite_gold, db_id=DB_ID)
        )
        slow = await server.verify(
            _make_verify_request("```sql\nSELECT pg_sleep(60)\n```", gt_sql=sqlite_gold, db_id=DB_ID)
        )
        await server._postgres.close()
        assert (good.reward, good.failure_reason) == (1.0, FailureCode.NONE)
        assert (bad.reward, bad.failure_reason) == (0.0, FailureCode.RESULT_MISMATCH)
        assert slow.failure_reason == FailureCode.EXECUTION_TIMEOUT


def _reader_connections(expected: int) -> int:
    """The test reader's server connections, polled for up to 2 s until they equal ``expected``: a backend leaves
    pg_stat_activity when its process exits, a moment after the client closes the socket."""
    import time

    import psycopg

    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        for _ in range(20):
            count = conn.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE usename = 'bird_pytest_reader' AND pid <> pg_backend_pid()"
            ).fetchone()[0]
            if count == expected:
                break
            time.sleep(0.1)
    return count


@requires_postgres
class TestPooling:
    @staticmethod
    async def _run(pool, sql, semaphore=None, timeout_s=30.0):
        from resources_servers.bird_sql.eval_utils import execute_postgres_async

        semaphore = semaphore or asyncio.Semaphore(pool.size)
        return await execute_postgres_async(pool, DB_ID, sql, semaphore, timeout_s, raise_on_timeout=True)

    @pytest.mark.asyncio
    async def test_grades_reuse_one_connection(self, postgres_env):
        async with _reader_pool(1) as pool:
            first = await self._run(pool, "SELECT pg_backend_pid()")
            second = await self._run(pool, "SELECT pg_backend_pid()")
        assert first == second

    @pytest.mark.asyncio
    async def test_concurrent_grades_never_use_more_than_size_connections(self, postgres_env):
        async with _reader_pool(2) as pool:
            semaphore = asyncio.Semaphore(8)  # more grades in flight than connections: the rest wait for one
            rows = await asyncio.gather(
                *(self._run(pool, "SELECT pg_backend_pid() FROM pg_sleep(0.1)", semaphore) for _ in range(8))
            )
        assert len({r[0][0] for r in rows}) <= 2

    @pytest.mark.asyncio
    async def test_a_timed_out_grade_returns_its_connection_for_reuse(self, postgres_env):
        async with _reader_pool(1) as pool:
            pid = await self._run(pool, "SELECT pg_backend_pid()")
            with pytest.raises(asyncio.TimeoutError):
                await self._run(pool, "SELECT 1 FROM pg_sleep(5)", timeout_s=0.5)
            assert await self._run(pool, "SELECT pg_backend_pid()") == pid

    @pytest.mark.asyncio
    async def test_a_query_cannot_change_settings_for_later_grades(self, postgres_env):
        # A pooled connection outlives the grade, so a model's SET must not reach the next rollout on it (DateStyle or
        # TimeZone change how results print, and so whether they match).
        settings = "SELECT current_setting('DateStyle'), current_setting('TimeZone'), current_setting('search_path')"
        async with _reader_pool(1) as pool:
            before = await self._run(pool, settings)
            await self._run(pool, "SELECT set_config('DateStyle', 'German', false)")
            assert await self._run(pool, "SET TimeZone = 'Asia/Tokyo'") is None
            await self._run(pool, "SELECT set_config('search_path', 'pg_catalog', false)")
            after = await self._run(pool, settings)
        assert after == before

    @pytest.mark.asyncio
    async def test_waiting_for_a_connection_does_not_count_against_the_query(self, postgres_env, monkeypatch):
        # Backstop 1.2 s; the third grade waits about 0.8 s for the only connection, then runs 0.4 s.
        monkeypatch.setattr("resources_servers.bird_sql.eval_utils._OUTER_TIMEOUT_BUFFER_S", 0.2)
        async with _reader_pool(1) as pool:
            semaphore = asyncio.Semaphore(3)
            results = await asyncio.gather(
                *(self._run(pool, "SELECT 1 FROM pg_sleep(0.4)", semaphore, timeout_s=1.0) for _ in range(3))
            )
        assert results == [[(1,)]] * 3

    @pytest.mark.asyncio
    async def test_a_backstop_cancel_discards_the_connection(self, postgres_env, monkeypatch):
        # The outer wait_for fires (0.2 s) before the server's statement_timeout (1 s): the server-side query is
        # cancelled and the connection, which may still be mid-query, is replaced rather than reused.
        monkeypatch.setattr("resources_servers.bird_sql.eval_utils._OUTER_TIMEOUT_BUFFER_S", -0.8)
        async with _reader_pool(1) as pool:
            pid = await self._run(pool, "SELECT pg_backend_pid()", timeout_s=1.0)
            with pytest.raises(asyncio.TimeoutError):
                await self._run(pool, "SELECT 1 FROM pg_sleep(5)", timeout_s=1.0)
            assert await self._run(pool, "SELECT pg_backend_pid()", timeout_s=1.0) != pid
        import psycopg

        with psycopg.connect(ADMIN_DSN) as conn:
            running = conn.execute(
                "SELECT count(*) FROM pg_stat_activity WHERE state = 'active' AND query LIKE '%%pg_sleep(5)%%'"
                " AND pid <> pg_backend_pid()"
            ).fetchone()[0]
        assert running == 0

    @pytest.mark.asyncio
    async def test_a_dropped_connection_raises(self, postgres_env):
        # A connection lost mid-query is the grader's failure, not the model's: it raises instead of scoring.
        import psycopg

        async with _reader_pool(1) as pool:
            pid = (await self._run(pool, "SELECT pg_backend_pid()"))[0][0]

            async def terminate():
                await asyncio.sleep(0.3)
                async with await psycopg.AsyncConnection.connect(ADMIN_DSN, autocommit=True) as admin:
                    await admin.execute("SELECT pg_terminate_backend(%s)", (pid,))

            with pytest.raises(psycopg.OperationalError):
                await asyncio.gather(self._run(pool, "SELECT 1 FROM pg_sleep(5)"), terminate())

    @pytest.mark.asyncio
    async def test_verify_keeps_one_pool_per_server(self, postgres_env, tmp_path, monkeypatch):
        monkeypatch.setattr("resources_servers.bird_sql.app.ensure_bird_sql", lambda _path: postgres_env.dev_dir)
        config = BirdSqlResourcesServerConfig(
            host="127.0.0.1",
            port=20099,
            entrypoint="",
            bird_sql_dir=str(tmp_path),
            dialect="postgres",
            postgres_dsn=READER_DSN,
            max_concurrency=2,
        )
        server = BirdSqlResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
        request = _make_verify_request("```sql\nSELECT x FROM t\n```", gt_sql="SELECT x FROM t", db_id=DB_ID)
        try:
            rewards = [r.reward for r in await asyncio.gather(*(server.verify(request) for _ in range(5)))]
            open_connections = _reader_connections(expected=2)
        finally:
            await server._postgres.close()
        assert rewards == [1.0] * 5
        assert open_connections == 2
        assert _reader_connections(expected=0) == 0
