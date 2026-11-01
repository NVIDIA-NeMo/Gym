# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PostgreSQL dialect tests.

Tests that need a server run only when ``BIRD_PG_TEST_ADMIN_DSN`` points at a reachable PostgreSQL server whose role
can create schemas and roles, e.g. a ``postgres:16`` container. They load one small schema (``bird_pytest_db``) and a
``bird_pytest_reader`` role, and drop the schema afterwards.
"""

import asyncio
import importlib.util
import os
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.server_utils import ServerClient
from resources_servers.bird_sql.app import BirdSqlResourcesServer, BirdSqlResourcesServerConfig, FailureCode
from resources_servers.bird_sql.tests.test_app import _make_verify_request


DB_ID = "bird_pytest_db"
ADMIN_DSN = os.environ.get("BIRD_PG_TEST_ADMIN_DSN", "")
READER_DSN = os.environ.get(
    "BIRD_PG_TEST_DSN", "postgresql://bird_pytest_reader:bird_pytest_reader@127.0.0.1:5432/bird"
)
_HAS_DEPS = all(importlib.util.find_spec(m) for m in ("psycopg", "pandas"))


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
    reason="PostgreSQL dialect tests need psycopg, pandas and a server at BIRD_PG_TEST_ADMIN_DSN",
)
requires_psycopg = pytest.mark.skipif(not _HAS_DEPS, reason="needs psycopg and pandas")


@requires_psycopg
@pytest.mark.asyncio
async def test_verify_postgres_dialect_routes_to_postgres(tmp_path, monkeypatch):
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
        host="127.0.0.1", port=20099, entrypoint="", bird_sql_dir=str(tmp_path), dialect="postgres", postgres_dsn="dsn"
    )
    server = BirdSqlResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    resp = await server.verify(_make_verify_request("```sql\nSELECT x FROM t\n```", gt_sql="SELECT x FROM t"))
    assert resp.reward == 1.0
    assert checked == [(tmp_path, "dsn")]
    assert (seen["dsn"], seen["db_id"]) == ("dsn", "TestDB")
    assert seen["db_path"] == tmp_path / "TestDB" / "TestDB.sqlite"  # gold runs on SQLite
    assert seen["gold_sql"] == "SELECT x FROM t"  # untouched, never transpiled


@requires_psycopg
@pytest.mark.asyncio
async def test_unreachable_server_is_not_scored_as_a_pred_error(tmp_path, monkeypatch):
    # A grader outage must not look like the model's mistake: it raises, and verify() reports unknown_error.
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
    conn.execute('CREATE TABLE "Mixed Case" ("Col Name" TEXT, n INTEGER, r REAL, dirty INTEGER, d DATE)')
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
        assert types == {
            "Col Name": "text C",
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

    def test_reader_role_cannot_write_even_outside_the_grader(self, postgres_env):
        import psycopg

        with psycopg.connect(READER_DSN, autocommit=True) as conn:
            with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                conn.execute(f'INSERT INTO "{DB_ID}".t VALUES (9)')
            conn.execute("SET default_transaction_read_only = off")
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(f'INSERT INTO "{DB_ID}".t VALUES (9)')

    def test_check_reports_missing_db_ids(self, postgres_env, tmp_path):
        from resources_servers.bird_sql.setup_bird_postgres import check_bird_postgres

        assert check_bird_postgres(postgres_env.dev_dir, READER_DSN) == [DB_ID]
        (tmp_path / "not_loaded").mkdir()
        (tmp_path / "not_loaded" / "not_loaded.sqlite").touch()
        with pytest.raises(RuntimeError, match="not_loaded"):
            check_bird_postgres(tmp_path, READER_DSN)


@requires_postgres
class TestExecuteAndComparePostgres:
    @staticmethod
    async def _compare(env, gold, pred, timeout_s=30.0):
        from resources_servers.bird_sql.eval_utils import execute_and_compare_postgres

        db_path = env.dev_dir / DB_ID / f"{DB_ID}.sqlite"
        return await execute_and_compare_postgres(
            db_path, READER_DSN, DB_ID, gold, pred, asyncio.Semaphore(2), timeout_s=timeout_s
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
    async def test_mixed_case_identifiers_must_be_quoted(self, postgres_env):
        gold = 'SELECT "Col Name" FROM "mixed case"'  # SQLite ignores identifier case
        quoted = await self._compare(postgres_env, gold, 'SELECT "Col Name" FROM "Mixed Case"')
        lowered = await self._compare(postgres_env, gold, 'SELECT "col name" FROM "Mixed Case"')
        assert (quoted[0], quoted[3]) == (True, None)
        assert lowered[3] == "pred_sql_error"

    @pytest.mark.asyncio
    async def test_text_sorts_bytewise_as_in_sqlite(self, postgres_env):
        # With the server's en_US collation 'a' sorts before 'B'; SQLite and the "C" collation put 'B' first.
        match, _gold, pred, err = await self._compare(
            postgres_env,
            'SELECT "Col Name" FROM "Mixed Case" ORDER BY 1 LIMIT 1',
            'SELECT "Col Name" FROM "Mixed Case" ORDER BY 1 LIMIT 1',
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
        assert (good.reward, good.failure_reason) == (1.0, FailureCode.NONE)
        assert (bad.reward, bad.failure_reason) == (0.0, FailureCode.RESULT_MISMATCH)
        assert slow.failure_reason == FailureCode.EXECUTION_TIMEOUT
