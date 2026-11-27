# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution-based evaluation utilities for BIRD text-to-SQL.

Mirrors the official BIRD eval (DAMO-ConvAI): execute both predicted and
ground-truth SQL against the per-db_id SQLite file, compare result sets via
``set(predicted) == set(ground_truth)``.

SQL execution uses ``asyncio.to_thread`` rather than Ray remote tasks.
When this resource server runs alongside a Ray-coordinated multi-node DP
vLLM (``gym env start`` attaches to the same Ray cluster), the vLLM engines
consume the Ray slots and ``@ray.remote`` sqlite tasks sit in the scheduler
queue past the per-query timeout, get cancelled, and every rollout reports
``gold_execution_error``. SQLite queries are fast and self-contained — no
reason to cross process boundaries; run them in the asyncio event loop's
default thread pool under a semaphore for bounded concurrency.
"""

import asyncio
import contextlib
import datetime
import decimal
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Optional


ResultRow = tuple[Any, ...]
ResultSet = list[ResultRow]


# Extra time the outer ``wait_for`` allows beyond ``timeout_s``; see ``execute_sqlite_async``.
_OUTER_TIMEOUT_BUFFER_S = 10.0
# How many SQLite VM instructions run between progress-handler checks.
_PROGRESS_HANDLER_INTERVAL = 1000


def execute_sqlite(db_path: Path, sql: str, timeout_s: Optional[float] = None) -> Optional[ResultSet]:
    """Execute SQL against a SQLite database file and return all rows.

    Returns ``None`` if execution raises (syntax error, missing table, etc.).

    When ``timeout_s`` is given, a SQLite progress handler aborts the query from *inside* SQLite once it has
    been running that long, and ``TimeoutError`` is raised (distinct from the ``None`` returned for query errors).
    The clock starts when this function starts executing in its thread, so time spent queued for a worker does
    not count. An outer ``asyncio.wait_for`` alone only stops *waiting*: the blocking call keeps running and
    keeps occupying a thread-pool worker, so later queries queue behind it and spuriously time out.
    """
    # `with sqlite3.connect(...)` only commits/rolls back; it does not close the connection, so close explicitly.
    conn = None
    timed_out = False
    try:
        conn = sqlite3.connect(str(db_path))
        conn.text_factory = lambda b: b.decode(errors="ignore")
        if timeout_s is not None:
            start = time.monotonic()

            def abort_if_timed_out() -> int:
                nonlocal timed_out
                timed_out = time.monotonic() - start > timeout_s
                return int(timed_out)

            conn.set_progress_handler(abort_if_timed_out, _PROGRESS_HANDLER_INTERVAL)
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        if timed_out:
            raise TimeoutError(f"SQLite query exceeded {timeout_s}s") from None
        return None
    finally:
        if conn is not None:
            conn.close()


async def execute_sqlite_async(
    db_path: Path,
    sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
    raise_on_timeout: bool = False,
) -> Optional[ResultSet]:
    """Execute SQL asynchronously in a worker thread, bounded by semaphore.

    Returns ``None`` on timeout (unless ``raise_on_timeout`` is True) or query exception.

    ``timeout_s`` is enforced inside the query by ``execute_sqlite`` so a timed-out query frees its thread
    before the semaphore is released. The outer ``wait_for`` is only a backstop with an extra
    ``_OUTER_TIMEOUT_BUFFER_S``: the progress handler checks elapsed time periodically and may fire slightly
    after the deadline, and the inner timeout must win so that no query is left running in the background.
    """
    async with semaphore:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(execute_sqlite, db_path, sql, timeout_s),
                timeout=timeout_s + _OUTER_TIMEOUT_BUFFER_S,
            )
        except asyncio.TimeoutError:
            if raise_on_timeout:
                raise
            return None


def result_sets_match(gold: ResultSet, pred: ResultSet) -> bool:
    """BIRD's result-set comparison: unordered set equality over tuple rows."""
    try:
        return set(gold) == set(pred)
    except TypeError:
        # Rows may contain unhashable types (bytes, lists) — fall back to sorted lists.
        try:
            return sorted(map(repr, gold)) == sorted(map(repr, pred))
        except Exception:
            return False


def normalize_rows(rows: list[tuple[Any, ...]]) -> ResultSet:
    """Make engine-specific Python types comparable: floats/decimals rounded, dates stringified, bytes decoded."""

    def norm(value: Any) -> Any:
        if isinstance(value, (float, decimal.Decimal)):
            return round(float(value), 6)
        if isinstance(value, (datetime.date, datetime.datetime)):
            return str(value)
        if isinstance(value, bytes):
            return value.decode(errors="replace")
        return value

    return [tuple(norm(v) for v in row) for row in rows]


class SparkUnavailableError(RuntimeError):
    """The Spark session cannot run any query (for example its JVM exited), as opposed to one query failing.

    The model's answer was never measured, so this is a verifier failure and must not be scored as a wrong answer.
    """


def _spark_is_alive(session: Any) -> bool:
    try:
        session.sql("SELECT 1").collect()
        return True
    except Exception:
        return False


def execute_spark(session: Any, sql: str, job_group: str) -> Optional[ResultSet]:
    """Execute a single read-only query on a Spark session and return all rows.

    Returns ``None`` if the query itself fails or ``sql`` is not exactly one query (``SELECT``/``WITH``/set
    operation): the Spark warehouse is shared by every rollout, so DDL/DML from a model must never run. Raises
    ``SparkUnavailableError`` if the query failed and a trivial ``SELECT 1`` fails too, i.e. Spark itself is down.
    """
    import sqlglot
    from sqlglot import exp

    try:
        statements = sqlglot.parse(sql, read="spark")
        if len(statements) != 1 or not isinstance(statements[0], exp.Query):
            return None
        session.sparkContext.setJobGroup(job_group, sql[:200], interruptOnCancel=True)
        return normalize_rows([tuple(row) for row in session.sql(sql.strip().rstrip(";")).collect()])
    except Exception as e:
        if not _spark_is_alive(session):
            raise SparkUnavailableError("The Spark session cannot run queries (did its JVM exit?)") from e
        return None


async def execute_spark_async(
    session: Any,
    sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
    raise_on_timeout: bool = False,
) -> Optional[ResultSet]:
    """Execute SQL on Spark in a worker thread, bounded by semaphore.

    Returns ``None`` on timeout (unless ``raise_on_timeout`` is True) or query exception. On timeout the Spark job
    group is cancelled; ``asyncio.wait_for`` alone would leave the job running.
    """
    async with semaphore:
        job_group = uuid.uuid4().hex
        try:
            return await asyncio.wait_for(asyncio.to_thread(execute_spark, session, sql, job_group), timeout=timeout_s)
        except asyncio.TimeoutError:
            session.sparkContext.cancelJobGroup(job_group)
            if raise_on_timeout:
                raise
            return None


# SQLSTATE classes that mean the server or connection failed, not the query: connection exception, insufficient
# resources, operator intervention (except 57014, a statement timeout), system error, internal error.
_POSTGRES_INFRA_SQLSTATE_CLASSES = ("08", "53", "57", "58", "XX")


class PostgresPool:
    """The grader's PostgreSQL connections: one pool of ``size`` connections per resources-server process.

    Connections are opened once and reused, so a grade pays no connection setup (a server process forked per
    connection, plus SCRAM authentication), and one process never holds more than ``size`` server connections. Size it
    to the server's ``max_concurrency``: a grade holding the semaphore then always finds a free connection, and grades
    queue in the semaphore as they did before pooling. Every transaction starts ``BEGIN READ ONLY``, which also holds
    behind a transaction-mode pooler such as PgBouncer.

    The pool opens on first use because psycopg's pool needs a running event loop. Waiting longer than ``timeout_s``
    to open it or to get a connection raises: the server is unreachable or full, which is not the model's mistake.
    """

    def __init__(self, dsn: str, size: int, timeout_s: float = 30.0) -> None:
        import psycopg_pool  # noqa: F401 - imported here so a missing install fails startup, not every grade

        self.dsn = dsn
        self.size = size
        self.timeout_s = timeout_s
        self._pool: Any = None
        self._lock = asyncio.Lock()

    async def _open(self) -> Any:
        import psycopg
        from psycopg_pool import AsyncConnectionPool

        async def read_only(conn: psycopg.AsyncConnection) -> None:
            await conn.set_read_only(True)

        async with self._lock:
            if self._pool is None:
                # Connect once first, so a wrong DSN or an unreachable server raises its own error now rather than a
                # PoolTimeout after the pool has retried for ``timeout_s``.
                await (await psycopg.AsyncConnection.connect(self.dsn)).close()
                pool = AsyncConnectionPool(
                    self.dsn,
                    min_size=self.size,
                    max_size=self.size,
                    open=False,
                    configure=read_only,
                    check=AsyncConnectionPool.check_connection,  # replaces connections a server restart broke
                    timeout=self.timeout_s,
                    name="bird_sql",
                )
                try:
                    await pool.open(wait=True, timeout=self.timeout_s)
                except BaseException:
                    await pool.close()
                    raise
                self._pool = pool
        return self._pool

    @contextlib.asynccontextmanager
    async def connection(self) -> AsyncIterator[Any]:
        pool = self._pool or await self._open()
        async with pool.connection() as conn:
            yield conn

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None


async def _run_postgres(conn: Any, db_id: str, sql: str, timeout_s: float) -> Optional[ResultSet]:
    """Run one query as a single prepared statement in a read-only transaction with ``search_path`` set to ``db_id``.

    A prepared statement cannot hold more than one command, so ``COMMIT; ...`` cannot leave the read-only
    transaction. The server's ``statement_timeout`` raises ``TimeoutError``; any other error caused by the query
    returns ``None``, including a result psycopg cannot convert (``'infinity'::date`` has no Python date). Connection
    and server failures raise, so they are not scored as the model's mistake.

    The transaction always rolls back: the connection goes back to the pool and serves other rollouts, so nothing the
    query sets (``SET TimeZone``, ``set_config(..., false)``) may outlive it.

    Parallel query is off: a parallel float ``SUM`` adds in a different order on every run (BIRD dev id 1476 gave 10
    different values in 10 runs), which would make the same answer pass or fail at random. It also keeps one rollout
    from taking several cores while hundreds run at once.
    """
    import psycopg

    try:
        async with conn.transaction(force_rollback=True):
            await conn.execute(
                "SELECT set_config('statement_timeout', %s, true), set_config('search_path', %s, true),"
                " set_config('max_parallel_workers_per_gather', '0', true)",
                (f"{int(timeout_s * 1000)}ms", psycopg.sql.Identifier(db_id).as_string(conn)),
            )
            cur = await conn.execute(sql, prepare=True)
            if cur.description is None:  # e.g. SET: no result set to compare
                return None
            return normalize_rows(await cur.fetchall())
    except psycopg.errors.QueryCanceled:
        raise TimeoutError(f"PostgreSQL query exceeded {timeout_s}s") from None
    except psycopg.Error as e:
        # psycopg's own errors carry no SQLSTATE: a lost or closed connection raises, and anything else (a DataError
        # for a value it cannot convert) is the query's. Server errors raise only for the infrastructure classes,
        # since OperationalError also covers query faults such as 54001 statement_too_complex.
        if e.sqlstate is None:
            if isinstance(e, (psycopg.OperationalError, psycopg.InterfaceError)):
                raise
        elif e.sqlstate[:2] in _POSTGRES_INFRA_SQLSTATE_CLASSES:
            raise
        return None
    except asyncio.CancelledError:
        # The backstop fired: stop the server-side query and close the connection, which may still be mid-query, so
        # the pool replaces it rather than handing it to the next grade.
        if not conn.closed:
            await conn.cancel_safe()
            await conn.close()
        raise


async def execute_postgres_async(
    pool: PostgresPool,
    db_id: str,
    sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
    raise_on_timeout: bool = False,
) -> Optional[ResultSet]:
    """Execute SQL on PostgreSQL schema ``db_id`` on a pooled connection, bounded by semaphore.

    Returns ``None`` on timeout (unless ``raise_on_timeout`` is True) or query error. ``timeout_s`` is enforced by the
    server (``statement_timeout``); the outer ``wait_for`` is a backstop that also cancels the server-side query. The
    connection is taken before the backstop starts, so time spent waiting for one never counts against the query.
    """
    async with semaphore, pool.connection() as conn:
        try:
            return await asyncio.wait_for(
                _run_postgres(conn, db_id, sql, timeout_s), timeout=timeout_s + _OUTER_TIMEOUT_BUFFER_S
            )
        except (asyncio.TimeoutError, TimeoutError):
            if raise_on_timeout:
                raise asyncio.TimeoutError from None
            return None


Runner = Callable[[str], Awaitable[Optional[ResultSet]]]


async def _compare(
    run_gold: Runner, run_pred: Runner, gold_sql: str, pred_sql: str
) -> tuple[bool, Optional[ResultSet], Optional[ResultSet], Optional[str]]:
    """Run gold then pred (runners raise ``asyncio.TimeoutError`` on timeout) and compare."""
    try:
        gold_rows = await run_gold(gold_sql)
    except asyncio.TimeoutError:
        return False, None, None, "gold_sql_timeout"
    if gold_rows is None:
        return False, None, None, "gold_sql_error"

    try:
        pred_rows = await run_pred(pred_sql)
    except asyncio.TimeoutError:
        return False, gold_rows, None, "pred_sql_timeout"
    if pred_rows is None:
        return False, gold_rows, None, "pred_sql_error"

    return result_sets_match(gold_rows, pred_rows), gold_rows, pred_rows, None


async def execute_and_compare(
    db_path: Path,
    gold_sql: str,
    pred_sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
) -> tuple[bool, Optional[ResultSet], Optional[ResultSet], Optional[str]]:
    """Execute both queries on SQLite and compare. Returns (match, gold, pred, error_tag)."""

    async def run(sql: str) -> Optional[ResultSet]:
        return await execute_sqlite_async(db_path, sql, semaphore, timeout_s, raise_on_timeout=True)

    return await _compare(run, run, gold_sql, pred_sql)


async def execute_and_compare_spark(
    db_path: Path,
    session: Any,
    gold_sql: str,
    pred_sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
) -> tuple[bool, Optional[ResultSet], Optional[ResultSet], Optional[str]]:
    """Run the BIRD gold SQL on SQLite and the model's Spark SQL on Spark, then compare. Same return as above.

    No gold transpilation: the gold stays the original SQLite query and only its result set is compared.
    """

    async def run_gold(sql: str) -> Optional[ResultSet]:
        rows = await execute_sqlite_async(db_path, sql, semaphore, timeout_s, raise_on_timeout=True)
        return None if rows is None else normalize_rows(rows)

    async def run_pred(sql: str) -> Optional[ResultSet]:
        return await execute_spark_async(session, sql, semaphore, timeout_s, raise_on_timeout=True)

    return await _compare(run_gold, run_pred, gold_sql, pred_sql)


async def execute_and_compare_postgres(
    db_path: Path,
    pool: PostgresPool,
    db_id: str,
    gold_sql: str,
    pred_sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
) -> tuple[bool, Optional[ResultSet], Optional[ResultSet], Optional[str]]:
    """Run the BIRD gold SQL on SQLite and the model's PostgreSQL on PostgreSQL, then compare. Same return as above.

    No gold transpilation: the gold stays the original SQLite query and only its result set is compared.
    """

    async def run_gold(sql: str) -> Optional[ResultSet]:
        rows = await execute_sqlite_async(db_path, sql, semaphore, timeout_s, raise_on_timeout=True)
        return None if rows is None else normalize_rows(rows)

    async def run_pred(sql: str) -> Optional[ResultSet]:
        return await execute_postgres_async(pool, db_id, sql, semaphore, timeout_s, raise_on_timeout=True)

    return await _compare(run_gold, run_pred, gold_sql, pred_sql)
