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
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable, Dict, FrozenSet, List, Optional


ResultRow = tuple[Any, ...]
ResultSet = list[ResultRow]


# Extra time the outer ``wait_for`` allows beyond ``timeout_s``; see ``execute_sqlite_async``.
_OUTER_TIMEOUT_BUFFER_S = 10.0
# How often a timed-out Snowflake query is interrupted again while its worker thread winds down.
_SNOWFLAKE_INTERRUPT_RETRY_S = 1.0
# A Snowflake query is timed from when it starts: queueing for a worker thread, the static check and connecting don't
# count. Getting that far taking longer than this is the engine's failure, not the query's.
_SNOWFLAKE_SETUP_TIMEOUT_S = 120.0
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


# Functions SQLite has and Snowflake does not (Snowflake spells them LISTAGG, IFF, POSITION, ...) that DuckDB, under the
# local Snowflake stand-in, still runs or rewrites. A Snowflake account refuses each of them (checked 2026-11-19).
_SQLITE_ONLY_FUNCTIONS = frozenset({"GROUP_CONCAT", "IIF", "INSTR", "JULIANDAY", "STRFTIME", "TOTAL"})
# Functions DuckDB has and Snowflake does not, seen in model SQL or probes. A Snowflake account refuses each of them
# as an unknown function (checked 2026-11-19; DIVIDE 2026-11-20); DuckDB, under the stand-in, runs them. Names are
# matched as written, like ``_SQLITE_ONLY_FUNCTIONS``.
_DUCKDB_ONLY_FUNCTIONS = frozenset({"DIVIDE", "EPOCH_MS", "LIST_VALUE", "REGEXP_MATCHES", "STRING_AGG", "STRPTIME"})
# Output columns of ``LATERAL FLATTEN(...)``.
_FLATTEN_COLUMNS = frozenset({"SEQ", "KEY", "PATH", "INDEX", "VALUE", "THIS"})


def _within(node: Any, ancestor: Any) -> bool:
    """Whether sqlglot node ``node`` is ``ancestor`` or inside it."""
    while node is not None:
        if node is ancestor:
            return True
        node = node.parent
    return False


def _is_boolean(node: Any) -> bool:
    """Whether a sqlglot expression is BOOLEAN: a comparison or logical operator, ``TRUE``/``FALSE``, or an ``IFF``,
    ``CASE`` or ``COALESCE`` whose every result is."""
    from sqlglot import exp

    node = node.unnest()
    if isinstance(node, (exp.Predicate, exp.Connector, exp.Not, exp.Boolean)):
        return True
    if isinstance(node, exp.If):
        results = [node.args.get("true"), node.args.get("false")]
    elif isinstance(node, exp.Case):
        results = [branch.args.get("true") for branch in node.args.get("ifs") or []] + [node.args.get("default")]
    elif isinstance(node, exp.Coalesce):
        results = [node.this, *node.expressions]
    else:
        return False
    return all(result is not None and _is_boolean(result) for result in results)


def snowflake_rejection(
    sql: str, db_id: str, tables: Dict[str, FrozenSet[str]], database: str = "BIRD"
) -> Optional[str]:
    """Why Snowflake would refuse ``sql``, though the local stand-in (DuckDB under fakesnow) might run it; else ``None``.

    ``tables`` is the ``db_id`` schema's catalog, ``{table: columns}``, with exact names. Checks, statically:

    - exactly one query (``SELECT``/``WITH``/set operation): the data is shared by every rollout;
    - identifiers resolve as on Snowflake, where an unquoted name means its upper-case spelling and a quoted one is
      exact (DuckDB matches names in any case);
    - no SQLite-only function (``_SQLITE_ONLY_FUNCTIONS``), ``LIMIT offset, count``, or two-argument ``MAX``/``MIN``;
    - no DuckDB-only function (``_DUCKDB_ONLY_FUNCTIONS``), and no aggregate ``FILTER (WHERE ...)`` clause;
    - no ``SUM``/``AVG`` of a boolean expression, and no ``SELECT DISTINCT`` ordered by a column it doesn't select.

    The identifier check knows the catalog, the query's own aliases and CTEs, and ``FLATTEN``'s columns; a column is
    accepted if any table of the schema has it. A column qualified by a catalog table (or its alias) must be a catalog
    column, and ``JOIN ... USING`` names are checked too. It is a static approximation, not Snowflake's resolver.
    """
    import sqlglot
    from sqlglot import exp
    from sqlglot.optimizer.normalize_identifiers import normalize_identifiers
    from sqlglot.optimizer.scope import traverse_scope
    from sqlglot.tokens import TokenType

    try:
        tokens = sqlglot.tokenize(sql, read="snowflake")
        parsed = sqlglot.parse(sql, read="snowflake")
    except sqlglot.errors.SqlglotError:
        return "unparsable"
    # A comment after the last semicolon (`SELECT 1; -- why`) parses as a Semicolon node: no statement.
    statements = [s for s in parsed if s is not None and not isinstance(s, exp.Semicolon)]
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        return "not_a_single_query"
    cte_names = {cte.alias.upper() for cte in statements[0].find_all(exp.CTE)}  # `WITH total(x) AS ...` is no call
    for i, tok in enumerate(tokens[:-1]):
        nxt = tokens[i + 1]
        if tok.token_type != TokenType.IDENTIFIER and nxt.token_type == TokenType.L_PAREN:
            if tok.text.upper() in _SQLITE_ONLY_FUNCTIONS and tok.text.upper() not in cte_names:
                return f"sqlite_function:{tok.text.upper()}"
            if tok.text.upper() in _DUCKDB_ONLY_FUNCTIONS and tok.text.upper() not in cte_names:
                return f"duckdb_function:{tok.text.upper()}"
        if tok.token_type == TokenType.LIMIT and i + 2 < len(tokens) and tokens[i + 2].token_type == TokenType.COMMA:
            return "limit_offset_comma"
    tree = normalize_identifiers(statements[0], dialect="snowflake")
    if any(f.expressions for f in tree.find_all(exp.Max, exp.Min)):
        return "scalar_max_min"
    if any(tree.find_all(exp.Filter)):
        # COUNT(*) FILTER (WHERE ...): DuckDB filters the aggregate, Snowflake has no such clause. Snowflake's own
        # FILTER(array, lambda) function is another node (ArrayFilter) and stays allowed.
        return "aggregate_filter"
    for agg in tree.find_all(exp.Sum, exp.Avg):
        # SUM("x" = 'y'): DuckDB counts the true rows, Snowflake refuses a BOOLEAN argument.
        arg = agg.this
        if isinstance(arg, exp.Distinct):
            if not arg.expressions:
                return "unparsable"  # SUM(DISTINCT)
            arg = arg.expressions[0]
        if _is_boolean(arg):
            return f"boolean_aggregate:{agg.key.upper()}"
    for select in tree.find_all(exp.Select):
        # SELECT DISTINCT ... ORDER BY a column it doesn't select: DuckDB sorts by it, Snowflake refuses it. An item
        # that repeats a selected expression, names an output, or uses only selected columns is fine (a column
        # qualified by another table is not the selected one); aggregates in the ORDER BY are left alone.
        distinct, order = select.args.get("distinct"), select.args.get("order")
        if not distinct or distinct.args.get("on") or not order or select.is_star:
            continue
        repeated = {e.unalias().sql() for e in select.expressions}
        bare = [e.unalias() for e in select.expressions if isinstance(e.unalias(), exp.Column)]
        outputs = set(select.named_selects)
        for ordered in order.expressions:
            if ordered.this.sql() in repeated:
                continue
            for column in ordered.find_all(exp.Column):
                if column.find_ancestor(exp.Select) is not select or column.find_ancestor(exp.AggFunc):
                    continue
                if not column.table and column.name in outputs:
                    continue
                if not any(
                    c.name == column.name and (not c.table or not column.table or c.table == column.table)
                    for c in bare
                ):
                    return f"distinct_order_by:{column.name}"

    ctes = {cte.alias for cte in tree.find_all(exp.CTE)}
    table_aliases = list(tree.find_all(exp.TableAlias))
    qualifiers = ctes | {a.name for a in table_aliases}
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            # A bare function in FROM, such as DuckDB's range() or query() (which runs any SQL): Snowflake has none.
            # Its table functions are wrapped in TABLE(...) or LATERAL, which parse to other nodes.
            return f"table_function:{(table.this.name or table.this.key).upper()}"
        if table.catalog and table.catalog != database:
            return f"unknown_database:{table.catalog}"
        if table.db and table.db != db_id:
            return f"unknown_schema:{table.db}"
        if not table.db and table.name in ctes:
            continue
        if table.name not in tables:
            return f"unknown_table:{table.name}"
        qualifiers.add(table.name)
    catalog_columns = set().union(*tables.values())
    aliases = list(tree.find_all(exp.Alias))
    # VALUES (...) outputs COLUMN1, COLUMN2, ... on Snowflake.
    values_width = max((len(row.expressions) for v in tree.find_all(exp.Values) for row in v.expressions), default=0)
    values_columns = {f"COLUMN{i + 1}" for i in range(values_width)}
    not_aliases = (
        catalog_columns | {c.name for a in table_aliases for c in a.columns} | _FLATTEN_COLUMNS | values_columns
    )
    columns = not_aliases | {a.alias for a in aliases}
    for join in tree.find_all(exp.Join):
        # A USING name is a column of both join inputs: a catalog column, or a column a CTE or derived table outputs.
        for name in join.args.get("using") or []:
            ident = name.this if isinstance(name, exp.Column) else name
            if (ident.name if ident.quoted else ident.name.upper()) not in columns:
                return f"unknown_column:{ident.name}"
    for column in tree.find_all(exp.Column):
        if isinstance(column.this, exp.Star):
            continue
        if column.name not in columns:
            return f"unknown_column:{column.name}"
        if column.name not in not_aliases and all(_within(column, a) for a in aliases if a.alias == column.name):
            # Only an alias has this name, and the column is inside that alias's own definition
            # (MAX(height_cm) AS height_cm): Snowflake looks for a real column there and finds none.
            return f"unknown_column:{column.name}"
        if column.table and column.table not in qualifiers:
            return f"unknown_qualifier:{column.table}"
        if column.db and column.db != db_id:
            return f"unknown_schema:{column.db}"
    # In each scope, a column qualified by a catalog table (by name or alias) must be a column of that table: not a
    # select alias, not another table's column. An unqualified column must belong to one of the catalog tables in
    # scope (or in an enclosing scope, for a correlated subquery); this is checked only when every source in reach
    # is a catalog table, and aliases are left to the checks above. Scopes matter: one alias can name a table in one
    # subquery and a derived table in another.
    try:
        scopes = traverse_scope(tree)
    except sqlglot.errors.SqlglotError:
        scopes = []
    for scope in scopes:
        reach, sources, node = set(), [], scope
        while node is not None:
            sources += list(node.sources.values())
            node = node.parent
        all_catalog = all(isinstance(src, exp.Table) and src.name in tables for src in sources)
        if all_catalog:
            reach = set().union(*(tables[src.name] for src in sources))
        for column in scope.columns:
            if isinstance(column.this, exp.Star):
                continue
            source = scope.sources.get(column.table) if column.table else None
            if isinstance(source, exp.Table) and source.name in tables and column.name not in tables[source.name]:
                return f"unknown_column:{column.table}.{column.name}"
            # A scope's columns can include a subquery's unqualified ones; each is judged in its own SELECT's scope.
            own = column.find_ancestor(exp.Select) is scope.expression
            if (
                not column.table
                and own
                and sources
                and all_catalog
                and column.name not in reach | (columns - catalog_columns)
            ):
                return f"unknown_column:{column.name}"
    return None


def _is_snowflake_infra_error(e: Exception) -> bool:
    """Whether ``e`` means the engine or its connection failed, not the query.

    The connector's errors carry a SQLSTATE (a closed connection is ``08003``), judged by the same classes as
    PostgreSQL's. Without one (the connector then says ``"n/a"``), its ``OperationalError`` and ``InterfaceError``
    are transport failures, such as a request that never reached the server (``250003``), as psycopg's are for
    PostgreSQL. DuckDB's own I/O, fatal, internal and connection errors can also surface through the stand-in.
    Anything else the query raised, including a result Python cannot hold (``'infinity'::date``), is the query's.
    """
    import duckdb
    from snowflake.connector import errors as snowflake_errors

    sqlstate = getattr(e, "sqlstate", None)
    if sqlstate and sqlstate != "n/a":
        return sqlstate[:2] in _POSTGRES_INFRA_SQLSTATE_CLASSES
    if isinstance(e, (snowflake_errors.OperationalError, snowflake_errors.InterfaceError)):
        return True
    return isinstance(
        e, (duckdb.IOException, duckdb.FatalException, duckdb.InternalException, duckdb.ConnectionException)
    )


def execute_snowflake(
    snowflake: Any,
    db_id: str,
    sql: str,
    connections: List[Any],
    cancelled: Optional[threading.Event] = None,
    on_start: Optional[Callable[[], None]] = None,
) -> Optional[ResultSet]:
    """Run one query on the Snowflake engine (see ``setup_bird_snowflake``) and return all rows.

    Returns ``None`` if ``snowflake_rejection`` finds a reason Snowflake would refuse the query, or if the query fails.
    Raises ``TimeoutError`` when the server cancels it at its statement timeout (SQLSTATE ``57014``), and re-raises
    engine and connection failures, so they are not scored as the model's mistake. The connection is appended to
    ``connections`` so a caller that times out can interrupt it. If the caller has given up by the time the connection
    is ready (``cancelled`` is set), the query doesn't run and ``TimeoutError`` is raised. ``on_start`` is called just
    before the query is sent, when its clock starts.
    """
    if snowflake_rejection(sql, db_id, snowflake.tables.get(db_id, {})) is not None:
        return None
    conn = snowflake.connect(db_id)
    connections.append(conn)
    if cancelled is not None and cancelled.is_set():
        conn.close()
        raise TimeoutError("Snowflake query timed out while connecting")
    try:
        cur = conn.cursor()
        if on_start is not None:
            on_start()
        cur.execute(sql.strip().rstrip(";"))
        return normalize_rows(snowflake.fetch_all(cur))
    except Exception as e:
        if getattr(e, "sqlstate", None) == "57014":
            raise TimeoutError("Snowflake query exceeded its statement timeout") from None
        if _is_snowflake_infra_error(e):
            raise
        return None
    finally:
        conn.close()


async def execute_snowflake_async(
    snowflake: Any,
    db_id: str,
    sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
    raise_on_timeout: bool = False,
) -> Optional[ResultSet]:
    """Execute SQL on the Snowflake engine in a worker thread, bounded by semaphore.

    Returns ``None`` on timeout (unless ``raise_on_timeout`` is True) or query error. ``timeout_s`` counts from when
    the query starts: waiting for a worker thread, the static check and connecting don't count, so a busy server
    doesn't time out (and score 0) a fast query; taking over ``_SNOWFLAKE_SETUP_TIMEOUT_S`` to get there raises, as
    the engine's failure. An engine whose server enforces ``timeout_s`` itself (``timeout_buffer_s``) gets that much
    longer, so the server's verdict arrives first. On timeout the query is interrupted (``asyncio.wait_for`` alone
    would leave it running), and the semaphore slot is held until the worker thread has stopped, so no more queries
    run at once than the semaphore allows.
    """
    async with semaphore:
        loop = asyncio.get_running_loop()
        connections: List[Any] = []
        cancelled = threading.Event()
        started: asyncio.Future = loop.create_future()

        def on_start() -> None:
            loop.call_soon_threadsafe(lambda: started.done() or started.set_result(time.monotonic()))

        worker = asyncio.ensure_future(
            asyncio.to_thread(execute_snowflake, snowflake, db_id, sql, connections, cancelled, on_start)
        )
        await asyncio.wait({worker, started}, timeout=_SNOWFLAKE_SETUP_TIMEOUT_S, return_when=asyncio.FIRST_COMPLETED)
        if not worker.done() and not started.done():
            await _stop_snowflake_worker(snowflake, worker, connections, cancelled)
            raise RuntimeError(f"Snowflake engine took over {_SNOWFLAKE_SETUP_TIMEOUT_S:.0f}s to start a query")
        deadline = timeout_s + getattr(snowflake, "timeout_buffer_s", 0.0)
        elapsed = time.monotonic() - started.result() if started.done() else 0.0
        try:
            return await asyncio.wait_for(asyncio.shield(worker), timeout=max(0.0, deadline - elapsed))
        except (asyncio.TimeoutError, TimeoutError):
            await _stop_snowflake_worker(snowflake, worker, connections, cancelled)
            if raise_on_timeout:
                raise asyncio.TimeoutError from None
            return None


async def _stop_snowflake_worker(
    snowflake: Any, worker: asyncio.Future, connections: List[Any], cancelled: threading.Event
) -> None:
    """Tell a Snowflake worker thread to stop and wait until it has: a worker still connecting won't run its query, and
    a running query is interrupted again every round, since one can start (or a first interrupt be lost) just after."""
    cancelled.set()
    while not worker.done():
        for conn in list(connections):
            snowflake.interrupt(conn)
        await asyncio.wait({worker}, timeout=_SNOWFLAKE_INTERRUPT_RETRY_S)
    if not worker.cancelled():
        worker.exception()  # retrieved here; the caller sees the timeout or failure instead


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


async def execute_and_compare_snowflake(
    db_path: Path,
    snowflake: Any,
    db_id: str,
    gold_sql: str,
    pred_sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
) -> tuple[bool, Optional[ResultSet], Optional[ResultSet], Optional[str]]:
    """Run the BIRD gold SQL on SQLite and the model's Snowflake SQL on the Snowflake engine, then compare. Same return
    as above.

    No gold transpilation: the gold stays the original SQLite query and only its result set is compared.
    """

    async def run_gold(sql: str) -> Optional[ResultSet]:
        rows = await execute_sqlite_async(db_path, sql, semaphore, timeout_s, raise_on_timeout=True)
        return None if rows is None else normalize_rows(rows)

    async def run_pred(sql: str) -> Optional[ResultSet]:
        return await execute_snowflake_async(snowflake, db_id, sql, semaphore, timeout_s, raise_on_timeout=True)

    return await _compare(run_gold, run_pred, gold_sql, pred_sql)
