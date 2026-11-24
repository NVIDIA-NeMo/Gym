# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BEAVER's execution accuracy (``ex_acc``): run a query on MySQL, compare two results.

This follows the rule of BEAVER's official evaluator (``eval/utils/ex_acc.py`` in https://github.com/beaverbench/beaver
at 81a2a88a1) and is checked against that file on the same predictions (``benchmarks/beaver/official_check.py``). It
is written for Gym, not copied: the official repository has no license file.

Running a query:

- Each query gets its own connection and 10 seconds, counted on its own thread from the start of the connection to
  the last row read into a ``pandas.DataFrame``. A query that runs longer is a timeout, whatever it then returns (rows,
  an error or a lost connection). The query is then stopped on the server (``KILL QUERY``); the official evaluator
  abandons it, which leaves it loading the server for later queries.
- A query that fails or times out, like a missing prediction, is an empty result. That holds for the gold query too.
- A failure to reach the data (no server, wrong password, unknown database) is not a result: it raises
  ``MySqlUnavailableError`` so the verifier can mask the row instead of scoring it.

Comparing two results (``results_match``):

- Both empty: a match. One empty: no match.
- Otherwise the rows are built into a ``pandas.DataFrame`` with the result's column names, every cell is converted
  with ``astype(str)`` and stripped, and the two sets of rows must be equal with the same number of columns. Row
  order, duplicate rows and column names don't matter.

The DataFrame step is kept on purpose. pandas infers a column's type from all its values, and that changes the text:
an integer column with a NULL becomes ``5.0`` and ``nan``, and a DATETIME column whose values are all at midnight
prints as dates. Converting each value on its own gives different verdicts in those cases. The pandas major version
matters too (pandas 3 keeps NULL as missing where pandas 2 prints ``None``); this server pins pandas 2, which is what
the official evaluator runs on (its setup is Python 3.10, and pandas 3 needs 3.11).
"""

import asyncio
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import pandas as pd

from resources_servers.beaver_sql.setup_beaver_mysql import parse_mysql_dsn


logger = logging.getLogger(__name__)

# The official evaluator's limits, in seconds.
QUERY_TIMEOUT_S = 10.0
CONNECTION_TIMEOUT_S = 10

# Any failure to connect means the grader couldn't reach the data, and so does losing the connection during a query
# (2006, 2013, 2055): the row is masked. Every other error is the query's, scored as an empty result. That includes
# access denied (1044, 1142): the grader's user may only read, so a query that writes, creates a temporary table or
# uses another database fails, as it would for a read-only user anywhere. It also includes the server running short of
# a resource (sort memory 1038, temporary table space 1114, ...): a runaway query causes those itself. (A gold query
# that errors for any reason is masked by the verifier instead; see app.py.)
_CONNECTION_LOST_ERRNOS = frozenset({2006, 2013, 2055})
# The query was stopped: by our KILL QUERY (1317) or by a server-wide max_execution_time (3024).
_INTERRUPTED_ERRNOS = frozenset({1317, 3024})
_FETCH_CHUNK_ROWS = 10_000
# A result over this size isn't read to the end or compared: it never matches. The largest BEAVER gold result (dw,
# dw_real, neutron) has 622,224 rows. Text and binary cells count by length, other cells as 8 bytes.
MAX_RESULT_ROWS = 5_000_000
MAX_RESULT_BYTES = 1 << 30
# A KILL connection may take this long to connect, send and read before it's given up on.
_KILL_IO_TIMEOUT_S = 2


class MySqlUnavailableError(RuntimeError):
    """The grader couldn't reach the data; the row must be masked, not scored."""


@dataclass
class QueryOutcome:
    """A query's result: ``frame`` when ``status`` is ``ok``, else an empty result, unless ``oversized``."""

    status: Literal["ok", "error", "timeout", "missing"]
    frame: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    seconds: float = 0.0
    # Returned more rows than MAX_RESULT_*: a non-empty result that matches nothing, as it would officially fail to
    # match any gold result (all far smaller).
    oversized: bool = False

    @property
    def empty(self) -> bool:
        return not self.oversized and (self.frame is None or self.frame.empty)


def missing_prediction() -> QueryOutcome:
    """No query to run: an empty result, as a missing prediction file is officially."""
    return QueryOutcome(status="missing")


def _approx_bytes(rows: List[Tuple[Any, ...]]) -> int:
    return sum(len(c) if isinstance(c, (str, bytes, bytearray)) else 8 for row in rows for c in row)


def _rows_as_text(frame: pd.DataFrame) -> List[Tuple[Any, ...]]:
    """Every cell as stripped text, typed column by column as pandas inferred it."""
    text = frame.astype(str)
    return [
        tuple(c.strip() if isinstance(c, str) else c for c in row) for row in text.itertuples(index=False, name=None)
    ]


def results_match(gold: QueryOutcome, pred: QueryOutcome) -> bool:
    """BEAVER's comparison: both empty, or equal sets of stripped-text rows with the same number of columns."""
    if gold.oversized or pred.oversized:
        return False
    if gold.empty or pred.empty:
        return gold.empty and pred.empty
    gold_rows, pred_rows = _rows_as_text(gold.frame), _rows_as_text(pred.frame)
    if len(gold_rows[0]) != len(pred_rows[0]):
        return False
    return set(gold_rows) == set(pred_rows)


@dataclass
class _Attempt:
    """State shared between the event loop and the thread running one query. ``started`` and ``ended`` are read on the
    query's own thread, so a busy event loop can't make a fast query look slow. ``finished`` is set when the outcome is
    known, ``done`` once the query's connection is cleaned up too: the outcome is decided on the first, the slot is
    released on the second."""

    notify: Callable[[], None] = lambda: None
    connection_id: Optional[int] = None
    outcome: Optional[QueryOutcome] = None
    unavailable: Optional[BaseException] = None
    started: Optional[float] = None
    ended: Optional[float] = None
    finished: threading.Event = field(default_factory=threading.Event)
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)


class MySqlRunner:
    """Runs queries on the BEAVER MySQL server, at most ``max_concurrency`` at a time.

    Each query runs on its own thread, which times it and wakes the event loop when it ends; the loop's own timer only
    decides when to stop waiting. A query keeps its slot until its thread has closed its connection. A timed-out query
    keeps it while ``KILL QUERY`` is repeated, until its thread ends or ``stop_grace_s`` passes, so stopped queries don't
    pile up on the server behind new ones, and one the server can't stop doesn't hold the slot forever. A request
    cancelled while its query runs stops the query the same way, and a cancellation that arrives while a query is being
    stopped waits for the stop, so the slot is never released early. A string with several statements runs them all, as
    the driver does officially, but only the first result counts, and the connection is killed before the rest can run
    on unwatched.
    """

    def __init__(
        self,
        dsn: str,
        *,
        password: Optional[str] = None,
        max_concurrency: int = 8,
        timeout_s: float = QUERY_TIMEOUT_S,
        connection_timeout_s: int = CONNECTION_TIMEOUT_S,
    ) -> None:
        self._params = parse_mysql_dsn(dsn, password)
        self._params.pop("database", None)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        # Its own threads for stopping queries, so they never queue behind other work on the default executor.
        self._executor = ThreadPoolExecutor(max_workers=max(4, 2 * max_concurrency), thread_name_prefix="beaver-stop")
        self.timeout_s = timeout_s
        self.connection_timeout_s = connection_timeout_s
        # How long a timed-out query may take to stop before its slot is released anyway.
        self.stop_grace_s = 10.0
        self.kill_failures = 0
        self._clock = time.monotonic

    async def run(self, db_id: str, sql: str) -> QueryOutcome:
        async with self._semaphore:
            loop = asyncio.get_running_loop()
            finished = asyncio.Event()

            def notify() -> None:
                try:
                    loop.call_soon_threadsafe(finished.set)
                except RuntimeError:
                    pass  # the loop has closed: nobody is waiting

            attempt = _Attempt(notify=notify)
            threading.Thread(target=self._execute, args=(db_id, sql, attempt), daemon=True).start()
            try:
                await asyncio.wait_for(finished.wait(), self.timeout_s)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                # The request went away: stop the query before the slot is released.
                attempt.cancelled.set()
                await self._settle(attempt)
                raise
            timed_out = not attempt.finished.is_set()
            if timed_out:
                attempt.cancelled.set()
            stopped = await self._settle(attempt)
            if timed_out:
                if attempt.connection_id is None:
                    # Still connecting after the whole limit: the server is unreachable or overloaded, which says
                    # nothing about the query. (The official evaluator would score this as an empty result.)
                    raise MySqlUnavailableError(
                        f"could not connect to MySQL within {self.timeout_s:g} s: {attempt.unavailable or 'no answer'}"
                    )
                if not stopped:
                    # The server didn't stop it: give up on the thread rather than hold the slot. Its statement runs on
                    # as the read-only user until the server ends it.
                    logger.warning(
                        "BEAVER query on %s still running %g s after KILL QUERY; abandoned", db_id, self.stop_grace_s
                    )
                return QueryOutcome(status="timeout", error="timeout", seconds=self.timeout_s)
            seconds = attempt.ended - attempt.started
            if attempt.connection_id is not None and seconds > self.timeout_s:
                # Ended after the limit (the loop was busy when it passed): the official evaluator has already given up
                # on it, whether it then returned rows, failed or lost its connection.
                return QueryOutcome(status="timeout", error="timeout", seconds=seconds)
            if attempt.unavailable is not None:
                raise MySqlUnavailableError(str(attempt.unavailable)) from attempt.unavailable
            outcome = attempt.outcome
            assert outcome is not None
            outcome.seconds = seconds
            return outcome

    async def _settle(self, attempt: _Attempt) -> bool:
        """Wait, up to ``stop_grace_s``, until the query's thread is done (its connection closed), stopping the query
        first if it was given up on. Returns whether the thread was done in time.

        A cancellation meanwhile doesn't cut the wait short: the slot stays held until the wait is over, and the
        cancellation is raised then.
        """
        if attempt.done.is_set():
            return True
        stopping = asyncio.get_running_loop().run_in_executor(self._executor, self._stop, attempt, self.stop_grace_s)
        cancelled: Optional[asyncio.CancelledError] = None
        while True:
            try:
                stopped = await asyncio.shield(stopping)
                break
            except asyncio.CancelledError as e:
                if stopping.done():
                    raise
                cancelled = e
        if cancelled is not None:
            raise cancelled
        return stopped

    def _connect(self, database: str):
        import mysql.connector

        return mysql.connector.connect(**self._params, database=database, connection_timeout=self.connection_timeout_s)

    def _execute(self, db_id: str, sql: str, attempt: _Attempt) -> None:
        import mysql.connector

        attempt.started = self._clock()
        conn = None
        try:
            try:
                # No session settings: the model's SQL sees the server's defaults, as under the official evaluator.
                conn = self._connect(db_id)
                cur = conn.cursor()
            except Exception as e:
                attempt.unavailable = e
                return
            if attempt.cancelled.is_set():
                # Connected only after the limit: the query never starts, and the caller reports the server.
                return
            attempt.connection_id = conn.connection_id
            try:
                cur.execute(sql)
                columns, rows = [], []
                if cur.description is not None:
                    columns = [d[0] for d in cur.description]
                    # In chunks, so a timed-out query stops reading a large result (KILL QUERY can't stop the client
                    # reading rows the server has already sent) and memory stays bounded.
                    size = 0
                    while chunk := cur.fetchmany(_FETCH_CHUNK_ROWS):
                        rows.extend(chunk)
                        size += _approx_bytes(chunk)
                        if attempt.cancelled.is_set():
                            attempt.outcome = QueryOutcome(status="timeout", error="timeout")
                            return
                        if len(rows) > MAX_RESULT_ROWS or size > MAX_RESULT_BYTES:
                            attempt.outcome = QueryOutcome(status="error", error="result too large", oversized=True)
                            return
                attempt.outcome = QueryOutcome(status="ok", frame=pd.DataFrame(rows, columns=columns))
            except mysql.connector.Error as e:
                if attempt.cancelled.is_set() or e.errno in _INTERRUPTED_ERRNOS:
                    attempt.outcome = QueryOutcome(status="timeout", error="timeout")
                elif e.errno in _CONNECTION_LOST_ERRNOS:
                    attempt.unavailable = e
                else:
                    attempt.outcome = QueryOutcome(status="error", error=f"mysql {e.errno}")
            except MemoryError as e:
                # The grader ran out of memory: not the query's fault.
                attempt.unavailable = e
            except Exception as e:
                # Anything else reading the result (an undecodable value, say) fails the query, as it does officially.
                attempt.outcome = QueryOutcome(status="error", error=type(e).__name__)
        finally:
            # The query's time ends here; cleaning up its connection isn't part of it.
            attempt.ended = self._clock()
            attempt.finished.set()
            attempt.notify()
            try:
                if conn is not None:
                    if attempt.connection_id is not None and _more_results_pending(conn):
                        # Statements after the first would run on after close, unwatched.
                        self._kill(attempt.connection_id, "CONNECTION")
                    try:
                        conn.close()
                    except Exception:
                        pass
            finally:
                attempt.done.set()

    def _stop(self, attempt: _Attempt, grace_s: float) -> bool:
        """Wait for a query's thread to be done; return whether it was within ``grace_s``. While a query that was given
        up on (``cancelled``) still runs, stop it on the server.

        ``KILL QUERY`` sent just before the query starts would stop nothing, so it is repeated every second. Each attempt
        is bounded by ``_KILL_IO_TIMEOUT_S``. The grader's user may kill its own connections' queries without extra
        privileges.
        """
        deadline = time.monotonic() + grace_s
        while not attempt.done.is_set() and time.monotonic() < deadline:
            if attempt.cancelled.is_set() and not attempt.finished.is_set() and attempt.connection_id is not None:
                self._kill(attempt.connection_id, "QUERY")
            attempt.done.wait(min(1.0, max(0.0, deadline - time.monotonic())))
        return attempt.done.is_set()

    def _kill(self, connection_id: int, what: Literal["QUERY", "CONNECTION"]) -> None:
        try:
            import mysql.connector

            conn = mysql.connector.connect(
                **self._params,
                connection_timeout=_KILL_IO_TIMEOUT_S,
                read_timeout=_KILL_IO_TIMEOUT_S,
                write_timeout=_KILL_IO_TIMEOUT_S,
            )
            try:
                conn.cursor().execute(f"KILL {what} {int(connection_id)}")
            finally:
                conn.close()
        except Exception as e:
            # The query may have just finished (1094, unknown thread), or the server is gone.
            self.kill_failures += 1
            logger.debug("BEAVER KILL %s %s failed: %s", what, connection_id, e)


def _more_results_pending(conn: Any) -> bool:
    """Whether the server has more result sets coming on this connection (a string with several statements)."""
    try:
        cmysql = getattr(conn, "_cmysql", None)
        if cmysql is not None:
            return bool(cmysql.more_results())
        return bool(getattr(conn, "_have_next_result", False))
    except Exception:
        return False


class CannedRunner:
    """A stand-in for ``MySqlRunner`` that answers each query from a table: ``{sql: (columns, rows) | "error" |
    "timeout" | "unavailable"}``. Used by the verifier fixture and the unit tests."""

    def __init__(self, answers: Dict[str, Any]) -> None:
        self.answers = answers
        self.calls: List[Tuple[str, str]] = []

    async def run(self, db_id: str, sql: str) -> QueryOutcome:
        self.calls.append((db_id, sql))
        answer = self.answers.get(sql, "error")
        if answer == "unavailable":
            raise MySqlUnavailableError("canned: server unavailable")
        if answer in ("error", "timeout"):
            return QueryOutcome(status=answer, error=answer)
        columns, rows = answer
        return QueryOutcome(status="ok", frame=pd.DataFrame(rows, columns=columns))
