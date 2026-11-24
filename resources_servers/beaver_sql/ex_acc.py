# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BEAVER's execution accuracy (``ex_acc``): run a query on MySQL, compare two results.

This follows the rule of BEAVER's official evaluator (``eval/utils/ex_acc.py`` in https://github.com/beaverbench/beaver
at 81a2a88a1) and is checked against that file on the same predictions (``benchmarks/beaver/official_check.py``). It
is written for Gym, not copied: the official repository has no license file.

Running a query:

- Each query gets its own connection and 10 seconds, counted from the start of the connection to the last row read
  into a ``pandas.DataFrame``. A query that runs longer is a timeout. The query is then stopped on the server
  (``KILL QUERY``); the official evaluator abandons it instead, which changes nothing in the score.
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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

from resources_servers.beaver_sql.setup_beaver_mysql import parse_mysql_dsn


logger = logging.getLogger(__name__)

# The official evaluator's limits, in seconds.
QUERY_TIMEOUT_S = 10.0
CONNECTION_TIMEOUT_S = 10

# Any failure to connect means the grader couldn't reach the data. Once the query runs, only a lost connection (2006,
# 2013, 2055) does; every other error is the query's, scored as an empty result. That includes access denied (1044,
# 1142): the grader's user may only read, so a query that writes, creates a temporary table or uses another database
# fails, as it would for a read-only user anywhere.
_CONNECTION_LOST_ERRNOS = frozenset({2006, 2013, 2055})
# The query was stopped: by our KILL QUERY (1317) or by the server's max_execution_time (3024).
_INTERRUPTED_ERRNOS = frozenset({1317, 3024})


class MySqlUnavailableError(RuntimeError):
    """The grader couldn't reach the data; the row must be masked, not scored."""


@dataclass
class QueryOutcome:
    """A query's result: ``frame`` when ``status`` is ``ok``, else an empty result."""

    status: Literal["ok", "error", "timeout", "missing"]
    frame: Optional[pd.DataFrame] = None
    error: Optional[str] = None
    seconds: float = 0.0

    @property
    def empty(self) -> bool:
        return self.frame is None or self.frame.empty


def missing_prediction() -> QueryOutcome:
    """No query to run: an empty result, as a missing prediction file is officially."""
    return QueryOutcome(status="missing")


def _rows_as_text(frame: pd.DataFrame) -> List[Tuple[Any, ...]]:
    """Every cell as stripped text, typed column by column as pandas inferred it."""
    text = frame.astype(str)
    return [
        tuple(c.strip() if isinstance(c, str) else c for c in row) for row in text.itertuples(index=False, name=None)
    ]


def results_match(gold: QueryOutcome, pred: QueryOutcome) -> bool:
    """BEAVER's comparison: both empty, or equal sets of stripped-text rows with the same number of columns."""
    if gold.empty or pred.empty:
        return gold.empty and pred.empty
    gold_rows, pred_rows = _rows_as_text(gold.frame), _rows_as_text(pred.frame)
    if len(gold_rows[0]) != len(pred_rows[0]):
        return False
    return set(gold_rows) == set(pred_rows)


@dataclass
class _Attempt:
    """State shared between the event loop and the thread running one query."""

    connection_id: Optional[int] = None
    outcome: Optional[QueryOutcome] = None
    unavailable: Optional[BaseException] = None
    done: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)


class MySqlRunner:
    """Runs queries on the BEAVER MySQL server, at most ``max_concurrency`` at a time.

    A timed-out query keeps its slot while ``KILL QUERY`` is repeated, until its thread ends or ``stop_grace_s`` passes,
    so stopped queries don't pile up on the server behind new ones, and one the server can't stop doesn't hold the slot
    forever.
    """

    def __init__(
        self,
        dsn: str,
        *,
        max_concurrency: int = 8,
        timeout_s: float = QUERY_TIMEOUT_S,
        connection_timeout_s: int = CONNECTION_TIMEOUT_S,
    ) -> None:
        self._params = parse_mysql_dsn(dsn)
        self._params.pop("database", None)
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout_s = timeout_s
        self.connection_timeout_s = connection_timeout_s
        # How long a timed-out query may take to stop before its slot is released anyway.
        self.stop_grace_s = 10.0
        self._clock = time.monotonic

    async def run(self, db_id: str, sql: str) -> QueryOutcome:
        async with self._semaphore:
            attempt = _Attempt()
            start = self._clock()
            thread = threading.Thread(target=self._execute, args=(db_id, sql, attempt), daemon=True)
            thread.start()
            finished = await asyncio.to_thread(attempt.done.wait, self.timeout_s)
            if not finished:
                attempt.cancelled.set()
                stopped = await asyncio.to_thread(self._stop, attempt, self.stop_grace_s)
                if attempt.connection_id is None:
                    # Still connecting after the whole limit: the server is unreachable or overloaded, which says
                    # nothing about the query. (The official evaluator would score this as an empty result.)
                    raise MySqlUnavailableError(
                        f"could not connect to MySQL within {self.timeout_s:g} s: {attempt.unavailable or 'no answer'}"
                    )
                if not stopped:
                    # The server didn't stop it: give up on the thread rather than hold the slot. The server-side
                    # max_execution_time still ends a SELECT; anything else runs on as a read-only statement.
                    logger.warning(
                        "BEAVER query on %s still running %g s after KILL QUERY; abandoned", db_id, self.stop_grace_s
                    )
                return QueryOutcome(status="timeout", error="timeout", seconds=self._clock() - start)
            await asyncio.to_thread(thread.join, self.stop_grace_s)
            if attempt.unavailable is not None:
                raise MySqlUnavailableError(str(attempt.unavailable)) from attempt.unavailable
            outcome = attempt.outcome
            assert outcome is not None
            outcome.seconds = self._clock() - start
            if outcome.status == "ok" and outcome.seconds > self.timeout_s:
                # Finished, but after the limit: the official evaluator has already given up on it.
                return QueryOutcome(status="timeout", error="timeout", seconds=outcome.seconds)
            return outcome

    def _connect(self, database: str):
        import mysql.connector

        return mysql.connector.connect(**self._params, database=database, connection_timeout=self.connection_timeout_s)

    def _execute(self, db_id: str, sql: str, attempt: _Attempt) -> None:
        import mysql.connector

        conn = None
        try:
            try:
                conn = self._connect(db_id)
                cur = conn.cursor()
                # A server-side backstop for SELECTs; the client-side limit above is the one that counts.
                cur.execute(f"SET SESSION max_execution_time = {int(self.timeout_s * 1000) + 1000}")
            except Exception as e:
                attempt.unavailable = e
                return
            if attempt.cancelled.is_set():
                # Connected only after the limit: the query never starts, and the caller reports the server.
                return
            attempt.connection_id = conn.connection_id
            try:
                cur.execute(sql)
                if cur.description is None:
                    columns, rows = [], []
                else:
                    columns = [d[0] for d in cur.description]
                    rows = cur.fetchall()
                attempt.outcome = QueryOutcome(status="ok", frame=pd.DataFrame(rows, columns=columns))
            except mysql.connector.Error as e:
                if e.errno in _INTERRUPTED_ERRNOS:
                    attempt.outcome = QueryOutcome(status="timeout", error="timeout")
                elif e.errno in _CONNECTION_LOST_ERRNOS:
                    attempt.unavailable = e
                else:
                    attempt.outcome = QueryOutcome(status="error", error=f"mysql {e.errno}")
            except Exception as e:
                attempt.outcome = QueryOutcome(status="error", error=type(e).__name__)
        finally:
            attempt.done.set()
            if conn is not None:
                try:
                    conn.close()
                except Exception:
                    pass

    def _stop(self, attempt: _Attempt, grace_s: float) -> bool:
        """Stop a timed-out query on the server; return whether its thread ended within ``grace_s``.

        ``KILL QUERY`` sent between a connection's ``SET`` and its query would stop nothing, so it is repeated every
        second. The grader's user may kill its own connections' queries without extra privileges.
        """
        deadline = time.monotonic() + grace_s
        while not attempt.done.is_set() and time.monotonic() < deadline:
            if attempt.connection_id is not None:
                self._kill_query(attempt.connection_id)
            attempt.done.wait(min(1.0, max(0.0, deadline - time.monotonic())))
        return attempt.done.is_set()

    def _kill_query(self, connection_id: int) -> None:
        try:
            import mysql.connector

            conn = mysql.connector.connect(**self._params, connection_timeout=self.connection_timeout_s)
            try:
                conn.cursor().execute(f"KILL QUERY {int(connection_id)}")
            finally:
                conn.close()
        except Exception:
            # The query may have just finished, or the server is gone; the caller keeps waiting for the thread.
            pass


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
