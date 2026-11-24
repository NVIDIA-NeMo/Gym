# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import datetime as dt
import importlib.util
import os
import threading
import time
from decimal import Decimal
from pathlib import Path
from typing import Any, Dict, List

import mysql.connector
import pandas as pd
import pytest

from resources_servers.beaver_sql.ex_acc import (
    CannedRunner,
    MySqlRunner,
    MySqlUnavailableError,
    QueryOutcome,
    missing_prediction,
    results_match,
)


def _ok(rows: List[tuple], columns: List[str] = None) -> QueryOutcome:
    if columns is None:
        columns = [f"c{i}" for i in range(len(rows[0]))] if rows else ["c0"]
    return QueryOutcome(status="ok", frame=pd.DataFrame(rows, columns=columns))


# (label, gold rows, predicted rows, expected verdict). The last group depends on pandas inferring each column's type
# from all its values, which a value-by-value conversion gets wrong.
CASES = [
    ("identical", [(1, "a")], [(1, "a")], True),
    ("row order", [(1, "a"), (2, "b")], [(2, "b"), (1, "a")], True),
    ("duplicate rows", [(1,)], [(1,), (1,)], True),
    ("column names ignored", [(1,)], [(1,)], True),
    ("decimal trailing zeros", [(Decimal("150.250"),)], [(Decimal("150.25"),)], False),
    ("decimal vs float", [(Decimal("150.25"),)], [(150.25,)], True),
    ("int vs text", [(1,)], [("1",)], True),
    ("whitespace stripped", [(" a",)], [("a ",)], True),
    ("column count", [(1,)], [(1, "x")], False),
    ("extra row", [(1,)], [(1,), (2,)], False),
    ("datetime at midnight vs date", [(dt.datetime(2020, 1, 1),)], [(dt.date(2020, 1, 1),)], True),
    ("datetime at midnight vs date text", [(dt.datetime(2020, 1, 1),)], [("2020-01-01",)], True),
    ("int column with NULL vs float column with NULL", [(5,), (None,)], [(5.0,), (None,)], True),
    ("NULL vs the text None", [("None",)], [(None,)], True),
    ("time vs text", [(dt.timedelta(hours=1),)], [("1:00:00",)], False),
    (
        "datetime column with NULL vs text",
        [(dt.datetime(2020, 1, 1, 3),), (None,)],
        [("2020-01-01 03:00:00",), ("None",)],
        False,
    ),
]


@pytest.mark.parametrize("label,gold,pred,expected", CASES, ids=[c[0] for c in CASES])
def test_results_match(label, gold, pred, expected) -> None:
    assert results_match(_ok(gold), _ok(pred)) is expected


def test_duplicate_column_names() -> None:
    assert results_match(_ok([(1, 1)], ["id", "id"]), _ok([(1, 1)], ["a", "b"]))


@pytest.mark.parametrize(
    "gold,pred,expected",
    [
        (_ok([]), missing_prediction(), True),
        (QueryOutcome(status="error"), QueryOutcome(status="timeout"), True),
        (_ok([(1,)]), missing_prediction(), False),
        (QueryOutcome(status="timeout"), _ok([(1,)]), False),
    ],
)
def test_empty_results(gold, pred, expected) -> None:
    assert results_match(gold, pred) is expected


_OFFICIAL = os.environ.get("BEAVER_OFFICIAL_EX_ACC")


@pytest.mark.skipif(not _OFFICIAL, reason="set BEAVER_OFFICIAL_EX_ACC to BEAVER's eval/utils/ex_acc.py to compare")
@pytest.mark.parametrize("label,gold,pred,expected", CASES, ids=[c[0] for c in CASES])
def test_agrees_with_official_evaluator(label, gold, pred, expected) -> None:
    spec = importlib.util.spec_from_file_location("beaver_official_ex_acc", Path(_OFFICIAL))
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    verdict, _ = official.compare_results(_ok(pred).frame, _ok(gold).frame)
    assert verdict is results_match(_ok(gold), _ok(pred)) is expected


def test_canned_runner() -> None:
    runner = CannedRunner({"A": (["x"], [(1,)]), "T": "timeout", "U": "unavailable"})
    assert asyncio.run(runner.run("dw", "A")).frame.shape == (1, 1)
    assert asyncio.run(runner.run("dw", "T")).status == "timeout"
    assert asyncio.run(runner.run("dw", "unknown")).status == "error"
    with pytest.raises(MySqlUnavailableError):
        asyncio.run(runner.run("dw", "U"))
    assert runner.calls == [("dw", "A"), ("dw", "T"), ("dw", "unknown"), ("dw", "U")]


# --- MySqlRunner against a fake driver -------------------------------------------------------------------------------


class _FakeServer:
    """Scripted answers per SQL text: (columns, rows), None (no result set), an Exception, or a float (sleep then
    rows). Records KILL QUERY and KILL CONNECTION targets; ``more_results`` is what a connection reports as still
    pending after its first result (a string with several statements)."""

    def __init__(self, script: Dict[str, Any], connect_delay: float = 0.0, connect_error: Exception = None) -> None:
        self.script = script
        self.connect_delay = connect_delay
        self.connect_error = connect_error
        self.fetch_delay = 0.0
        self.executed: List[str] = []
        self.killed: List[int] = []
        self.killed_connections: List[int] = []
        self.more_results = False
        self.connects: List[Dict[str, Any]] = []
        self.next_id = 100
        self.lock = threading.Lock()
        self.interrupt = threading.Event()

    def connect(self, **params: Any) -> "_FakeConnection":
        self.connects.append(params)
        if self.connect_delay:
            time.sleep(self.connect_delay)
        if self.connect_error is not None:
            raise self.connect_error
        with self.lock:
            self.next_id += 1
            return _FakeConnection(self, self.next_id)


class _FakeCMySql:
    def __init__(self, server: _FakeServer) -> None:
        self.server = server

    def more_results(self) -> bool:
        return self.server.more_results


class _FakeConnection:
    def __init__(self, server: _FakeServer, connection_id: int) -> None:
        self.server, self.connection_id = server, connection_id
        self._cmysql = _FakeCMySql(server)

    def cursor(self) -> "_FakeCursor":
        return _FakeCursor(self)

    def close(self) -> None:
        pass


class _FakeCursor:
    def __init__(self, conn: _FakeConnection) -> None:
        self.conn = conn
        self.description = None
        self._rows: List[tuple] = []

    def execute(self, sql: str) -> None:
        server = self.conn.server
        server.executed.append(sql)
        if sql.startswith("KILL CONNECTION"):
            server.killed_connections.append(int(sql.split()[-1]))
            return
        if sql.startswith("KILL QUERY"):
            server.killed.append(int(sql.split()[-1]))
            server.interrupt.set()
            return
        answer = server.script[sql]
        if isinstance(answer, tuple) and answer[0] == "ignores kill":
            time.sleep(answer[1])
            answer = (["x"], [(1,)])
        if isinstance(answer, float):
            if server.interrupt.wait(answer):
                raise mysql.connector.errors.DatabaseError("Query execution was interrupted", errno=1317)
            answer = (["x"], [(1,)])
        if isinstance(answer, Exception):
            raise answer
        if answer is None:
            return
        columns, self._rows = answer
        self.description = [(c,) for c in columns]

    def fetchmany(self, size: int) -> List[tuple]:
        chunk, self._rows = self._rows[:size], self._rows[size:]
        if chunk and self.conn.server.fetch_delay:
            time.sleep(self.conn.server.fetch_delay)
        return chunk


@pytest.fixture
def fake(monkeypatch):
    def install(script: Dict[str, Any], **kwargs: Any) -> _FakeServer:
        server = _FakeServer(script, **kwargs)
        monkeypatch.setattr(mysql.connector, "connect", server.connect)
        return server

    return install


def _run(runner: MySqlRunner, sql: str) -> QueryOutcome:
    return asyncio.run(runner.run("dw", sql))


def test_runner_rows_and_params(fake) -> None:
    server = fake({"Q": (["a", "b"], [(1, "x")]), "SET @x = 1": None})
    runner = MySqlRunner("mysql://reader:p%40ss@db.example:3307/ignored", max_concurrency=2)
    outcome = _run(runner, "Q")
    assert outcome.status == "ok" and list(outcome.frame.columns) == ["a", "b"] and not outcome.empty
    assert server.connects[0] == {
        "user": "reader",
        "password": "p@ss",
        "host": "db.example",
        "port": 3307,
        "database": "dw",
        "connection_timeout": 10,
    }
    no_result_set = _run(runner, "SET @x = 1")
    assert no_result_set.status == "ok" and no_result_set.empty
    # Nothing but the model's SQL runs on its connection: no session settings it could observe.
    assert server.executed == ["Q", "SET @x = 1"]


@pytest.mark.parametrize(
    "error,status",
    [
        (mysql.connector.errors.ProgrammingError("syntax", errno=1064), "error"),
        (mysql.connector.errors.DatabaseError("max_execution_time", errno=3024), "timeout"),
        # Refused because the grader's user may only read: the query's error, not the server's.
        (mysql.connector.errors.ProgrammingError("access denied to database", errno=1044), "error"),
        (mysql.connector.errors.ProgrammingError("unknown database", errno=1049), "error"),
        (ValueError("bad value"), "error"),
    ],
)
def test_runner_query_failures(fake, error, status) -> None:
    fake({"Q": error})
    assert _run(MySqlRunner("mysql://r:p@h"), "Q").status == status


@pytest.mark.parametrize("errno", [1038, 1114, 1037])
def test_runner_server_resource_errors_are_the_query_failure(fake, errno) -> None:
    # A runaway sort or GROUP BY causes these itself; officially they're errors, scored as an empty result.
    fake({"Q": mysql.connector.errors.DatabaseError("out of sort memory", errno=errno)})
    assert _run(MySqlRunner("mysql://r:p@h"), "Q").status == "error"


@pytest.mark.parametrize("errno", [2013, 2006])
def test_runner_unavailable_during_query(fake, errno) -> None:
    fake({"Q": mysql.connector.errors.OperationalError("gone", errno=errno)})
    with pytest.raises(MySqlUnavailableError):
        _run(MySqlRunner("mysql://r:p@h"), "Q")


def test_runner_unavailable_at_connect(fake) -> None:
    fake({}, connect_error=mysql.connector.errors.InterfaceError("refused", errno=2003))
    with pytest.raises(MySqlUnavailableError):
        _run(MySqlRunner("mysql://r:p@h"), "Q")


def test_runner_timeout_kills_the_query(fake) -> None:
    server = fake({"SLOW": 5.0})
    runner = MySqlRunner("mysql://r:p@h", timeout_s=0.2)
    start = time.monotonic()
    outcome = _run(runner, "SLOW")
    assert outcome.status == "timeout" and outcome.empty
    assert server.killed and server.killed[0] >= 101
    assert time.monotonic() - start < 3


def test_runner_releases_a_query_the_server_will_not_stop(fake) -> None:
    server = fake({"STUCK": ("ignores kill", 3.0)})
    runner = MySqlRunner("mysql://r:p@h", timeout_s=0.2)
    runner.stop_grace_s = 0.5
    start = time.monotonic()
    assert _run(runner, "STUCK").status == "timeout"
    assert time.monotonic() - start < 2.0 and server.killed


def test_runner_stops_reading_a_large_result_after_the_limit(fake, monkeypatch) -> None:
    # The server has sent every row; only the client is still reading when the limit passes.
    monkeypatch.setattr("resources_servers.beaver_sql.ex_acc._FETCH_CHUNK_ROWS", 10)
    server = fake({"BIG": (["x"], [(i,) for i in range(1000)])})
    server.fetch_delay = 0.05
    runner = MySqlRunner("mysql://r:p@h", timeout_s=0.2)
    start = time.monotonic()
    assert _run(runner, "BIG").status == "timeout"
    assert time.monotonic() - start < 2.0


def test_runner_reads_a_result_in_chunks(fake, monkeypatch) -> None:
    monkeypatch.setattr("resources_servers.beaver_sql.ex_acc._FETCH_CHUNK_ROWS", 3)
    fake({"Q": (["x"], [(i,) for i in range(10)])})
    assert _run(MySqlRunner("mysql://r:p@h"), "Q").frame.shape == (10, 1)


def test_runner_slow_connect_is_unavailable(fake) -> None:
    fake({"Q": (["x"], [(1,)])}, connect_delay=0.5)
    with pytest.raises(MySqlUnavailableError, match="could not connect"):
        _run(MySqlRunner("mysql://r:p@h", timeout_s=0.2), "Q")


def test_runner_result_after_limit_is_a_timeout(fake) -> None:
    fake({"Q": (["x"], [(1,)])})
    runner = MySqlRunner("mysql://r:p@h", timeout_s=5.0)
    clock = iter([0.0, 6.0])
    runner._clock = lambda: next(clock)
    assert _run(runner, "Q").status == "timeout"


def test_kill_query_failure_is_ignored(fake, monkeypatch) -> None:
    fake({})

    def refuse(**params: Any):
        raise mysql.connector.errors.InterfaceError("refused", errno=2003)

    monkeypatch.setattr(mysql.connector, "connect", refuse)
    runner = MySqlRunner("mysql://r:p@h")
    runner._kill(7, "QUERY")
    assert runner.kill_failures == 1


def test_runner_grader_out_of_memory_is_unavailable(fake) -> None:
    fake({"Q": MemoryError()})
    with pytest.raises(MySqlUnavailableError):
        _run(MySqlRunner("mysql://r:p@h"), "Q")


def test_runner_result_too_large_never_matches(fake, monkeypatch) -> None:
    monkeypatch.setattr("resources_servers.beaver_sql.ex_acc._FETCH_CHUNK_ROWS", 10)
    monkeypatch.setattr("resources_servers.beaver_sql.ex_acc.MAX_RESULT_ROWS", 25)
    fake({"BIG": (["x"], [(i,) for i in range(100)]), "WIDE": (["x"], [("y" * 100,) for _ in range(5)])})
    big = _run(MySqlRunner("mysql://r:p@h"), "BIG")
    assert (big.status, big.error, big.empty, big.oversized) == ("error", "result too large", False, True)
    # Officially a huge result is non-empty, so it doesn't match an empty gold result either.
    assert not results_match(QueryOutcome(status="ok", frame=pd.DataFrame([], columns=["x"])), big)
    monkeypatch.setattr("resources_servers.beaver_sql.ex_acc.MAX_RESULT_BYTES", 300)
    assert _run(MySqlRunner("mysql://r:p@h"), "WIDE").error == "result too large"


def test_cancelled_request_stops_its_query_before_releasing_the_slot(fake) -> None:
    server = fake({"SLOW": 5.0})
    runner = MySqlRunner("mysql://r:p@h", max_concurrency=1, timeout_s=10.0)

    async def scenario():
        task = asyncio.create_task(runner.run("dw", "SLOW"))
        await asyncio.sleep(0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # The query was killed before the slot came back.
        assert server.killed
        start = time.monotonic()
        server.script["FAST"] = (["x"], [(1,)])
        assert (await runner.run("dw", "FAST")).status == "ok"
        assert time.monotonic() - start < 2

    asyncio.run(scenario())


def test_kill_connection_is_bounded(fake, monkeypatch) -> None:
    seen = {}

    def connect(**params):
        seen.update(params)
        raise mysql.connector.errors.InterfaceError("refused", errno=2003)

    monkeypatch.setattr(mysql.connector, "connect", connect)
    MySqlRunner("mysql://r:p@h")._kill(7, "QUERY")
    assert seen["read_timeout"] == seen["write_timeout"] == seen["connection_timeout"] == 2


def test_a_busy_event_loop_does_not_time_out_a_fast_query(fake) -> None:
    # The query is timed on its own thread: blocking the loop past the limit doesn't turn a 0.05 s query into a timeout.
    fake({"FAST": 0.05})
    runner = MySqlRunner("mysql://r:p@h", timeout_s=1.0)

    async def scenario():
        task = asyncio.create_task(runner.run("dw", "FAST"))
        await asyncio.sleep(0.01)
        time.sleep(1.5)  # the loop is stuck, as during a large comparison
        return await task

    outcome = asyncio.run(scenario())
    assert outcome.status == "ok" and outcome.seconds < 1.0


def test_several_statements_kill_the_connection_after_the_first_result(fake) -> None:
    server = fake({"SELECT 1; DO SLEEP(60)": (["1"], [(1,)])})
    server.more_results = True
    outcome = _run(MySqlRunner("mysql://r:p@h"), "SELECT 1; DO SLEEP(60)")
    assert outcome.status == "ok" and outcome.frame.iloc[0, 0] == 1
    # The query's own connection (the first one opened), from a second connection.
    assert server.killed_connections == [101]


def test_one_statement_kills_nothing(fake) -> None:
    server = fake({"Q": (["x"], [(1,)])})
    _run(MySqlRunner("mysql://r:p@h"), "Q")
    assert server.killed_connections == [] and server.killed == []


@pytest.mark.parametrize("errno", [1049, 1045])
def test_runner_connect_errors_are_unavailable(fake, errno) -> None:
    fake({}, connect_error=mysql.connector.errors.ProgrammingError("not loaded", errno=errno))
    with pytest.raises(MySqlUnavailableError):
        _run(MySqlRunner("mysql://r:p@h"), "Q")
