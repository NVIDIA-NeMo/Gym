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
    rows). Records KILL QUERY targets."""

    def __init__(self, script: Dict[str, Any], connect_delay: float = 0.0, connect_error: Exception = None) -> None:
        self.script = script
        self.connect_delay = connect_delay
        self.connect_error = connect_error
        self.killed: List[int] = []
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


class _FakeConnection:
    def __init__(self, server: _FakeServer, connection_id: int) -> None:
        self.server, self.connection_id = server, connection_id

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
        if sql.startswith("SET SESSION max_execution_time"):
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

    def fetchall(self) -> List[tuple]:
        return self._rows


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
    MySqlRunner("mysql://r:p@h")._kill_query(7)
