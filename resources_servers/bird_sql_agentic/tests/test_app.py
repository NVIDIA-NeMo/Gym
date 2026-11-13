# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import sqlite3
import time
from http.cookiejar import CookieJar
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.bird_sql.app import BirdSqlVerifyRequest, FailureCode
from resources_servers.bird_sql_agentic.app import (
    BirdSqlAgenticResourcesServer,
    BirdSqlAgenticResourcesServerConfig,
)
from resources_servers.bird_sql_agentic.sql_tool import (
    check_sql_allowed,
    format_rows,
    run_readonly_query,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def db_root(tmp_path) -> Path:
    """Two tiny databases laid out like BIRD: <root>/<db_id>/<db_id>.sqlite."""
    for db_id, names in (("SchoolDB", ["Ann", "Bob", "Cat"]), ("OtherDB", ["Zed"])):
        (tmp_path / db_id).mkdir()
        conn = sqlite3.connect(str(tmp_path / db_id / f"{db_id}.sqlite"))
        conn.execute("CREATE TABLE students (id INTEGER PRIMARY KEY, name TEXT, score REAL)")
        conn.executemany(
            "INSERT INTO students (name, score) VALUES (?, ?)", [(n, 80 + i) for i, n in enumerate(names)]
        )
        conn.execute("CREATE TABLE numbers (n INTEGER)")
        conn.executemany("INSERT INTO numbers VALUES (?)", [(i,) for i in range(50)])
        conn.commit()
        conn.close()
    return tmp_path


@pytest.fixture
def db_path(db_root) -> Path:
    return db_root / "SchoolDB" / "SchoolDB.sqlite"


@pytest.fixture
def server(db_root, monkeypatch) -> BirdSqlAgenticResourcesServer:
    monkeypatch.setattr("resources_servers.bird_sql.app.ensure_bird_sql", lambda _path: db_root)
    config = BirdSqlAgenticResourcesServerConfig(
        host="127.0.0.1",
        port=20099,
        entrypoint="",
        bird_sql_dir=str(db_root),
        max_concurrency=4,
        sql_execution_timeout_s=5.0,
    )
    return BirdSqlAgenticResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def client(server) -> TestClient:
    class StatelessCookieJar(CookieJar):
        def extract_cookies(self, response: object, request: object) -> None:
            # Sessions must come from explicit request cookies, like in a real rollout.
            pass

    return TestClient(server.setup_webserver(), cookies=StatelessCookieJar())


def _seed(client: TestClient, db_id: str):
    response = client.post("/seed_session", json={"db_id": db_id, "question": "q", "gt_sql": "SELECT 1"})
    assert response.status_code == 200
    return response.cookies


def _run(db_path: Path, sql: str, timeout_s: float = 5.0, max_rows: int = 20, max_cell_chars: int = 100) -> dict:
    return run_readonly_query(db_path, sql, timeout_s, max_rows, max_cell_chars)


# ---------------------------------------------------------------------------
# check_sql_allowed / format_rows / run_readonly_query
# ---------------------------------------------------------------------------


class TestCheckSqlAllowed:
    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT * FROM students",
            "  select name from students;",
            "WITH t AS (SELECT 1) SELECT * FROM t",
            "PRAGMA table_info(students)",
            "SELECT 'a;b' FROM students",
        ],
    )
    def test_allows_read_only_single_statements(self, sql) -> None:
        assert check_sql_allowed(sql) is None

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM students",
            "DROP TABLE students",
            "INSERT INTO students (name) VALUES ('x')",
            "PRAGMA writable_schema = ON",
            "ATTACH DATABASE 'other.sqlite' AS o",
        ],
    )
    def test_rejects_non_read_queries(self, sql) -> None:
        assert check_sql_allowed(sql) is not None

    def test_rejects_multiple_statements(self) -> None:
        assert check_sql_allowed("SELECT 1; DROP TABLE students") is not None

    def test_rejects_empty(self) -> None:
        assert check_sql_allowed("   ;  ") is not None


class TestFormatRows:
    def test_truncates_rows_and_reports_total(self) -> None:
        out = format_rows(
            ["n"],
            [(0,), (1,), (2,)],
            total_rows=50,
            max_rows=3,
            max_cell_chars=100,
        )
        assert out["rows"] == [[0], [1], [2]]
        assert out["total_rows"] == 50
        assert out["truncated"] is True
        assert out["error"] is None

    def test_shortens_long_cells(self) -> None:
        out = format_rows(
            ["n"],
            [("x" * 150,)],
            total_rows=1,
            max_rows=3,
            max_cell_chars=100,
        )
        assert out["rows"] == [["x" * 100 + "..."]]
        assert out["truncated"] is False


class TestRunReadonlyQuery:
    def test_returns_columns_and_rows(self, db_path) -> None:
        out = _run(db_path, "SELECT name FROM students ORDER BY id")
        assert out["columns"] == ["name"]
        assert out["rows"] == [["Ann"], ["Bob"], ["Cat"]]
        assert out["total_rows"] == 3
        assert out["error"] is None

    def test_truncates_large_results(self, db_path) -> None:
        out = _run(db_path, "SELECT * FROM numbers", max_rows=20)
        assert len(out["rows"]) == 20
        assert out["total_rows"] == 50
        assert out["truncated"] is True

    def test_sql_error_is_returned_not_raised(self, db_path) -> None:
        out = _run(db_path, "SELECT nme FROM students")
        assert "no such column" in out["error"]

    def test_write_is_rejected_and_database_unchanged(self, db_path) -> None:
        assert _run(db_path, "DELETE FROM students")["error"]
        assert _run(db_path, "SELECT COUNT(*) FROM students")["rows"] == [[3]]

    def test_connection_itself_is_read_only(self, db_path) -> None:
        # Even if the statement check were bypassed, mode=ro stops writes at the SQLite level.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("DELETE FROM students")
        conn.close()

    def test_timeout_stops_long_query(self, db_path) -> None:
        # Recursive query with no stop condition: runs forever unless the timeout fires.
        sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT COUNT(*) FROM c"
        start = time.monotonic()
        out = _run(db_path, sql, timeout_s=0.5)
        elapsed = time.monotonic() - start
        assert "timed out" in out["error"]
        assert elapsed < 5


# ---------------------------------------------------------------------------
# Server: session binding and the /execute_sql endpoint
# ---------------------------------------------------------------------------


class TestExecuteSqlEndpoint:
    def test_tool_uses_the_database_bound_in_seed_session(self, client) -> None:
        cookies = _seed(client, "SchoolDB")
        response = client.post("/execute_sql", json={"sql": "SELECT COUNT(*) FROM students"}, cookies=cookies)
        assert response.status_code == 200
        assert response.json()["rows"] == [[3]]

    def test_sessions_are_isolated(self, client) -> None:
        school = _seed(client, "SchoolDB")
        other = _seed(client, "OtherDB")
        sql = {"sql": "SELECT COUNT(*) FROM students"}
        assert client.post("/execute_sql", json=sql, cookies=school).json()["rows"] == [[3]]
        assert client.post("/execute_sql", json=sql, cookies=other).json()["rows"] == [[1]]

    def test_no_seeded_session_returns_error(self, client) -> None:
        response = client.post("/execute_sql", json={"sql": "SELECT 1"})
        assert response.status_code == 200
        assert "No database" in response.json()["error"]

    def test_errors_come_back_as_tool_output(self, client) -> None:
        cookies = _seed(client, "SchoolDB")
        body = client.post("/execute_sql", json={"sql": "DROP TABLE students"}, cookies=cookies).json()
        assert body["error"]
        assert body["rows"] == []

    def test_rejects_path_like_db_id(self, client) -> None:
        response = client.post("/seed_session", json={"db_id": "../SchoolDB", "question": "q", "gt_sql": "SELECT 1"})
        assert response.status_code == 422
        assert "Invalid db_id" in response.text


# ---------------------------------------------------------------------------
# verify: bird_sql scoring + tool-use counts
# ---------------------------------------------------------------------------


def _message(text: str) -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg",
        content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _verify_request(output: list, gt_sql: str = "SELECT COUNT(*) FROM students") -> BirdSqlVerifyRequest:
    response = NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=output,
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )
    return BirdSqlVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=response,
        question="How many students?",
        gt_sql=gt_sql,
        db_id="SchoolDB",
        difficulty="simple",
        id=0,
    )


def _tool_turn(call_id: str, sql: str, result: dict) -> list:
    return [
        NeMoGymResponseFunctionToolCall(
            call_id=call_id, name="execute_sql", arguments=json.dumps({"sql": sql}), type="function_call"
        ),
        NeMoGymFunctionCallOutput(call_id=call_id, output=json.dumps(result), type="function_call_output"),
    ]


class TestVerify:
    async def test_correct_after_fixing_an_error(self, server) -> None:
        output = [
            *_tool_turn("c1", "SELECT COUNT(*) FROM student", {"error": "SQL error: no such table: student"}),
            *_tool_turn("c2", "SELECT COUNT(*) FROM students", {"rows": [[3]], "error": None}),
            _message("```sql\nSELECT COUNT(*) FROM students\n```"),
        ]
        result = await server.verify(_verify_request(output))
        assert result.reward == 1.0
        assert result.num_tool_calls == 2
        assert result.num_sql_errors == 1

    async def test_wrong_answer(self, server) -> None:
        result = await server.verify(_verify_request([_message("```sql\nSELECT 99\n```")]))
        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.RESULT_MISMATCH
        assert result.num_tool_calls == 0

    async def test_no_sql_block(self, server) -> None:
        result = await server.verify(_verify_request([_message("I am not sure.")]))
        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.NO_SQL_EXTRACTED

    def test_compute_metrics_reports_tool_use(self, server) -> None:
        tasks = [
            [{"reward": 1.0, "extracted_sql": "a", "difficulty": "simple", "num_tool_calls": 2, "num_sql_errors": 1}],
            [{"reward": 0.0, "extracted_sql": "b", "difficulty": "simple", "num_tool_calls": 4, "num_sql_errors": 0}],
        ]
        metrics = server.compute_metrics(tasks)
        assert metrics["mean/num_tool_calls"] == 3
        assert metrics["mean/num_sql_errors"] == 0.5
