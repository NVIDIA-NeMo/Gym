# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import datetime
import io
import json
import shutil
import sqlite3
import zipfile
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from resources_servers.spider2_lite.app import (
    FailureCode,
    Spider2LiteResourcesServer,
    Spider2LiteResourcesServerConfig,
    Spider2LiteVerifyRequest,
)
from resources_servers.spider2_lite.eval_utils import (
    _coerce,
    _normalize,
    compare_multi_result_sets,
    compare_result_sets,
    execute_and_compare,
    execute_bigquery,
    execute_bigquery_async,
    execute_sqlite,
    execute_sqlite_async,
)
from resources_servers.text_to_sql.app import extract_sql_from_response


# ---------------------------------------------------------------------------
# TestEvalUtils
# ---------------------------------------------------------------------------


class TestEvalUtils:
    def test_normalize_none(self):
        assert _normalize(None) == 0

    def test_normalize_nan(self):
        assert _normalize(float("nan")) == 0

    def test_normalize_value(self):
        assert _normalize(42) == 42
        assert _normalize("hello") == "hello"

    def test_coerce_string_int(self):
        assert _coerce("1") == 1
        assert isinstance(_coerce("1"), int)

    def test_coerce_string_float(self):
        assert _coerce("13664.08") == pytest.approx(13664.08)
        assert isinstance(_coerce("13664.08"), float)

    def test_coerce_string_stays_string(self):
        assert _coerce("abc") == "abc"

    def test_coerce_non_string_unchanged(self):
        assert _coerce(42) == 42
        assert _coerce(3.14) == pytest.approx(3.14)
        assert _coerce(None) is None

    def test_execute_sqlite_select(self, tiny_db):
        rows = execute_sqlite(tiny_db, "SELECT id, name FROM items ORDER BY id")
        assert rows == [(1, "alpha"), (2, "beta")]

    def test_execute_sqlite_empty(self, tiny_db):
        rows = execute_sqlite(tiny_db, "SELECT * FROM items WHERE id = 999")
        assert rows == []

    def test_execute_sqlite_bad_sql(self, tiny_db):
        assert execute_sqlite(tiny_db, "NOT VALID SQL") == None

    def test_execute_sqlite_async_ok(self, tiny_db):
        sem = asyncio.Semaphore(1)
        rows = asyncio.run(execute_sqlite_async(tiny_db, "SELECT id FROM items ORDER BY id", sem))
        assert rows == [(1,), (2,)]

    def test_compare_result_sets_exact(self):
        gold = [(1, "a"), (2, "b")]
        pred = [(1, "a"), (2, "b")]
        assert compare_result_sets(gold, pred, ignore_order=False)

    def test_compare_result_sets_order(self):
        gold = [(1,), (2,)]
        pred = [(2,), (1,)]
        assert compare_result_sets(gold, pred, ignore_order=True)
        assert not compare_result_sets(gold, pred, ignore_order=False)

    def test_compare_result_sets_float(self):
        gold = [(1.0,)]
        pred = [(1.005,)]
        assert compare_result_sets(gold, pred)

    def test_compare_result_sets_float_fail(self):
        gold = [(1.0,)]
        pred = [(1.02,)]
        assert not compare_result_sets(gold, pred)

    def test_compare_result_sets_condition_cols(self):
        gold = [(1, "x"), (2, "y")]
        pred = [(1, "z"), (2, "w")]
        # Compare only col 0 (the integers)
        assert compare_result_sets(gold, pred, condition_cols=[0], ignore_order=True)
        # Compare all cols, should fail
        assert not compare_result_sets(gold, pred, condition_cols=None, ignore_order=True)

    def test_compare_result_sets_empty(self):
        assert compare_result_sets([], [])

    def test_compare_result_sets_pred_empty(self):
        assert not compare_result_sets([(1,)], [])

    def test_compare_result_sets_extra_pred_cols(self):
        gold = [(1,), (2,)]
        pred = [(1, "extra"), (2, "extra")]
        assert compare_result_sets(gold, pred, ignore_order=True)

    def test_coerce_in_comparison(self):
        # Gold from CSV as strings, pred from sqlite3 as ints
        gold = [("1",), ("2",)]
        pred = [(1,), (2,)]
        assert compare_result_sets(gold, pred, ignore_order=True)

    def test_compare_multi_flat_condition_cols(self):
        gold_a = [(1, "a"), (2, "b")]
        gold_b = [(3, "c"), (4, "d")]
        pred = [(1, "z"), (2, "w")]
        assert compare_multi_result_sets([gold_a, gold_b], pred, multi_condition_cols=[0], ignore_order=True)

    def test_compare_multi_list_condition_cols(self):
        gold_a = [(0, 1, "a")]
        gold_b = [(0, 2, "b")]
        pred = [(99, 1, "z")]
        # Compare col 1 for gold_a, col 1 for gold_b
        assert compare_multi_result_sets([gold_a, gold_b], pred, multi_condition_cols=[[1], [1]], ignore_order=True)

    def test_compare_multi_any_match(self):
        gold_wrong = [(99,)]
        gold_right = [(1,)]
        pred = [(1,)]
        assert compare_multi_result_sets([gold_wrong, gold_right], pred)

    def test_compare_multi_empty_gold(self):
        assert not compare_multi_result_sets([], [(1,)])

    def test_execute_and_compare_match(self, tiny_db):
        sem = asyncio.Semaphore(4)
        match, gold, pred, err = asyncio.run(
            execute_and_compare(tiny_db, "SELECT id FROM items ORDER BY id", "SELECT id FROM items ORDER BY id", sem)
        )
        assert match is True
        assert err is None

    def test_execute_and_compare_mismatch(self, tiny_db):
        sem = asyncio.Semaphore(4)
        match, gold, pred, err = asyncio.run(
            execute_and_compare(tiny_db, "SELECT id FROM items", "SELECT name FROM items", sem)
        )
        assert match is False
        assert err is None

    def test_execute_and_compare_gold_error(self, tiny_db):
        sem = asyncio.Semaphore(4)
        match, gold, pred, err = asyncio.run(execute_and_compare(tiny_db, "INVALID SQL", "SELECT 1", sem))
        assert match is False
        assert err is not None and "gold_sql_error" in err

    def test_execute_and_compare_pred_error(self, tiny_db):
        sem = asyncio.Semaphore(4)
        match, gold, pred, err = asyncio.run(execute_and_compare(tiny_db, "SELECT id FROM items", "INVALID SQL", sem))
        assert match is False
        assert err is not None and "pred_sql_error" in err


# ---------------------------------------------------------------------------
# TestExtractSqlReuse
# ---------------------------------------------------------------------------


class TestExtractSqlReuse:
    def test_import_extract_sql_from_response(self):
        assert callable(extract_sql_from_response)

    def test_extract_from_sql_block(self):
        text = "Here is the query:\n```sql\nSELECT 1;\n```"
        assert extract_sql_from_response(text) == "SELECT 1;"

    def test_returns_none_on_no_sql(self):
        assert extract_sql_from_response("No SQL here.") is None


# ---------------------------------------------------------------------------
# TestSpider2LiteServerUnit
# ---------------------------------------------------------------------------


class TestSpider2LiteServerUnit:
    async def test_verify_gold_sql_match(self, server):
        req = _make_verify_request(
            "```sql\nSELECT id, name, val FROM items ORDER BY id;\n```",
            gold_sql="SELECT id, name, val FROM items ORDER BY id",
        )
        result = await server.verify(req)
        assert result.reward == 1.0
        assert result.execution_match is True
        assert result.failure_reason == FailureCode.NONE

    async def test_verify_gold_sql_mismatch(self, server):
        req = _make_verify_request("```sql\nSELECT 1;\n```", gold_sql="SELECT id FROM items")
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.execution_match is False

    async def test_verify_no_sql_extracted(self, server):
        req = _make_verify_request("I don't know.", gold_sql="SELECT 1")
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.NO_SQL_EXTRACTED

    async def test_verify_empty_response(self, server):
        req = _make_verify_request("", gold_sql="SELECT 1")
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.NO_SQL_EXTRACTED

    async def test_verify_gold_result_mode(self, server):
        # Gold result = [[("alpha",), ("beta",)]] as lists
        req = Spider2LiteVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("```sql\nSELECT name FROM items ORDER BY id;\n```"),
            db_id="TestDB",
            question="q",
            gold_result=[[["alpha"], ["beta"]]],
            ignore_order=True,
            condition_cols=None,
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_verify_gold_result_mismatch(self, server):
        req = Spider2LiteVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("```sql\nSELECT id FROM items;\n```"),
            db_id="TestDB",
            question="q",
            gold_result=[[["alpha"], ["beta"]]],
            ignore_order=True,
        )
        result = await server.verify(req)
        assert result.reward == 0.0

    async def test_verify_execution_error_pred(self, server):
        req = _make_verify_request("```sql\nSELECT * FROM nonexistent_table;\n```", gold_sql="SELECT 1")
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.EXECUTION_ERROR

    async def test_verify_gold_execution_error(self, server):
        req = _make_verify_request("```sql\nSELECT 1;\n```", gold_sql="SELECT * FROM nonexistent_table")
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.EXECUTION_ERROR

    async def test_verify_no_gold_data(self, server):
        req = Spider2LiteVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("```sql\nSELECT 1;\n```"),
            db_id="TestDB",
            question="q",
        )
        with pytest.raises(ValueError):
            await server.verify(req)

    async def test_verify_ignore_order(self, server):
        # Gold returns rows in a different order than pred, should still match with ignore_order=True
        req = _make_verify_request(
            "```sql\nSELECT name FROM items ORDER BY name DESC;\n```",
            gold_sql="SELECT name FROM items ORDER BY name ASC",
            ignore_order=True,
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_verify_condition_cols(self, server):
        # Compare only col 0 (id), pred has wrong col 1 (name) but correct id
        req = _make_verify_request(
            "```sql\nSELECT id, val FROM items ORDER BY id;\n```",
            gold_sql="SELECT id, name FROM items ORDER BY id",
            condition_cols=[0],
            ignore_order=True,
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_verify_multi_gold_result(self, server):
        # Two possible gold sets; pred matches the second
        req = Spider2LiteVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("```sql\nSELECT name FROM items ORDER BY id;\n```"),
            db_id="TestDB",
            question="q",
            gold_result=[[["wrong"]], [["alpha"], ["beta"]]],
            ignore_order=True,
        )
        result = await server.verify(req)
        assert result.reward == 1.0

    async def test_verify_uuid_passthrough(self, server):
        req = _make_verify_request("```sql\nSELECT 1;\n```", gold_sql="SELECT 1", uuid="task-42")
        result = await server.verify(req)
        assert result.uuid == "task-42"


# ---------------------------------------------------------------------------
# TestSetupSpider2
# ---------------------------------------------------------------------------


class TestSetupSpider2:
    def test_ensure_idempotent(self, tmp_path):
        from resources_servers.spider2_lite.setup_spider2 import ensure_spider2_lite

        sqlite_dir = tmp_path / "sqlite"
        sqlite_dir.mkdir()
        (sqlite_dir / "Fake.sqlite").write_bytes(b"data")

        with patch("resources_servers.spider2_lite.setup_spider2.urllib.request.urlopen") as mock_open:
            result = ensure_spider2_lite(tmp_path)
            mock_open.assert_not_called()
        assert result == sqlite_dir

    def test_ensure_downloads_on_empty_dir(self, tmp_path):
        from resources_servers.spider2_lite.setup_spider2 import ensure_spider2_lite

        zip_bytes = _make_zip_bytes({"databases/IPL.sqlite": b"fake-db", "databases/Pagila.sqlite": b"fake-db2"})

        mock_resp = MagicMock()
        mock_resp.read.return_value = zip_bytes
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch("resources_servers.spider2_lite.setup_spider2.urllib.request.urlopen", return_value=mock_resp):
            result = ensure_spider2_lite(tmp_path)

        assert (result / "IPL.sqlite").exists()
        assert (result / "Pagila.sqlite").exists()


# ---------------------------------------------------------------------------
# TestCoerceBigQueryTypes
# ---------------------------------------------------------------------------


class TestCoerceBigQueryTypes:
    def test_decimal_integral(self):
        assert _coerce(Decimal("42")) == 42
        assert isinstance(_coerce(Decimal("42")), int)

    def test_decimal_fractional(self):
        assert _coerce(Decimal("3.14")) == pytest.approx(3.14)
        assert isinstance(_coerce(Decimal("3.14")), float)

    def test_date(self):
        assert _coerce(datetime.date(2024, 1, 2)) == "2024-01-02"

    def test_datetime(self):
        assert _coerce(datetime.datetime(2024, 1, 2, 3, 4, 5)) == "2024-01-02T03:04:05"

    def test_bytes_utf8(self):
        assert _coerce(b"abc") == "abc"


# ---------------------------------------------------------------------------
# TestBigQueryExecution (unit test, the real client is mocked)
# ---------------------------------------------------------------------------


class TestBigQueryExecution:
    def test_execute_bigquery_returns_tuples(self):
        client = _FakeBigQueryClient(rows=[("alpha", 1), ("beta", 2)])
        rows = execute_bigquery(client, "SELECT 1")
        assert rows == [("alpha", 1), ("beta", 2)]
        assert client.queries == ["SELECT 1"]

    @pytest.mark.parametrize(
        "exc_kwarg,exc",
        [("query_exc", RuntimeError("test error")), ("result_exc", TimeoutError("timed out"))],
    )
    def test_execute_bigquery_returns_none_on_error(self, exc_kwarg, exc):
        client = _FakeBigQueryClient(**{exc_kwarg: exc})
        assert execute_bigquery(client, "SELECT 1") is None

    def test_execute_bigquery_async_returns_rows(self):
        client = _FakeBigQueryClient(rows=[("x",)])

        async def _run():
            sem = asyncio.Semaphore(1)
            return await execute_bigquery_async(client, "SELECT 1", sem)

        rows = asyncio.run(_run())
        assert rows == [("x",)]


# ---------------------------------------------------------------------------
# TestSpider2LiteServerBigQuery (verify() with mocked BigQuery client)
# ---------------------------------------------------------------------------


class TestSpider2LiteServerBigQuery:
    async def test_bigquery_match_via_mocked_client(self, server):
        fake = _FakeBigQueryClient(rows=[("alpha",), ("beta",)])
        self._make_bq_server(server, fake)
        req = _make_bq_request(
            "```sql\nSELECT name FROM `bigquery-public-data.x.items`;\n```",
            gold_result=[[["alpha"], ["beta"]]],
        )
        result = await server.verify(req)
        assert result.engine == "bigquery"
        assert result.reward == 1.0
        assert result.failure_reason == FailureCode.NONE
        assert fake.queries, "BigQuery client should have received the predicted SQL"

    async def test_bigquery_mismatch(self, server):
        fake = _FakeBigQueryClient(rows=[("wrong",)])
        self._make_bq_server(server, fake)
        req = _make_bq_request(
            "```sql\nSELECT name FROM `bigquery-public-data.x.items`;\n```",
            gold_result=[[["alpha"], ["beta"]]],
        )
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.EXECUTION_ERROR

    async def test_bigquery_query_error_marks_execution_error(self, server):
        fake = _FakeBigQueryClient(query_exc=RuntimeError("permission denied"))
        self._make_bq_server(server, fake)
        req = _make_bq_request("```sql\nSELECT 1;\n```", gold_result=[[["alpha"]]])
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.EXECUTION_ERROR

    async def test_bigquery_requires_gold_result(self, server):
        self._make_bq_server(server, _FakeBigQueryClient(rows=[]))
        # gold_sql is not accepted on BigQuery, which only takes gold_result.
        req = _make_bq_request("```sql\nSELECT 1;\n```", gold_sql="SELECT 1")
        with pytest.raises(ValueError):
            await server.verify(req)

    async def test_bigquery_unavailable_when_creds_missing(self, server, tmp_path):
        server.config.bigquery_credentials_path = str(tmp_path / "missing.json")
        req = _make_bq_request("```sql\nSELECT 1;\n```", gold_result=[[["alpha"]]])
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.BIGQUERY_UNAVAILABLE

    async def test_bigquery_unavailable_when_adc_has_no_project(self, server, tmp_path):
        _write_adc(tmp_path, project=None)
        server.config.bigquery_credentials_path = str(tmp_path / "application_default_credentials.json")
        req = _make_bq_request("```sql\nSELECT 1;\n```", gold_result=[[["alpha"]]])
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.BIGQUERY_UNAVAILABLE

    async def test_default_engine_is_sqlite(self, server):
        req = Spider2LiteVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("```sql\nSELECT id FROM items ORDER BY id;\n```"),
            db_id="TestDB",
            question="q",
            gold_sql="SELECT id FROM items ORDER BY id",
        )
        # Omitting the engine field must route to sqlite.
        assert req.engine == "sqlite"
        result = await server.verify(req)
        assert result.engine == "sqlite"
        assert result.reward == 1.0

    async def test_bigquery_only_skips_sqlite_download(self, server):
        """A BigQuery verify must not invoke the SQLite download/extract path."""
        self._make_bq_server(server, _FakeBigQueryClient(rows=[("alpha",)]))
        with patch("resources_servers.spider2_lite.app.ensure_spider2_lite") as mock_ensure:
            req = _make_bq_request(
                "```sql\nSELECT name FROM `bigquery-public-data.x.items`;\n```",
                gold_result=[[["alpha"]]],
            )
            result = await server.verify(req)
        mock_ensure.assert_not_called()
        assert result.engine == "bigquery"
        assert result.reward == 1.0

    def _make_bq_server(self, server, fake_client):
        """Rebind the server's BigQuery client to the provided fake."""
        server._bigquery_client = fake_client
        server._bigquery_init_error = None
        return server


# ---------------------------------------------------------------------------
# TestPrepareDatasetBigQuery
# ---------------------------------------------------------------------------


class TestPrepareDatasetBigQuery:
    def test_build_row_bigquery_marks_engine(self):
        from resources_servers.spider2_lite.scripts.prepare_dataset import (
            SYSTEM_PROMPT_BIGQUERY,
            build_row,
        )

        row = build_row(
            "bq001",
            {"db_id": "my_db", "question": "q"},
            {"ignore_order": True, "condition_cols": []},
            schema="CREATE TABLE t (id INT)",
            gold_result=[[["1"]]],
            engine="bigquery",
        )
        assert row["engine"] == "bigquery"
        assert row["responses_create_params"]["input"][0]["content"] == SYSTEM_PROMPT_BIGQUERY

    def test_build_row_sqlite_default(self):
        from resources_servers.spider2_lite.scripts.prepare_dataset import (
            SYSTEM_PROMPT_SQLITE,
            build_row,
        )

        row = build_row(
            "local001",
            {"db_id": "my_db", "question": "q"},
            {"ignore_order": True, "condition_cols": []},
            schema="CREATE TABLE t (id INT)",
            gold_sql="SELECT 1",
        )
        assert row["engine"] == "sqlite"
        assert row["responses_create_params"]["input"][0]["content"] == SYSTEM_PROMPT_SQLITE

    def test_load_gold_result_keeps_later_letters_when_a_is_missing(self, tmp_path):
        from resources_servers.spider2_lite.scripts.prepare_dataset import load_gold_result

        exec_dir = tmp_path / "evaluation_suite" / "gold" / "exec_result"
        exec_dir.mkdir(parents=True)
        (exec_dir / "bq305_b.csv").write_text("USER_ID\n1\n")
        (exec_dir / "bq305_c.csv").write_text("USER_ID\n2\n")
        (exec_dir / "bq085_c.csv").write_text("country\nSpain\n")
        (exec_dir / "bq001_a.csv").write_text("x\n1\n")
        (exec_dir / "bq001_c.csv").write_text("x\n3\n")
        (exec_dir / "local001.csv").write_text("x\n1\n")
        (exec_dir / "local001_b.csv").write_text("x\n9\n")

        assert load_gold_result(tmp_path, "bq305") == [[[1]], [[2]]]
        assert load_gold_result(tmp_path, "bq085") == [[["Spain"]]]
        assert load_gold_result(tmp_path, "bq001") == [[[1]], [[3]]]
        assert load_gold_result(tmp_path, "local001") == [[[1]]]

    def test_parse_value_rejects_non_finite(self):
        """NaN/Inf in gold_result CSVs must become ``None`` so the emitted
        JSONL is valid for strict parsers (Rust ``serde_json``, etc)."""
        import json

        from resources_servers.spider2_lite.scripts.prepare_dataset import parse_value

        assert parse_value("nan") is None
        assert parse_value("NaN") is None
        assert parse_value("inf") is None
        assert parse_value("-inf") is None
        assert parse_value("Infinity") is None
        # Regular values stay typed as before.
        assert parse_value("1") == 1
        assert parse_value("1.5") == 1.5
        assert parse_value("hello") == "hello"
        # Round-tripping a row with these values must produce strict JSON.
        row = [parse_value(x) for x in ["1.5", "nan", "inf"]]
        encoded = json.dumps(row, allow_nan=False)
        assert encoded == "[1.5, null, null]"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_verify_request(
    text: str, db_id: str = "TestDB", gold_sql: str = "SELECT 1", **kwargs
) -> Spider2LiteVerifyRequest:
    return Spider2LiteVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_make_response(text),
        db_id=db_id,
        question="test question",
        gold_sql=gold_sql,
        **kwargs,
    )


def _make_bq_request(sql: str, **kwargs) -> Spider2LiteVerifyRequest:
    """Build a Spider2LiteVerifyRequest preset for the BigQuery engine."""
    defaults = {
        "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input=[]),
        "response": _make_response(sql),
        "db_id": "any",
        "question": "q",
        "engine": "bigquery",
        "ignore_order": True,
    }
    defaults.update(kwargs)
    return Spider2LiteVerifyRequest(**defaults)


def _make_response(text: str) -> NeMoGymResponse:
    return NeMoGymResponse(
        id="r",
        created_at=0.0,
        model="m",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="msg",
                content=[NeMoGymResponseOutputText(annotations=[], text=text, type="output_text")],
                role="assistant",
                status="completed",
                type="message",
            )
        ],
        parallel_tool_calls=True,
        tool_choice="auto",
        tools=[],
    )


def _make_zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def _write_adc(tmp_path: Path, project: str | None = "canary-project") -> Path:
    info = {"type": "authorized_user", "client_id": "x", "client_secret": "y", "refresh_token": "z"}
    if project is not None:
        info["quota_project_id"] = project
    path = tmp_path / "application_default_credentials.json"
    path.write_text(json.dumps(info))
    return path


class _FakeBigQueryClient:
    def __init__(self, rows=None, result_exc=None, query_exc=None):
        self._rows = rows
        self._result_exc = result_exc
        self._query_exc = query_exc
        self.queries = []

    def query(self, sql):
        self.queries.append(sql)
        if self._query_exc is not None:
            raise self._query_exc
        return _FakeJob(rows=self._rows, result_exc=self._result_exc)


class _FakeJob:
    def __init__(self, rows=None, result_exc=None, query_exc=None):
        self._rows = rows or []
        self._result_exc = result_exc
        self._query_exc = query_exc

    def result(self, timeout=None):
        if self._result_exc is not None:
            raise self._result_exc
        return [_FakeRow(r) for r in self._rows]


class _FakeRow:
    def __init__(self, values):
        self._values = values

    def values(self):
        return self._values


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def server(tmp_path, tiny_db):
    # Pre-populate the sqlite dir so the server's lazy loader finds it already
    # cached and skips the Google Drive download.
    sqlite_dir = tmp_path / ".spider2_lite" / "sqlite"
    sqlite_dir.mkdir(parents=True)
    shutil.copy(tiny_db, sqlite_dir / "TestDB.sqlite")

    cfg = Spider2LiteResourcesServerConfig(
        host="127.0.0.1",
        port=20099,
        entrypoint="",
        spider2_lite_dir=str(tmp_path / ".spider2_lite"),
        # Point at a non-existent file so BigQuery stays unavailable, keeping SQLite tests
        # isolated from any real ADC on the host.
        bigquery_credentials_path=str(tmp_path / "no_such_adc.json"),
    )
    return Spider2LiteResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))


@pytest.fixture
def tiny_db(tmp_path) -> Path:
    """Create a tiny SQLite DB with one table."""
    db = tmp_path / "TestDB.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT, val REAL)")
    conn.executemany("INSERT INTO items VALUES (?,?,?)", [(1, "alpha", 1.0), (2, "beta", 2.0)])
    conn.commit()
    conn.close()
    return db
