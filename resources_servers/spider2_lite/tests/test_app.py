# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import io
import sqlite3
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
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
    execute_sqlite,
    execute_sqlite_async,
)
from resources_servers.text_to_sql.app import extract_sql_from_response


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


@pytest.fixture
def server(tmp_path, tiny_db):
    sqlite_dir = tmp_path / ".spider2_lite" / "sqlite"
    sqlite_dir.mkdir(parents=True)
    # Move the tiny db into the sqlite_dir
    import shutil

    shutil.copy(tiny_db, sqlite_dir / "TestDB.sqlite")

    with patch("resources_servers.spider2_lite.app.ensure_spider2_lite", return_value=sqlite_dir):
        cfg = Spider2LiteResourcesServerConfig(
            host="127.0.0.1",
            port=20099,
            entrypoint="",
            spider2_lite_dir=str(tmp_path / ".spider2_lite"),
        )
        srv = Spider2LiteResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))
    return srv


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
        with pytest.raises(sqlite3.Error):
            execute_sqlite(tiny_db, "NOT VALID SQL")

    def test_execute_sqlite_async_ok(self, tiny_db):
        sem = asyncio.Semaphore(1)
        rows = asyncio.get_event_loop().run_until_complete(
            execute_sqlite_async(tiny_db, "SELECT id FROM items ORDER BY id", sem)
        )
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
        # Compare all cols — should fail
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
        match, gold, pred, err = asyncio.get_event_loop().run_until_complete(
            execute_and_compare(tiny_db, "SELECT id FROM items ORDER BY id", "SELECT id FROM items ORDER BY id", sem)
        )
        assert match is True
        assert err is None

    def test_execute_and_compare_mismatch(self, tiny_db):
        sem = asyncio.Semaphore(4)
        match, gold, pred, err = asyncio.get_event_loop().run_until_complete(
            execute_and_compare(tiny_db, "SELECT id FROM items", "SELECT name FROM items", sem)
        )
        assert match is False
        assert err is None

    def test_execute_and_compare_gold_error(self, tiny_db):
        sem = asyncio.Semaphore(4)
        match, gold, pred, err = asyncio.get_event_loop().run_until_complete(
            execute_and_compare(tiny_db, "INVALID SQL", "SELECT 1", sem)
        )
        assert match is False
        assert err is not None and "gold_sql_error" in err

    def test_execute_and_compare_pred_error(self, tiny_db):
        sem = asyncio.Semaphore(4)
        match, gold, pred, err = asyncio.get_event_loop().run_until_complete(
            execute_and_compare(tiny_db, "SELECT id FROM items", "INVALID SQL", sem)
        )
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
        assert result.failure_reason == FailureCode.GOLD_EXECUTION_ERROR

    async def test_verify_no_gold_data(self, server):
        req = Spider2LiteVerifyRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
            response=_make_response("```sql\nSELECT 1;\n```"),
            db_id="TestDB",
            question="q",
        )
        result = await server.verify(req)
        assert result.reward == 0.0
        assert result.failure_reason == FailureCode.UNKNOWN_ERROR

    async def test_verify_ignore_order(self, server):
        # Gold returns rows in a different order than pred — should still match with ignore_order=True
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


def _make_zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


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
