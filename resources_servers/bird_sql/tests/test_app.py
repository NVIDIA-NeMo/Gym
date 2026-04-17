# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import io
import sqlite3
import zipfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.server_utils import ServerClient
from resources_servers.bird_sql.app import (
    _NO_ANSWER_FILLER,
    BirdSqlResourcesServer,
    BirdSqlResourcesServerConfig,
    BirdSqlVerifyRequest,
    FailureCode,
    extract_sql_from_response,
    has_sql_codeblock,
)
from resources_servers.bird_sql.eval_utils import (
    execute_and_compare,
    execute_sqlite,
    execute_sqlite_async,
    result_sets_match,
)
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


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
    text: str,
    db_id: str = "TestDB",
    gt_sql: str = "SELECT 1",
    difficulty: str = "simple",
    id_: int = 0,
    **kwargs,
) -> BirdSqlVerifyRequest:
    return BirdSqlVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]),
        response=_make_response(text),
        question="test question",
        gt_sql=gt_sql,
        db_id=db_id,
        difficulty=difficulty,
        id=id_,
        **kwargs,
    )


@pytest.fixture
def tiny_db(tmp_path) -> Path:
    """Create a tiny SQLite DB with one table."""
    db_dir = tmp_path / "TestDB"
    db_dir.mkdir()
    db = db_dir / "TestDB.sqlite"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.executemany("INSERT INTO t VALUES (?)", [(1,), (2,), (3,)])
    conn.commit()
    conn.close()
    return db


@pytest.fixture
def server_with_mocked_dbs(tmp_path, monkeypatch):
    """Build a BirdSqlResourcesServer whose DB root points at tmp_path (no download)."""
    monkeypatch.setattr(
        "resources_servers.bird_sql.app.ensure_bird_sql",
        lambda _path: tmp_path,
    )
    config = BirdSqlResourcesServerConfig(
        host="127.0.0.1",
        port=20099,
        entrypoint="",
        bird_sql_dir=str(tmp_path),
        max_concurrency=4,
        sql_execution_timeout_s=10.0,
    )
    server = BirdSqlResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return server


# ---------------------------------------------------------------------------
# extract_sql_from_response
# ---------------------------------------------------------------------------


class TestExtractSqlFromResponse:
    """Strict parity tests against NeMo Skills ``BirdEvaluator._extract_answer``."""

    def test_empty_returns_filler(self):
        # Skills returns the "SELECT 1" no-op filler whenever extraction fails.
        assert extract_sql_from_response("") == _NO_ANSWER_FILLER
        assert extract_sql_from_response(None) == _NO_ANSWER_FILLER

    def test_no_codeblock_returns_filler(self):
        # Raw SQL without ```sql fences is NOT extracted — CODEBLOCK mode only.
        assert extract_sql_from_response("SELECT * FROM t;") == _NO_ANSWER_FILLER

    def test_single_codeblock(self):
        text = "Let me think...\n```sql\nSELECT * FROM t\n```"
        assert extract_sql_from_response(text) == "SELECT * FROM t"

    def test_last_codeblock_wins(self):
        text = "```sql\nSELECT 1\n```\nthen\n```sql\nSELECT 2\n```"
        assert extract_sql_from_response(text) == "SELECT 2"

    def test_inline_block_comment_stripped(self):
        text = "```sql\nSELECT 1 /* inline */ FROM t\n```"
        assert extract_sql_from_response(text) == "SELECT 1 FROM t"

    def test_line_comment_eats_to_eos(self):
        # Faithful port of Skills' `--.*?$` regex with DOTALL and no MULTILINE.
        # A `--` line inside the block swallows the rest of the string — this
        # is a Skills "bug" we replicate for parity.
        text = "```sql\n-- comment\nSELECT 1 FROM t\n```"
        assert extract_sql_from_response(text) == ""

    def test_strips_leading_bold_header(self):
        # Skills' `^\*\*.*\*\*` requires the ** at the very start — after the
        # preceding whitespace collapse that means no space-before-** either.
        text = "```sql**Answer** SELECT 1 FROM t```"
        assert extract_sql_from_response(text) == "SELECT 1 FROM t"

    def test_bold_header_not_stripped_when_preceded_by_whitespace(self):
        # Faithful to Skills: newline before ** means the anchor mismatches
        # and the bold marker survives.
        text = "```sql\n**Answer** SELECT 1 FROM t\n```"
        assert extract_sql_from_response(text) == "**Answer** SELECT 1 FROM t"

    def test_requires_alpha_inside_block(self):
        # Skills regex requires at least one a-z letter inside the fenced block.
        text = "```sql\n12345\n```"
        assert extract_sql_from_response(text) == _NO_ANSWER_FILLER


class TestHasSqlCodeblock:
    def test_present(self):
        assert has_sql_codeblock("```sql\nSELECT 1\n```") is True

    def test_absent(self):
        assert has_sql_codeblock("just prose") is False
        assert has_sql_codeblock("") is False
        assert has_sql_codeblock(None) is False

    def test_requires_letter(self):
        assert has_sql_codeblock("```sql\n12345\n```") is False


# ---------------------------------------------------------------------------
# result_sets_match
# ---------------------------------------------------------------------------


class TestResultSetsMatch:
    def test_empty_both(self):
        assert result_sets_match([], []) is True

    def test_exact_match(self):
        assert result_sets_match([(1,), (2,)], [(1,), (2,)]) is True

    def test_unordered_match(self):
        assert result_sets_match([(1,), (2,)], [(2,), (1,)]) is True

    def test_duplicate_rows_collapsed(self):
        # BIRD uses set equality — duplicates don't matter.
        assert result_sets_match([(1,), (1,), (2,)], [(1,), (2,)]) is True

    def test_different(self):
        assert result_sets_match([(1,)], [(2,)]) is False

    def test_unhashable_fallback(self):
        # Lists are unhashable — fallback path (sorted repr) must still decide.
        a = [([1, 2],)]
        b = [([1, 2],)]
        assert result_sets_match(a, b) is True

    def test_fallback_failure_returns_false(self):
        # An object whose repr() raises drives both the set() path (unhashable)
        # and the sorted(map(repr, ...)) path (repr raises) to fail,
        # exercising the final "except Exception: return False" arm.
        class _BadRepr:
            __hash__ = None  # unhashable → triggers the TypeError fallback

            def __repr__(self):
                raise RuntimeError("repr boom")

        a = [(_BadRepr(),)]
        b = [(_BadRepr(),)]
        assert result_sets_match(a, b) is False


# ---------------------------------------------------------------------------
# execute_sqlite (sync) against a real DB
# ---------------------------------------------------------------------------


class TestExecuteSqlite:
    def test_returns_rows(self, tiny_db):
        assert execute_sqlite(tiny_db, "SELECT x FROM t ORDER BY x") == [(1,), (2,), (3,)]

    def test_syntax_error_returns_none(self, tiny_db):
        assert execute_sqlite(tiny_db, "SELEC BROKEN") is None

    def test_missing_table_returns_none(self, tiny_db):
        assert execute_sqlite(tiny_db, "SELECT * FROM does_not_exist") is None


# ---------------------------------------------------------------------------
# execute_sqlite_async and execute_and_compare (asyncio.to_thread)
# ---------------------------------------------------------------------------


def _mock_execute_sqlite(monkeypatch, gold_result, pred_result):
    """Patch eval_utils.execute_sqlite to return canned results based on SQL string."""
    calls = []

    def fake_execute(db_path, sql):
        calls.append(sql)
        return gold_result if "GOLD_SQL" in sql else pred_result

    monkeypatch.setattr("resources_servers.bird_sql.eval_utils.execute_sqlite", fake_execute)
    return calls


@pytest.mark.asyncio
async def test_execute_sqlite_async_success(monkeypatch):
    _mock_execute_sqlite(monkeypatch, gold_result=[(1,)], pred_result=[(1,)])
    sem = asyncio.Semaphore(4)
    rows = await execute_sqlite_async(Path("/x.sqlite"), "SELECT 1 -- PRED", sem, timeout_s=5)
    assert rows == [(1,)]


@pytest.mark.asyncio
async def test_execute_sqlite_async_timeout(monkeypatch):
    """A slow execute trips the timeout path and returns None."""
    import time

    def slow_execute(db_path, sql):
        time.sleep(10)
        return [(1,)]

    monkeypatch.setattr("resources_servers.bird_sql.eval_utils.execute_sqlite", slow_execute)

    sem = asyncio.Semaphore(4)
    rows = await execute_sqlite_async(Path("/x.sqlite"), "SELECT 1", sem, timeout_s=0.05)
    assert rows is None


@pytest.mark.asyncio
async def test_execute_and_compare_match(monkeypatch):
    _mock_execute_sqlite(monkeypatch, gold_result=[(1,), (2,)], pred_result=[(2,), (1,)])
    sem = asyncio.Semaphore(4)
    match, gold, pred, err = await execute_and_compare(
        Path("/x.sqlite"), gold_sql="SELECT GOLD_SQL", pred_sql="SELECT 1", semaphore=sem, timeout_s=5
    )
    assert match is True and err is None


@pytest.mark.asyncio
async def test_execute_and_compare_pred_error(monkeypatch):
    _mock_execute_sqlite(monkeypatch, gold_result=[(1,)], pred_result=None)
    sem = asyncio.Semaphore(4)
    match, _gold, pred, err = await execute_and_compare(
        Path("/x.sqlite"), gold_sql="SELECT GOLD_SQL", pred_sql="BROKEN", semaphore=sem, timeout_s=5
    )
    assert match is False and err == "pred_sql_error" and pred is None


@pytest.mark.asyncio
async def test_execute_and_compare_gold_error(monkeypatch):
    _mock_execute_sqlite(monkeypatch, gold_result=None, pred_result=[(1,)])
    sem = asyncio.Semaphore(4)
    match, gold, pred, err = await execute_and_compare(
        Path("/x.sqlite"), gold_sql="SELECT GOLD_SQL", pred_sql="SELECT 1", semaphore=sem, timeout_s=5
    )
    assert match is False and err == "gold_sql_error" and gold is None and pred is None


# ---------------------------------------------------------------------------
# BirdSqlResourcesServer.verify end-to-end (mock execute_and_compare)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_verify_no_codeblock_filler_runs(server_with_mocked_dbs, monkeypatch):
    """No ```sql``` block → Skills returns "SELECT 1" filler, which executes
    and mismatches almost any GT. We still record NO_SQL_EXTRACTED so the
    diagnostic is preserved, but the reward matches Skills."""

    async def fake(*, pred_sql, gold_sql, **_kw):
        assert pred_sql == _NO_ANSWER_FILLER, "Should have used the Skills filler"
        return False, [(42,)], [(1,)], None

    monkeypatch.setattr("resources_servers.bird_sql.app.execute_and_compare", fake)

    body = _make_verify_request("the model produced only prose, no SQL fence")
    resp = await server_with_mocked_dbs.verify(body)
    assert resp.reward == 0.0
    assert resp.execution_match is False
    assert resp.had_codeblock is False
    assert resp.extracted_sql == _NO_ANSWER_FILLER
    assert resp.failure_reason == FailureCode.NO_SQL_EXTRACTED


@pytest.mark.asyncio
async def test_verify_empty_response_filler_runs(server_with_mocked_dbs, monkeypatch):
    async def fake(*, pred_sql, **_kw):
        assert pred_sql == _NO_ANSWER_FILLER
        return False, [(42,)], [(1,)], None

    monkeypatch.setattr("resources_servers.bird_sql.app.execute_and_compare", fake)

    body = _make_verify_request("")
    resp = await server_with_mocked_dbs.verify(body)
    assert resp.reward == 0.0
    assert resp.had_codeblock is False
    assert resp.failure_reason == FailureCode.NO_SQL_EXTRACTED


@pytest.mark.asyncio
async def test_verify_match(server_with_mocked_dbs, monkeypatch):
    async def fake(**_kw):
        return True, [(1,)], [(1,)], None

    monkeypatch.setattr("resources_servers.bird_sql.app.execute_and_compare", fake)

    body = _make_verify_request("```sql\nSELECT x FROM t\n```", gt_sql="SELECT x FROM t")
    resp = await server_with_mocked_dbs.verify(body)
    assert resp.reward == 1.0
    assert resp.execution_match is True
    assert resp.failure_reason == FailureCode.NONE
    assert resp.extracted_sql == "SELECT x FROM t"
    assert resp.had_codeblock is True


@pytest.mark.asyncio
async def test_verify_mismatch_with_codeblock(server_with_mocked_dbs, monkeypatch):
    async def fake(**_kw):
        return False, [(1,)], [(2,)], None

    monkeypatch.setattr("resources_servers.bird_sql.app.execute_and_compare", fake)

    body = _make_verify_request("```sql\nSELECT 2\n```")
    resp = await server_with_mocked_dbs.verify(body)
    assert resp.reward == 0.0
    assert resp.execution_match is False
    assert resp.had_codeblock is True
    assert resp.failure_reason == FailureCode.EXECUTION_ERROR


@pytest.mark.asyncio
async def test_verify_pred_sql_error_with_codeblock(server_with_mocked_dbs, monkeypatch):
    async def fake(**_kw):
        return False, [(1,)], None, "pred_sql_error"

    monkeypatch.setattr("resources_servers.bird_sql.app.execute_and_compare", fake)

    body = _make_verify_request("```sql\nBROKEN SQL\n```")
    resp = await server_with_mocked_dbs.verify(body)
    assert resp.reward == 0.0
    assert resp.failure_reason == FailureCode.EXECUTION_ERROR


@pytest.mark.asyncio
async def test_verify_gold_sql_error(server_with_mocked_dbs, monkeypatch):
    async def fake(**_kw):
        return False, None, None, "gold_sql_error"

    monkeypatch.setattr("resources_servers.bird_sql.app.execute_and_compare", fake)

    body = _make_verify_request("```sql\nSELECT 1\n```")
    resp = await server_with_mocked_dbs.verify(body)
    assert resp.reward == 0.0
    assert resp.failure_reason == FailureCode.GOLD_EXECUTION_ERROR


@pytest.mark.asyncio
async def test_verify_unknown_exception(server_with_mocked_dbs, monkeypatch):
    async def fake(**_kw):
        raise RuntimeError("boom")

    monkeypatch.setattr("resources_servers.bird_sql.app.execute_and_compare", fake)

    body = _make_verify_request("```sql\nSELECT 1\n```")
    resp = await server_with_mocked_dbs.verify(body)
    assert resp.reward == 0.0
    assert resp.failure_reason == FailureCode.UNKNOWN_ERROR


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestComputeMetrics:
    @staticmethod
    def _task_result(reward: float, difficulty: str = "simple", extracted: str = "SELECT 1") -> dict:
        return {
            "reward": reward,
            "difficulty": difficulty,
            "extracted_sql": extracted,
        }

    def test_overall_and_subset(self, server_with_mocked_dbs):
        # Two tasks × two rollouts each.
        tasks = [
            # Task A (simple): [pass, fail]
            [
                self._task_result(1.0, "simple"),
                self._task_result(0.0, "simple"),
            ],
            # Task B (challenging): [fail, pass]
            [
                self._task_result(0.0, "challenging"),
                self._task_result(1.0, "challenging"),
            ],
        ]
        metrics = server_with_mocked_dbs.compute_metrics(tasks)

        # Overall pass@1[avg-of-2]/accuracy = avg of per-task means = (0.5 + 0.5) / 2 = 0.5 → 50.0
        assert metrics["pass@1[avg-of-2]/accuracy"] == pytest.approx(50.0)
        # Per-difficulty subset metrics must be present.
        assert "simple/pass@1[avg-of-2]/accuracy" in metrics
        assert "challenging/pass@1[avg-of-2]/accuracy" in metrics
        assert metrics["simple/pass@1[avg-of-2]/accuracy"] == pytest.approx(50.0)
        assert metrics["challenging/pass@1[avg-of-2]/accuracy"] == pytest.approx(50.0)

    def test_all_correct(self, server_with_mocked_dbs):
        tasks = [[self._task_result(1.0, "moderate")], [self._task_result(1.0, "moderate")]]
        metrics = server_with_mocked_dbs.compute_metrics(tasks)
        assert metrics["pass@1[avg-of-1]/accuracy"] == pytest.approx(100.0)

    def test_all_incorrect(self, server_with_mocked_dbs):
        tasks = [[self._task_result(0.0, "simple")], [self._task_result(0.0, "simple")]]
        metrics = server_with_mocked_dbs.compute_metrics(tasks)
        assert metrics["pass@1[avg-of-1]/accuracy"] == pytest.approx(0.0)

    def test_get_key_metrics_filters(self, server_with_mocked_dbs):
        agent_metrics = {
            "pass@1[avg-of-4]/accuracy": 60.0,
            "pass@4/accuracy": 80.0,
            "mean/output_tokens": 4000,
            "simple/pass@1[avg-of-4]/accuracy": 70.0,
            "challenging/pass@1[avg-of-4]/accuracy": 40.0,
            "unrelated": 0,
        }
        key = server_with_mocked_dbs.get_key_metrics(agent_metrics)
        assert "pass@1[avg-of-4]/accuracy" in key
        assert "pass@4/accuracy" in key
        assert "mean/output_tokens" in key
        assert "simple/pass@1[avg-of-4]/accuracy" in key
        assert "challenging/pass@1[avg-of-4]/accuracy" in key
        assert "unrelated" not in key


# ---------------------------------------------------------------------------
# setup_bird_sql.ensure_bird_sql
# ---------------------------------------------------------------------------


def _make_fake_bird_archive() -> bytes:
    """Build the nested zip layout BIRD publishes: dev.zip contains dev_20240627/,
    which contains dev.json + dev_databases.zip, which in turn contains
    per-db_id subdirs with <db_id>.sqlite files.
    """
    inner_buf = io.BytesIO()
    with zipfile.ZipFile(inner_buf, "w") as inner:
        inner.writestr("dev_databases/TestDB/TestDB.sqlite", b"\x00SQLITE3")
        inner.writestr("dev_databases/OtherDB/OtherDB.sqlite", b"\x00SQLITE3")
    inner_bytes = inner_buf.getvalue()

    outer_buf = io.BytesIO()
    with zipfile.ZipFile(outer_buf, "w") as outer:
        outer.writestr("dev_20240627/dev.json", "[]")
        outer.writestr("dev_20240627/dev_databases.zip", inner_bytes)
    return outer_buf.getvalue()


class _FakeResponse:
    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0

    def read(self, n: int = -1) -> bytes:
        if n < 0 or self._offset + n >= len(self._payload):
            chunk = self._payload[self._offset :]
            self._offset = len(self._payload)
            return chunk
        chunk = self._payload[self._offset : self._offset + n]
        self._offset += n
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class TestEnsureBirdSql:
    def test_downloads_and_extracts(self, tmp_path, monkeypatch):
        archive = _make_fake_bird_archive()
        monkeypatch.setattr(
            "resources_servers.bird_sql.setup_bird_sql.urllib.request.urlopen",
            lambda *_a, **_kw: _FakeResponse(archive),
        )

        out = ensure_bird_sql(tmp_path)

        assert out == tmp_path / "dev_20240627" / "dev_databases"
        assert (out / "TestDB" / "TestDB.sqlite").exists()
        assert (out / "OtherDB" / "OtherDB.sqlite").exists()
        assert (tmp_path / "dev_20240627" / "dev.json").exists()

    def test_skips_download_when_already_extracted(self, tmp_path, monkeypatch):
        # Pre-create one .sqlite file — ensure_bird_sql must short-circuit.
        dbs = tmp_path / "dev_20240627" / "dev_databases" / "TestDB"
        dbs.mkdir(parents=True)
        (dbs / "TestDB.sqlite").write_bytes(b"\x00")

        def _boom(*_a, **_kw):
            raise AssertionError("should not download when already present")

        monkeypatch.setattr("resources_servers.bird_sql.setup_bird_sql.urllib.request.urlopen", _boom)
        out = ensure_bird_sql(tmp_path)
        assert out == tmp_path / "dev_20240627" / "dev_databases"

    def test_default_dir_used_when_none_passed(self, tmp_path, monkeypatch):
        from resources_servers.bird_sql import setup_bird_sql

        monkeypatch.setattr(setup_bird_sql, "_DEFAULT_DIR", tmp_path / "default")
        monkeypatch.setattr(
            setup_bird_sql.urllib.request,
            "urlopen",
            lambda *_a, **_kw: _FakeResponse(_make_fake_bird_archive()),
        )
        out = setup_bird_sql.ensure_bird_sql()
        assert out == tmp_path / "default" / "dev_20240627" / "dev_databases"

    def test_missing_inner_zip_raises(self, tmp_path, monkeypatch):
        # Outer zip with no dev_databases.zip inside → ensure_bird_sql must raise.
        bad_outer = io.BytesIO()
        with zipfile.ZipFile(bad_outer, "w") as zf:
            zf.writestr("dev_20240627/dev.json", "[]")
        monkeypatch.setattr(
            "resources_servers.bird_sql.setup_bird_sql.urllib.request.urlopen",
            lambda *_a, **_kw: _FakeResponse(bad_outer.getvalue()),
        )
        with pytest.raises(RuntimeError, match="Expected nested archive"):
            ensure_bird_sql(tmp_path)

    def test_missing_dev_databases_dir_after_extract_raises(self, tmp_path, monkeypatch):
        # Inner zip is empty of any dev_databases/ entries → post-extract check fails.
        inner_empty = io.BytesIO()
        with zipfile.ZipFile(inner_empty, "w") as zf:
            zf.writestr("stray.txt", "")
        outer = io.BytesIO()
        with zipfile.ZipFile(outer, "w") as zf:
            zf.writestr("dev_20240627/dev.json", "[]")
            zf.writestr("dev_20240627/dev_databases.zip", inner_empty.getvalue())
        monkeypatch.setattr(
            "resources_servers.bird_sql.setup_bird_sql.urllib.request.urlopen",
            lambda *_a, **_kw: _FakeResponse(outer.getvalue()),
        )
        with pytest.raises(RuntimeError, match="dev_databases directory missing"):
            ensure_bird_sql(tmp_path)
