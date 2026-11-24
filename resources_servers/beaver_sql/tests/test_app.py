# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import (
    VERIFIER_ERROR,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.verification import exercise_verifier_fixture
from resources_servers.beaver_sql import app as app_module
from resources_servers.beaver_sql.app import (
    VERIFIER_FIXTURE,
    BeaverSqlResourcesServer,
    BeaverSqlResourcesServerConfig,
    BeaverSqlVerifier,
    BeaverSqlVerifyRequest,
    FailureCode,
    extract_sql_from_response,
)
from resources_servers.beaver_sql.ex_acc import CannedRunner


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


def _request(text: str, gt_sql: str = "GOLD", **extra: Any) -> BeaverSqlVerifyRequest:
    return BeaverSqlVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "q"}]),
        response=_make_response(text),
        question="q",
        gt_sql=gt_sql,
        db_id="dw",
        id="t1",
        category="base",
        **extra,
    )


def _verifier(answers: Dict[str, Any]) -> BeaverSqlVerifier:
    verifier = BeaverSqlVerifier()
    verifier._runner = CannedRunner(answers)
    return verifier


def _verify(answers: Dict[str, Any], text: str, **extra: Any):
    return asyncio.run(_verifier(answers).verify(_request(text, **extra)))


def _sql(query: str) -> str:
    return f"Here it is:\n```sql\n{query}\n```"


ONE = (["x"], [(1,)])
EMPTY = (["x"], [])


def test_verifier_fixture() -> None:
    asyncio.run(
        exercise_verifier_fixture(
            VERIFIER_FIXTURE,
            reward_range=(0.0, 1.0),
            higher_is_better=True,
            determinism="unknown",
        )
    )


class TestExtractSql:
    def test_last_block_wins(self) -> None:
        assert extract_sql_from_response("```sql\nSELECT 1\n``` then ```sql\nSELECT 2\n```") == "SELECT 2"

    def test_tag_case_and_bold_header(self) -> None:
        assert extract_sql_from_response("```SQL\n**Query**\nSELECT 1\n```") == "SELECT 1"

    def test_comments_kept(self) -> None:
        assert extract_sql_from_response("```sql\n-- count\nSELECT 1\n```") == "-- count\nSELECT 1"

    @pytest.mark.parametrize(
        "text",
        ["", None, "SELECT 1", "```sqlite\nSELECT 1\n```", "```sql\n123\n```", "```sql\nSELECT 1"],
    )
    def test_no_block(self, text) -> None:
        assert extract_sql_from_response(text) is None

    def test_block_without_a_letter_does_not_swallow_the_next(self) -> None:
        text = "```sql\n123\n```\nThen:\n```sql\nSELECT 1\n```"
        assert extract_sql_from_response(text) == "SELECT 1"
        assert extract_sql_from_response("```sql\nSELECT 1\n```\n```sql\n42\n```") == "SELECT 1"


class TestVerify:
    def test_match(self) -> None:
        r = _verify({"GOLD": ONE, "SELECT 1": (["one"], [(1,)])}, _sql("SELECT 1"))
        assert (r.reward, r.failure_reason, r.gold_result, r.pred_status) == (
            1.0,
            FailureCode.NONE,
            "nonempty_gold",
            "ok",
        )
        assert r.extracted_sql == "SELECT 1" and r.had_codeblock and r.execution_match

    def test_mismatch(self) -> None:
        r = _verify({"GOLD": ONE, "SELECT 2": (["x"], [(2,)])}, _sql("SELECT 2"))
        assert (r.reward, r.failure_reason) == (0.0, FailureCode.RESULT_MISMATCH)

    def test_no_sql_against_nonempty_gold(self) -> None:
        r = _verify({"GOLD": ONE}, "I don't know")
        assert (r.reward, r.failure_reason, r.pred_status, r.had_codeblock) == (
            0.0,
            FailureCode.NO_SQL_EXTRACTED,
            "missing",
            False,
        )

    def test_no_sql_against_empty_gold_scores_one(self) -> None:
        # Officially a missing prediction is an empty result, and two empty results match.
        r = _verify({"GOLD": EMPTY}, "I don't know")
        assert (r.reward, r.failure_reason, r.gold_result, r.pred_status) == (
            1.0,
            FailureCode.NONE,
            "empty_gold",
            "missing",
        )

    @pytest.mark.parametrize(
        "status,code", [("error", FailureCode.EXECUTION_ERROR), ("timeout", FailureCode.EXECUTION_TIMEOUT)]
    )
    def test_failed_prediction(self, status, code) -> None:
        r = _verify({"GOLD": ONE, "BAD": status}, _sql("BAD"))
        assert (r.reward, r.failure_reason, r.pred_status) == (0.0, code, status)
        r = _verify({"GOLD": EMPTY, "BAD": status}, _sql("BAD"))
        assert (r.reward, r.failure_reason, r.pred_status) == (1.0, FailureCode.NONE, status)

    @pytest.mark.parametrize("status", ["timeout"])
    def test_gold_timeout_is_an_empty_result(self, status) -> None:
        r = _verify({"GOLD": status, "SELECT 1": ONE}, _sql("SELECT 1"))
        assert (r.reward, r.gold_status, r.gold_result, r.mask_sample) == (0.0, status, "empty_gold", False)
        r = _verify({"GOLD": status, "EMPTY": EMPTY}, _sql("EMPTY"))
        assert (r.reward, r.failure_reason) == (1.0, FailureCode.NONE)

    def test_unavailable_database_is_masked(self) -> None:
        r = _verify({"GOLD": "unavailable"}, _sql("SELECT 1"))
        assert r.mask_sample and r.failure_kind == VERIFIER_ERROR
        assert (r.reward, r.failure_reason, r.extracted_sql) == (0.0, FailureCode.DATABASE_UNAVAILABLE, "SELECT 1")

    def test_crash_is_masked(self) -> None:
        verifier = BeaverSqlVerifier()
        verifier._runner = MagicMock()
        verifier._runner.run.side_effect = KeyError("boom")
        r = asyncio.run(verifier.verify(_request(_sql("SELECT 1"))))
        assert r.mask_sample and r.failure_kind == VERIFIER_ERROR
        assert r.failure_reason == FailureCode.UNKNOWN_ERROR

    def test_gold_query_error_is_masked(self) -> None:
        r = _verify({"GOLD": "error", "EMPTY": EMPTY}, _sql("EMPTY"))
        assert r.mask_sample and r.failure_kind == VERIFIER_ERROR
        assert (r.reward, r.failure_reason) == (0.0, FailureCode.DATABASE_UNAVAILABLE)

    def test_saved_grading_fields_in_the_request_are_ignored(self) -> None:
        stale = {
            "model_output": "old answer",
            "reward": 1.0,
            "extracted_sql": "SELECT 2",
            "failure_reason": "none",
            "execution_match": True,
            "gold_result": "empty_gold",
            "mask_sample": True,
            "failure_kind": "x",
        }
        r = _verify({"GOLD": ONE, "SELECT 1": ONE}, _sql("SELECT 1"), **stale)
        assert (r.reward, r.extracted_sql, r.gold_result, r.mask_sample) == (1.0, "SELECT 1", "nonempty_gold", False)
        assert r.model_output.startswith("Here it is")

    def test_schema_not_echoed(self) -> None:
        r = _verify({"GOLD": ONE, "SELECT 1": ONE}, _sql("SELECT 1"), sql_context="CREATE TABLE t (x INT);")
        assert "sql_context" not in r.model_dump()

    def test_gold_runs_before_prediction(self) -> None:
        verifier = _verifier({"GOLD": ONE, "SELECT 1": ONE})
        asyncio.run(verifier.verify(_request(_sql("SELECT 1"))))
        assert verifier._runner.calls == [("dw", "GOLD"), ("dw", "SELECT 1")]


def _row(reward: float, gold_result: str, pred_status: str, category: str = "base", **extra: Any) -> Dict:
    return {
        "reward": reward,
        "gold_result": gold_result,
        "pred_status": pred_status,
        "gold_status": extra.pop("gold_status", "ok"),
        "category": category,
        "extracted_sql": extra.pop("extracted_sql", "SELECT 1"),
        **extra,
    }


def _config(**kwargs: Any) -> BeaverSqlResourcesServerConfig:
    return BeaverSqlResourcesServerConfig(host="127.0.0.1", port=20099, entrypoint="", **kwargs)


class TestMetrics:
    def _server(self, monkeypatch) -> BeaverSqlResourcesServer:
        from resources_servers.beaver_sql import setup_beaver_mysql

        monkeypatch.setattr(setup_beaver_mysql, "check_beaver_mysql", lambda *a: None)
        return BeaverSqlResourcesServer(config=_config(), server_client=MagicMock(spec=ServerClient))

    def test_nonempty_gold_and_counts(self, monkeypatch) -> None:
        tasks = [
            [_row(1.0, "nonempty_gold", "ok")],
            [_row(0.0, "nonempty_gold", "error")],
            [_row(1.0, "empty_gold", "missing", category="cte", extracted_sql=None)],
            [_row(0.0, "nonempty_gold", "timeout", gold_status="timeout")],
        ]
        server = self._server(monkeypatch)
        metrics = server.compute_metrics(tasks)
        assert metrics["pass@1[avg-of-1]/accuracy"] == pytest.approx(50.0)
        assert metrics["nonempty_gold/pass@1[avg-of-1]/accuracy"] == pytest.approx(100 / 3)
        assert metrics["empty_gold/pass@1[avg-of-1]/accuracy"] == pytest.approx(100.0)
        assert metrics["cte/pass@1[avg-of-1]/accuracy"] == pytest.approx(100.0)
        assert metrics["rollouts_count"] == 4
        assert (metrics["no_sql_count"], metrics["pred-sql-error_count"], metrics["pred-sql-timeout_count"]) == (
            1,
            1,
            1,
        )
        assert (metrics["gold-sql-timeout_count"], metrics["gold-sql-error_count"], metrics["empty-gold_count"]) == (
            1,
            0,
            1,
        )
        key = server.get_key_metrics({**metrics, "mean/input_tokens": 10.0})
        assert key["mean/input_tokens"] == 10.0
        assert key["pass@1[avg-of-1]/accuracy"] == pytest.approx(50.0)
        assert key["nonempty_gold/pass@1[avg-of-1]/accuracy"] == pytest.approx(100 / 3)
        assert key["no_sql_count"] == 1 and key["rollouts_count"] == 4


class TestStartup:
    def test_checks_and_builds_runner(self, monkeypatch) -> None:
        from resources_servers.beaver_sql import setup_beaver_mysql

        calls = []
        monkeypatch.setattr(setup_beaver_mysql, "ensure_beaver_mysql", lambda *a: calls.append(("ensure", a)))
        monkeypatch.setattr(setup_beaver_mysql, "check_beaver_mysql", lambda *a: calls.append(("check", a)))
        server = BeaverSqlResourcesServer(
            config=_config(mysql_dsn="mysql://r:p@h:1", databases=["dw", "neutron"]),
            server_client=MagicMock(spec=ServerClient),
        )
        assert calls == [("check", ("mysql://r:p@h:1", ["dw", "neutron"], 10))]
        assert isinstance(server._runner, app_module.MySqlRunner)
        assert server._runner.timeout_s == 10.0

    def test_admin_dsn_loads_first(self, monkeypatch) -> None:
        from resources_servers.beaver_sql import setup_beaver_mysql

        calls = []
        monkeypatch.setattr(setup_beaver_mysql, "ensure_beaver_mysql", lambda *a: calls.append("ensure"))
        monkeypatch.setattr(setup_beaver_mysql, "check_beaver_mysql", lambda *a: calls.append("check"))
        BeaverSqlResourcesServer(
            config=_config(mysql_admin_dsn="mysql://root:p@h:1", cache_dir="/tmp/beaver-cache"),
            server_client=MagicMock(spec=ServerClient),
        )
        assert calls == ["ensure", "check"]

    def test_failed_check_stops_startup(self, monkeypatch) -> None:
        from resources_servers.beaver_sql import setup_beaver_mysql

        def _fail(*a):
            raise RuntimeError("BEAVER MySQL is not ready: dw: not loaded")

        monkeypatch.setattr(setup_beaver_mysql, "check_beaver_mysql", _fail)
        with pytest.raises(RuntimeError, match="not ready"):
            BeaverSqlResourcesServer(config=_config(), server_client=MagicMock(spec=ServerClient))

    def test_resolve_dir(self) -> None:
        assert BeaverSqlResourcesServer._resolve_dir("/abs").as_posix() == "/abs"
        assert BeaverSqlResourcesServer._resolve_dir("rel/x").as_posix().endswith("rel/x")
