# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BEAVER execution-based text-to-SQL resources server.

Runs the model's MySQL query and the gold query on BEAVER's MySQL databases and scores them with BEAVER's official
execution accuracy (``ex_acc.py``): 10 seconds per query, a failed, timed-out or missing query is an empty result,
and two results match when both are empty or their rows, as stripped text, are equal sets.

Every task is scored, including one whose gold query fails (an empty gold result, as officially). Each row records
``gold_result`` (``empty_gold`` or ``nonempty_gold``), and the metrics report accuracy over all tasks and over
``nonempty_gold`` tasks: an empty gold result is matched by any query that returns nothing, so BEAVER publishes both.

When the grader can't reach the data (no MySQL, wrong password, a database not loaded) the row is masked with
``failure_kind`` ``verifier_error`` instead of entering the score. Startup checks that every configured database is
loaded and visible to the grader's user, so a setup gap fails there rather than scoring every query empty.
"""

import asyncio
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import ConfigDict

from nemo_gym.core import (
    VERIFIER_ERROR,
    FailureBoundary,
    NgFailureEvent,
    current_verifier_invocation_id,
    observe_failure,
)
from nemo_gym.metrics import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)
from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
    observe_programmatic_grading,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.verification import VerifierFixture
from resources_servers.beaver_sql.ex_acc import (
    CONNECTION_TIMEOUT_S,
    QUERY_TIMEOUT_S,
    CannedRunner,
    MySqlRunner,
    MySqlUnavailableError,
    missing_prediction,
    results_match,
)


logger = logging.getLogger(__name__)


class FailureCode(str, Enum):
    """Why ``reward`` is what it is. The first five are scored outcomes of the model's query; the rest are verifier
    failures, on which the row is masked and ``failure_kind`` is set."""

    NONE = "none"
    NO_SQL_EXTRACTED = "no_sql_extracted"
    RESULT_MISMATCH = "result_mismatch"
    EXECUTION_ERROR = "execution_error"
    EXECUTION_TIMEOUT = "execution_timeout"
    DATABASE_UNAVAILABLE = "database_unavailable"
    UNKNOWN_ERROR = "unknown_error"


def extract_sql_from_response(text: Optional[str]) -> Optional[str]:
    """The SQL in the last complete ```` ```sql ```` block that contains a letter, or ``None`` when there is none.

    As ``bird_sql``'s ``extract_sql_from_response``, the fence tag is matched case-insensitively, comments are left in,
    and a leading ``**bold**`` header is dropped. Unlike it, each block ends at its own closing fence before the letter
    check, so a block without a letter (```` ```sql 123``` ````) can't swallow the text after it.
    """
    if not text:
        return None
    opening = re.compile(r"```sql\b", re.IGNORECASE)
    last_content = None
    position = 0
    while start := opening.search(text, position):
        end = text.find("```", start.end())
        if end == -1:
            break
        content = text[start.end() : end]
        if re.search(r"[a-zA-Z]", content):
            last_content = content
        position = end + 3
    if last_content is None:
        return None
    return re.sub(r"^\*\*.*\*\*", "", last_content.strip()).strip()


class BeaverSqlResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    name: str = "beaver_sql"
    # The grader's (read-only) user; the database is chosen per task. Password in the DSN; pass it with an
    # `${oc.env:...}` override so it stays off the command line.
    mysql_dsn: str = "mysql://beaver_reader@127.0.0.1:3306"
    # With an admin DSN, startup downloads the pinned dumps, loads any missing database and creates the grader's user.
    mysql_admin_dsn: Optional[str] = None
    # Databases checked (and, with mysql_admin_dsn, loaded) at startup. dw_real questions use dw.
    databases: List[str] = ["dw"]
    cache_dir: str = "resources_servers/beaver_sql/.beaver"
    max_concurrency: int = 8
    query_timeout_s: float = QUERY_TIMEOUT_S
    connection_timeout_s: int = CONNECTION_TIMEOUT_S


class BeaverSqlVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    question: str
    gt_sql: str
    db_id: str
    id: Optional[str] = None
    category: Optional[str] = None


class BeaverSqlVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    question: str
    gt_sql: str
    db_id: str
    id: Optional[str] = None
    category: Optional[str] = None
    model_output: str
    extracted_sql: Optional[str] = None
    had_codeblock: bool = False
    execution_match: bool = False
    failure_reason: Optional[FailureCode] = None
    # "ok", "error" or "timeout"; a failed gold query is scored as an empty result.
    gold_status: Optional[str] = None
    gold_result: Optional[str] = None  # "empty_gold" or "nonempty_gold"
    pred_status: Optional[str] = None  # "ok", "error", "timeout" or "missing"


def _observe_verifier_failure(reason: str, *, cause: Optional[BaseException] = None) -> None:
    """Record a verifier failure for this request's failure report; the reply row names the same kind."""
    failure = NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind=VERIFIER_ERROR,
        failure_reason=reason,
        operation="verify",
        operation_id=current_verifier_invocation_id(),
        stage="verification",
        exception_type=type(cause).__name__ if cause is not None else None,
    )
    failure.__cause__ = cause
    observe_failure(failure)


# Response fields this verifier sets; an incoming request's copies are ignored.
_COMPUTED_FIELDS = (
    "reward",
    "model_output",
    "extracted_sql",
    "had_codeblock",
    "execution_match",
    "failure_reason",
    "failure_kind",
    "mask_sample",
    "gold_status",
    "gold_result",
    "pred_status",
)

# Per-rollout 0/1 scores reported next to accuracy, each over every rollout. Names are the metric names users see.
_FAILURE_SCORE_NAMES = (
    "no_sql",
    "pred-sql-error",
    "pred-sql-timeout",
    "gold-sql-error",
    "gold-sql-timeout",
    "empty-gold",
)


class BeaverSqlVerifier:
    """Scores one model answer with ``self._runner`` (a ``MySqlRunner``, or a ``CannedRunner`` in tests). Holds no
    state between requests."""

    _runner: Any

    async def verify(self, body: BeaverSqlVerifyRequest) -> BeaverSqlVerifyResponse:
        generated = body.response.output_text or ""
        payload = body.model_dump()
        # The schema is in the prompt already; a second copy per rollout only makes the files bigger. Fields this verifier
        # computes are dropped too, so a re-verified row that still carries them is graded afresh.
        for f in ("question", "gt_sql", "db_id", "id", "category", "sql_context", *_COMPUTED_FIELDS):
            payload.pop(f, None)

        def _response(**kwargs: Any) -> BeaverSqlVerifyResponse:
            return BeaverSqlVerifyResponse(
                **payload,
                question=body.question,
                gt_sql=body.gt_sql,
                db_id=body.db_id,
                id=body.id,
                category=body.category,
                model_output=generated,
                **kwargs,
            )

        extracted_sql = extract_sql_from_response(generated)

        def _verifier_failure(code: FailureCode, reason: str, cause: BaseException) -> BeaverSqlVerifyResponse:
            # The verifier failed, not the model: the row keeps the answer and is masked, and a stateless reverify can
            # grade the saved answer once the cause is fixed.
            _observe_verifier_failure(reason, cause=cause)
            logger.error("BEAVER verify on id=%s db_id=%s: %s: %s", body.id, body.db_id, reason, cause)
            return _response(
                reward=0.0,
                mask_sample=True,
                failure_kind=VERIFIER_ERROR,
                failure_reason=code,
                extracted_sql=extracted_sql,
                had_codeblock=extracted_sql is not None,
            )

        try:
            gold = await self._runner.run(body.db_id, body.gt_sql)
            if gold.status == "error":
                # Every gold query runs on a correctly set-up server (benchmarks/beaver/gold_check.py), so one that
                # fails here means the data or server isn't what was checked. (Officially it would be an empty result,
                # which an empty or failed answer matches.) A gold query that times out is still scored as empty.
                raise MySqlUnavailableError(f"gold query failed: {gold.error}")
            pred = missing_prediction() if extracted_sql is None else await self._runner.run(body.db_id, extracted_sql)
            # Off the event loop: a large result takes a while to compare.
            match = await asyncio.to_thread(results_match, gold, pred)
        except MySqlUnavailableError as e:
            return _verifier_failure(FailureCode.DATABASE_UNAVAILABLE, "BEAVER MySQL is unavailable", e)
        except Exception as e:
            return _verifier_failure(FailureCode.UNKNOWN_ERROR, f"BEAVER verifier crashed: {type(e).__name__}", e)

        def _grade() -> BeaverSqlVerifyResponse:
            # A missing or failed prediction is an empty result, so it matches an empty gold result: then the row
            # reads NONE, and pred_status still says what happened to the query.
            if match:
                reason = FailureCode.NONE
            elif extracted_sql is None:
                reason = FailureCode.NO_SQL_EXTRACTED
            elif pred.status == "error":
                reason = FailureCode.EXECUTION_ERROR
            elif pred.status == "timeout":
                reason = FailureCode.EXECUTION_TIMEOUT
            else:
                reason = FailureCode.RESULT_MISMATCH
            return _response(
                reward=1.0 if match else 0.0,
                execution_match=match,
                extracted_sql=extracted_sql,
                had_codeblock=extracted_sql is not None,
                failure_reason=reason,
                gold_status=gold.status,
                gold_result="empty_gold" if gold.empty else "nonempty_gold",
                pred_status=pred.status,
            )

        return observe_programmatic_grading(_grade)


class BeaverSqlResourcesServer(BeaverSqlVerifier, SimpleResourcesServer):
    config: BeaverSqlResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # Imported here so the unit tests and the verifier fixture need no MySQL driver.
        from resources_servers.beaver_sql.setup_beaver_mysql import check_beaver_mysql, ensure_beaver_mysql

        if self.config.mysql_admin_dsn:
            ensure_beaver_mysql(
                self._resolve_dir(self.config.cache_dir),
                self.config.databases,
                self.config.mysql_admin_dsn,
                self.config.mysql_dsn,
            )
        check_beaver_mysql(self.config.mysql_dsn, self.config.databases, self.config.connection_timeout_s)
        self._runner = MySqlRunner(
            self.config.mysql_dsn,
            max_concurrency=self.config.max_concurrency,
            timeout_s=self.config.query_timeout_s,
            connection_timeout_s=self.config.connection_timeout_s,
        )

    @staticmethod
    def _resolve_dir(path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).parent.parent.parent / p
        return p

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, float]:
        """Accuracy plus the failure-breakdown scores for one rollout; every score is 0/1."""
        scores = {
            "accuracy": r.get("reward", 0.0) > 0,
            "no_sql": r.get("pred_status") == "missing",
            "pred-sql-error": r.get("pred_status") == "error",
            "pred-sql-timeout": r.get("pred_status") == "timeout",
            "gold-sql-error": r.get("gold_status") == "error",
            "gold-sql-timeout": r.get("gold_status") == "timeout",
            "empty-gold": r.get("gold_result") == "empty_gold",
        }
        return {name: float(value) for name, value in scores.items()}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """pass@k over all tasks, and per ``gold_result`` (BEAVER's accuracy without empty gold) and ``category``."""
        metrics, *_ = compute_pass_majority_metrics(tasks, score_fn=self._score_fn, answer_key="extracted_sql")
        for field in ("gold_result", "category"):
            metrics.update(compute_subset_metrics(tasks, field, self._score_fn, "extracted_sql"))
        rollouts = [r for task in tasks for r in task]
        scores = [self._score_fn(r) for r in rollouts]
        metrics["rollouts_count"] = len(rollouts)
        for name in _FAILURE_SCORE_NAMES:
            metrics[f"{name}_count"] = int(sum(score[name] for score in scores))
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "nonempty_gold/pass@1[avg-of-{k}]", score_names=["accuracy"]))
        for name in ("rollouts", *_FAILURE_SCORE_NAMES):
            if f"{name}_count" in agent_metrics:
                key[f"{name}_count"] = agent_metrics[f"{name}_count"]
        return key


def _fixture_verifier() -> BeaverSqlVerifier:
    """A verifier whose 'database' is a table of canned results, for ``gym env test``."""
    verifier = BeaverSqlVerifier()
    verifier._runner = CannedRunner(
        {
            "SELECT 1": (["1"], [(1,)]),
            "SELECT 1 AS one": (["one"], [(1,)]),
            "SELECT 2": (["2"], [(2,)]),
            "SELECT 1 FROM DUAL WHERE FALSE": (["1"], []),
        }
    )
    return verifier


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=_fixture_verifier,
    request_model=BeaverSqlVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    BeaverSqlResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = BeaverSqlResourcesServer.run_webserver()  # noqa: F401
