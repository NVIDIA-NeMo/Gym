# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BIRD execution-based Text-to-SQL resource server.

Verifies a model-generated SQL query by executing it against the per-``db_id``
SQLite database from the BIRD dev split, then comparing the result set against
the ground-truth query's result set via unordered set equality (the official
BIRD evaluator's rule).

With ``dialect: spark`` the model is expected to write Spark SQL: it is executed on a local PySpark copy of
the databases (see ``setup_bird_spark.py``) and its result set is compared against the original SQLite
gold query's result set. The gold SQL is never transpiled.

``dialect: postgres`` works the same way against a PostgreSQL server holding one schema per ``db_id`` (see
``setup_bird_postgres.py``); the model's query runs in a read-only transaction with a server-side timeout, on one of
``max_concurrency`` pooled connections.

Only the model's query is scored. When the verifier itself cannot measure it (the
database is missing or corrupt, the gold query cannot run, the executor crashes)
the row keeps the model's answer, is masked and names ``failure_kind``
``verifier_error`` instead of entering the score as a wrong answer.
"""

import asyncio
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Literal, Optional

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
from resources_servers.bird_sql.eval_utils import (
    PostgresPool,
    SparkUnavailableError,
    execute_and_compare,
    execute_and_compare_postgres,
    execute_and_compare_spark,
)
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql, ensure_bird_train


logger = logging.getLogger(__name__)


class FailureCode(str, Enum):
    """Why ``reward`` is what it is. The first five are scored outcomes of the model's query; the
    rest are verifier failures, on which the row is masked and ``failure_kind`` is set."""

    NONE = "none"
    NO_SQL_EXTRACTED = "no_sql_extracted"
    RESULT_MISMATCH = "result_mismatch"
    EXECUTION_ERROR = "execution_error"
    EXECUTION_TIMEOUT = "execution_timeout"
    GOLD_EXECUTION_ERROR = "gold_execution_error"
    GOLD_EXECUTION_TIMEOUT = "gold_execution_timeout"
    DATABASE_MISSING = "database_missing"
    SPARK_UNAVAILABLE = "spark_unavailable"
    UNKNOWN_ERROR = "unknown_error"


def _observe_verifier_failure(reason: str, *, cause: Optional[BaseException] = None) -> None:
    """Record a verifier failure for this request's failure report; the reply row names the same kind."""
    operation_id = current_verifier_invocation_id()
    failure = NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind=VERIFIER_ERROR,
        failure_reason=reason,
        operation="verify",
        operation_id=operation_id,
        stage="verification",
        exception_type=type(cause).__name__ if cause is not None else None,
    )
    failure.__cause__ = cause
    observe_failure(failure)


def extract_sql_from_response(text: Optional[str]) -> Optional[str]:
    """Extract SQL from a model response (CODEBLOCK mode).

    Behavior:
    - No ` ```sql ``` ` block found → return ``None``. The caller scores this
      as a hard 0 without attempting execution (no query to run). The ``sql``
      fence tag is matched case-insensitively (e.g. ` ```SQL ` also matches);
      the captured SQL content's case is left untouched.
    - Multiple blocks → use the LAST one.
    - SQL comments (``--...``, ``/*...*/``) are left as-is: SQLite's parser
      ignores them natively, so stripping them before execution is unnecessary.
      Internal newlines are preserved (only leading/trailing whitespace is
      trimmed) so a ``--`` line comment can't merge onto the same line as
      the SQL that follows it.
    - Drop a leading ``**bold**`` header that some models emit before the query.
    """
    if not text:
        return None

    # \b after "sql" so IGNORECASE doesn't also match unrelated tags that merely start with
    # those letters (```SQLite, ```sqlalchemy, ...).
    opening = re.compile(r"```sql\b", re.IGNORECASE)
    letter = re.compile(r"[a-zA-Z]", re.IGNORECASE)
    last_content = None
    position = 0
    # Match the same non-overlapping spans as the former lazy regex, without
    # backtracking over long unfinished responses. In particular, its required
    # letter can occur after an intervening fence; retain that existing behavior.
    while start := opening.search(text, position):
        first_letter = letter.search(text, start.end())
        if first_letter is None:
            break
        end = text.find("```", first_letter.end())
        if end == -1:
            break
        last_content = text[start.end() : end]
        position = end + 3
    if last_content is None:
        return None

    ans = last_content.strip()
    ans = re.sub(r"^\*\*.*\*\*", "", ans).strip()
    return ans


class BirdSqlResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS
    name: str = "bird_sql"
    bird_sql_dir: str = "resources_servers/bird_sql/.bird_sql"
    dialect: Literal["sqlite", "spark", "postgres"] = "sqlite"
    # Which BIRD split's databases to verify against. ``train`` uses a different db_id set than dev, so pair it with
    # its own ``spark_warehouse_dir`` when ``dialect`` is ``spark``.
    split: Literal["dev", "train"] = "dev"
    spark_warehouse_dir: str = "resources_servers/bird_sql/.bird_sql/spark_warehouse"
    # ``dialect: postgres``: the grader connects with ``postgres_dsn`` (a read-only role). With ``postgres_admin_dsn``
    # set, startup loads any missing db_id and creates that role; without it, the data must already be loaded.
    # Dev and train db_ids differ, so both splits can share one database. Each server process keeps a pool of
    # ``max_concurrency`` connections, so the server needs ``max_connections`` above that times the processes.
    postgres_dsn: str = "postgresql://bird_reader@127.0.0.1:5432/bird"  # password via the DSN or PGPASSWORD
    postgres_admin_dsn: Optional[str] = None
    max_concurrency: int = 32
    sql_execution_timeout_s: float = 30.0


class BirdSqlVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    question: str
    gt_sql: str
    db_id: str
    difficulty: Optional[str] = None
    id: Optional[int] = None


class BirdSqlVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    question: str
    gt_sql: str
    db_id: str
    difficulty: Optional[str] = None
    id: Optional[int] = None
    model_output: str
    extracted_sql: Optional[str] = None
    had_codeblock: bool = False
    execution_match: bool = False
    failure_reason: Optional[FailureCode] = None


# Execution failures that are logged as they happen, with running totals, so a bad run can be spotted and stopped.
_EXECUTION_FAILURE_LABELS = {
    FailureCode.GOLD_EXECUTION_ERROR: "gold-sql-error",
    FailureCode.GOLD_EXECUTION_TIMEOUT: "gold-sql-timeout",
    FailureCode.EXECUTION_ERROR: "pred-sql-error",
    FailureCode.EXECUTION_TIMEOUT: "pred-sql-timeout",
}

# Per-rollout 0/1 scores reported next to accuracy, each over every rollout. Names are the metric names users see.
_FAILURE_SCORE_NAMES = (
    "no_sql",
    "gold-sql-timeout",
    "pred-sql-timeout",
    "sql-timeout",
    "gold-sql-error",
    "pred-sql-error",
    "sql_errors",
    "unclean",
)


class BirdSqlResourcesServer(SimpleResourcesServer):
    config: BirdSqlResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        base_dir = self._resolve_dir(self.config.bird_sql_dir)
        if self.config.split == "train":
            self._databases_dir: Path = ensure_bird_train(base_dir) / "train_databases"
        else:
            self._databases_dir = ensure_bird_sql(base_dir)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self._spark_sessions: Dict[str, Any] = {}
        self._verified_count = 0
        self._failure_counts: Dict[str, int] = {label: 0 for label in _EXECUTION_FAILURE_LABELS.values()}
        if self.config.dialect == "spark":
            # Imported here so the default SQLite path needs neither pandas nor pyspark/JVM.
            from resources_servers.bird_sql.setup_bird_spark import ensure_bird_spark

            self._spark_sessions = ensure_bird_spark(
                self._databases_dir, self._resolve_dir(self.config.spark_warehouse_dir)
            )
        elif self.config.dialect == "postgres":
            # Imported here so the default SQLite path needs neither psycopg nor pandas.
            from resources_servers.bird_sql.setup_bird_postgres import check_bird_postgres, ensure_bird_postgres

            if self.config.postgres_admin_dsn:
                ensure_bird_postgres(self._databases_dir, self.config.postgres_admin_dsn, self.config.postgres_dsn)
            else:
                check_bird_postgres(self._databases_dir, self.config.postgres_dsn)
            self._postgres = PostgresPool(self.config.postgres_dsn, self.config.max_concurrency)

    @staticmethod
    def _resolve_dir(path: str) -> Path:
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).parent.parent.parent / p
        return p

    def _db_path(self, db_id: str) -> Path:
        return self._databases_dir / db_id / f"{db_id}.sqlite"

    async def verify(self, body: BirdSqlVerifyRequest) -> BirdSqlVerifyResponse:
        generated = body.response.output_text or ""
        db_path = self._db_path(body.db_id)

        base_payload = body.model_dump()
        for f in ("question", "gt_sql", "db_id", "difficulty", "id"):
            base_payload.pop(f, None)

        def _response(*, reward: float, execution_match: bool = False, **kwargs) -> BirdSqlVerifyResponse:
            return BirdSqlVerifyResponse(
                **base_payload,
                reward=reward,
                question=body.question,
                gt_sql=body.gt_sql,
                db_id=body.db_id,
                difficulty=body.difficulty,
                id=body.id,
                model_output=generated,
                execution_match=execution_match,
                **kwargs,
            )

        extracted_sql = extract_sql_from_response(generated)

        if extracted_sql is None:
            # No fenced ```sql``` block at all -- nothing to execute. Scored as a hard 0
            # rather than running a filler query like "SELECT 1" against the gold query:
            # that filler always mismatches anyway, so it added execution cost and a
            # misleading EXECUTION_ERROR-shaped failure without changing the reward.
            return observe_programmatic_grading(
                lambda: _response(
                    reward=0.0,
                    extracted_sql=None,
                    failure_reason=FailureCode.NO_SQL_EXTRACTED,
                    had_codeblock=False,
                )
            )

        def _verifier_failure(
            code: FailureCode, reason: str, cause: Optional[BaseException] = None
        ) -> BirdSqlVerifyResponse:
            # The verifier failed, not the model: its query was never measured against the gold
            # result. The row keeps the answer and is masked under a typed kind rather than scored
            # as a wrong answer; the event joins the request's failure report so the collector can
            # group the outage. A stateless reverify can grade the saved answer once the cause is fixed.
            _observe_verifier_failure(reason, cause=cause)
            self._log_execution_failure(body, code)
            return _response(
                reward=0.0,
                mask_sample=True,
                failure_kind=VERIFIER_ERROR,
                failure_reason=code,
                extracted_sql=extracted_sql,
                had_codeblock=True,
            )

        if not db_path.is_file():
            # Checked before connecting: sqlite3.connect would create an empty database at this
            # path and then report the gold query as broken, hiding the missing data behind a
            # per-row execution error.
            logger.error("BIRD verify on id=%s: database %s is missing", body.id, db_path)
            return _verifier_failure(
                FailureCode.DATABASE_MISSING, f"BIRD database for db_id={body.db_id!r} is missing"
            )

        try:
            if self.config.dialect == "spark":
                match, _gold, _pred, err = await execute_and_compare_spark(
                    db_path=db_path,
                    session=self._spark_sessions[body.db_id],
                    gold_sql=body.gt_sql,
                    pred_sql=extracted_sql,
                    semaphore=self._semaphore,
                    timeout_s=self.config.sql_execution_timeout_s,
                )
            elif self.config.dialect == "postgres":
                match, _gold, _pred, err = await execute_and_compare_postgres(
                    db_path=db_path,
                    pool=self._postgres,
                    db_id=body.db_id,
                    gold_sql=body.gt_sql,
                    pred_sql=extracted_sql,
                    semaphore=self._semaphore,
                    timeout_s=self.config.sql_execution_timeout_s,
                )
            else:
                match, _gold, _pred, err = await execute_and_compare(
                    db_path=db_path,
                    gold_sql=body.gt_sql,
                    pred_sql=extracted_sql,
                    semaphore=self._semaphore,
                    timeout_s=self.config.sql_execution_timeout_s,
                )
        except SparkUnavailableError as e:
            logger.error("BIRD verify on id=%s db_id=%s: %s", body.id, body.db_id, e)
            return _verifier_failure(FailureCode.SPARK_UNAVAILABLE, "BIRD Spark session is unavailable", cause=e)
        except Exception as e:
            logger.exception("BIRD verify execution error on id=%s db_id=%s: %s", body.id, body.db_id, e)
            return _verifier_failure(FailureCode.UNKNOWN_ERROR, f"BIRD verifier crashed: {type(e).__name__}", cause=e)

        if err == "gold_sql_error":
            # The ground-truth query could not run against its database (a corrupt or incomplete
            # database, a broken gold query): no model output can be measured until that is fixed.
            return _verifier_failure(
                FailureCode.GOLD_EXECUTION_ERROR, f"BIRD gold SQL failed to execute on db_id={body.db_id!r}"
            )
        if err == "gold_sql_timeout":
            return _verifier_failure(
                FailureCode.GOLD_EXECUTION_TIMEOUT, f"BIRD gold SQL timed out on db_id={body.db_id!r}"
            )

        def _grade() -> BirdSqlVerifyResponse:
            # The model's own query failing or timing out is its answer being wrong, as in the
            # official BIRD evaluator, so these stay scored zeros.
            if err == "pred_sql_error":
                execution_match, failure_reason = False, FailureCode.EXECUTION_ERROR
            elif err == "pred_sql_timeout":
                execution_match, failure_reason = False, FailureCode.EXECUTION_TIMEOUT
            else:
                execution_match = match
                failure_reason = FailureCode.NONE if match else FailureCode.RESULT_MISMATCH
            self._log_execution_failure(body, failure_reason)
            return _response(
                reward=1.0 if execution_match else 0.0,
                execution_match=execution_match,
                extracted_sql=extracted_sql,
                failure_reason=failure_reason,
                had_codeblock=True,
            )

        return observe_programmatic_grading(_grade)

    def _log_execution_failure(self, body: BirdSqlVerifyRequest, failure_reason: FailureCode) -> None:
        """Log each gold/pred SQL error or timeout as it happens, with running totals."""
        self._verified_count += 1
        label = _EXECUTION_FAILURE_LABELS.get(failure_reason)
        if label is None:
            return
        self._failure_counts[label] += 1
        totals = " ".join(f"{name}={count}" for name, count in self._failure_counts.items())
        logger.warning(
            "BIRD %s on id=%s db_id=%s | running totals over %d verified: %s",
            label,
            body.id,
            body.db_id,
            self._verified_count,
            totals,
        )

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, float]:
        """Accuracy plus the failure-breakdown scores for one rollout; every score is 0/1.

        A failed LLM request is not scored here: it aborts the run (``route_failures_to_sidecar`` stays off), so
        every row seen is a completed rollout. A row without a ``failure_reason`` is counted ``unclean``.
        """
        reason = r.get("failure_reason")
        gold_timeout = reason == FailureCode.GOLD_EXECUTION_TIMEOUT
        pred_timeout = reason == FailureCode.EXECUTION_TIMEOUT
        gold_error = reason == FailureCode.GOLD_EXECUTION_ERROR
        pred_error = reason == FailureCode.EXECUTION_ERROR
        scores = {
            "accuracy": r.get("reward", 0.0) > 0,
            "no_sql": reason == FailureCode.NO_SQL_EXTRACTED,
            "gold-sql-timeout": gold_timeout,
            "pred-sql-timeout": pred_timeout,
            "sql-timeout": gold_timeout or pred_timeout,
            "gold-sql-error": gold_error,
            "pred-sql-error": pred_error,
            "sql_errors": gold_error or pred_error,
            # Clean = ran to a verdict: correct, or a well-formed query with a wrong answer.
            "unclean": reason not in (FailureCode.NONE, FailureCode.RESULT_MISMATCH),
        }
        return {name: float(value) for name, value in scores.items()}

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        """BIRD metrics: overall pass@k + per-difficulty (simple/moderate/challenging) pass@k."""
        metrics, *_ = compute_pass_majority_metrics(
            tasks,
            score_fn=self._score_fn,
            answer_key="extracted_sql",
        )
        metrics.update(
            compute_subset_metrics(
                tasks,
                "difficulty",
                self._score_fn,
                "extracted_sql",
            )
        )
        rollouts = [r for task in tasks for r in task]
        scores = [self._score_fn(r) for r in rollouts]
        metrics["rollouts_count"] = len(rollouts)
        for name in _FAILURE_SCORE_NAMES:
            metrics[f"{name}_count"] = int(sum(score[name] for score in scores))
        logger.info(
            "BIRD run breakdown over %d rollouts: %s",
            len(rollouts),
            " ".join(f"{name}={metrics[f'{name}_count']}" for name in _FAILURE_SCORE_NAMES),
        )
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=list(_FAILURE_SCORE_NAMES)))
        for name in ("rollouts", *_FAILURE_SCORE_NAMES):
            if f"{name}_count" in agent_metrics:
                key[f"{name}_count"] = agent_metrics[f"{name}_count"]

        for prefix in {k.split("/pass@")[0] for k in agent_metrics if "/pass@" in k and k[0].islower()}:
            key.update(highest_k_metrics(agent_metrics, f"{prefix}/pass@1[avg-of-{{k}}]", score_names=["accuracy"]))

        return key


if __name__ == "__main__":
    BirdSqlResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = BirdSqlResourcesServer.run_webserver()  # noqa: F401
