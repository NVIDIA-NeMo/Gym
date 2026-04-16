# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BIRD execution-based Text-to-SQL resource server.

Verifies a model-generated SQL query by executing it against the per-``db_id``
SQLite database from the BIRD dev split, then comparing the result set against
the ground-truth query's result set (unordered set equality — matches the
official BIRD evaluator in NeMo Skills).
"""

import asyncio
import logging
import re
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import (
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)
from resources_servers.bird_sql.eval_utils import execute_and_compare
from resources_servers.bird_sql.setup_bird_sql import ensure_bird_sql


logger = logging.getLogger(__name__)


class FailureCode(str, Enum):
    NONE = "none"
    NO_SQL_EXTRACTED = "no_sql_extracted"
    EXECUTION_ERROR = "execution_error"
    GOLD_EXECUTION_ERROR = "gold_execution_error"
    UNKNOWN_ERROR = "unknown_error"


_NO_ANSWER_FILLER = "SELECT 1"


def has_sql_codeblock(text: Optional[str]) -> bool:
    """True iff the response contains a ` ```sql ... ``` ` block with at least one letter."""
    if not text:
        return False
    return bool(re.search(r"(?:```sql)(.*?[a-zA-Z].*?)(?:```)", text, flags=re.DOTALL))


def extract_sql_from_response(text: Optional[str]) -> str:
    """Extract SQL from a model response — exact port of the Skills BIRD evaluator.

    Port of ``BirdEvaluator._extract_answer`` with ``answer_format="CODEBLOCK"``
    from ``nemo_skills/evaluation/evaluator/bird.py``. Contract matches Skills:

    - No ` ```sql ``` ` block found → return ``"SELECT 1"`` (no-op filler that
      executes harmlessly but mismatches almost any BIRD gold query).
    - Multiple blocks → use the LAST one.
    - Strip SQL comments (``--...``, ``/*...*/``) via Skills' exact regex with
      ``re.DOTALL``; note that ``--.*?$`` without ``re.MULTILINE`` eats to
      end-of-string, so a comment line inside an otherwise valid block can
      swallow the rest — replicated intentionally for parity.
    - Collapse whitespace and drop a leading ``**bold**`` header.
    - Post-strip empty output is returned as ``""`` (Skills returns the empty
      string; downstream execution then fails and the reward is 0).
    """
    if not text:
        return _NO_ANSWER_FILLER

    matches = re.findall(r"(?:```sql)(.*?[a-zA-Z].*?)(?:```)", text, flags=re.DOTALL)
    if not matches:
        return _NO_ANSWER_FILLER

    ans = matches[-1]
    ans = re.sub(r"--.*?$|/\*.*?\*/", "", ans, flags=re.DOTALL)
    ans = re.sub(r"\s+", " ", ans)
    ans = re.sub(r"^\*\*.*\*\*", "", ans).strip()
    return ans


class BirdSqlResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "bird_sql"
    bird_sql_dir: str = "resources_servers/bird_sql/.bird_sql"
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


class BirdSqlResourcesServer(SimpleResourcesServer):
    config: BirdSqlResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._dev_databases_dir: Path = ensure_bird_sql(self._resolve_bird_sql_dir())
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

    def _resolve_bird_sql_dir(self) -> Path:
        p = Path(self.config.bird_sql_dir)
        if not p.is_absolute():
            p = Path(__file__).parent.parent.parent / p
        return p

    def _db_path(self, db_id: str) -> Path:
        return self._dev_databases_dir / db_id / f"{db_id}.sqlite"

    async def verify(self, body: BirdSqlVerifyRequest) -> BirdSqlVerifyResponse:
        generated = body.response.output_text or ""
        db_path = self._db_path(body.db_id)

        reward = 0.0
        execution_match = False

        base_payload = body.model_dump()
        for f in ("question", "gt_sql", "db_id", "difficulty", "id"):
            base_payload.pop(f, None)

        def _response(**kwargs) -> BirdSqlVerifyResponse:
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

        # Skills' extractor always returns a string: either the parsed SQL,
        # the empty string (when comment stripping ate everything), or the
        # filler ``"SELECT 1"`` when no fenced block was found. We mirror
        # that exactly and let execution decide the reward.
        extracted_sql = extract_sql_from_response(generated)
        had_codeblock = has_sql_codeblock(generated)

        try:
            match, _gold, _pred, err = await execute_and_compare(
                db_path=db_path,
                gold_sql=body.gt_sql,
                pred_sql=extracted_sql,
                semaphore=self._semaphore,
                timeout_s=self.config.sql_execution_timeout_s,
            )
        except Exception as e:
            logger.exception("BIRD verify execution error on id=%s db_id=%s: %s", body.id, body.db_id, e)
            return _response(
                extracted_sql=extracted_sql,
                failure_reason=FailureCode.UNKNOWN_ERROR,
                had_codeblock=had_codeblock,
            )

        if err == "gold_sql_error":
            failure_reason = FailureCode.GOLD_EXECUTION_ERROR
        elif err == "pred_sql_error":
            failure_reason = FailureCode.NO_SQL_EXTRACTED if not had_codeblock else FailureCode.EXECUTION_ERROR
        else:
            execution_match = match
            if match:
                failure_reason = FailureCode.NONE
            else:
                failure_reason = FailureCode.NO_SQL_EXTRACTED if not had_codeblock else FailureCode.EXECUTION_ERROR

        reward = 1.0 if execution_match else 0.0

        return _response(
            extracted_sql=extracted_sql,
            failure_reason=failure_reason,
            had_codeblock=had_codeblock,
        )

    @staticmethod
    def _score_fn(r: dict) -> Dict[str, float]:
        return {"accuracy": float(r.get("reward", 0.0) > 0)}

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
        return metrics

    def get_key_metrics(self, agent_metrics: Dict[str, Any]) -> Dict[str, Any]:
        key: Dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]

        key.update(highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=["accuracy"]))
        key.update(highest_k_metrics(agent_metrics, "pass@{k}", score_names=["accuracy"]))

        for prefix in {k.split("/pass@")[0] for k in agent_metrics if "/pass@" in k and k[0].islower()}:
            key.update(highest_k_metrics(agent_metrics, f"{prefix}/pass@1[avg-of-{{k}}]", score_names=["accuracy"]))

        return key


if __name__ == "__main__":
    BirdSqlResourcesServer.run_webserver()
