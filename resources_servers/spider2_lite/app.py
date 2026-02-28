# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spider 2.0-Lite execution-based Text-to-SQL resource server."""

import asyncio
import logging
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pydantic import ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.spider2_lite.eval_utils import (
    compare_multi_result_sets,
    execute_and_compare,
    execute_sqlite_async,
)
from resources_servers.spider2_lite.setup_spider2 import ensure_spider2_lite
from resources_servers.text_to_sql.app import extract_sql_from_response


logger = logging.getLogger(__name__)


class FailureCode(str, Enum):
    NONE = "none"
    NO_SQL_EXTRACTED = "no_sql_extracted"
    EXECUTION_ERROR = "execution_error"
    GOLD_EXECUTION_ERROR = "gold_execution_error"
    UNKNOWN_ERROR = "unknown_error"


class Spider2LiteResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "spider2_lite"
    spider2_lite_dir: str = "resources_servers/spider2_lite/.spider2_lite"
    max_concurrency: int = 32
    sql_execution_timeout_s: float = 30.0


class Spider2LiteVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    instance_id: Optional[str] = None
    db_id: str
    question: str

    gold_sql: Optional[str] = None
    gold_result: Optional[list[list[list[Any]]]] = None

    ignore_order: bool = True
    condition_cols: Optional[list] = None
    metadata: Optional[dict[str, Any]] = None


class Spider2LiteVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    instance_id: Optional[str] = None
    db_id: str
    question: str
    model_output: str
    extracted_sql: Optional[str] = None
    execution_match: bool = False
    failure_reason: Optional[FailureCode] = None
    gold_sql: Optional[str] = None


class Spider2LiteResourcesServer(SimpleResourcesServer):
    config: Spider2LiteResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._sqlite_dir: Path = ensure_spider2_lite(self._resolve_spider2_dir())
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)

    def _resolve_spider2_dir(self) -> Path:
        p = Path(self.config.spider2_lite_dir)
        if not p.is_absolute():
            p = Path(__file__).parent.parent.parent / p
        return p

    def _db_path(self, db_id: str) -> Path:
        return self._sqlite_dir / f"{db_id}.sqlite"

    async def verify(self, body: Spider2LiteVerifyRequest) -> Spider2LiteVerifyResponse:
        generated = body.response.output_text or ""
        db_path = self._db_path(body.db_id)

        reward = 0.0
        execution_match = False
        failure_reason = None
        extracted_sql = None

        base_payload = body.model_dump()
        for f in ("db_id", "question", "gold_sql", "gold_result", "ignore_order", "condition_cols"):
            base_payload.pop(f, None)

        def _response(**kwargs):
            return Spider2LiteVerifyResponse(
                **base_payload,
                reward=reward,
                db_id=body.db_id,
                question=body.question,
                model_output=generated,
                execution_match=execution_match,
                gold_sql=body.gold_sql,
                **kwargs,
            )

        if not generated:
            return _response(extracted_sql=None, failure_reason=FailureCode.NO_SQL_EXTRACTED)

        try:
            extracted_sql = extract_sql_from_response(generated)
            if not extracted_sql:
                return _response(extracted_sql=None, failure_reason=FailureCode.NO_SQL_EXTRACTED)

            if body.gold_sql:
                match, _gold, _pred, err = await execute_and_compare(
                    db_path=db_path,
                    gold_sql=body.gold_sql,
                    pred_sql=extracted_sql,
                    semaphore=self._semaphore,
                    condition_cols=body.condition_cols or [],
                    ignore_order=body.ignore_order,
                    timeout_s=self.config.sql_execution_timeout_s,
                )
                if err and err.startswith("gold_sql_error"):
                    failure_reason = FailureCode.GOLD_EXECUTION_ERROR
                elif err:
                    failure_reason = FailureCode.EXECUTION_ERROR
                else:
                    execution_match = match
                    failure_reason = FailureCode.NONE if match else FailureCode.EXECUTION_ERROR

            elif body.gold_result:
                try:
                    pred_rows = await execute_sqlite_async(
                        db_path,
                        extracted_sql,
                        self._semaphore,
                        timeout_s=self.config.sql_execution_timeout_s,
                    )
                    gold_sets = [[tuple(row) for row in gold] for gold in body.gold_result]
                    execution_match = compare_multi_result_sets(
                        gold_sets=gold_sets,
                        pred=pred_rows,
                        multi_condition_cols=body.condition_cols,
                        ignore_order=body.ignore_order,
                    )
                    failure_reason = FailureCode.NONE if execution_match else FailureCode.EXECUTION_ERROR
                except Exception as e:
                    failure_reason = FailureCode.EXECUTION_ERROR
                    logger.warning("pred execution error: %s %s", type(e).__name__, e)
            else:
                raise ValueError("verifier_metadata must contain either 'gold_sql' or 'gold_result'")

            reward = 1.0 if execution_match else 0.0

        except Exception as e:
            failure_reason = FailureCode.UNKNOWN_ERROR
            logger.warning("Unknown error in verify: %s %s", type(e).__name__, e)

        return _response(extracted_sql=extracted_sql, failure_reason=failure_reason)


if __name__ == "__main__":
    Spider2LiteResourcesServer.run_webserver()
