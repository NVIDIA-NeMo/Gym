# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spider 2.0-Lite execution-based Text-to-SQL resource server."""

import asyncio
import json
import logging
import os
import re
import threading
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import ConfigDict

from nemo_gym.server_runtime import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.spider2_lite.eval_utils import (
    compare_multi_result_sets,
    execute_and_compare,
    execute_bigquery_async,
    execute_sqlite_async,
)
from resources_servers.spider2_lite.setup_spider2 import ensure_spider2_lite


logger = logging.getLogger(__name__)


def extract_sql_from_response(text: str) -> Optional[str]:
    """Extract SQL query from model response.

    Attempts to extract SQL in the following order:
    1. SQL wrapped in ```sql ... ``` code blocks
    2. SQL wrapped in ``` ... ``` code blocks
    3. Raw SQL statements starting with SELECT/INSERT/UPDATE/DELETE/WITH

    Returns:
        Extracted SQL query or None if no SQL found.
    """
    if not text:
        return None

    sql_block_pattern = r"```sql\s*([\s\S]*?)\s*```"
    matches = re.findall(sql_block_pattern, text, re.IGNORECASE)
    if matches:
        return matches[-1].strip()

    generic_block_pattern = r"```\s*([\s\S]*?)\s*```"
    matches = re.findall(generic_block_pattern, text)
    if matches:
        for match in reversed(matches):
            content = match.strip()
            if re.match(r"^\s*(SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP)\s", content, re.IGNORECASE):
                return content

    sql_start = re.search(r"(?:SELECT|INSERT|UPDATE|DELETE|WITH)\s", text, re.IGNORECASE)
    if sql_start:
        last_semicolon = text.rfind(";")
        if last_semicolon >= sql_start.start():
            extracted = text[sql_start.start() : last_semicolon + 1].strip()
        else:
            extracted = text[sql_start.start() :].strip()
        return extracted.rstrip(";") + ";"

    return None


class FailureCode(str, Enum):
    NONE = "none"
    NO_SQL_EXTRACTED = "no_sql_extracted"
    EXECUTION_ERROR = "execution_error"
    GOLD_EXECUTION_ERROR = "gold_execution_error"
    BIGQUERY_UNAVAILABLE = "bigquery_unavailable"
    UNKNOWN_ERROR = "unknown_error"


class Spider2LiteResourcesServerConfig(BaseResourcesServerConfig):
    name: str = "spider2_lite"
    spider2_lite_dir: str = "resources_servers/spider2_lite/.spider2_lite"
    sqlite_max_concurrency: int = 32
    sqlite_execution_timeout_s: float = 30.0
    bigquery_credentials_path: str = "~/.config/gcloud/application_default_credentials.json"
    bigquery_project: Optional[str] = None
    bigquery_execution_timeout_s: float = 120.0
    bigquery_max_concurrency: int = 8


class Spider2LiteVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    uuid: Optional[str | int] = None
    instance_id: Optional[str] = None
    db_id: str
    question: str
    engine: Literal["sqlite", "bigquery"] = "sqlite"

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
    engine: Literal["sqlite", "bigquery"] = "sqlite"
    model_output: str
    extracted_sql: Optional[str] = None
    execution_match: bool = False
    failure_reason: Optional[FailureCode] = None
    gold_sql: Optional[str] = None


class Spider2LiteResourcesServer(SimpleResourcesServer):
    ray_enabled = True
    config: Spider2LiteResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._sqlite_dir: Optional[Path] = None
        self._sqlite_lock = threading.Lock()
        self._semaphore = asyncio.Semaphore(self.config.sqlite_max_concurrency)
        self._bigquery_client: Any = None
        self._bigquery_init_error: Optional[str] = None
        self._bigquery_lock = threading.Lock()
        self._bigquery_semaphore = asyncio.Semaphore(self.config.bigquery_max_concurrency)

    def _resolve_spider2_dir(self) -> Path:
        p = Path(self.config.spider2_lite_dir)
        if not p.is_absolute():
            p = Path(__file__).parent.parent.parent / p
        return p

    def _db_path(self, db_id: str) -> Path:
        return self._ensure_sqlite_dir() / f"{db_id}.sqlite"

    def _ensure_sqlite_dir(self) -> Path:
        """Download and extract the Spider 2.0-Lite SQLite databases on first use."""
        if self._sqlite_dir is not None:
            return self._sqlite_dir
        with self._sqlite_lock:
            if self._sqlite_dir is None:
                self._sqlite_dir = ensure_spider2_lite(self._resolve_spider2_dir())
        return self._sqlite_dir

    def _resolve_credentials_path(self) -> Path:
        return Path(os.path.expanduser(self.config.bigquery_credentials_path))

    def _get_bigquery_client(self) -> tuple[Any, Optional[str]]:
        """Return a cached BigQuery client, or ``(None, error_message)``. In-band so one bad task can't kill the server."""
        cached = self._cached_bigquery_result()
        if cached is not None:
            return cached

        with self._bigquery_lock:
            cached = self._cached_bigquery_result()
            if cached is not None:
                return cached

            bigquery, err = self._import_bigquery()
            if err:
                return self._record_bigquery_disabled(err)

            cred_path = self._resolve_credentials_path()
            if not cred_path.is_file():
                return self._record_bigquery_disabled(
                    f"BigQuery credentials file not found: {cred_path}. "
                    "Run `gcloud auth application-default login` first."
                )

            project, err = self._resolve_bigquery_project(cred_path)
            if err:
                return self._record_bigquery_disabled(err)

            os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", str(cred_path))
            try:
                self._bigquery_client = bigquery.Client(project=project)
            except Exception as exc:
                return self._record_bigquery_disabled(f"Failed to construct BigQuery client: {exc}")

            logger.info("BigQuery client ready for project %s", project)
        return self._bigquery_client, None

    def _cached_bigquery_result(self) -> Optional[tuple[Any, Optional[str]]]:
        """Return the cached (client, None) or (None, error) if init already settled, else None."""
        if self._bigquery_client is not None:
            return self._bigquery_client, None
        if self._bigquery_init_error is not None:
            return None, self._bigquery_init_error
        return None

    def _record_bigquery_disabled(self, message: str) -> tuple[None, str]:
        """Store, log, and return a BigQuery init failure."""
        self._bigquery_init_error = message
        logger.warning("BigQuery disabled: %s", message)
        return None, message

    def _import_bigquery(self) -> tuple[Any, Optional[str]]:
        try:
            from google.cloud import bigquery
        except ImportError as exc:
            return (
                None,
                f"google-cloud-bigquery is not installed ({exc}). Add it to the environment to run BigQuery tasks.",
            )
        return bigquery, None

    def _resolve_bigquery_project(self, cred_path: Path) -> tuple[Optional[str], Optional[str]]:
        if self.config.bigquery_project:
            return self.config.bigquery_project, None
        try:
            with open(cred_path) as fh:
                info = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            return None, f"Could not read BigQuery credentials from {cred_path}: {exc}"
        project = info.get("quota_project_id") or info.get("project_id")
        if not project:
            return None, (
                f"No quota_project_id or project_id in {cred_path}. "
                "Run `gcloud auth application-default set-quota-project <PROJECT_ID>` "
                "or set `bigquery_project` in the server config."
            )
        return project, None

    async def verify(self, body: Spider2LiteVerifyRequest) -> Spider2LiteVerifyResponse:
        generated = body.response.output_text or ""
        engine = body.engine

        reward = 0.0
        execution_match = False
        failure_reason = None
        extracted_sql = None

        base_payload = body.model_dump()
        for f in ("db_id", "question", "engine", "gold_sql", "gold_result", "ignore_order", "condition_cols"):
            base_payload.pop(f, None)

        def _response(**kwargs):
            return Spider2LiteVerifyResponse(
                **base_payload,
                reward=reward,
                db_id=body.db_id,
                question=body.question,
                engine=engine,
                model_output=generated,
                execution_match=execution_match,
                gold_sql=body.gold_sql,
                **kwargs,
            )

        if not generated:
            return _response(extracted_sql=None, failure_reason=FailureCode.NO_SQL_EXTRACTED)

        extracted_sql = extract_sql_from_response(generated)
        if not extracted_sql:
            return _response(extracted_sql=None, failure_reason=FailureCode.NO_SQL_EXTRACTED)

        if engine == "bigquery":
            execution_match, failure_reason = await self._verify_bigquery(body, extracted_sql)
            reward = 1.0 if execution_match else 0.0
            return _response(extracted_sql=extracted_sql, failure_reason=failure_reason)

        db_path = self._db_path(body.db_id)

        if body.gold_sql:
            match, _gold, _pred, err = await execute_and_compare(
                db_path=db_path,
                gold_sql=body.gold_sql,
                pred_sql=extracted_sql,
                semaphore=self._semaphore,
                condition_cols=body.condition_cols or [],
                ignore_order=body.ignore_order,
                timeout_s=self.config.sqlite_execution_timeout_s,
            )
            if err:
                failure_reason = FailureCode.EXECUTION_ERROR
            else:
                execution_match = match
                failure_reason = FailureCode.NONE if match else FailureCode.EXECUTION_ERROR

        elif body.gold_result:
            pred_rows = await execute_sqlite_async(
                db_path,
                extracted_sql,
                self._semaphore,
                timeout_s=self.config.sqlite_execution_timeout_s,
            )
            if pred_rows is None:
                execution_match = False
                failure_reason = FailureCode.EXECUTION_ERROR
            else:
                gold_sets = [[tuple(row) for row in gold] for gold in body.gold_result]
                execution_match = compare_multi_result_sets(
                    gold_sets=gold_sets,
                    pred=pred_rows,
                    multi_condition_cols=body.condition_cols,
                    ignore_order=body.ignore_order,
                )
                failure_reason = FailureCode.NONE if execution_match else FailureCode.EXECUTION_ERROR
        else:
            raise ValueError("verifier_metadata must contain either 'gold_sql' or 'gold_result'")

        reward = 1.0 if execution_match else 0.0

        return _response(extracted_sql=extracted_sql, failure_reason=failure_reason)

    async def _verify_bigquery(
        self,
        body: Spider2LiteVerifyRequest,
        extracted_sql: str,
    ) -> tuple[bool, FailureCode]:
        """Score a BigQuery row. Returns (execution_match, failure_reason)."""
        if not body.gold_result:
            raise ValueError("BigQuery tasks require 'gold_result'; running gold SQL on BigQuery is not supported.")
        client, err = self._get_bigquery_client()
        if client is None:
            logger.warning("Skipping BigQuery verification for %s: %s", body.instance_id, err)
            return False, FailureCode.BIGQUERY_UNAVAILABLE
        pred_rows = await execute_bigquery_async(
            client,
            extracted_sql,
            self._bigquery_semaphore,
            timeout_s=self.config.bigquery_execution_timeout_s,
        )
        if pred_rows is None:
            return False, FailureCode.EXECUTION_ERROR
        gold_sets = [[tuple(row) for row in gold] for gold in body.gold_result]
        match = compare_multi_result_sets(
            gold_sets=gold_sets,
            pred=pred_rows,
            multi_condition_cols=body.condition_cols,
            ignore_order=body.ignore_order,
        )
        return match, FailureCode.NONE if match else FailureCode.EXECUTION_ERROR


if __name__ == "__main__":
    Spider2LiteResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # With num_workers > 1, uvicorn imports this module in each worker and serves its module-level `app`.
    app = Spider2LiteResourcesServer.run_webserver()  # noqa: F401
