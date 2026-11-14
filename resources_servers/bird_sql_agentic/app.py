# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-turn BIRD text-to-SQL resource server with a read-only ``execute_sql`` tool.

Same data and same reward as ``bird_sql``; the only difference is that the model can run
read-only queries against the task's database before giving its final answer.

Flow of one rollout (legacy ``/run`` path used by ``simple_agent``):

1. ``simple_agent`` POSTs the whole task row to ``/seed_session``. We remember which
   ``db_id`` belongs to this session (the session id lives in a cookie).
2. Each time the model calls ``execute_sql``, ``simple_agent`` POSTs ``{"sql": ...}`` to
   ``/execute_sql`` with the same cookie. We look up the session's ``db_id`` -- the model
   never chooses the database.
3. At the end, ``/verify`` scores the last ```sql``` block exactly like ``bird_sql``.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Union

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.client import SESSION_ID_KEY
from nemo_gym.server_runtime import (
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.bird_sql.app import (
    BirdSqlResourcesServer,
    BirdSqlResourcesServerConfig,
    BirdSqlVerifyRequest,
    BirdSqlVerifyResponse,
)
from resources_servers.bird_sql_agentic.sql_tool import run_readonly_query


TOOL_NAME = "execute_sql"
_DB_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


class BirdSqlAgenticResourcesServerConfig(BirdSqlResourcesServerConfig):
    name: str = "bird_sql_agentic"
    max_result_rows: int = 20
    max_cell_chars: int = 100


class BirdSqlAgenticSeedSessionRequest(BaseSeedSessionRequest):
    """The legacy seed body is the whole task row; we only need ``db_id`` from it."""

    model_config = ConfigDict(extra="allow")

    db_id: str


class ExecuteSqlRequest(BaseModel):
    sql: str = Field(description="One read-only SQL query (SELECT, WITH, or PRAGMA table_info).")


class ExecuteSqlResponse(BaseModel):
    columns: List[str] = Field(default_factory=list)
    rows: List[List[Any]] = Field(default_factory=list)
    total_rows: int = 0
    truncated: bool = False
    error: Optional[str] = None


class BirdSqlAgenticVerifyResponse(BirdSqlVerifyResponse):
    num_tool_calls: int = 0
    num_sql_errors: int = 0


class BirdSqlAgenticResourcesServer(BirdSqlResourcesServer):
    config: BirdSqlAgenticResourcesServerConfig
    session_id_to_db_id: Dict[str, str] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post(f"/{TOOL_NAME}")(self.execute_sql)
        return app

    def _bind_db(self, session_id: str, db_id: str) -> None:
        if not _DB_ID_PATTERN.match(db_id):
            raise ResourcesSeedRejectedError(f"Invalid db_id: {db_id!r}")
        self.session_id_to_db_id[session_id] = db_id

    async def seed_session(
        self,
        request: Request,
        body: Union[ResourcesSeedSessionRequest, BirdSqlAgenticSeedSessionRequest],
    ) -> Union[ResourcesSeedSessionResponse, BaseSeedSessionResponse]:
        if isinstance(body, ResourcesSeedSessionRequest):
            # Environment Server path: task fields arrive under task_data.
            request.session[SESSION_ID_KEY] = body.resources_session_id
            self._bind_db(body.resources_session_id, str(body.task_data["db_id"]))
            return ResourcesSeedSessionResponse(resources_session_id=body.resources_session_id)

        self._bind_db(request.session[SESSION_ID_KEY], body.db_id)
        return BaseSeedSessionResponse()

    async def execute_sql(self, request: Request, body: ExecuteSqlRequest) -> ExecuteSqlResponse:
        db_id = self.session_id_to_db_id.get(request.session[SESSION_ID_KEY])
        if db_id is None:
            return ExecuteSqlResponse(error="No database is bound to this session.")

        timeout_s = self.config.sql_execution_timeout_s
        async with self._semaphore:
            try:
                # run_readonly_query stops the query itself at timeout_s; the outer wait_for is a backstop.
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        run_readonly_query,
                        self._db_path(db_id),
                        body.sql,
                        timeout_s,
                        self.config.max_result_rows,
                        self.config.max_cell_chars,
                    ),
                    timeout=timeout_s + 5,
                )
            except asyncio.TimeoutError:
                result = {"error": f"Query timed out after {timeout_s:g} seconds."}
        return ExecuteSqlResponse(**result)

    async def verify(self, body: BirdSqlVerifyRequest) -> BirdSqlAgenticVerifyResponse:
        # Scoring is exactly bird_sql's: last ```sql``` block, executed and compared with the gold SQL.
        base = await super().verify(body)

        num_tool_calls = 0
        num_sql_errors = 0
        for item in body.response.output:
            if item.type == "function_call" and item.name == TOOL_NAME:
                num_tool_calls += 1
            elif item.type == "function_call_output":
                try:
                    if json.loads(item.output).get("error"):
                        num_sql_errors += 1
                except (json.JSONDecodeError, AttributeError, TypeError):
                    num_sql_errors += 1

        return BirdSqlAgenticVerifyResponse(
            **base.model_dump(),
            num_tool_calls=num_tool_calls,
            num_sql_errors=num_sql_errors,
        )

    def compute_metrics(self, tasks: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
        metrics = super().compute_metrics(tasks)
        rollouts = [r for task in tasks for r in task]
        if rollouts:
            for key in ("num_tool_calls", "num_sql_errors"):
                metrics[f"mean/{key}"] = sum(r.get(key, 0) for r in rollouts) / len(rollouts)
        return metrics


if __name__ == "__main__":
    BirdSqlAgenticResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = BirdSqlAgenticResourcesServer.run_webserver()
