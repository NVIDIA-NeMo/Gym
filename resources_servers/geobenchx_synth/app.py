# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NemoGym resources server for synthetic GeoBenchX-style tasks with outcome rewards."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, ClassVar, Literal

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ReverifyMode,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.geobenchx.catalog import TOOL_DEFINITIONS
from resources_servers.geobenchx.upstream_backend import UpstreamToolPool
from resources_servers.geobenchx_synth.catalog import UNUSED_TOOLS, offered_states
from resources_servers.geobenchx_synth.scoring import score
from resources_servers.geobenchx_synth.sql_tool import SQL_TOOL_NAME, SqlSandbox, StateDatabases


WORKER_SCRIPT = Path(__file__).with_name("synth_worker.py")


class GeoBenchXSynthResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    upstream_python: str
    upstream_src: str
    geo_data_path: str
    scratch_path: str
    num_tool_workers: int = 16
    tool_timeout_s: float = 300.0
    max_sessions_per_worker: int = 64
    # run_sql: the output of prepare/build_sql.py. Unset, run_sql reports that no database is available.
    sql_data_path: str | None = None
    # Per-state DuckDB files are built here on first use (default: <scratch_path>/sql).
    sql_db_path: str | None = None
    # DuckDB extension directory holding the spatial extension (default: DuckDB's own).
    sql_extension_dir: str | None = None
    sql_timeout_s: float = 60.0
    sql_max_rows: int = 50


class GeoBenchXSynthVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    task_id: str
    activity: Literal["geobenchx_synth"]
    template: str
    task_kind: Literal["solve", "reject", "control"]
    answers: list[float]
    rel_tol: float
    abs_tol: float


class GeoBenchXSynthVerifyResponse(BaseVerifyResponse):
    answer: float | None
    rejected: bool
    tool_calls: int


class ToolBody(BaseModel):
    model_config = ConfigDict(extra="allow")


class SeedBody(BaseModel):
    model_config = ConfigDict(extra="allow")


class GeoBenchXSynthResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    config: GeoBenchXSynthResourcesServerConfig
    _pool: UpstreamToolPool | None = PrivateAttr(default=None)
    _sql: SqlSandbox | None = PrivateAttr(default=None)
    _entries: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        c = self.config
        catalog = Path(c.geo_data_path) / "catalog.json"
        if not catalog.is_file():
            raise FileNotFoundError(
                f"no catalog.json in geo_data_path {c.geo_data_path}; run prepare/build_catalog.py"
            )
        self._pool = UpstreamToolPool(
            python=c.upstream_python,
            upstream_src=c.upstream_src,
            stat_data_path=c.geo_data_path,
            geo_data_path=c.geo_data_path,
            scratch_path=c.scratch_path,
            num_workers=c.num_tool_workers,
            tool_timeout_s=c.tool_timeout_s,
            max_sessions_per_worker=c.max_sessions_per_worker,
            worker_script=WORKER_SCRIPT,
            worker_args=("--catalog", str(catalog)),
        )
        self._entries = json.loads(catalog.read_text())
        if c.sql_data_path:
            databases = StateDatabases(
                c.sql_data_path, c.sql_db_path or str(Path(c.scratch_path) / "sql"), c.sql_extension_dir
            )
            self._sql = SqlSandbox(databases, timeout_s=c.sql_timeout_s, max_rows=c.sql_max_rows)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        for name in TOOL_DEFINITIONS:
            if name not in UNUSED_TOOLS:
                app.post(f"/{name}")(self._handler(name))
        app.post(f"/{SQL_TOOL_NAME}")(self.run_sql)
        app.router.add_event_handler("shutdown", self._pool.close)
        if self._sql:
            app.router.add_event_handler("shutdown", self._sql.close)
        return app

    async def run_sql(self, request: Request, body: ToolBody) -> dict[str, Any]:
        if self._sql is None:
            return {"result": "Error: no spatial database is available for this task"}
        session_id = request.session.get(SESSION_ID_KEY, "")
        query = (body.model_extra or {}).get("query", "")
        return {"result": await asyncio.to_thread(self._sql.run, session_id, str(query))}

    def _handler(self, tool_name: str):
        async def handler(request: Request, body: ToolBody) -> dict[str, Any]:
            session_id = request.session.get(SESSION_ID_KEY, "")
            out = await self._pool.call(session_id, tool_name, body.model_dump(exclude_none=True))
            return {"result": out["result"]}

        handler.__name__ = f"handle_{tool_name}"
        return handler

    async def seed_session(self, request: Request, body: SeedBody) -> BaseSeedSessionResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id:
            await self._pool.reset(session_id)
            if self._sql:
                task = body.model_extra or {}
                states = offered_states(task.get("state", ""), task.get("split", ""), self._entries)
                await asyncio.to_thread(self._sql.reset, session_id, states)
        return BaseSeedSessionResponse()

    async def verify(self, body: GeoBenchXSynthVerifyRequest) -> GeoBenchXSynthVerifyResponse:
        scored = score(body.model_dump(), body.response.output)
        return GeoBenchXSynthVerifyResponse(**body.model_dump(), **scored)


if __name__ == "__main__":
    GeoBenchXSynthResourcesServer.run_webserver()
