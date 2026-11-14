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
from resources_servers.geobenchx_synth.catalog import UNUSED_TOOLS
from resources_servers.geobenchx_synth.scoring import score


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

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        for name in TOOL_DEFINITIONS:
            if name not in UNUSED_TOOLS:
                app.post(f"/{name}")(self._handler(name))
        app.router.add_event_handler("shutdown", self._pool.close)
        return app

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
        return BaseSeedSessionResponse()

    async def verify(self, body: GeoBenchXSynthVerifyRequest) -> GeoBenchXSynthVerifyResponse:
        scored = score(body.model_dump(), body.response.output)
        return GeoBenchXSynthVerifyResponse(**body.model_dump(), **scored)


if __name__ == "__main__":
    GeoBenchXSynthResourcesServer.run_webserver()
