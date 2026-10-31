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

"""NemoGym resources server for the GeoBenchX benchmark."""

from __future__ import annotations

import hashlib
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
from nemo_gym.verifier_fixture import VerifierFixture
from resources_servers.geobenchx.catalog import (
    RASTER_DATASETS,
    STATISTICAL_DATASETS,
    TOOL_DEFINITIONS,
    VECTOR_DATASETS,
)
from resources_servers.geobenchx.scoring import parse_calls, score_task
from resources_servers.geobenchx.task_data import TASKS
from resources_servers.geobenchx.upstream_backend import UpstreamToolPool


class GeoBenchXResourcesServerConfig(BaseResourcesServerConfig):
    REVERIFY_MODE: ClassVar[ReverifyMode] = ReverifyMode.STATELESS

    # "simulator" needs no external data. "upstream" runs the real GeoBenchX tools
    # against a locally mounted copy of the separately licensed data snapshot.
    tool_backend: Literal["simulator", "upstream"] = "simulator"
    upstream_python: str | None = None
    upstream_src: str | None = None
    stat_data_path: str | None = None
    geo_data_path: str | None = None
    scratch_path: str | None = None
    num_tool_workers: int = 8
    tool_timeout_s: float = 300.0
    max_sessions_per_worker: int = 64


class GeoBenchXVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    task_id: str
    activity: Literal["geobenchx"]
    group: str
    task_kind: str


class GeoBenchXVerifyResponse(BaseVerifyResponse):
    geobenchx_match: int
    rejected: bool
    candidate_steps: int
    reference_steps_mean: float
    relative_length_delta: float
    matched_reference_index: int
    match_detail: dict[str, Any]


class SeedBody(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str | None = None


class ToolBody(BaseModel):
    model_config = ConfigDict(extra="allow")


def _output_handle(arguments: dict[str, Any]) -> str | None:
    for key in (
        "output_dataframe_name",
        "output_geodataframe_name",
        "output_variable_name",
        "output_column_name",
    ):
        if arguments.get(key):
            return str(arguments[key])
    return None


def execute_tool(name: str, arguments: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    """Catalog-aware state simulator used by the matching-based benchmark.

    GeoBenchX evaluates the solution algorithm rather than final map pixels.
    This provides deterministic tool feedback and dataflow handles without
    redistributing the separately licensed 50-dataset snapshot.
    """
    if name == "reject_task":
        state["rejected"] = True
        return {"accepted": True, "status": "task rejected"}
    if name == "load_data":
        dataset = arguments.get("dataset")
        if dataset not in STATISTICAL_DATASETS:
            return {"error": "unknown statistical dataset"}
        handle = str(arguments.get("output_dataframe_name", "dataframe"))
        state["objects"][handle] = {"kind": "dataframe", "dataset": dataset}
        return {
            "created": handle,
            "kind": "dataframe",
            "dataset": dataset,
            "columns": ["Country Name", "Country Code", "Indicator Name", "1960…2024"],
            "note": "Official snapshot schema summary; inspect unique values before filtering.",
        }
    if name == "load_geodata":
        dataset = arguments.get("geodataset")
        if dataset not in VECTOR_DATASETS:
            return {"error": "unknown vector dataset"}
        handle = str(arguments.get("output_geodataframe_name", "geodataframe"))
        state["objects"][handle] = {"kind": "geodataframe", "dataset": dataset}
        columns = (
            ["NAME_EN", "WB_A3", "ISO_A3", "CONTINENT", "REGION_UN", "SUBREGION", "geometry"]
            if dataset == "Countries"
            else ["dataset attributes", "geometry"]
        )
        return {"created": handle, "kind": "geodataframe", "dataset": dataset, "columns": columns}
    if name == "get_raster_path":
        dataset = arguments.get("rasterdataset")
        if dataset not in RASTER_DATASETS:
            return {"error": "unknown raster dataset"}
        slug = hashlib.sha256(str(dataset).encode()).hexdigest()[:16]
        path = f"snapshot://rasters/{slug}.tif"
        state["objects"][path] = {"kind": "raster", "dataset": dataset}
        return {"raster_path": path, "dataset": dataset, "available": True}

    handle = _output_handle(arguments)
    if handle:
        state["objects"][handle] = {
            "kind": "derived",
            "operation": name,
            "arguments": arguments,
        }
    if name.startswith("make_") or name == "visualize_geographies":
        state["artifacts"].append({"tool": name, "arguments": arguments})
        return {
            "status": "map created",
            "artifact_id": f"map_{len(state['artifacts'])}",
            "arguments": arguments,
        }
    if name == "get_unique_values":
        return {
            "dataframe": arguments.get("dataframe_name"),
            "column": arguments.get("column"),
            "instruction": "Use exact snapshot values for the subsequent categorical filter.",
        }
    return {
        "status": "completed",
        "operation": name,
        "created": handle,
        "arguments": arguments,
    }


class GeoBenchXVerifier:
    async def verify(self, body: GeoBenchXVerifyRequest) -> GeoBenchXVerifyResponse:
        task = TASKS.get(body.task_id)
        if task is None:
            return GeoBenchXVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                geobenchx_match=0,
                rejected=False,
                candidate_steps=0,
                reference_steps_mean=0.0,
                relative_length_delta=0.0,
                matched_reference_index=-1,
                match_detail={},
                failure_reason=f"unknown GeoBenchX task_id {body.task_id}",
            )
        scored = score_task(parse_calls(body.response.output), task)
        return GeoBenchXVerifyResponse(**body.model_dump(), **scored)


class GeoBenchXResourcesServer(GeoBenchXVerifier, SimpleResourcesServer):
    ray_enabled = False
    config: GeoBenchXResourcesServerConfig
    _sessions: dict[str, dict[str, Any]] = PrivateAttr(default_factory=dict)
    _upstream: UpstreamToolPool | None = PrivateAttr(default=None)

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.config.tool_backend == "upstream":
            self._upstream = _build_upstream_pool(self.config)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        for name in TOOL_DEFINITIONS:
            app.post(f"/{name}")(self._handler(name))
        if self._upstream is not None:
            app.router.add_event_handler("shutdown", self._upstream.close)
        return app

    def _handler(self, tool_name: str):
        async def handler(request: Request, body: ToolBody) -> dict[str, Any]:
            session_id = request.session.get(SESSION_ID_KEY, "")
            arguments = body.model_dump(exclude_none=True)
            if self._upstream is not None:
                out = await self._upstream.call(session_id, tool_name, arguments)
                return {"result": out["result"]}
            state = self._sessions.setdefault(
                session_id, {"task_id": None, "objects": {}, "artifacts": [], "rejected": False}
            )
            return execute_tool(tool_name, arguments, state)

        handler.__name__ = f"handle_{tool_name}"
        return handler

    async def seed_session(self, request: Request, body: SeedBody) -> BaseSeedSessionResponse:
        session_id = request.session.get(SESSION_ID_KEY)
        if session_id and self._upstream is not None:
            await self._upstream.reset(session_id)
        elif session_id:
            self._sessions[session_id] = {
                "task_id": body.task_id,
                "objects": {},
                "artifacts": [],
                "rejected": False,
            }
        return BaseSeedSessionResponse()


def _build_upstream_pool(config: GeoBenchXResourcesServerConfig) -> UpstreamToolPool:
    required = {
        "upstream_python": config.upstream_python,
        "upstream_src": config.upstream_src,
        "stat_data_path": config.stat_data_path,
        "geo_data_path": config.geo_data_path,
        "scratch_path": config.scratch_path,
    }
    missing = [key for key, value in required.items() if not value]
    if missing:
        raise ValueError(f"tool_backend=upstream requires: {', '.join(missing)}")
    return UpstreamToolPool(
        python=config.upstream_python,
        upstream_src=config.upstream_src,
        stat_data_path=config.stat_data_path,
        geo_data_path=config.geo_data_path,
        scratch_path=config.scratch_path,
        num_workers=config.num_tool_workers,
        tool_timeout_s=config.tool_timeout_s,
        max_sessions_per_worker=config.max_sessions_per_worker,
    )


VERIFIER_FIXTURE = VerifierFixture(
    server_factory=GeoBenchXVerifier,
    request_model=GeoBenchXVerifyRequest,
    cases_path=Path(__file__).parent / "tests" / "verifier_cases.jsonl",
)


if __name__ == "__main__":
    GeoBenchXResourcesServer.run_webserver()
