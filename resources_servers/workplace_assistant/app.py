# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import asyncio
import math
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, JsonValue

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.workplace_assistant.utils import get_tools, is_correct


_TOOLKITS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]


def _encode_frame(frame: pd.DataFrame) -> dict[str, Any]:
    # Column by column, so each cell becomes a plain Python value.
    # Float NaN (empty CSV cells) and None (values a tool call stored) both serialize as null,
    # so NaN positions are listed separately to restore them exactly.
    columns = []
    for name, series in frame.items():
        values = series.tolist()
        columns.append(
            {
                "name": name,
                "dtype": str(series.dtype),
                "values": [None if isinstance(value, float) and math.isnan(value) else value for value in values],
                "nan": [i for i, value in enumerate(values) if isinstance(value, float) and math.isnan(value)],
            }
        )
    return {"index": frame.index.tolist(), "columns": columns}


def _decode_frame(payload: dict[str, Any]) -> pd.DataFrame:
    index = payload["index"]
    data = {}
    for column in payload["columns"]:
        values = list(column["values"])
        if len(values) != len(index):
            raise ValueError(f"column {column['name']!r} has {len(values)} values for {len(index)} rows")
        for position in column["nan"]:
            values[position] = float("nan")
        data[column["name"]] = pd.Series(values, dtype=object).astype(column["dtype"])
    frame = pd.DataFrame(data)
    # Tools append rows by label (``.loc[len(frame)]``), so the index labels are part of the state.
    frame.index = pd.RangeIndex(len(index)) if index == list(range(len(index))) else pd.Index(index)
    return frame


def _export_tool_env(tool_env: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            attribute: _encode_frame(frame)
            for attribute, frame in vars(container).items()
            if isinstance(frame, pd.DataFrame)
        }
        for name, container in tool_env["containers"].items()
    }


def _restore_tool_env(state: dict[str, Any]) -> dict[str, Any]:
    tool_env = get_tools(_TOOLKITS)
    containers = tool_env["containers"]
    if set(state) != set(containers):
        raise ValueError(f"expected containers {sorted(containers)}, got {sorted(state)}")
    for name, frames in state.items():
        container = containers[name]
        expected = {attribute for attribute, value in vars(container).items() if isinstance(value, pd.DataFrame)}
        if set(frames) != expected:
            raise ValueError(f"expected {name} tables {sorted(expected)}, got {sorted(frames)}")
        for attribute, payload in frames.items():
            setattr(container, attribute, _decode_frame(payload))
    return tool_env


class WorkbenchResourcesServerConfig(BaseResourcesServerConfig):
    pass


class WorkbenchRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchResponse(BaseModel):
    model_config = ConfigDict(extra="allow")


class WorkbenchVerifyRequest(BaseVerifyRequest):
    ground_truth: list[Dict[str, str]] | str
    id: int
    category: str
    environment_name: str


class WorkbenchVerifyResponse(BaseVerifyResponse):
    pass


class WorkbenchResourcesServer(SimpleResourcesServer):
    ray_enabled = False
    # Each session's tool environment is a set of in-memory tables,
    # exported whole at a checkpoint. /verify keeps the default "wait"
    # because it discards the session's tool environment.
    checkpoint_mode = "exported"
    config: WorkbenchResourcesServerConfig
    session_id_to_tool_env: Dict[str, Any] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.route_to_python_function)
        return app

    # Register all 27 workplace tools as MCP tools via the catch-all route when expose_tools_over_mcp is enabled.
    def mcp_tools(self, harvested, catchall):
        specs = get_tools(_TOOLKITS)["schemas"]
        return harvested + [catchall.tool(s["name"], s["parameters"], s.get("description")) for s in specs]

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        # init session once for each sample.
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_tool_env[session_id] = get_tools(_TOOLKITS)
        return BaseSeedSessionResponse()

    async def route_to_python_function(self, path: str, body: WorkbenchRequest, request: Request) -> WorkbenchResponse:
        session_id = request.session[SESSION_ID_KEY]

        # Check if session exists
        if session_id not in self.session_id_to_tool_env:
            raise HTTPException(
                status_code=400,
                detail="Session not initialized. Please call seed_session first.",
            )

        tool_env = self.session_id_to_tool_env[session_id]
        args = {key: value for key, value in body.model_dump(exclude_unset=True).items() if value is not None}

        try:
            function = tool_env["functions"][path]
            result = function(**args)
            return WorkbenchResponse(output=result)
        except Exception as e:
            return WorkbenchResponse(
                output=f"Error executing tool '{path}': {str(e)}"
            )  # return error to model so that it can correct itself

    async def verify(self, request: Request, body: WorkbenchVerifyRequest) -> WorkbenchVerifyResponse:
        session_id = request.session[SESSION_ID_KEY]
        try:
            ground_truth = body.ground_truth
            response = body.response.output

            total_score = 0.0

            # Convert list of ResponseFunctionToolCall objects into list of dictionaries
            predicted_function_calls = []

            for message in response:
                if message.type == "function_call":
                    predicted_function_calls.append(message.model_dump())

            predicted_chat_content = []

            for message in response:
                if message.type == "output_text":
                    predicted_chat_content.append(message.model_dump())

            total_score += is_correct(predicted_function_calls, ground_truth, None) * 1.0
            return WorkbenchVerifyResponse(**body.model_dump(), reward=total_score)
        finally:
            self.session_id_to_tool_env.pop(session_id, None)

    async def export_session_states(self, session_ids: list[str]) -> dict[str, JsonValue]:
        # A session verify already discarded, for example because it raised, is left out.
        tool_envs = {
            session_id: self.session_id_to_tool_env[session_id]
            for session_id in session_ids
            if session_id in self.session_id_to_tool_env
        }
        # Encoding every table is CPU work; admission is closed during a commit, so no tool call mutates them.
        return await asyncio.to_thread(
            lambda: {session_id: _export_tool_env(tool_env) for session_id, tool_env in tool_envs.items()}
        )

    async def restore_session_states(self, states: dict[str, JsonValue]) -> None:
        def rebuild() -> dict[str, Any]:
            restored = {}
            for session_id, state in states.items():
                try:
                    restored[session_id] = _restore_tool_env(state)
                except (AttributeError, KeyError, TypeError, ValueError) as error:
                    raise ValueError(f"invalid workplace assistant state for session {session_id}: {error}") from error
            return restored

        # Rebuild every session before installing any, so an invalid state installs nothing.
        self.session_id_to_tool_env.update(await asyncio.to_thread(rebuild))

    async def retire_session_state(self, session_id: str) -> None:
        self.session_id_to_tool_env.pop(session_id, None)


if __name__ == "__main__":
    WorkbenchResourcesServer.run_webserver()
