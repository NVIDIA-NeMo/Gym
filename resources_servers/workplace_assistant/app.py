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
from io import StringIO
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym._checkpoint import ResourceSnapshot
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.rollout_correlation import current_attempt_index, current_logical_rollout_id
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.workplace_assistant.utils import get_tools, is_correct


_TOOLKITS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]


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
    config: WorkbenchResourcesServerConfig
    session_id_to_tool_env: Dict[str, Any] = Field(default_factory=dict)
    execution_to_session: Dict[tuple[str, int], str] = Field(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.route_to_python_function)
        return app

    # Register all 27 workplace tools as MCP tools via the catch-all route when expose_tools_over_mcp is enabled.
    def mcp_tools(self, harvested, catchall):
        specs = get_tools(["email", "calendar", "analytics", "project_management", "customer_relationship_manager"])[
            "schemas"
        ]
        return harvested + [catchall.tool(s["name"], s["parameters"], s.get("description")) for s in specs]

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        # init session once for each sample.
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_tool_env[session_id] = get_tools(_TOOLKITS)
        identity = self._current_identity()
        if identity is not None:
            self.execution_to_session[identity] = session_id
        return BaseSeedSessionResponse()

    async def route_to_python_function(self, path: str, body: WorkbenchRequest, request: Request) -> WorkbenchResponse:
        identity = self._current_identity()
        session_id = self.execution_to_session.get(identity) if identity is not None else None
        if session_id is None:
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
        identity = self._current_identity()
        session_id = self.execution_to_session.get(identity) if identity is not None else None
        if session_id is None:
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
            if identity is not None:
                self.execution_to_session.pop(identity, None)

    @staticmethod
    def _current_identity() -> tuple[str, int] | None:
        rollout_id = current_logical_rollout_id()
        attempt_index = current_attempt_index()
        if rollout_id is None or attempt_index is None:
            return None
        return rollout_id, attempt_index

    def checkpoint_state_enabled(self) -> bool:
        return True

    async def export_checkpoint_state(self, rollout_id: str, attempt_index: int) -> dict[str, Any]:
        session_id = self.execution_to_session[(rollout_id, attempt_index)]
        tool_env = self.session_id_to_tool_env[session_id]
        return {
            "containers": {
                name: {
                    attribute: frame.to_json(orient="split")
                    for attribute, frame in vars(container).items()
                    if isinstance(frame, pd.DataFrame)
                }
                for name, container in tool_env["containers"].items()
            }
        }

    async def restore_checkpoint_states(self, snapshots: list[ResourceSnapshot]) -> None:
        restored: list[tuple[ResourceSnapshot, str, dict[str, Any]]] = []
        for snapshot in snapshots:
            tool_env = get_tools(_TOOLKITS)
            for name, frames in snapshot.state["containers"].items():
                container = tool_env["containers"][name]
                for attribute, payload in frames.items():
                    setattr(
                        container,
                        attribute,
                        pd.read_json(StringIO(payload), orient="split", dtype=False, convert_dates=False),
                    )
            session_id = f"checkpoint:{snapshot.rollout_id}:a{snapshot.attempt_index}"
            restored.append((snapshot, session_id, tool_env))

        replacement_environments = dict(self.session_id_to_tool_env)
        replacement_index = dict(self.execution_to_session)
        for snapshot, session_id, tool_env in restored:
            replacement_environments[session_id] = tool_env
            replacement_index[(snapshot.rollout_id, snapshot.attempt_index)] = session_id
        self.session_id_to_tool_env = replacement_environments
        self.execution_to_session = replacement_index


if __name__ == "__main__":
    WorkbenchResourcesServer.run_webserver()
