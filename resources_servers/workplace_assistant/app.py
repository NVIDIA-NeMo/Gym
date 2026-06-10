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
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

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

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/{path}")(self.route_to_python_function)
        return app

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        # init session once for each sample.
        session_id = request.session[SESSION_ID_KEY]
        toolkits = [
            "email",
            "calendar",
            "analytics",
            "project_management",
            "customer_relationship_manager",
        ]
        self.session_id_to_tool_env[session_id] = get_tools(toolkits)
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

    async def verify(self, body: WorkbenchVerifyRequest) -> WorkbenchVerifyResponse:
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


if __name__ == "__main__":
    WorkbenchResourcesServer.run_webserver()
