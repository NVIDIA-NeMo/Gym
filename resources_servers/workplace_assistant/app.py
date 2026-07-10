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
from typing import Any, Callable, Dict

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
    gym_tool,
)
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.workplace_assistant.utils import get_tools, is_correct


TOOLKITS = [
    "email",
    "calendar",
    "analytics",
    "project_management",
    "customer_relationship_manager",
]


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

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        # Register every workbench tool over both transports (HTTP POST /<name> and MCP). The
        # hand-authored schema dicts are advertised verbatim and call arguments pass through raw.
        for schema in get_tools(TOOLKITS)["schemas"]:
            gym_tool(
                self._make_tool_closure(schema["name"]),
                name=schema["name"],
                description=schema["description"],
                input_schema=schema["parameters"],
                owner=self,
            )

    def _make_tool_closure(self, name: str) -> Callable:
        def call_workbench_tool(session_id: str, **args: Any) -> WorkbenchResponse:
            # Check if session exists
            if session_id not in self.session_id_to_tool_env:
                raise HTTPException(
                    status_code=400,
                    detail="Session not initialized. Please call seed_session first.",
                )

            tool_env = self.session_id_to_tool_env[session_id]
            args = {key: value for key, value in args.items() if value is not None}

            try:
                function = tool_env["functions"][name]
                result = function(**args)
                return WorkbenchResponse(output=result)
            except Exception as e:
                return WorkbenchResponse(
                    output=f"Error executing tool '{name}': {str(e)}"
                )  # return error to model so that it can correct itself

        return call_workbench_tool

    async def seed_session(self, request: Request, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        # init session once for each sample.
        session_id = request.session[SESSION_ID_KEY]
        self.session_id_to_tool_env[session_id] = get_tools(TOOLKITS)
        return BaseSeedSessionResponse()

    async def handle_unknown_tool(self, tool_name: str, request: Request) -> WorkbenchResponse:
        # Preserve the historical catch-all dispatcher contract: unseeded sessions get the 400,
        # seeded sessions get the 200 soft error produced by the KeyError on the function lookup
        # ("Error executing tool '<name>': '<name>'").
        session_id = request.session[SESSION_ID_KEY]

        try:
            raw = await request.json()
        except Exception:
            raw = {}
        body = WorkbenchRequest.model_validate(raw) if isinstance(raw, dict) else WorkbenchRequest()
        args = {key: value for key, value in body.model_dump(exclude_unset=True).items() if value is not None}

        return self._make_tool_closure(tool_name)(session_id, **args)

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
