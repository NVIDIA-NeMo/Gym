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

from __future__ import annotations

import os
from abc import abstractmethod
from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.browserbase_dom import BrowserbaseDOMRuntime, BrowserbaseDOMSessionManager
from nemo_gym.server_utils import SESSION_ID_KEY


class BrowserbaseDOMResourcesServerConfig(BaseResourcesServerConfig):
    project_id: Optional[str] = Field(
        default=None,
        description="Browserbase project ID. Falls back to browserbase_project_id_var.",
    )
    browserbase_project_id_var: str = Field(
        default="BROWSERBASE_PROJECT_ID",
        description="Environment variable containing the Browserbase project ID.",
    )
    browserbase_api_key: Optional[str] = Field(
        default=None,
        description="Optional direct Browserbase API key override for testing or local development.",
    )
    browserbase_api_key_var: str = Field(
        default="BROWSERBASE_API_KEY",
        description="Environment variable containing the Browserbase API key.",
    )
    model_api_key: Optional[str] = Field(
        default=None,
        description="Optional direct Stagehand model API key override for testing or local development.",
    )
    model_api_key_var: str = Field(
        default="MODEL_API_KEY",
        description="Environment variable containing the Stagehand model API key.",
    )
    stagehand_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Stagehand model name used for DOM browser actions.",
    )
    proxies: bool = Field(
        default=False,
        description="Enable Browserbase proxies for DOM sessions.",
    )
    advanced_stealth: bool = Field(
        default=False,
        description="Enable Browserbase advanced stealth for DOM sessions.",
    )


class BrowserbaseDOMSeedSessionRequest(BaseSeedSessionRequest):
    start_url: Optional[str] = None


class BrowserbaseNavigateRequest(BaseModel):
    url: str


class BrowserbaseObserveRequest(BaseModel):
    instruction: str


class BrowserbaseActRequest(BaseModel):
    instruction: str


class BrowserbaseExtractRequest(BaseModel):
    instruction: str
    schema_json: str


def browserbase_dom_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "navigate",
            "description": "Navigate the current browser session to a URL.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Absolute URL to navigate to.",
                    }
                },
                "required": ["url"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "observe",
            "description": "Find candidate actions on the current page that match an instruction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Natural-language description of the element or action to find.",
                    }
                },
                "required": ["instruction"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "act",
            "description": "Perform a natural-language browser action on the current page.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "Precise natural-language action to perform.",
                    }
                },
                "required": ["instruction"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "type": "function",
            "name": "extract",
            "description": "Extract structured data from the current page using a JSON schema.",
            "parameters": {
                "type": "object",
                "properties": {
                    "instruction": {
                        "type": "string",
                        "description": "What data to extract from the page.",
                    },
                    "schema_json": {
                        "type": "string",
                        "description": "JSON-serialized schema describing the expected structure.",
                    },
                },
                "required": ["instruction", "schema_json"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


class BrowserbaseDOMResourcesServer(SimpleResourcesServer):
    config: BrowserbaseDOMResourcesServerConfig
    runtime_manager: Any = None
    session_id_to_runtime: Dict[str, BrowserbaseDOMRuntime] = Field(default_factory=dict)

    def model_post_init(self, context: Any) -> None:
        browserbase_api_key = self.config.browserbase_api_key or os.getenv(self.config.browserbase_api_key_var)
        project_id = self.config.project_id or os.getenv(self.config.browserbase_project_id_var)
        model_api_key = self.config.model_api_key or os.getenv(self.config.model_api_key_var)

        self.runtime_manager = BrowserbaseDOMSessionManager(
            browserbase_api_key=browserbase_api_key or "",
            project_id=project_id or "",
            model_api_key=model_api_key or "",
            stagehand_model=self.config.stagehand_model,
            proxies=self.config.proxies,
            advanced_stealth=self.config.advanced_stealth,
        )
        self.session_id_to_runtime = {}

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/navigate")(self.navigate)
        app.post("/observe")(self.observe)
        app.post("/act")(self.act)
        app.post("/extract")(self.extract)
        return app

    async def seed_session(self, request: Request, body: BrowserbaseDOMSeedSessionRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        await self._close_session(session_id)
        runtime = await self.runtime_manager.create_runtime(start_url=body.start_url)
        self.session_id_to_runtime[session_id] = runtime
        return BaseSeedSessionResponse()

    async def navigate(self, request: Request, body: BrowserbaseNavigateRequest) -> PlainTextResponse:
        return await self._run_tool(request, lambda runtime: self.runtime_manager.navigate(runtime, body.url))

    async def observe(self, request: Request, body: BrowserbaseObserveRequest) -> PlainTextResponse:
        return await self._run_tool(request, lambda runtime: self.runtime_manager.observe(runtime, body.instruction))

    async def act(self, request: Request, body: BrowserbaseActRequest) -> PlainTextResponse:
        return await self._run_tool(request, lambda runtime: self.runtime_manager.act(runtime, body.instruction))

    async def extract(self, request: Request, body: BrowserbaseExtractRequest) -> PlainTextResponse:
        return await self._run_tool(
            request,
            lambda runtime: self.runtime_manager.extract(runtime, body.instruction, body.schema_json),
        )

    async def _run_tool(self, request: Request, tool_fn: Any) -> PlainTextResponse:
        runtime = self._get_runtime(request)
        if runtime is None:
            return PlainTextResponse("No active session. Call /seed_session first.")

        try:
            output = await tool_fn(runtime)
        except Exception as exc:
            output = f"Tool execution error: {exc}"
        return PlainTextResponse(output)

    def _get_runtime(self, request: Request) -> Optional[BrowserbaseDOMRuntime]:
        session_id = request.session[SESSION_ID_KEY]
        return self.session_id_to_runtime.get(session_id)

    async def _close_session(self, session_id: str) -> None:
        runtime = self.session_id_to_runtime.pop(session_id, None)
        if runtime is not None:
            await self.runtime_manager.cleanup_runtime(runtime)

    async def teardown(self) -> None:
        for session_id in list(self.session_id_to_runtime):
            await self._close_session(session_id)
        await self.runtime_manager.close()

    @abstractmethod
    async def verify(self, request: Request, body: BaseVerifyRequest) -> BaseVerifyResponse:
        pass
