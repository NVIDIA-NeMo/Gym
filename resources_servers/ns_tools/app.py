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

"""
NeMo Skills Tools Resources Server.

This resources server provides:
- Integration with nemo_skills ToolManager for tool execution (e.g., PythonTool)
- Verification delegation to math_with_judge
"""

import asyncio
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from nemo_skills.mcp.tool_manager import ToolManager
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.server_utils import SESSION_ID_KEY


logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================


class NSToolsConfig(BaseResourcesServerConfig):
    """Config for the NeMo Skills tools resources server."""

    # Default verifier (typically math_with_judge)
    default_verifier: str = "math_with_judge"

    # Map of verifier names to server references
    # At minimum, should include math_with_judge
    verifiers: Dict[str, ResourcesServerRef] = Field(default_factory=dict)

    # NeMo Skills tool modules to load (e.g., "nemo_skills.mcp.servers.python_tool.PythonTool")
    nemo_skills_tools: List[str] = Field(default_factory=list)

    # Per-tool overrides for nemo_skills tools
    nemo_skills_tool_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    # Sandbox configuration for code execution tools
    sandbox_host: str = "127.0.0.1"
    sandbox_port: str = "6000"


# ============================================================
# Run/Verify Request/Response Models
# ============================================================


class NSToolsRunRequest(BaseRunRequest):
    """Run request that allows extra fields from the sample."""

    model_config = ConfigDict(extra="allow")

    # Per-sample verifier selection (optional, falls back to default_verifier)
    verifier_type: Optional[str] = None

    # Fields for math_with_judge verifier
    question: Optional[str] = None
    expected_answer: Optional[str] = None


class NSToolsVerifyRequest(NSToolsRunRequest, BaseVerifyRequest):
    pass


class NSToolsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    delegated_response: Optional[Dict[str, Any]] = None


# ============================================================
# Resources Server Implementation
# ============================================================


class NSToolsResourcesServer(SimpleResourcesServer):
    config: NSToolsConfig
    tool_manager: Optional[Any] = None
    _tool_name_map: Dict[str, str] = {}  # Maps tool names to qualified names

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Initialize nemo_skills ToolManager if tools are configured
        if self.config.nemo_skills_tools:
            self._initialize_nemo_skills_tools()

            # Register a catch-all endpoint for tool execution
            # This handles any tool name dynamically
            app.post("/{tool_name}")(self.execute_tool)

        return app

    def _initialize_nemo_skills_tools(self):
        """Initialize the nemo_skills ToolManager with configured tools."""

        logger.info(f"Initializing NeMo Skills ToolManager with tools: {self.config.nemo_skills_tools}")

        # Build context with sandbox config for PythonTool
        context = {
            "sandbox": {
                "sandbox_type": "local",
                "host": self.config.sandbox_host,
                "port": self.config.sandbox_port,
            }
        }

        self.tool_manager = ToolManager(
            module_specs=self.config.nemo_skills_tools,
            overrides=self.config.nemo_skills_tool_overrides,
            context=context,
        )

        # Load tools and build name mapping
        async def _load_tools():
            tools = await self.tool_manager.list_all_tools()
            for tool in tools:
                self._tool_name_map[tool["name"]] = tool["name"]
            logger.info(f"Loaded {len(tools)} nemo_skills tools: {list(self._tool_name_map.keys())}")

        asyncio.get_event_loop().run_until_complete(_load_tools())
        logger.info("NeMo Skills ToolManager initialized successfully")

    async def execute_tool(self, tool_name: str, request: Request) -> PlainTextResponse:
        """
        Execute a nemo_skills tool by name.

        Uses the nemo-gym session ID as the request_id for stateful tools.
        Returns the result as plain text for simple_agent compatibility.
        """
        if not self.tool_manager:
            return PlainTextResponse(json.dumps({"error": "No tools configured"}))

        # Check if tool is in our known tools
        if tool_name not in self._tool_name_map:
            logger.error(f"Unknown tool requested: {tool_name}")
            return PlainTextResponse(json.dumps({"error": f"Unknown tool: {tool_name}"}))

        try:
            body = await request.json()

            # Get session ID for stateful execution
            session_id = request.session.get(SESSION_ID_KEY)
            if not session_id:
                session_id = str(uuid.uuid4())
                logger.warning(f"No session ID found, using fallback: {session_id}")

            # Execute the tool
            result = await self.tool_manager.execute_tool(
                raw_name=tool_name,
                args=body,
                extra_args={"request_id": session_id},
            )

            # Return result as plain text to avoid double JSON serialization
            if isinstance(result, str):
                return PlainTextResponse(result)
            return PlainTextResponse(json.dumps(result))

        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}: {e}")
            return PlainTextResponse(json.dumps({"error": str(e)}))

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    async def verify(self, body: NSToolsVerifyRequest) -> NSToolsVerifyResponse:
        """
        Verify the model's response by delegating to the configured verifier.

        The verifier is selected by:
        1. Per-sample `verifier_type` field (if present)
        2. Config `default_verifier` (fallback)
        """
        # Select verifier
        verifier_type = body.verifier_type or self.config.default_verifier

        if verifier_type not in self.config.verifiers:
            raise ValueError(
                f"Unknown verifier: {verifier_type}. Configure it in 'verifiers' or check 'default_verifier'."
            )

        verifier_ref = self.config.verifiers[verifier_type]

        # Delegate to the verifier
        response = await self.server_client.post(
            server_name=verifier_ref.name,
            url_path="/verify",
            json=body.model_dump(),
        )

        result = await response.json()

        # Hard fail if no reward in response
        if "reward" not in result:
            raise ValueError(f"Verifier did not return 'reward' field. Response: {result}")

        return NSToolsVerifyResponse(
            **body.model_dump(),
            reward=result["reward"],
            delegated_response=result,
        )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    async def shutdown(self):
        """Cleanup resources on server shutdown."""
        if self.tool_manager:
            await self.tool_manager.shutdown()


if __name__ == "__main__":
    NSToolsResourcesServer.run_webserver()
