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
Custom Tools Resources Server with Dynamic Verifier Delegation and NeMo Skills Integration.

This resources server provides:
- Custom tool implementations (search_database, execute_calculation, fetch_data)
- Integration with nemo_skills ToolManager for additional tools (e.g., PythonTool)
- Dynamic verification delegation to other resources servers
"""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
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


class MyCustomToolsConfig(BaseResourcesServerConfig):
    """Config for the custom tools resources server."""

    # Map of verifier names to server references
    available_verifiers: Dict[str, ResourcesServerRef] = Field(default_factory=dict)

    # Default verifier if not specified in sample
    default_verifier: str = "xlam_fc"

    # NeMo Skills tool modules to load (e.g., "nemo_skills.mcp.servers.python_tool.PythonTool")
    nemo_skills_tools: List[str] = Field(default_factory=list)

    # Per-tool overrides for nemo_skills tools
    nemo_skills_tool_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


# ============================================================
# Tool Request/Response Models
# ============================================================


class SearchDatabaseRequest(BaseModel):
    query: str
    database: str = "default"


class SearchDatabaseResponse(BaseModel):
    results: List[Dict[str, Any]]
    total_count: int


class ExecuteCalculationRequest(BaseModel):
    expression: str
    precision: int = 4


class ExecuteCalculationResponse(BaseModel):
    result: float
    formatted: str


class FetchDataRequest(BaseModel):
    source_id: str
    fields: List[str] = Field(default_factory=list)


class FetchDataResponse(BaseModel):
    data: Dict[str, Any]
    source: str


# Generic request for nemo_skills tools
class NemoSkillsToolRequest(BaseModel):
    """Request body for calling nemo_skills tools."""

    model_config = ConfigDict(extra="allow")


# ============================================================
# Run/Verify Request/Response Models
# ============================================================


class MyCustomToolsRunRequest(BaseRunRequest):
    """
    Run request that allows extra fields from the sample.
    The verifier_type field determines which verifier to use.
    """

    model_config = ConfigDict(extra="allow")

    # Per-sample verifier selection
    verifier_type: Optional[str] = None

    # Fields that different verifiers might need
    expected_answers: Optional[List[Dict[str, Any]]] = None  # for xlam_fc
    expected_answer: Optional[str] = None  # for mcqa / equivalence_llm_judge
    options: Optional[List[Dict[str, str]]] = None  # for mcqa
    expected_output: Optional[str] = None  # for python_output verifier


class MyCustomToolsVerifyRequest(MyCustomToolsRunRequest, BaseVerifyRequest):
    pass


class MyCustomToolsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    verifier_used: str
    delegated_response: Optional[Dict[str, Any]] = None


# ============================================================
# Resources Server Implementation
# ============================================================


class MyCustomToolsResourcesServer(SimpleResourcesServer):
    config: MyCustomToolsConfig
    tool_manager: Optional[Any] = None  # Will be ToolManager if nemo_skills tools are configured

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Register custom tool endpoints (always available)
        app.post("/search_database")(self.search_database)
        app.post("/execute_calculation")(self.execute_calculation)
        app.post("/fetch_data")(self.fetch_data)

        # Initialize nemo_skills ToolManager if tools are configured
        if self.config.nemo_skills_tools:
            self._initialize_nemo_skills_tools()

            # Register a generic endpoint for nemo_skills tools
            # Tool calls will be routed by tool name
            app.post("/nemo_skills/{tool_name}")(self.execute_nemo_skills_tool)

            # Also register individual tool endpoints for direct access
            app.get("/nemo_skills/tools")(self.list_nemo_skills_tools)

            # Register stateful_python_code_exec at root level for simpler agent access
            # This allows the agent to call /stateful_python_code_exec directly
            app.post("/stateful_python_code_exec")(self.stateful_python_code_exec)

        return app

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        """
        Initialize session state for a new rollout.
        
        For stateful tools like PythonTool, the session is automatically managed
        via the nemo-gym session ID (accessed from the Request in tool execution).
        This method can be extended to perform additional initialization if needed.
        """
        # Call parent implementation
        return await super().seed_session(body)

    def _initialize_nemo_skills_tools(self):
        """Initialize the nemo_skills ToolManager with configured tools."""
        try:
            import asyncio

            from nemo_skills.mcp.tool_manager import ToolManager

            logger.info(f"Initializing NeMo Skills ToolManager with tools: {self.config.nemo_skills_tools}")

            # Build context with sandbox config for PythonTool
            # The sandbox config tells PythonTool how to connect to the code execution sandbox
            context = {
                "sandbox": {
                    "sandbox_type": "local",
                    "host": "127.0.0.1",
                    "port": "6000",
                }
            }

            self.tool_manager = ToolManager(
                module_specs=self.config.nemo_skills_tools,
                overrides=self.config.nemo_skills_tool_overrides,
                context=context,
            )

            # IMPORTANT: Call list_all_tools() to populate the tool mappings
            # This is required before execute_tool() can be called
            async def _load_tools():
                tools = await self.tool_manager.list_all_tools()
                logger.info(f"Loaded {len(tools)} nemo_skills tools: {[t['name'] for t in tools]}")

            asyncio.get_event_loop().run_until_complete(_load_tools())

            logger.info("NeMo Skills ToolManager initialized successfully")

        except ImportError as e:
            logger.error(f"Failed to import nemo_skills: {e}")
            logger.error("Make sure nemo_skills is installed: pip install nemo-skills")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize ToolManager: {e}")
            raise

    # --------------------------------------------------------
    # NeMo Skills Tool Endpoints
    # --------------------------------------------------------

    async def list_nemo_skills_tools(self) -> Dict[str, Any]:
        """List all available nemo_skills tools and their schemas."""
        if not self.tool_manager:
            return {"tools": [], "error": "No nemo_skills tools configured"}

        tools = await self.tool_manager.list_all_tools()
        return {"tools": tools}

    async def execute_nemo_skills_tool(self, tool_name: str, request: Request) -> Dict[str, Any]:
        """
        Execute a nemo_skills tool by name.
        
        Uses the nemo-gym session ID as the request_id for stateful tools like PythonTool.
        This ensures that all tool calls within the same rollout share the same session,
        allowing stateful execution (e.g., variables persist across multiple Python code cells).
        """
        if not self.tool_manager:
            return {"error": "No nemo_skills tools configured"}

        try:
            # Parse the request body
            body = await request.json()

            # Use nemo-gym's session ID for stateful tool execution
            # This is set by SessionMiddleware and persists across all tool calls
            # within the same rollout (from seed_session through all tool calls)
            session_id = request.session.get(SESSION_ID_KEY)
            if not session_id:
                # Fallback if session not available (shouldn't happen in normal flow)
                import uuid
                session_id = str(uuid.uuid4())
                logger.warning(f"No session ID found, using fallback: {session_id}")

            # Execute the tool via ToolManager
            # The request_id is used by PythonTool to maintain stateful sessions
            result = await self.tool_manager.execute_tool(
                raw_name=tool_name,
                args=body,
                extra_args={"request_id": session_id},
            )

            return {"result": result, "tool_name": tool_name}

        except KeyError as e:
            return {"error": f"Unknown tool: {tool_name}", "available_tools": list(self.tool_manager._raw_to_qualified_map.keys())}
        except Exception as e:
            logger.exception(f"Error executing tool {tool_name}: {e}")
            return {"error": str(e), "tool_name": tool_name}

    # --------------------------------------------------------
    # Convenience endpoint for stateful Python code execution
    # --------------------------------------------------------

    async def stateful_python_code_exec(self, request: Request) -> str:
        """
        Convenience endpoint for stateful Python code execution.
        
        This wraps the PythonTool's stateful_python_code_exec for easier access.
        The session is automatically managed via nemo-gym's session ID, so
        variables defined in one call will persist across subsequent calls
        within the same rollout.
        
        Request body should contain:
        - code: str - The Python code to execute
        
        Returns the execution output as a string (matching the format expected
        by the simple_agent).
        """
        result = await self.execute_nemo_skills_tool("stateful_python_code_exec", request)
        
        # Return just the result string for compatibility with simple_agent
        if isinstance(result, dict):
            if "error" in result:
                return json.dumps(result)
            return result.get("result", json.dumps(result))
        return str(result)

    # --------------------------------------------------------
    # Custom Tool Implementations
    # --------------------------------------------------------

    async def search_database(self, body: SearchDatabaseRequest) -> SearchDatabaseResponse:
        """Search a database with the given query."""
        # Mock implementation - replace with your actual logic
        results = [{"id": 1, "name": f"Result for: {body.query}", "database": body.database}]
        return SearchDatabaseResponse(results=results, total_count=len(results))

    async def execute_calculation(self, body: ExecuteCalculationRequest) -> ExecuteCalculationResponse:
        """Execute a mathematical calculation."""
        # Safe evaluation with limited builtins
        safe_builtins = {
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum,
            "pow": pow,
        }
        try:
            result = eval(body.expression, {"__builtins__": safe_builtins}, {})
            result = round(float(result), body.precision)
        except Exception:
            result = 0.0
        return ExecuteCalculationResponse(result=result, formatted=f"{result:.{body.precision}f}")

    async def fetch_data(self, body: FetchDataRequest) -> FetchDataResponse:
        """Fetch data from a specified source."""
        # Mock implementation - replace with your actual logic
        data = {"source_id": body.source_id, "fields": body.fields, "fetched": True}
        return FetchDataResponse(data=data, source=body.source_id)

    # --------------------------------------------------------
    # Dynamic Verification
    # --------------------------------------------------------

    async def verify(self, body: MyCustomToolsVerifyRequest) -> MyCustomToolsVerifyResponse:
        """
        Verify the model's response using the appropriate verifier.
        The verifier is selected based on the sample's verifier_type field.
        """
        # Determine which verifier to use
        verifier_type = body.verifier_type or self.config.default_verifier

        # Handle the built-in python_output verifier
        if verifier_type == "python_output":
            return self._verify_python_output(body)

        # Check if the verifier is available
        if verifier_type not in self.config.available_verifiers:
            return MyCustomToolsVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                verifier_used=f"unknown:{verifier_type}",
                delegated_response={
                    "error": f"Unknown verifier: {verifier_type}. "
                    f"Available: {list(self.config.available_verifiers.keys()) + ['python_output']}"
                },
            )

        verifier_ref = self.config.available_verifiers[verifier_type]

        # Delegate to the appropriate verifier
        response = await self.server_client.post(
            server_name=verifier_ref.name,
            url_path="/verify",
            json=body.model_dump(),  # Pass through ALL fields
        )

        result = await response.json()
        reward = result.get("reward", 0.0)

        return MyCustomToolsVerifyResponse(
            **body.model_dump(),
            reward=reward,
            verifier_used=verifier_type,
            delegated_response=result,
        )

    def _verify_python_output(self, body: MyCustomToolsVerifyRequest) -> MyCustomToolsVerifyResponse:
        """
        Verify Python code execution by checking if:
        1. The correct tool was called (stateful_python_code_exec)
        2. The tool output contains the expected output string
        """
        expected_output = body.expected_output
        if expected_output is None:
            return MyCustomToolsVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                verifier_used="python_output",
                delegated_response={"error": "No expected_output specified"},
            )

        # Find tool calls and their outputs in the response
        response = body.response
        tool_outputs = []
        correct_tool_called = False

        for output_item in response.output:
            if output_item.type == "function_call" and output_item.name == "stateful_python_code_exec":
                correct_tool_called = True
            elif output_item.type == "function_call_output":
                tool_outputs.append(output_item.output)

        if not correct_tool_called:
            return MyCustomToolsVerifyResponse(
                **body.model_dump(),
                reward=0.0,
                verifier_used="python_output",
                delegated_response={
                    "error": "Tool stateful_python_code_exec was not called",
                    "correct_tool_called": False,
                },
            )

        # Check if any tool output contains the expected output
        output_matches = any(expected_output in output for output in tool_outputs)

        reward = 1.0 if output_matches else 0.0

        return MyCustomToolsVerifyResponse(
            **body.model_dump(),
            reward=reward,
            verifier_used="python_output",
            delegated_response={
                "correct_tool_called": correct_tool_called,
                "tool_outputs": tool_outputs,
                "expected_output": expected_output,
                "output_matches": output_matches,
            },
        )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    async def shutdown(self):
        """Cleanup resources on server shutdown."""
        if self.tool_manager:
            await self.tool_manager.shutdown()


if __name__ == "__main__":
    MyCustomToolsResourcesServer.run_webserver()
