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
"""The workplace_assistant tools for one episode, served over MCP (streamable HTTP) from the tools sidecar.

The 27 tools are the Resources Server's own code (``resources_servers.workplace_assistant.utils.get_tools``), vendored
unchanged next to this file by the adapter. The sidecar runs for one episode, so one process holds one fresh set of
tables and needs no session map. Every call that reaches the server is appended to a journal (name and arguments, in
order) before it runs; the verifier replays that journal with the environment's own ``is_correct``. The journal lives
only in this container: the agent can add to it only by calling a tool, as the native verifier counts every
``function_call`` the model made.

Each call answers as the Resources Server's ``POST /{path}`` route does over Gym's MCP exposure: ``None`` arguments are
dropped, an exception comes back as ``Error executing tool ...`` text, and the reply is the JSON of
``{"output": <result>}``.
"""

import contextlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

import mcp.types as types
import uvicorn
from fastapi.encoders import jsonable_encoder
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, ConfigDict
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route


sys.path.insert(0, os.environ.get("WA_CODE_DIR", "/opt/wa"))

from resources_servers.workplace_assistant.utils import get_tools  # noqa: E402


TOOLKITS = ["email", "calendar", "analytics", "project_management", "customer_relationship_manager"]
JOURNAL = Path(os.environ.get("WA_JOURNAL", "/var/lib/wa-journal/calls.jsonl"))
PORT = int(os.environ.get("WA_PORT", "8000"))
# The Compose sidecar listens on all interfaces: main reaches it by service name.
HOST = os.environ.get("WA_HOST", "0.0.0.0")


class WorkbenchResponse(BaseModel):
    """The Resources Server's reply model (``app.WorkbenchResponse``)."""

    model_config = ConfigDict(extra="allow")


TOOL_ENV = get_tools(TOOLKITS)
TOOLS = [
    types.Tool(name=schema["name"], description=schema.get("description"), inputSchema=schema["parameters"])
    for schema in TOOL_ENV["schemas"]
]
TOOL_NAMES = sorted(tool.name for tool in TOOLS)
# Tool calls change shared tables; one at a time, in journal order.
LOCK = threading.Lock()


def journal(name: str, arguments: dict[str, Any]) -> None:
    """Append one call, as the native verifier reads a ``function_call``: its name and its arguments as JSON text."""
    with JOURNAL.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"name": name, "arguments": json.dumps(arguments)}) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_tool(name: str, arguments: dict[str, Any]) -> types.CallToolResult:
    """``route_to_python_function`` of the Resources Server, then Gym's MCP rendering of its reply."""
    args = {key: value for key, value in arguments.items() if value is not None}
    try:
        function = TOOL_ENV["functions"][name]
        result = function(**args)
        response = WorkbenchResponse(output=result)
    except Exception as error:
        response = WorkbenchResponse(output=f"Error executing tool '{name}': {str(error)}")
    payload = jsonable_encoder(response)
    text = JSONResponse(payload).body.decode("utf-8")
    return types.CallToolResult(content=[types.TextContent(type="text", text=text)], structuredContent=payload)


async def on_list_tools(ctx: Any, params: Any) -> types.ListToolsResult:
    return types.ListToolsResult(tools=TOOLS)


async def on_call_tool(ctx: Any, params: types.CallToolRequestParams) -> types.CallToolResult:
    name, arguments = params.name, dict(params.arguments or {})
    with LOCK:
        journal(name, arguments)
        if name not in TOOL_ENV["functions"]:
            text = f"Unknown tool: {name!r}. Available tools: {TOOL_NAMES}"
            return types.CallToolResult(content=[types.TextContent(type="text", text=text)], isError=True)
        try:
            return run_tool(name, arguments)
        except Exception as error:
            # As Gym's MCP exposure answers a reply it cannot render (for example a NaN in a table row).
            return types.CallToolResult(content=[types.TextContent(type="text", text=str(error))], isError=True)


def build_app() -> Starlette:
    server = Server("workplace_assistant", on_list_tools=on_list_tools, on_call_tool=on_call_tool)
    manager = StreamableHTTPSessionManager(
        app=server,
        json_response=True,
        stateless=True,
        security_settings=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        async with manager.run():
            yield

    class Endpoint:
        async def __call__(self, scope, receive, send):
            await manager.handle_request(scope, receive, send)

    # Both "/mcp" and "/mcp/..." reach the endpoint, with no redirect in between.
    endpoint = Endpoint()
    return Starlette(routes=[Route("/mcp", endpoint), Mount("/mcp", app=endpoint)], lifespan=lifespan)


if __name__ == "__main__":
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    JOURNAL.touch()
    uvicorn.run(build_app(), host=HOST, port=PORT, log_level="warning")
