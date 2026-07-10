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
"""Shared helpers for MCP wire tests (test-only; not part of the runtime API).

Every dual-registered server's test suite drives the same JSON-RPC lifecycle against a
``TestClient``: initialize -> notifications/initialized -> tools/list -> tools/call, plus the
transport-parity route scan. These helpers hold that scaffolding once so per-server test files
only contain server-specific assertions.
"""

from typing import Any, Optional

from nemo_gym.base_resources_server import NEMO_GYM_MCP_SESSION_TOKEN_HEADER


TOKEN_HEADER = NEMO_GYM_MCP_SESSION_TOKEN_HEADER
RPC_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
# Paths every tool-bearing server has that are NOT tools (harness endpoints, the MCP mount, and
# the unknown-tool catch-all), excluded from transport-parity comparisons.
NON_TOOL_PATHS = frozenset({"/seed_session", "/verify", "/aggregate_metrics", "/mcp", "/{tool_name}"})


def mcp_rpc(client, method: str, params: Optional[dict] = None, token: Optional[str] = None, rpc_id: int = 1):
    """POST one JSON-RPC message to /mcp (notifications carry no id, per the protocol)."""
    headers = dict(RPC_HEADERS)
    if token is not None:
        headers[TOKEN_HEADER] = token
    payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if not method.startswith("notifications/"):
        payload["id"] = rpc_id
    return client.post("/mcp", headers=headers, json=payload, follow_redirects=False)


def mcp_handshake(client, token: Optional[str] = None) -> None:
    """Drive the MCP lifecycle prologue: initialize + the initialized notification."""
    resp = mcp_rpc(
        client,
        "initialize",
        {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "pytest", "version": "0"}},
        token=token,
    )
    assert resp.status_code == 200, resp.text
    resp = mcp_rpc(client, "notifications/initialized", token=token)
    assert resp.status_code in (200, 202)


def mcp_list_tools(client, token: Optional[str] = None) -> dict[str, dict]:
    """tools/list -> {tool name: tool object}."""
    resp = mcp_rpc(client, "tools/list", token=token, rpc_id=2)
    assert resp.status_code == 200, resp.text
    return {tool["name"]: tool for tool in resp.json()["result"]["tools"]}


def mcp_call(client, name: str, arguments: dict, token: Optional[str] = None) -> dict:
    """tools/call -> the JSON-RPC result object (check result.get("isError") yourself)."""
    resp = mcp_rpc(client, "tools/call", {"name": name, "arguments": arguments}, token=token, rpc_id=3)
    assert resp.status_code == 200, resp.text
    return resp.json()["result"]


def mcp_result_payload(result: dict) -> Any:
    """The tool's return payload: structuredContent when present, else the JSON text block."""
    import json

    if result.get("structuredContent") is not None:
        return result["structuredContent"]
    return json.loads(result["content"][0]["text"])


def seed_token(client, body: Optional[dict] = None) -> str:
    """POST /seed_session and return the minted per-rollout MCP session token."""
    resp = client.post("/seed_session", json=body if body is not None else {})
    assert resp.status_code == 200, resp.text
    return resp.json()["mcp"]["headers"][TOKEN_HEADER]


def http_tool_names(app, extra_non_tool_paths: tuple = ()) -> set[str]:
    """The set of HTTP tool routes on the app (POST routes minus harness endpoints)."""
    excluded = set(NON_TOOL_PATHS) | set(extra_non_tool_paths)
    names = set()
    for route in app.router.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None) or set()
        if path and "POST" in methods and path not in excluded:
            names.add(path.lstrip("/"))
    return names


def assert_transport_parity(app, client, expected_tools: set, extra_non_tool_paths: tuple = ()) -> None:
    """The E9 invariant: unfiltered MCP tools/list == HTTP tool routes == the expected inventory."""
    mcp_names = set(mcp_list_tools(client))
    http_names = http_tool_names(app, extra_non_tool_paths)
    assert mcp_names == http_names == set(expected_tools), (mcp_names, http_names, set(expected_tools))
