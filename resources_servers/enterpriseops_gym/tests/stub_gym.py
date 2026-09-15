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
"""In-process stub of an EnterpriseOps MCP gym server, backed by per-database sqlite.

Implements the exact HTTP surface the resources server depends on — POST /mcp (JSON-RPC
initialize / notifications / tools/list / tools/call), POST /api/seed-database,
DELETE /api/delete-database, POST /api/sql-runner — so the whole seed -> tool -> verify
flow is testable offline with genuine SQL state mutations.
"""

import json
import sqlite3
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, Response


STUB_TOOLS = [
    {
        "name": "update_entitlement",
        "description": "Update an entitlement's coverage hours and/or monthly case limit.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "entitlement_id": {"type": "integer", "description": "Entitlement to update"},
                "coverage_hours": {"type": ["string", "null"], "description": "e.g. h24x7"},
                "max_cases_per_month": {"type": ["integer", "null"], "description": "0 = unlimited"},
            },
            "required": ["entitlement_id", "coverage_hours", "max_cases_per_month"],
        },
    },
    {
        "name": "get_entitlement",
        "description": "Fetch an entitlement row.",
        "inputSchema": {
            "type": "object",
            "properties": {"entitlement_id": {"type": "integer", "description": ""}},
            "required": ["entitlement_id"],
        },
    },
]


class StubGymState:
    """Shared, inspectable state: live sqlite databases and an event log for assertions."""

    def __init__(self) -> None:
        self.dbs: Dict[str, sqlite3.Connection] = {}
        self.seed_events: List[str] = []
        self.delete_events: List[str] = []
        self.tool_calls: List[Dict[str, Any]] = []
        self.sql_calls: List[Dict[str, Any]] = []

    def reset(self) -> None:
        for connection in self.dbs.values():
            connection.close()
        self.dbs.clear()
        self.seed_events.clear()
        self.delete_events.clear()
        self.tool_calls.clear()
        self.sql_calls.clear()


def _rows_as_dicts(cursor: sqlite3.Cursor) -> List[Dict[str, Any]]:
    columns = [d[0] for d in cursor.description] if cursor.description else []
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _jsonrpc_tool_result(request_id: Any, text: str, is_error: bool = False) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"content": [{"type": "text", "text": text}], "isError": is_error},
    }


def create_stub_gym_app(state: StubGymState) -> FastAPI:
    app = FastAPI()

    @app.post("/api/seed-database")
    async def seed_database(request: Request) -> Dict[str, Any]:
        payload = await request.json()
        database_id = payload["database_id"]
        connection = sqlite3.connect(":memory:", check_same_thread=False)
        connection.executescript(payload["sql_content"])
        connection.commit()
        state.dbs[database_id] = connection
        state.seed_events.append(database_id)
        return {"status": "ok", "database_id": database_id}

    @app.delete("/api/delete-database")
    async def delete_database(request: Request) -> Dict[str, Any]:
        payload = await request.json()
        database_id = payload["database_id"]
        connection = state.dbs.pop(database_id, None)
        if connection is not None:
            connection.close()
        state.delete_events.append(database_id)
        return {"status": "deleted", "database_id": database_id}

    @app.post("/api/sql-runner")
    async def sql_runner(request: Request) -> Response:
        payload = await request.json()
        database_id = payload.get("database_id") or request.headers.get("x-database-id")
        state.sql_calls.append(
            {"query": payload.get("query"), "database_id": database_id, "headers": dict(request.headers)}
        )

        connection = state.dbs.get(database_id)
        if connection is None:
            return Response(
                content=json.dumps({"detail": f"Unknown database: {database_id}"}),
                media_type="application/json",
                status_code=404,
            )
        try:
            cursor = connection.execute(payload["query"])
            return Response(content=json.dumps({"data": _rows_as_dicts(cursor)}), media_type="application/json")
        except sqlite3.Error as e:
            return Response(
                content=json.dumps({"detail": f"SQL error: {e}"}), media_type="application/json", status_code=400
            )

    def _call_tool(
        name: str, arguments: Dict[str, Any], database_id: Optional[str], request_id: Any
    ) -> Dict[str, Any]:
        connection = state.dbs.get(database_id)
        if connection is None:
            return _jsonrpc_tool_result(request_id, f"Unknown database: {database_id}", is_error=True)

        if name == "update_entitlement":
            sets, params = [], []
            for column in ("coverage_hours", "max_cases_per_month"):
                if arguments.get(column) is not None:
                    sets.append(f"{column} = ?")
                    params.append(arguments[column])
            if not sets:
                return _jsonrpc_tool_result(request_id, "Nothing to update", is_error=True)
            params.append(arguments["entitlement_id"])
            connection.execute(f"UPDATE entitlement SET {', '.join(sets)} WHERE entitlement_id = ?", params)
            connection.commit()
            return _jsonrpc_tool_result(
                request_id, json.dumps({"success": True, "entitlement_id": arguments["entitlement_id"]})
            )

        if name == "get_entitlement":
            cursor = connection.execute(
                "SELECT * FROM entitlement WHERE entitlement_id = ?", [arguments["entitlement_id"]]
            )
            rows = _rows_as_dicts(cursor)
            return _jsonrpc_tool_result(request_id, json.dumps(rows[0] if rows else None))

        return _jsonrpc_tool_result(request_id, f"Error: Tool '{name}' not found.", is_error=True)

    @app.post("/mcp")
    async def mcp(request: Request) -> Response:
        payload = await request.json()
        method = payload.get("method")
        request_id = payload.get("id")

        if method == "initialize":
            body = {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "stub-gym", "version": "1.0.0"},
                },
            }
            return Response(
                content=json.dumps(body),
                media_type="application/json",
                headers={"mcp-session-id": "stub-mcp-session"},
            )

        if method == "notifications/initialized":
            return Response(content=json.dumps({}), media_type="application/json")

        if method == "tools/list":
            body = {"jsonrpc": "2.0", "id": request_id, "result": {"tools": STUB_TOOLS}}
            return Response(content=json.dumps(body), media_type="application/json")

        if method == "tools/call":
            params = payload.get("params") or {}
            name = params.get("name")
            arguments = params.get("arguments") or {}
            database_id = request.headers.get("x-database-id")
            state.tool_calls.append(
                {"tool": name, "arguments": arguments, "database_id": database_id, "headers": dict(request.headers)}
            )
            return Response(
                content=json.dumps(_call_tool(name, arguments, database_id, request_id)),
                media_type="application/json",
            )

        body = {"jsonrpc": "2.0", "id": request_id, "error": {"code": -32601, "message": f"Unknown method {method}"}}
        return Response(content=json.dumps(body), media_type="application/json")

    return app
