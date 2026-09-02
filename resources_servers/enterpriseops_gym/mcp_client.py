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
"""Async MCP client for the EnterpriseOps-Gym MCP servers.

A functional port of EnterpriseOps-Gym's ``benchmark/mcp_client.py`` with the transport
swapped from per-request ``httpx.AsyncClient`` instances to NeMo Gym's pooled global
aiohttp client (``nemo_gym.server_utils.request``), and all per-call state (database_id,
context) passed per call instead of mutated on the client — the original mutates
``client.database_id``, which is racy under concurrent rollouts.

Return shapes intentionally mirror the EOG client 1:1 ({"success": ..., "result": ...,
"error": ...}) so the verifier engine port stays line-for-line comparable.

Why not the official MCP Python SDK (a core Gym dependency since #1682)? Evaluated and
rejected 2026-07-08: (1) the SDK's streamable-HTTP client rides on httpx, which this repo
bans for high-concurrency async paths — we'd need an aiohttp transport adapter anyway;
(2) it binds headers at session level, but per-rollout isolation here requires PER-CALL
``x-database-id``/context headers on a shared client (a session per rollout would
reintroduce the per-task handshake overhead this port removed); (3) roughly half of this
module talks to the gyms' NON-MCP REST endpoints (/api/seed-database, /api/delete-database,
/api/sql-runner) which no MCP client covers; (4) the upstream gym containers are frozen at
protocolVersion 2024-11-05, so SDK protocol-evolution tracking buys nothing. Note #1682's
``MCPResourcesServer`` solves the INVERSE problem (exposing Gym-owned tools AS an MCP
server to MCP-native agents) — see PARITY.md follow-ups for how that could let MCP-native
agents like Claude Code run against these EnterpriseOps tools.
"""

import asyncio
import json
import logging
import random
import string
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

from aiohttp import ClientTimeout

from nemo_gym.server_utils import request


logger = logging.getLogger(__name__)


_SEED_SQL_CACHE: Dict[str, str] = {}


def load_seed_sql(sql_file_path: str) -> str:
    """Read a seed SQL file, caching contents in memory (files are reused across many rollouts)."""
    cached = _SEED_SQL_CACHE.get(sql_file_path)
    if cached is None:
        with open(sql_file_path, "r", encoding="utf-8") as f:
            cached = f.read()
        _SEED_SQL_CACHE[sql_file_path] = cached
    return cached


def generate_database_id() -> str:
    """Generate a unique database id, matching the EOG format (db_<millis>_<9 alnum>)."""
    timestamp = int(time.time() * 1000)
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=9))
    return f"db_{timestamp}_{suffix}"


def context_to_headers(context: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """Convert EOG task context key-values to x-* headers (user_id -> x-user-id), as EOG does."""
    headers: Dict[str, str] = {}
    if context and isinstance(context, dict):
        for key, value in context.items():
            if not key.lower().startswith("x-"):
                header_key = f"x-{key.lower().replace('_', '-')}"
            else:
                header_key = key
            headers[header_key] = str(value)
    return headers


class MCPGymClient:
    """JSON-RPC (streamable HTTP) MCP client for one EnterpriseOps gym server.

    One instance per (gym server URL) — NOT per rollout. The MCP session handshake runs
    once lazily; per-rollout isolation comes from the ``x-database-id`` header on each call.
    """

    def __init__(
        self,
        base_url: str,
        auth_config: Optional[Dict[str, Any]] = None,
        mcp_endpoint: str = "/mcp",
        tool_call_timeout_seconds: float = 30.0,
        sql_timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.mcp_endpoint = mcp_endpoint
        self.auth_config = auth_config
        self.tool_call_timeout_seconds = tool_call_timeout_seconds
        self.sql_timeout_seconds = sql_timeout_seconds

        self.mcp_session_id: Optional[str] = None
        self._request_id = 1
        self._initialize_lock = asyncio.Lock()

    def _next_request_id(self) -> int:
        self._request_id += 1
        return self._request_id

    def _get_auth_headers(self) -> Dict[str, str]:
        """Get authentication headers for HTTP requests (verbatim EOG semantics)."""
        if not self.auth_config:
            return {}

        auth_type = self.auth_config.get("type")
        token = self.auth_config.get("token")
        header_name = self.auth_config.get("header_name", "Authorization")

        if auth_type == "bearer":
            return {header_name: f"Bearer {token}"}
        elif auth_type == "api_key":
            return {header_name: token}

        return {}

    async def _send_jsonrpc(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        is_notification: bool = False,
        timeout_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Send a JSON-RPC request (or notification) to the MCP endpoint."""
        payload: Dict[str, Any] = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        if not is_notification:
            payload["id"] = self._next_request_id()

        headers = {"Accept": "application/json, text/event-stream"}
        headers.update(self._get_auth_headers())
        if self.mcp_session_id:
            headers["mcp-session-id"] = self.mcp_session_id
        if extra_headers:
            headers.update(extra_headers)

        try:
            response = await request(
                method="POST",
                url=f"{self.base_url}{self.mcp_endpoint}",
                json=payload,
                headers=headers,
                timeout=ClientTimeout(total=timeout_seconds or self.tool_call_timeout_seconds),
            )
        except Exception as e:
            logger.error(f"MCP request exception: {e}")
            return {"success": False, "error": str(e)}

        # Capture session ID from response headers
        if "mcp-session-id" in response.headers:
            self.mcp_session_id = response.headers["mcp-session-id"]

        if is_notification:
            return {"success": response.status in (200, 204), "status_code": response.status}

        if response.status == 200:
            data = json.loads(await response.read())
            return {"success": True, "data": data}

        error_msg = f"MCP request failed: {response.status} - {(await response.read()).decode(errors='replace')}"
        logger.error(error_msg)
        return {"success": False, "error": error_msg}

    async def ensure_initialized(self) -> None:
        """Run the MCP initialize handshake once per client (concurrency-safe)."""
        if self.mcp_session_id is not None:
            return
        async with self._initialize_lock:
            if self.mcp_session_id is not None:
                return
            params = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "nemo-gym-enterpriseops", "version": "1.0.0"},
            }
            result = await self._send_jsonrpc("initialize", params)
            if not result.get("success"):
                raise RuntimeError(f"Failed to initialize MCP session with {self.base_url}: {result.get('error')}")
            await self._send_jsonrpc("notifications/initialized", {}, is_notification=True)
            # Some servers do not set an mcp-session-id header. Use a sentinel so we don't re-initialize.
            if self.mcp_session_id is None:
                self.mcp_session_id = ""

    async def list_tools(self) -> List[Dict[str, Any]]:
        """List all available tools (raw MCP tool dicts with inputSchema)."""
        await self.ensure_initialized()
        result = await self._send_jsonrpc("tools/list", {})
        if result.get("success"):
            return result.get("data", {}).get("result", {}).get("tools", [])
        return []

    async def call_tool(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
        database_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Call a tool against a specific database. Mirrors the EOG return shape."""
        await self.ensure_initialized()

        extra_headers: Dict[str, str] = {}
        if database_id:
            extra_headers["x-database-id"] = database_id
        extra_headers.update(context_to_headers(context))

        params = {"name": tool_name, "arguments": arguments or {}}
        result = await self._send_jsonrpc("tools/call", params, extra_headers=extra_headers)
        if result.get("success"):
            data = result.get("data", {})
            return {"success": True, "result": data.get("result"), "error": data.get("error")}
        return result

    async def run_sql(
        self,
        query: str,
        database_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a SQL query via the gym's /api/sql-runner endpoint (EOG verifier transport)."""
        headers = {"x-database-id": database_id}
        headers.update(self._get_auth_headers())
        headers.update(context_to_headers(context))

        try:
            response = await request(
                method="POST",
                url=f"{self.base_url}/api/sql-runner",
                json={"query": query, "database_id": database_id},
                headers=headers,
                timeout=ClientTimeout(total=self.sql_timeout_seconds),
            )
        except Exception as e:
            logger.error(f"SQL query execution failed: {e}")
            return {"success": False, "error": str(e)}

        if response.status == 200:
            return {"success": True, "result": json.loads(await response.read())}

        # Mirror EOG's HTTP error detail extraction
        response_text = (await response.read()).decode(errors="replace")
        try:
            error_json = json.loads(response_text)
            if isinstance(error_json, dict):
                error_details = error_json.get("detail", error_json.get("message", response_text))
            else:
                error_details = response_text
        except json.JSONDecodeError:
            error_details = response_text

        logger.error(f"HTTP error calling sql-runner API: {response.status} - {error_details}")
        return {"success": False, "error": f"SQL API call failed (HTTP {response.status}): {error_details}"}

    async def seed_database(self, sql_content: str, seed_semaphore: Optional[asyncio.Semaphore] = None) -> str:
        """Create a new database from SQL content and return its database_id.

        Timeout formula matches EOG: max(1200, 120 + len(sql)/102400) seconds.
        """
        database_id = generate_database_id()
        db_name = f"Auto DB {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        payload = {
            "database_id": database_id,
            "name": db_name,
            "description": "Auto-created by nemo-gym enterpriseops_gym resources server",
            "sql_content": sql_content,
        }
        timeout_seconds = max(1200, int(120 + len(sql_content) / 102400))

        async def _do_seed() -> None:
            response = await request(
                method="POST",
                url=f"{self.base_url}/api/seed-database",
                json=payload,
                headers=self._get_auth_headers(),
                timeout=ClientTimeout(total=timeout_seconds),
            )
            if response.status != 200:
                body = (await response.read()).decode(errors="replace")
                raise RuntimeError(f"Failed to seed database on {self.base_url} (HTTP {response.status}): {body}")

        if seed_semaphore is not None:
            async with seed_semaphore:
                await _do_seed()
        else:
            await _do_seed()

        logger.info(f"Seeded database {database_id} on {self.base_url}")
        return database_id

    async def delete_database(self, database_id: str) -> bool:
        """Delete a database. Tolerates servers without a delete API (404/405), as EOG does."""
        try:
            response = await request(
                method="DELETE",
                url=f"{self.base_url}/api/delete-database",
                json={"database_id": database_id},
                headers=self._get_auth_headers(),
                timeout=ClientTimeout(total=30),
            )
        except Exception as e:
            logger.error(f"Error deleting database {database_id}: {e}")
            return False

        if response.status in (404, 405):
            logger.warning(f"Server {self.base_url} does not support database deletion (HTTP {response.status})")
            return False
        if response.status != 200:
            body = (await response.read()).decode(errors="replace")
            logger.error(f"Error deleting database {database_id} (HTTP {response.status}): {body}")
            return False

        logger.info(f"Deleted database {database_id} on {self.base_url}")
        return True
