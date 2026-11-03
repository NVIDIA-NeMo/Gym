# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP client for one EnterpriseOps-Gym server running in a sandbox.

A gym server answers MCP JSON-RPC at ``/mcp`` with plain JSON responses and no MCP session, and a REST API that
seeds (``/api/seed-database``) and queries (``/api/sql-runner``) per-task SQLite databases. Every call names its
database with the ``x-database-id`` header; the task's context, such as the acting user's token, travels as further
``x-`` headers.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import time
from dataclasses import dataclass
from typing import Any, Optional

from aiohttp import ClientError, ClientTimeout

from nemo_gym.client import request
from nemo_gym.sandbox import SandboxEndpoint
from resources_servers.enterpriseops_gym.task_data import DATABASE_ID_HEADER


# The proxy in front of a sandbox port answers these while the server behind it is unreachable.
_UNAVAILABLE_STATUSES = frozenset({502, 503, 504})


class GymUnavailableError(RuntimeError):
    """The gym server could not be reached, or answered as if it were down: an infrastructure failure."""


class GymRequestError(RuntimeError):
    """The gym server refused a request, such as a seed whose SQL does not run or a verifier query with an error."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class ToolCallResult:
    """What a tool call returned: the text the model sees, and whether the server flagged it as an error."""

    text: str
    is_error: bool


class GymServerClient:
    """Calls one gym server through its sandbox endpoint."""

    def __init__(self, endpoint: SandboxEndpoint, *, verify_tls: bool, timeout_s: float) -> None:
        self._base_url = endpoint.endpoint.rstrip("/")
        self._headers = dict(endpoint.headers)
        self._ssl = None if verify_tls else False
        self._timeout = ClientTimeout(total=timeout_s)
        self._request_ids = itertools.count(1)

    async def _post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        headers: Optional[dict[str, str]] = None,
        resend_after_disconnect: bool = True,
    ) -> tuple[int, Any]:
        """POST JSON and return the status and the decoded body; raise GymUnavailableError for infrastructure failures."""
        try:
            response = await request(
                "POST",
                f"{self._base_url}{path}",
                json=body,
                # The endpoint's provider headers last, so task context cannot replace them.
                headers={"Accept": "application/json", **(headers or {}), **self._headers},
                ssl=self._ssl,
                timeout=self._timeout,
                _max_num_tries=3,
                _resend_after_disconnect=resend_after_disconnect,
            )
            raw = await response.read()
        except (ClientError, asyncio.TimeoutError) as error:
            raise GymUnavailableError(f"{path}: {type(error).__name__}: {error}") from error
        if response.status in _UNAVAILABLE_STATUSES:
            raise GymUnavailableError(f"{path}: HTTP {response.status}: {raw[:500].decode(errors='replace')}")
        try:
            return response.status, json.loads(raw) if raw else None
        except ValueError:
            return response.status, raw.decode(errors="replace")

    async def wait_until_healthy(self, timeout_s: float, *, poll_interval_s: float = 1.0) -> None:
        """Poll ``/health`` until the server answers 200."""
        deadline = time.monotonic() + timeout_s
        last_error = "no answer"
        while True:
            try:
                response = await request(
                    "GET",
                    f"{self._base_url}/health",
                    headers=self._headers,
                    ssl=self._ssl,
                    timeout=ClientTimeout(total=min(10.0, timeout_s)),
                    _max_num_tries=1,
                )
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
            except (ClientError, asyncio.TimeoutError) as error:
                last_error = f"{type(error).__name__}: {error}"
            if time.monotonic() + poll_interval_s > deadline:
                raise GymUnavailableError(f"gym server not healthy after {timeout_s:g}s: {last_error}")
            await asyncio.sleep(poll_interval_s)

    async def seed_database(self, database_id: str, sql: str) -> None:
        """Create the database ``database_id`` from a SQL dump."""
        status, body = await self._post(
            "/api/seed-database",
            {
                "database_id": database_id,
                "name": database_id,
                "description": "NeMo Gym task database",
                "sql_content": sql,
            },
            resend_after_disconnect=False,
        )
        # Gym servers word their success differently; like upstream, a 200 that does not report a failure is one.
        if status != 200 or (isinstance(body, dict) and body.get("success") is False):
            raise GymRequestError(status, _detail(body))

    async def call_tool(self, name: str, arguments: dict[str, Any], *, headers: dict[str, str]) -> ToolCallResult:
        """Call an MCP tool. A call that reached the server is never sent twice: tools change the database."""
        status, body = await self._post(
            "/mcp",
            {
                "jsonrpc": "2.0",
                "id": next(self._request_ids),
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
            headers=headers,
            resend_after_disconnect=False,
        )
        if status != 200 or not isinstance(body, dict):
            return ToolCallResult(text=f"Tool call failed with HTTP {status}: {_detail(body)}", is_error=True)
        if "error" in body:
            return ToolCallResult(text=f"Tool call failed: {_detail(body['error'])}", is_error=True)
        result = body.get("result") or {}
        texts = [block.get("text", "") for block in result.get("content") or [] if block.get("type") == "text"]
        return ToolCallResult(text="\n".join(texts), is_error=bool(result.get("isError")))

    async def run_sql(self, query: str, *, headers: dict[str, str]) -> dict[str, Any]:
        """Run a read query against the database named in ``headers`` and return the server's JSON answer."""
        status, body = await self._post(
            "/api/sql-runner", {"query": query, "database_id": headers[DATABASE_ID_HEADER]}, headers=headers
        )
        if status != 200 or not isinstance(body, dict):
            raise GymRequestError(status, _detail(body))
        return body


def _detail(body: Any) -> str:
    if isinstance(body, dict):
        for key in ("detail", "message", "error"):
            if body.get(key):
                return str(body[key])[:2000]
    return str(body)[:2000]
