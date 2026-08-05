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

import asyncio
import json

import aiohttp


_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json, text/event-stream",
    "MCP-Protocol-Version": "2025-06-18",
}


class MCPClient:
    def __init__(self, url, timeout=45, pool_size=8, max_concurrency=16):
        self.url = url
        self.timeout = timeout
        self.pool_size = pool_size
        self._session = None
        self._sids = []
        self._rr = 0
        self._id = 0
        self._init_lock = asyncio.Lock()
        self._sem = asyncio.Semaphore(max_concurrency)

    async def _http(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def _post(self, method, params=None, notify=False, sid=None):
        self._id += 1
        msg = {"jsonrpc": "2.0", "method": method}
        if not notify:
            msg["id"] = self._id
        if params is not None:
            msg["params"] = params
        headers = dict(_HEADERS)
        if sid:
            headers["Mcp-Session-Id"] = sid
        session = await self._http()
        async with session.post(
            self.url,
            data=json.dumps(msg).encode(),
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=self.timeout),
        ) as resp:
            new_sid = resp.headers.get("Mcp-Session-Id")
            ctype = resp.headers.get("Content-Type") or ""
            body = await resp.text()
        parsed = None
        if not notify:
            if "text/event-stream" in ctype:
                for line in body.splitlines():
                    if line.startswith("data:"):
                        try:
                            parsed = json.loads(line[5:].strip())
                            break
                        except Exception:
                            pass
                if parsed is None:
                    parsed = {}
            else:
                parsed = json.loads(body)
        return parsed, new_sid

    async def _open_session(self):
        _, sid = await self._post(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "nemo-gym", "version": "0.1"},
            },
        )
        if sid:
            await self._post("notifications/initialized", notify=True, sid=sid)
        return sid

    async def _ensure_pool(self):
        if self._sids:
            return
        async with self._init_lock:
            if self._sids:
                return
            sids = []
            for _ in range(self.pool_size):
                try:
                    sid = await self._open_session()
                    if sid:
                        sids.append(sid)
                except Exception:
                    pass
            self._sids = sids or [None]

    def _next_sid(self):
        self._rr = (self._rr + 1) % len(self._sids)
        return self._sids[self._rr]

    async def list_tools(self):
        await self._ensure_pool()
        result, _ = await self._post("tools/list", {}, sid=self._sids[0])
        return ((result or {}).get("result") or {}).get("tools") or []

    async def call_tool(self, name, arguments):
        async with self._sem:
            try:
                await self._ensure_pool()
                result, _ = await self._post(
                    "tools/call", {"name": name, "arguments": arguments}, sid=self._next_sid()
                )
            except Exception as exc:
                return f"Tool call failed: {exc}"
        items = ((result or {}).get("result") or {}).get("content") or []
        blocks = [it.get("text", "") for it in items if it.get("type") == "text" and it.get("text")]
        return "\n\n".join(blocks) if blocks else "No results."
