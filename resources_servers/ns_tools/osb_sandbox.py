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
"""NeMo-Skills sandbox backend that routes through an OpenSandbox pod pool.

Registers ``sandbox_type: opensandbox_pool`` with the nemo_skills sandbox registry. The
class IS a ``LocalSandbox`` — same request preparation, same session bookkeeping — with the
transport re-pointed: each request resolves (base_url, headers) from the pool by session
uuid, and rides a shared AIOHTTP session (httpx/httpcore's O(n^2) connection pooling
collapses at high concurrency — see CLAUDE.md; measured on cell-2: health-only GETs fell
from 87 to 8 calls/s between 64 and 512 in-flight on httpx). Exception TYPES stay httpx
because the nemo_skills base class's execute_code catches those; anything non-200 or
transport-level is normalized to the NS timeout contract so infra failures degrade rewards
without new error shapes.

Importing this module is the opt-in: the default ``local`` backend never imports it.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import aiohttp
import httpx
from nemo_skills.code_execution import sandbox as ns_sandbox

from osb_pool import OpenSandboxPool


LOGGER = logging.getLogger(__name__)

# The pool the owning server can warm up at lifespan startup (set by the first construction).
CURRENT_POOL: Optional[OpenSandboxPool] = None


class OpenSandboxPoolSandbox(ns_sandbox.LocalSandbox):
    """LocalSandbox with the transport routed through an OpenSandbox pod pool over aiohttp."""

    def __init__(self, pool: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not pool:
            raise ValueError("sandbox_type=opensandbox_pool requires a 'pool' config dict")
        global CURRENT_POOL
        self._pool = OpenSandboxPool(**pool)
        self._aiohttp: Optional[aiohttp.ClientSession] = None
        CURRENT_POOL = self._pool

    def _session(self) -> aiohttp.ClientSession:
        if self._aiohttp is None or self._aiohttp.closed:
            connector = aiohttp.TCPConnector(limit=4096, limit_per_host=4096, ttl_dns_cache=300)
            self._aiohttp = aiohttp.ClientSession(connector=connector)
        return self._aiohttp

    @staticmethod
    def _parse_output_text(text: str) -> Dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            LOGGER.error("Error during parsing output: %s", text[:500])
            return {"process_status": "error", "stdout": "", "stderr": "Unknown error"}

    async def _post_execute(self, base_url: str, headers: Dict[str, str], payload: str, timeout: float):
        async with self._session().post(
            f"{base_url}/execute",
            data=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=timeout + 5.0),
        ) as response:
            return response.status, await response.text()

    async def _send_request(self, request: Dict[str, Any], timeout: float):
        session_id = request.pop("session_id", None)
        base_url, pool_headers = await self._pool.route(str(session_id) if session_id is not None else None)
        headers = {"Content-Type": "application/json", **pool_headers}
        if session_id is not None:
            headers["X-Session-ID"] = str(session_id)
        payload = json.dumps(request)

        try:
            status, text = await self._post_execute(base_url, headers, payload, timeout)
            if status == 502:
                # A proxy-minted 502 means the pod never received the request, so ONE retry
                # is idempotency-safe even for stateful ipython.
                status, text = await self._post_execute(base_url, headers, payload, timeout)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            raise httpx.TimeoutException(f"sandbox pool transport error: {exc!r}") from exc
        if status != 200:
            # Normalize every infra failure to the shape the NS client already tolerates.
            raise httpx.TimeoutException(f"sandbox pool transport returned HTTP {status}")
        return self._parse_output_text(text)

    async def delete_session(self, session_id: str) -> None:
        """Delete the session on the pod it is pinned to, then release the pin."""
        try:
            base_url, pool_headers = await self._pool.route(str(session_id))
        except httpx.TimeoutException:
            self._pool.release(str(session_id))
            self.session_histories.pop(str(session_id), None)
            return
        try:
            async with self._session().delete(
                f"{base_url}/sessions/{session_id}",
                headers={**pool_headers, "X-Session-ID": str(session_id)},
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as response:
                if response.status not in (200, 404):
                    LOGGER.warning("delete_session %s returned HTTP %d", session_id, response.status)
        except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
            LOGGER.warning("delete_session %s failed (pod TTL/idle reaper will clean up): %s", session_id, exc)
        finally:
            self._pool.release(str(session_id))
            self.session_histories.pop(str(session_id), None)

    async def close(self) -> None:
        try:
            await self._pool.aclose()
        finally:
            if self._aiohttp is not None and not self._aiohttp.closed:
                await self._aiohttp.close()
            await super().close()


ns_sandbox.sandboxes["opensandbox_pool"] = OpenSandboxPoolSandbox
