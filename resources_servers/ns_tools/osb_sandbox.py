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
class IS a ``LocalSandbox`` — same request preparation, same response parsing, same session
bookkeeping — with the transport re-pointed: each request resolves (base_url, headers) from
the pool by session uuid instead of using a fixed host:port. Anything non-200 from the
transport is normalized to the NS timeout contract so infra failures degrade rewards
without ever surfacing new error shapes to the model.

Importing this module is the opt-in: the default ``local`` backend never imports it.
"""

import json
import logging
from typing import Any, Dict, Optional

import httpx
from nemo_skills.code_execution import sandbox as ns_sandbox

from osb_pool import OpenSandboxPool


LOGGER = logging.getLogger(__name__)

# The pool the owning server can warm up at lifespan startup (set by the first construction).
CURRENT_POOL: Optional[OpenSandboxPool] = None


class OpenSandboxPoolSandbox(ns_sandbox.LocalSandbox):
    """LocalSandbox with the transport routed through an OpenSandbox pod pool."""

    def __init__(self, pool: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if not pool:
            raise ValueError("sandbox_type=opensandbox_pool requires a 'pool' config dict")
        global CURRENT_POOL
        self._pool = OpenSandboxPool(**pool)
        CURRENT_POOL = self._pool

    async def _send_request(self, request: Dict[str, Any], timeout: float):
        session_id = request.pop("session_id", None)
        base_url, pool_headers = await self._pool.route(str(session_id) if session_id is not None else None)
        headers = {"Content-Type": "application/json", **pool_headers}
        if session_id is not None:
            headers["X-Session-ID"] = str(session_id)

        output = await self.http_session.post(
            url=f"{base_url}/execute",
            content=json.dumps(request),
            timeout=timeout + 5.0,
            headers=headers,
        )
        if output.status_code == 502:
            # A proxy-minted 502 means the pod never received the request, so ONE retry is
            # idempotency-safe even for stateful ipython.
            output = await self.http_session.post(
                url=f"{base_url}/execute",
                content=json.dumps(request),
                timeout=timeout + 5.0,
                headers=headers,
            )
        if output.status_code != 200:
            # Normalize every infra failure to the shape the NS client already tolerates.
            raise httpx.TimeoutException(f"sandbox pool transport returned HTTP {output.status_code}")
        return self._parse_request_output(output)

    async def delete_session(self, session_id: str) -> None:
        """Delete the session on the pod it is pinned to, then release the pin."""
        try:
            base_url, pool_headers = await self._pool.route(str(session_id))
        except httpx.TimeoutException:
            self._pool.release(str(session_id))
            self.session_histories.pop(str(session_id), None)
            return
        try:
            response = await self.http_session.delete(
                url=f"{base_url}/sessions/{session_id}",
                timeout=10.0,
                headers={**pool_headers, "X-Session-ID": str(session_id)},
            )
            if response.status_code not in (200, 404):
                LOGGER.warning("delete_session %s returned HTTP %d", session_id, response.status_code)
        except httpx.HTTPError as exc:
            LOGGER.warning("delete_session %s failed (pod TTL/idle reaper will clean up): %s", session_id, exc)
        finally:
            self._pool.release(str(session_id))
            self.session_histories.pop(str(session_id), None)

    async def close(self) -> None:
        try:
            await self._pool.aclose()
        finally:
            await super().close()


ns_sandbox.sandboxes["opensandbox_pool"] = OpenSandboxPoolSandbox
