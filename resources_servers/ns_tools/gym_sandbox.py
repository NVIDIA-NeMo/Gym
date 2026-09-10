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
"""NeMo-Skills backend that routes sessions to sandboxes owned by the resources server.

The resources server owns the sandbox lifetimes (a shared :class:`sandbox_pool.SandboxPool`
or one-per-session :class:`session_sandboxes.SessionSandboxes`) and registers this backend
when selected. Subclassing ``LocalSandbox`` preserves its request and session behavior
while replacing only the HTTP transport. The owning backend only needs ``request``,
``request_existing``, ``release`` and ``report_failure``; how a request reaches the
sandbox (exec+curl or direct HTTP through the endpoint proxy) is the backend's choice.
"""

import json
import logging
from typing import Any, Dict

import httpx
from nemo_skills.code_execution import sandbox as ns_sandbox


LOGGER = logging.getLogger(__name__)


class GymSandbox(ns_sandbox.LocalSandbox):
    """LocalSandbox with service requests routed through the owning backend."""

    def __init__(self, pool: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pool = pool

    async def _send_request(self, request: Dict[str, Any], timeout: float):
        session_id = request.pop("session_id", None)
        sid = str(session_id) if session_id is not None else None
        headers = {"Content-Type": "application/json"}
        if sid is not None:
            headers["X-Session-ID"] = sid
        try:
            status, text = await self._pool.request(
                sid, "POST", "/execute", headers=headers, payload=json.dumps(request), timeout_s=timeout + 5.0
            )
        except httpx.TimeoutException:
            await self._pool.report_failure(sid)
            raise
        if status != 200:
            await self._pool.report_failure(sid)
            # Normalize every infra failure to the shape the NS client already tolerates.
            raise httpx.TimeoutException(f"sandbox pool transport returned HTTP {status}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            LOGGER.error("Error during parsing output: %s", text[:500])
            return {"process_status": "error", "stdout": "", "stderr": "Unknown error"}

    async def delete_session(self, session_id: str) -> None:
        """Delete the IPython session on the sandbox that holds it, then forget the binding.

        These are nemo_skills' session semantics (also invoked mid-rollout before a state
        restore); the sandbox itself belongs to the owning backend (shared pool: stays
        up; per-session: deleted when ``app.py`` ends the rollout session).
        """
        sid = str(session_id)
        try:
            # request_existing never creates a sandbox just to receive a 404.
            result = await self._pool.request_existing(
                sid, "DELETE", f"/sessions/{sid}", headers={"X-Session-ID": sid}, timeout_s=10.0
            )
            if result is not None and result[0] not in (200, 404):
                LOGGER.warning("delete_session %s returned HTTP %d", sid, result[0])
        except httpx.TimeoutException as exc:
            LOGGER.warning("delete_session %s failed (sandbox teardown/TTL will clean up): %s", sid, exc)
        finally:
            self._pool.release(sid)
            self.session_histories.pop(sid, None)
