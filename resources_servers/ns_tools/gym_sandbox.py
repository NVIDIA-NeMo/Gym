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
"""NeMo-Skills backend that routes sessions through a shared sandbox pool.

The resources server owns the pool lifetime and registers this backend when selected.
Subclassing ``LocalSandbox`` preserves its request and session behavior while replacing
only the HTTP transport.
"""

import json
import logging
from typing import Any, Dict

import httpx
from nemo_skills.code_execution import sandbox as ns_sandbox
from sandbox_pool import SandboxPool, sandbox_request


LOGGER = logging.getLogger(__name__)


class GymSandbox(ns_sandbox.LocalSandbox):
    """LocalSandbox with service requests routed through the pooled sandbox exec API."""

    def __init__(self, pool: SandboxPool, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._pool = pool

    async def _send_request(self, request: Dict[str, Any], timeout: float):
        session_id = request.pop("session_id", None)
        sandbox = await self._pool.route(str(session_id) if session_id is not None else None)
        headers = {"Content-Type": "application/json"}
        if session_id is not None:
            headers["X-Session-ID"] = str(session_id)
        status, text = await sandbox_request(
            sandbox,
            self._pool._port,
            "POST",
            "/execute",
            headers=headers,
            payload=json.dumps(request),
            timeout_s=timeout + 5.0,
        )
        if status != 200:
            # Normalize every infra failure to the shape the NS client already tolerates.
            raise httpx.TimeoutException(f"sandbox pool transport returned HTTP {status}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            LOGGER.error("Error during parsing output: %s", text[:500])
            return {"process_status": "error", "stdout": "", "stderr": "Unknown error"}

    async def delete_session(self, session_id: str) -> None:
        """Delete the session on the pod it is pinned to, then release the pin."""
        try:
            if str(session_id) not in self._pool._session_to_slot:
                # Avoid pinning an already-deleted session just to receive a 404.
                return
            sandbox = await self._pool.route(str(session_id))
            status, _ = await sandbox_request(
                sandbox,
                self._pool._port,
                "DELETE",
                f"/sessions/{session_id}",
                headers={"X-Session-ID": str(session_id)},
                timeout_s=10.0,
            )
            if status not in (200, 404):
                LOGGER.warning("delete_session %s returned HTTP %d", session_id, status)
        except httpx.TimeoutException as exc:
            LOGGER.warning("delete_session %s failed (pod TTL/idle reaper will clean up): %s", session_id, exc)
        finally:
            self._pool.release(str(session_id))
            self.session_histories.pop(str(session_id), None)
