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

"""Client for the Kimina Lean Server ``/verify`` endpoint.

CombiBench's upstream harness checks proofs through
https://github.com/project-numina/kimina-lean-server (MIT). The server wraps
the Lean REPL, splits each submission into an import header and a body, and
reuses a REPL that has already loaded the same header, so ``import Mathlib``
costs seconds once rather than per proof.

Only the backward-compatible ``/verify`` route is used, with the request shape
upstream's ``Lean4Client`` sends. All HTTP goes through Gym's shared aiohttp
client, as the repository requires.
"""

import logging
import uuid
from typing import Any, Optional

from aiohttp import ClientTimeout

from nemo_gym.server_utils import request
from resources_servers.combibench.fine_eval import LeanResult


LOG = logging.getLogger(__name__)

# Network and REPL start-up allowance on top of the Lean timeout, so a
# legitimate slow compile is reported by the server as its own timeout rather
# than cut off by the client first.
HTTP_TIMEOUT_MARGIN_SECONDS = 30.0


class KiminaLeanClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def verify(self, code: str, timeout_seconds: int) -> LeanResult:
        """Compile ``code`` once and report what Lean said.

        Any failure to obtain a well-formed reply is a transport failure: the
        model cannot cause it, so callers attribute it to the harness.
        """
        payload = {
            "codes": [{"custom_id": uuid.uuid4().hex, "proof": code}],
            "timeout": int(timeout_seconds),
            "infotree_type": None,
            "disable_cache": False,
        }
        try:
            response = await request(
                "POST",
                f"{self.base_url}/verify",
                json=payload,
                headers=self._headers(),
                timeout=ClientTimeout(total=timeout_seconds + HTTP_TIMEOUT_MARGIN_SECONDS),
            )
            if response.status != 200:
                text = await response.text()
                LOG.warning("Lean server returned HTTP %s: %s", response.status, text[:500])
                return LeanResult(error=f"HTTP {response.status}: {text[:500]}", transport_failure=True)
            body = await response.json()
        except Exception as exc:  # network errors, timeouts, bad JSON
            LOG.warning("Lean server request failed: %r", exc)
            return LeanResult(error=f"{type(exc).__name__}: {exc}", transport_failure=True)
        return parse_verify_response(body)


def parse_verify_response(body: Any) -> LeanResult:
    """Turn the ``/verify`` JSON body into a ``LeanResult``.

    The reply is ``{"results": [{"custom_id", "error", "response": {"messages",
    "sorries", "env", "time"}}]}``. A body without exactly one result is a
    transport failure, not a Lean verdict.
    """
    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        return LeanResult(error="malformed Lean server reply", transport_failure=True)
    result = results[0]
    error = result.get("error")
    payload = result.get("response") or {}
    if not isinstance(payload, dict):
        payload = {}
    messages = payload.get("messages") or []
    sorries = payload.get("sorries") or []
    return LeanResult(
        error=str(error) if error else None,
        messages=[m for m in messages if isinstance(m, dict)],
        sorries=[s for s in sorries if isinstance(s, dict)],
        time=payload.get("time") if isinstance(payload.get("time"), (int, float)) else None,
    )
