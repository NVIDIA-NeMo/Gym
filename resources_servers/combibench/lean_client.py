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

import asyncio
import logging
import uuid
from typing import Any, Optional

from aiohttp import ClientTimeout

from nemo_gym.server_utils import request
from resources_servers.combibench.fine_eval import LeanResult
from resources_servers.lean_proof.toolchain import TOOLCHAIN_PROBE, parse_lean_version


LOG = logging.getLogger(__name__)

# Network and REPL start-up allowance on top of the Lean timeout, so a
# legitimate slow compile is reported by the server as its own timeout rather
# than cut off by the client first.
HTTP_TIMEOUT_MARGIN_SECONDS = 30.0

# The Kimina default (``LEAN_SERVER_MAX_REPLS``). Rollout fan-out is unbounded,
# so without a matching bound here every extra request is a connection queued
# against a server that can only run this many REPLs anyway.
DEFAULT_MAX_CONCURRENCY = 8

# Kimina exposes no version endpoint, so the toolchain is read by compiling a
# one-line program through the same path a submission takes. ``TOOLCHAIN_PROBE``
# is the shared one, so every Lean benchmark asks the question the same way; its
# header is the one submissions use, so the probe also pays the cold
# ``import Mathlib`` that would otherwise land on the first scored rollout.


class KiminaLeanClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, max_concurrency: int = DEFAULT_MAX_CONCURRENCY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._version: Optional[str] = None
        self._version_probed = False
        self._version_lock = asyncio.Lock()

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
            async with self._semaphore:
                response = await request(
                    "POST",
                    f"{self.base_url}/verify",
                    json=payload,
                    headers=self._headers(),
                    timeout=ClientTimeout(total=timeout_seconds + HTTP_TIMEOUT_MARGIN_SECONDS),
                    # A compile that exhausts the client timeout has already cost
                    # the server a REPL for that long. Retrying it twice more
                    # triples the cost and cannot change the answer, so the shared
                    # client's default of 3 tries is turned off here.
                    _max_connection_retries=1,
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

    async def toolchain_version(self, timeout_seconds: int = 120) -> Optional[str]:
        """Lean version the server actually runs, probed once and cached.

        A server built for another Lean version scores every row
        ``proof_failed`` and nothing in the rollouts says why. The probe pays
        one cold ``import`` on the first call. ``None`` means the probe itself
        did not return a version; it is cached like a hit, so an unreachable
        server is not re-probed once per rollout.
        """
        if self._version_probed:
            return self._version
        async with self._version_lock:
            if self._version_probed:
                return self._version
            result = await self.verify(TOOLCHAIN_PROBE, timeout_seconds)
            # parse_lean_version reads a sandbox's stdout/stderr; the REPL answers in
            # structured messages instead, so they are joined into the shape it expects
            # rather than the version regex being written a second time.
            self._version = parse_lean_version(
                {"stdout": "\n".join(str(message.get("data", "")) for message in result.messages)}
            )
            self._version_probed = True
            if self._version is not None:
                LOG.info("Lean server at %s reports Lean %s", self.base_url, self._version)
            else:
                LOG.warning(
                    "Lean version probe returned no version (error=%r): the toolchain behind %s is unknown",
                    result.error,
                    self.base_url,
                )
        return self._version


# Keys that mark the per-item ``response`` object as something other than a
# command response. Kimina's ``Error`` TypedDict is ``{"message": str}``
# (``client/kimina_client/models.py``); ``error`` and ``stderr`` are the two
# upstream CombiBench's own ``is_error`` looks for in the same object before it
# reads ``messages`` (``evaluation/client/lean_client.py``).
PAYLOAD_ERROR_KEYS = ("message", "error", "stderr")


def _payload_failure(payload: dict[str, Any]) -> Optional[str]:
    """Describe a per-item ``response`` that is not a Lean verdict, else None.

    The top-level ``error`` is not the only way ``/verify`` reports a failure.
    ``server/repl.py`` returns ``ReplResponse(response=cmd_response)`` with
    ``cmd_response`` taken straight from ``json.loads`` of the REPL's stdout and
    validated nowhere, and the client's ``extend()`` explicitly admits the
    ``{"message": ...}`` shape, so a reply can carry an Error object in
    ``response`` with no top-level ``error`` at all. Reading only the outer key
    would leave such a reply with no error, no messages and no sorries — which
    is exactly what a clean compile looks like — and reward it.

    This fails closed, and it fails to ``transport_failure`` (``sandbox_error``,
    masked) rather than to a model-attributable status: a REPL that answered
    with an Error object instead of a command response did not evaluate the
    model's proof, so there is no verdict to charge to the model. Upstream fails
    the submission on ``error``/``stderr`` instead, which reaches the same reward
    of 0.0 by a different route; the difference only shows up in whether the
    rollout is counted in the denominator, and a non-verdict should not be.
    """
    for key in PAYLOAD_ERROR_KEYS:
        if payload.get(key):
            return f"Lean server reported {key}: {str(payload[key])[:500]}"
    return None


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
    if not error:
        failed = _payload_failure(payload)
        if failed is not None:
            LOG.warning("Lean server returned an error payload: %s", failed)
            return LeanResult(error=failed, transport_failure=True)
    messages = payload.get("messages") or []
    sorries = payload.get("sorries") or []
    return LeanResult(
        error=str(error) if error else None,
        messages=[m for m in messages if isinstance(m, dict)],
        sorries=[s for s in sorries if isinstance(s, dict)],
        time=payload.get("time") if isinstance(payload.get("time"), (int, float)) else None,
    )
