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

# The ``LEAN_SERVER_MAX_REPLS`` this repository's own ``kimina_image`` sets, *not*
# a Kimina default: Kimina's default is ``max((os.cpu_count() or 1) - 1, 1)``
# (``server/settings.py``), which is a property of whatever host it lands on and
# so cannot be matched from here. Rollout fan-out is unbounded, so without a bound
# matching the server's every extra request is a connection queued against a
# server that can only run that many REPLs anyway. Operators pointing this at a
# differently configured server must set both sides to the same number.
DEFAULT_MAX_CONCURRENCY = 8

# "Try again": the server is out of REPLs (429) or not accepting work yet (503).
# Unlike a timeout, these cost the server no REPL time, and they are what a
# saturated server answers, so they are worth another try rather than becoming a
# masked rollout that shrinks the denominator.
SATURATION_STATUSES = (429, 503)
MAX_SATURATION_ATTEMPTS = 3
SATURATION_BACKOFF_SECONDS = 1.0  # doubled each attempt: 1s, then 2s

# How many times the toolchain probe may be re-attempted across a run before the
# mismatch guard is given up on. Bounded because the probe pays a cold
# ``import Mathlib``; more than one because caching a transient failure as an
# answer would silently switch the guard off for the whole run.
MAX_VERSION_PROBES = 3

# Lean timeout for the probe. Shorter than a submission's because nothing waits
# on it: the probe runs as a background task and a rollout is scored with
# ``lean_version`` still unknown rather than behind it. Against a server that
# accepts connections and never answers, this bounds how long one of the
# ``max_concurrency`` slots is held by a request no verdict depends on.
VERSION_PROBE_TIMEOUT_SECONDS = 30

# Kimina exposes no version endpoint, so the toolchain is read by compiling a
# one-line program through the same path a submission takes. ``TOOLCHAIN_PROBE``
# is the shared one, so every Lean benchmark asks the question the same way, and
# its header is the one submissions use, so the probe warms the same REPL pool
# rather than a second one.


def _log_probe_task(task: "asyncio.Task") -> None:
    """Consume a finished background probe so a failure is logged, never swallowed."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        LOG.warning("Lean toolchain probe raised: %r", exc)


class KiminaLeanClient:
    def __init__(self, base_url: str, api_key: Optional[str] = None, max_concurrency: int = DEFAULT_MAX_CONCURRENCY):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._version: Optional[str] = None
        self._version_probes = 0
        self._version_lock = asyncio.Lock()
        self._version_task: Optional[asyncio.Task] = None

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    async def verify(self, code: str, timeout_seconds: int) -> LeanResult:
        """Compile ``code`` once and report what Lean said.

        A failure the model could not have caused — the connection refused, the
        client timing out, a reply that is not the documented JSON, saturation
        still unresolved after the retries — is a transport failure: callers
        attribute it to the harness and mask the rollout. A 5xx is *not* one of
        those: Kimina raises it per snippet, from executing this submission, so
        it is charged to the model. See ``_verify_once`` and the README's
        "Who a failure is charged to".

        A saturation reply (``SATURATION_STATUSES``) is retried with backoff,
        because it is the one failure that costs the server no REPL time and is
        exactly what a busy server returns: left unretried it becomes a masked
        ``sandbox_error`` and silently shrinks the measured denominator. Every
        other failure, timeouts included, is returned on the first try.
        """
        payload = {
            "codes": [{"custom_id": uuid.uuid4().hex, "proof": code}],
            "timeout": int(timeout_seconds),
            "infotree_type": None,
            "disable_cache": False,
        }
        for attempt in range(MAX_SATURATION_ATTEMPTS):
            result = await self._verify_once(payload, timeout_seconds)
            if not isinstance(result, int):
                return result
            if attempt + 1 == MAX_SATURATION_ATTEMPTS:
                LOG.warning("Lean server still saturated (HTTP %s) after %s tries", result, attempt + 1)
                return LeanResult(error=f"HTTP {result}: Lean server saturated", transport_failure=True)
            backoff = SATURATION_BACKOFF_SECONDS * 2**attempt
            LOG.warning("Lean server is saturated (HTTP %s), retrying in %.1fs", result, backoff)
            # Slept outside the semaphore, so the slot goes to a request that can
            # use it rather than being held by this one while it waits.
            await asyncio.sleep(backoff)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _verify_once(self, payload: dict[str, Any], timeout_seconds: int) -> LeanResult | int:
        """One ``/verify`` round trip: a ``LeanResult``, or the HTTP status to back off from."""
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
                    # client's default of 3 tries is turned off here; the caller
                    # retries saturation instead, which costs no REPL time.
                    _max_connection_retries=1,
                )
                if response.status in SATURATION_STATUSES:
                    # Released explicitly: nothing below reads this body, and an
                    # unread response keeps its connection checked out of the
                    # shared aiohttp pool until it is garbage-collected. That is
                    # exactly the wrong moment to leak one — the server is
                    # saturated and this request is about to be retried. The other
                    # early returns consume the body (``text()`` / ``json()``),
                    # which releases it, and the ``except`` branch never got one.
                    await response.release()
                    return response.status
                if response.status != 200:
                    text = await response.text()
                    LOG.warning("Lean server returned HTTP %s: %s", response.status, text[:500])
                    # A 5xx is Kimina reporting that executing *this snippet*
                    # failed, so it is charged to the model rather than masked.
                    # ``server/routers/check.py`` wraps every non-timeout
                    # exception raised while getting a REPL, running the header
                    # or running the body into ``HTTPException(500, str(e))``,
                    # per snippet, and one of those exceptions is
                    # ``LeanError``, which ``server/repl.py`` raises whenever
                    # the REPL wrote anything to stderr. Model output reaches
                    # that path: ``native_decide`` is allowed by design and the
                    # model's own ``import`` lines become the pooled REPL's
                    # header. Masking it would let a rollout that Lean refused
                    # to evaluate be deleted from the denominator instead of
                    # scored 0, which upstream never does.
                    #
                    # Everything else non-200 (401, 404, 422, ...) is this
                    # client or its credentials being wrong, which the model
                    # cannot cause, so it stays a masked transport failure.
                    server_error = 500 <= response.status < 600
                    return LeanResult(
                        error=f"HTTP {response.status}: {text[:500]}",
                        transport_failure=not server_error,
                        server_error=server_error,
                    )
                body = await response.json()
        except Exception as exc:  # network errors, timeouts, bad JSON
            LOG.warning("Lean server request failed: %r", exc)
            return LeanResult(error=f"{type(exc).__name__}: {exc}", transport_failure=True)
        return parse_verify_response(body)

    @property
    def lean_version(self) -> Optional[str]:
        """The probed Lean version if it has already answered, else None.

        Reading this never blocks and never starts a probe. ``None`` is the
        documented meaning "not known", which is what a caller sees until the
        background probe has come back.
        """
        return self._version

    def start_version_probe(self) -> None:
        """Start the toolchain probe in the background, at most one at a time.

        Scoring must not wait on this. The probe compiles a one-line program
        through the same path a submission takes, so against a server that
        accepts connections but never answers it costs the full Lean timeout
        plus the HTTP margin; awaiting it before every compile put that latency
        in front of the first rollouts and turned a server outage into agent and
        eval timeouts instead of the masked ``sandbox_error`` this server
        intends. Calling this on each use keeps round 2's "probe once, cache the
        answer, re-probe a transient failure" behaviour: the call is a no-op once
        a version is known or ``MAX_VERSION_PROBES`` have been spent, and a probe
        that failed is re-attempted by the next call rather than by a retry loop
        nothing is waiting on.
        """
        if self._version is not None or self._version_probes >= MAX_VERSION_PROBES:
            return
        if self._version_task is not None and not self._version_task.done():
            return
        self._version_task = asyncio.ensure_future(self.toolchain_version(VERSION_PROBE_TIMEOUT_SECONDS))
        # Nothing awaits the task, so its result has to be retrieved here or asyncio
        # reports it as never retrieved at garbage-collection time.
        self._version_task.add_done_callback(_log_probe_task)

    async def toolchain_version(self, timeout_seconds: int = VERSION_PROBE_TIMEOUT_SECONDS) -> Optional[str]:
        """Lean version the server actually runs, probed until it answers and then cached.

        A server built for another Lean version scores every row
        ``proof_failed`` and nothing in the rollouts says why, which is what the
        version in every response exists to make visible. The probe pays one cold
        ``import`` on its first call.

        Normally driven by ``start_version_probe`` rather than awaited by a
        caller: no verdict depends on the answer, so nothing should wait for it.

        A probe that returns no version is *not* cached as an answer: caching it
        would let one transient failure — a server still starting, a single
        dropped connection — disable the mismatch guard for the whole run while
        the README still advertises it. It is retried on the next call instead,
        bounded at ``MAX_VERSION_PROBES`` so an unreachable server costs a few
        probes rather than one per rollout. After the last failed attempt the
        guard really is off for the run, and that is logged at ERROR.
        """
        if self._version is not None or self._version_probes >= MAX_VERSION_PROBES:
            return self._version
        async with self._version_lock:
            if self._version is not None or self._version_probes >= MAX_VERSION_PROBES:
                return self._version
            self._version_probes += 1
            result = await self.verify(TOOLCHAIN_PROBE, timeout_seconds)
            # parse_lean_version reads a sandbox's stdout/stderr; the REPL answers in
            # structured messages instead, so they are joined into the shape it expects
            # rather than the version regex being written a second time.
            self._version = parse_lean_version(
                {"stdout": "\n".join(str(message.get("data", "")) for message in result.messages)}
            )
            if self._version is not None:
                LOG.info("Lean server at %s reports Lean %s", self.base_url, self._version)
            elif self._version_probes < MAX_VERSION_PROBES:
                LOG.warning(
                    "Lean version probe %s/%s returned no version (error=%r); will retry",
                    self._version_probes,
                    MAX_VERSION_PROBES,
                    result.error,
                )
            else:
                LOG.error(
                    "Lean version probe failed %s times (error=%r): the toolchain behind %s stays unknown "
                    "for the rest of this run, so a toolchain mismatch will not be reported",
                    self._version_probes,
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

    The test is key *presence*, not truthiness, because that is what upstream's
    ``is_error`` does: ``if "error" in feedback`` / ``if "stderr" in feedback``
    (``evaluation/client/lean_client.py``). A payload carrying ``{"error": None}``
    or ``{"stderr": ""}`` is one upstream fails, so a truthiness test here would
    reward precisely the reply upstream rejects. A genuine command response
    carries neither key, so nothing legitimate is caught by this.
    """
    for key in PAYLOAD_ERROR_KEYS:
        if key in payload:
            # ``repr`` when the value is falsy, so ``{"error": None}`` and
            # ``{"stderr": ""}`` read as themselves rather than as a blank message.
            value = payload[key]
            shown = str(value)[:500] if value else repr(value)
            return f"Lean server reported {key}: {shown}"
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
