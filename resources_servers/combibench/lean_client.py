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
import json
import logging
import uuid
from typing import Any, Optional

from aiohttp import ClientTimeout

from nemo_gym.server_utils import request
from resources_servers.combibench.fine_eval import LeanResult
from resources_servers.lean_proof.toolchain import TOOLCHAIN_PROBE, parse_lean_version


LOG = logging.getLogger(__name__)

# Network allowance on top of the *server's* own worst case, so a legitimate slow
# compile is reported by the server as its own timeout rather than cut off by the
# client first. See ``http_budget_seconds`` for the arithmetic this is added to.
HTTP_TIMEOUT_MARGIN_SECONDS = 30.0

# ``LEAN_SERVER_MAX_WAIT``: how long Kimina's ``manager.get_repl`` waits for a
# REPL to come free before answering 429 (``server/settings.py``: ``max_wait:
# int = 60``; this repository's ``kimina_image/Dockerfile`` sets the same 60).
# Configurable because it is a property of the server being pointed at, and the
# client budget below is derived from it rather than guessed.
DEFAULT_LEAN_SERVER_MAX_WAIT_SECONDS = 60


def http_budget_seconds(
    timeout_seconds: float, lean_server_max_wait: float = DEFAULT_LEAN_SERVER_MAX_WAIT_SECONDS
) -> float:
    """Client-side HTTP budget for one ``/verify``, sized so the server answers first.

    Kimina spends, for a single snippet (``server/routers/check.py::run_one``):

    * up to ``lean_server_max_wait`` inside ``manager.get_repl`` waiting for a
      free REPL (``server/manager.py``: the ``max_wait`` deadline, after which it
      raises ``NoAvailableReplError`` → 429),
    * then up to ``timeout`` running the import header on a cold REPL
      (``manager.prep``, which is given the request's own ``timeout``),
    * then up to ``timeout`` running the body.

    So the worst case is ``max_wait + 2 * timeout``, and a client budget below
    that gives up before the server does. That matters for attribution, not just
    for latency: a client-side timeout is a masked ``sandbox_error``, so a
    non-terminating proof — the model's own output, which the server would have
    returned as its own timeout and charged to the model — would instead be
    deleted from the denominator whenever the client blinked first. The flat
    ``timeout + 30`` this used to be did exactly that at the default 60 s.

    ``HTTP_TIMEOUT_MARGIN_SECONDS`` on top is the network and queueing slack.
    """
    return lean_server_max_wait + 2 * timeout_seconds + HTTP_TIMEOUT_MARGIN_SECONDS


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

# Kimina answers 5xx for one snippet in four shapes. Three are
# ``HTTPException(500, str(e))`` raised inside ``run_one`` in
# ``server/routers/check.py`` (a grep for ``HTTPException`` over ``server/``
# finds no other 5xx: the rest are 401, 429 and 499), and the fourth is
# Starlette answering for an exception ``run_one`` never caught:
#
# * ``:84``  — ``manager.get_repl`` raised something other than
#   ``NoAvailableReplError``. Not a failure to *spawn* Lean, despite where it
#   sits: spawning is ``repl.start()``, which runs inside ``manager.prep``, so a
#   real spawn failure arrives at ``:118`` as "Failed to start REPL". What
#   ``get_repl`` does that can raise is ``Repl.create``, which writes a row
#   through prisma when ``LEAN_SERVER_DATABASE_URL`` is set — so against the
#   shipped image, which sets it empty, this site is close to unreachable.
#   Infrastructure.
# * ``:118`` — ``manager.prep`` raised, which ``server/manager.py:197-215``
#   normalises to ``ReplError("Failed to start REPL")`` — the Lean process would
#   not start, infrastructure — or ``ReplError("Failed to run header on REPL")``
#   — the *import header* raised. Kimina's header is the submission's own leading
#   ``import`` run (``server/split.py``), so that one may well be the model's;
#   see ``is_header_run_failure``.
# * ``:159`` — executing the *body* raised. This is where a model's proof lands:
#   ``server/repl.py`` raises ``LeanError("Lean process broken pipe")`` or
#   ``LeanError("Failed to write to REPL stdin")`` when the REPL is no longer
#   there to write to, and ``ReplError("JSON decode error")`` when what it wrote
#   back is not JSON. Model code reaches all three by killing the REPL — it runs
#   under an ``RLIMIT_AS`` cap (``repl.py::start``) and ``native_decide`` is
#   allowed by design. (``repl.py:305`` also raises ``LeanError`` when the REPL
#   wrote to stderr, but that check is dead at this pin: ``error_file`` is a
#   ``TemporaryFile`` that ``create_subprocess_exec`` is never given — it is
#   passed ``stderr=PIPE``, which nothing reads — so the file is always empty.)
#   The model's.
# * an exception the endpoint never caught at all: the prisma writes at
#   ``check.py:96-106``/``129-140``/``169-186``, again only with a database
#   configured. Starlette's default handler answers those with a 500 whose body
#   is not JSON, so no ``detail`` matches and it is charged. See the README's
#   "Who a failure is charged to" for why that is the safe direction.
#
# FastAPI serialises the three ``HTTPException``s as ``{"detail": str(e)}``, so
# the two ``manager.prep`` cases arrive as those fixed strings and are told apart
# by them. ``:84`` carries whatever the failure said and so cannot be matched; it
# falls through to "charged", which is the safe direction of the two — the rule
# this server keeps is that a failure the model *could* have caused is never
# masked — and, being all but unreachable against the shipped image, it is a
# residual rather than a live tradeoff.
REPL_START_FAILURE_DETAIL = "Failed to start REPL"
HEADER_RUN_FAILURE_DETAIL = "Failed to run header on REPL"

# The two details ``manager.prep`` normalises its failures to. Only the first is
# excused unconditionally; the second depends on whose header it was, which this
# module cannot see. See ``is_header_run_failure``.
REPL_LIFECYCLE_DETAILS = (REPL_START_FAILURE_DETAIL, HEADER_RUN_FAILURE_DETAIL)


def _server_error_detail(text: str) -> str:
    """The ``detail`` of a FastAPI error body, or the raw text if it is not one."""
    try:
        body = json.loads(text)
    except ValueError:
        return text
    if isinstance(body, dict) and isinstance(body.get("detail"), str):
        return body["detail"]
    return text


def is_model_attributable_server_error(status: int, text: str) -> bool:
    """Whether a 5xx is Kimina reporting that *this submission's body* blew up.

    Three ways it is not:

    * The status is a 5xx Kimina never emits. Only 500 comes out of
      ``check.py``; 502/504 (and 501, 505, ...) are what a proxy or load
      balancer in front of the server answers when the server itself did not,
      which no model can cause. (503 never reaches here: it is in
      ``SATURATION_STATUSES`` and is retried.) Masked.
    * The detail is ``REPL_START_FAILURE_DETAIL`` — the server could not start a
      Lean process at all, which is REPL lifecycle and nothing to do with this
      proof. Masked.
    * The detail is ``HEADER_RUN_FAILURE_DETAIL`` — the *import header* raised.
      Neither charged nor masked here: whose header it was decides that, and
      only the caller has the submission and the reference statement to compare.
      ``is_header_run_failure`` names that case and
      ``app.CombibenchVerifier.verify`` settles it, exactly as it already does
      for the header *timeout*.

    This does key on the wording of a server message, deliberately: the message
    is the only thing distinguishing the sites that share status 500, and
    ``fine_eval.HEADER_TIMEOUT_MARKER`` already keys the ``header_timeout``
    status on Kimina's literal "header command timed out". Both strings are
    pinned upstream text, and both fail in the charged direction if upstream
    reworded them, which is the direction that cannot silently inflate a score.
    """
    if status != 500:
        return False
    return not any(marker in _server_error_detail(text) for marker in REPL_LIFECYCLE_DETAILS)


def is_header_run_failure(status: int, text: str) -> bool:
    """Whether a 500 is Kimina saying the *import header* raised.

    The twin of ``fine_eval.HEADER_TIMEOUT_MARKER`` for the non-timeout case.
    ``manager.prep`` re-raises a ``TimeoutError`` from the header command
    unchanged — which ``check.py`` turns into the "header command timed out"
    verdict — but normalises *everything else* to ``ReplError("Failed to run
    header on REPL")``, which becomes a 500 with no trace of what actually
    failed. Since Kimina's header is the submission's own leading ``import`` run,
    that 500 is a failure the model can cause by writing ``import Foo``, and
    excusing it unconditionally would delete such a rollout from the denominator.
    Who it belongs to is decided by the caller from
    ``fine_eval.header_is_harness_supplied``; this only reports the shape.
    """
    return status == 500 and HEADER_RUN_FAILURE_DETAIL in _server_error_detail(text)


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
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        lean_server_max_wait: int = DEFAULT_LEAN_SERVER_MAX_WAIT_SECONDS,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.lean_server_max_wait = lean_server_max_wait
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
        attribute it to the harness and mask the rollout. A Kimina 500 raised
        from executing *this submission* is not one of those: it is charged to
        the model. A 500 raised from failing to start a REPL, and any 5xx Kimina
        does not emit at all, is. A 500 from the import header raising is left
        open for the caller to attribute (``header_error``). See
        ``is_model_attributable_server_error``, ``is_header_run_failure``,
        ``_verify_once`` and the README's "Who a failure is charged to".

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
                    timeout=ClientTimeout(total=http_budget_seconds(timeout_seconds, self.lean_server_max_wait)),
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
                    # A Kimina 500 raised from *executing this snippet's body* is
                    # the model's: model code can kill the REPL (memory cap,
                    # ``native_decide``), and every way it does — broken pipe,
                    # unwritable stdin, stdout that is not JSON — comes back as
                    # this 500, so masking it would let a rollout Lean refused to
                    # evaluate be deleted from the denominator instead of scored
                    # 0, which upstream never does. A 500 from *failing to start a
                    # REPL* is not, and neither is a gateway 5xx Kimina never
                    # emits; see ``is_model_attributable_server_error``. A 500 from
                    # the *import header* raising is neither until the caller says
                    # whose header it was — ``header_error`` carries that question
                    # up, the same way a header timeout already travels up as its
                    # own status.
                    #
                    # Everything else non-200 (401, 404, 422, ...) is this
                    # client or its credentials being wrong, which the model
                    # cannot cause, so it stays a masked transport failure.
                    server_error = is_model_attributable_server_error(response.status, text)
                    header_error = is_header_run_failure(response.status, text)
                    return LeanResult(
                        error=f"HTTP {response.status}: {text[:500]}",
                        transport_failure=not (server_error or header_error),
                        server_error=server_error,
                        header_error=header_error,
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

# Of those three, the one that is charged to the model rather than masked.
# ``client/kimina_client/proof_utils.py::parse_error_message`` turns a
# ``{"message": ...}`` payload into a single ``FinalMessage`` of severity
# ``"error"`` — Kimina's own client reads that shape as a Lean error *on the
# snippet*, not as an infrastructure failure. ``error`` and ``stderr`` have no
# such reading; they stay masked. See ``_payload_failure``.
PAYLOAD_MODEL_ERROR_KEYS = ("message",)


def _payload_failure(payload: dict[str, Any]) -> Optional[tuple[str, str]]:
    """The key and description of a per-item ``response`` that is not a Lean verdict, else None.

    The top-level ``error`` is not the only way ``/verify`` reports a failure.
    ``server/repl.py`` returns ``ReplResponse(response=cmd_response)`` with
    ``cmd_response`` taken straight from ``json.loads`` of the REPL's stdout and
    validated nowhere, and the client's ``extend()`` explicitly admits the
    ``{"message": ...}`` shape, so a reply can carry an Error object in
    ``response`` with no top-level ``error`` at all. Reading only the outer key
    would leave such a reply with no error, no messages and no sorries — which
    is exactly what a clean compile looks like — and reward it.

    This fails closed either way; which way depends on the key. ``error`` and
    ``stderr`` fail to ``transport_failure`` (``sandbox_error``, masked): a REPL
    that answered with one of those instead of a command response did not
    evaluate the model's proof, so there is no verdict to charge to the model.
    ``message`` is charged (``lean_error``) — Kimina's own client maps that
    payload to a ``FinalMessage(severity="error")``
    (``client/kimina_client/proof_utils.py::parse_error_message``), i.e. reads
    it as Lean erroring on the snippet, and a model-authored bad import is one
    way to produce it. See ``PAYLOAD_MODEL_ERROR_KEYS``. Upstream fails the
    submission on ``error``/``stderr`` and has no case for ``message`` at all
    (it would score such a reply 1.0); for the two it does read, it reaches the
    same reward of 0.0 by a different route, and the difference only shows up in
    whether the rollout is counted in the denominator, where a non-verdict
    should not be.

    The test is key *presence*, not truthiness, because that is what upstream's
    ``is_error`` does: ``if "error" in feedback`` / ``if "stderr" in feedback``
    (``evaluation/client/lean_client.py``). A payload carrying ``{"error": None}``
    or ``{"stderr": ""}`` is one upstream fails, so a truthiness test here would
    reward precisely the reply upstream rejects. A genuine command response
    carries neither key, so nothing legitimate is caught by this.

    Earlier rounds left "can a *model-authored* bad import reach this path, and
    is masking it therefore wrong?" open, because the Lean REPL's own behaviour
    on an unknown module could not be read (``leanprover-community/repl`` is not
    vendored in the pinned Kimina tree, only referenced by URL from its
    ``Dockerfile``/``setup.sh``). It is settled now, from Kimina's side rather
    than the REPL's: ``proof_utils.parse_error_message`` classifies a ``message``
    payload as a severity-``error`` Lean message, so Kimina itself treats that
    shape as the snippet failing. That is why ``message`` is charged.
    """
    for key in PAYLOAD_ERROR_KEYS:
        if key in payload:
            # ``repr`` when the value is falsy, so ``{"error": None}`` and
            # ``{"stderr": ""}`` read as themselves rather than as a blank message.
            value = payload[key]
            shown = str(value)[:500] if value else repr(value)
            return key, f"Lean server reported {key}: {shown}"
    return None


def parse_verify_response(body: Any) -> LeanResult:
    """Turn the ``/verify`` JSON body into a ``LeanResult``.

    The reply is ``{"results": [{"custom_id", "error", "response": {"messages",
    "sorries", "env", "time"}}]}``. A body without exactly one result is a
    transport failure, not a Lean verdict, and so is a result that carries
    neither an ``error`` nor a ``response`` object.
    """
    results = body.get("results") if isinstance(body, dict) else None
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], dict):
        return LeanResult(error="malformed Lean server reply", transport_failure=True)
    result = results[0]
    error = result.get("error")
    payload = result.get("response")
    if not error and not isinstance(payload, dict):
        # Neither a verdict nor a failure: ``{"custom_id": "x"}`` and
        # ``{"custom_id": "x", "response": null}`` used to fall through to "no
        # error, no messages, no sorries" — which is exactly what a clean
        # compile looks like — and were rewarded 1.0.
        #
        # Defence against a malformed or non-Kimina server, not a shape the
        # pinned server produces: ``ReplResponse``
        # (``client/kimina_client/models.py``) carries a ``@model_validator``
        # ``require_error_or_response`` that raises when neither field is set, so
        # a well-behaved Kimina cannot construct one. ``/verify`` being declared
        # ``response_model_exclude_none=True`` (``server/routers/backward.py``)
        # is what would *serialise* such an object to neither key if one ever
        # existed. Kept anyway: this is the same class of bug as
        # ``_payload_failure`` one level up — an under-specified reply read as a
        # success — so it fails closed the same way, and to the same masked
        # ``sandbox_error``: a reply carrying no verdict is not a verdict on the
        # model's proof.
        LOG.warning("Lean server result carried neither an error nor a response: %r", result)
        return LeanResult(error="Lean server result carried neither an error nor a response", transport_failure=True)
    if not isinstance(payload, dict):
        payload = {}
    if not error:
        failed = _payload_failure(payload)
        if failed is not None:
            key, description = failed
            LOG.warning("Lean server returned an error payload: %s", description)
            charged = key in PAYLOAD_MODEL_ERROR_KEYS
            return LeanResult(error=description, transport_failure=not charged, server_error=charged)
    messages = payload.get("messages") or []
    sorries = payload.get("sorries") or []
    return LeanResult(
        error=str(error) if error else None,
        messages=[m for m in messages if isinstance(m, dict)],
        sorries=[s for s in sorries if isinstance(s, dict)],
        time=payload.get("time") if isinstance(payload.get("time"), (int, float)) else None,
    )
