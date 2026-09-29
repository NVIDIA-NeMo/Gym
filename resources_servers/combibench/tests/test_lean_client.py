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

"""The Lean client must turn every transport problem into a harness fault, never an exception."""

import asyncio
from typing import Any

import pytest

from resources_servers.combibench import lean_client
from resources_servers.combibench.fine_eval import classify_lean_result
from resources_servers.combibench.lean_client import (
    DEFAULT_LEAN_SERVER_MAX_WAIT_SECONDS,
    HEADER_RUN_FAILURE_DETAIL,
    HTTP_TIMEOUT_MARGIN_SECONDS,
    MAX_SATURATION_ATTEMPTS,
    MAX_VERSION_PROBES,
    REPL_LIFECYCLE_DETAILS,
    REPL_START_FAILURE_DETAIL,
    SATURATION_BACKOFF_SECONDS,
    SATURATION_STATUSES,
    KiminaLeanClient,
    http_budget_seconds,
    is_header_run_failure,
)
from resources_servers.lean_proof.status import STATUS_COMPILE_ERROR, STATUS_SANDBOX_ERROR


class _FakeResponse:
    def __init__(self, status: int, body: Any = None, text: str = ""):
        self.status = status
        self._body = body
        self._text = text
        self.released = False

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    async def text(self):
        return self._text

    async def release(self):
        self.released = True


def _patch_request(monkeypatch, response=None, exc: Exception | None = None) -> list[dict]:
    calls: list[dict] = []

    async def fake_request(method, url, **kwargs):
        calls.append({"method": method, "url": url, **kwargs})
        if exc is not None:
            raise exc
        return response

    monkeypatch.setattr(lean_client, "request", fake_request)
    return calls


class TestKiminaLeanClient:
    async def test_sends_upstreams_verify_shape(self, monkeypatch) -> None:
        body = {"results": [{"custom_id": "x", "response": {"messages": [], "env": 1, "time": 0.2}}]}
        calls = _patch_request(monkeypatch, _FakeResponse(200, body))
        client = KiminaLeanClient("http://lean:8000/", api_key="secret")
        result = await client.verify("import Mathlib\nexample : True := trivial", timeout_seconds=45)

        assert result.transport_failure is False and result.error is None and result.time == 0.2
        call = calls[0]
        assert call["method"] == "POST" and call["url"] == "http://lean:8000/verify"
        assert call["json"]["timeout"] == 45 and call["json"]["disable_cache"] is False
        assert call["json"]["codes"][0]["proof"].startswith("import Mathlib")
        assert call["headers"]["Authorization"] == "Bearer secret"
        assert call["timeout"].total == http_budget_seconds(45)

    async def test_no_api_key_sends_no_authorization_header(self, monkeypatch) -> None:
        calls = _patch_request(monkeypatch, _FakeResponse(200, {"results": [{"custom_id": "x", "response": {}}]}))
        await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert "Authorization" not in calls[0]["headers"]

    @pytest.mark.parametrize("status", [400, 401, 404, 422])
    async def test_a_client_side_http_error_is_a_transport_failure(self, monkeypatch, status) -> None:
        """A wrong URL, a missing key or a rejected request shape is this harness, not the model."""
        _patch_request(monkeypatch, _FakeResponse(status, text="nope"))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is True and result.server_error is False
        assert f"HTTP {status}" in result.error
        assert classify_lean_result(result) == STATUS_SANDBOX_ERROR

    async def test_a_snippet_execution_500_is_charged_to_the_model(self, monkeypatch) -> None:
        """Kimina raises 500 per snippet when executing *this submission* blew up.

        ``server/routers/check.py:159`` wraps every non-timeout exception from
        running the body into ``HTTPException(500, str(e))`` for that snippet
        alone, and ``server/repl.py`` gets there by raising ``LeanError("Lean
        process broken pipe")`` or ``ReplError("JSON decode error")`` once the
        REPL is no longer answering. Model output reaches that path — the REPL
        runs under an ``RLIMIT_AS`` cap and ``native_decide`` is allowed by
        design — so a masked ``sandbox_error`` here would delete the attempt
        from the denominator instead of scoring it 0, which upstream never does.
        (``repl.py:305`` also raises ``LeanError`` on REPL stderr, but that
        check is dead at this pin: ``error_file`` is a ``TemporaryFile`` that
        ``create_subprocess_exec`` is never handed, so it is always empty.)
        """
        _patch_request(monkeypatch, _FakeResponse(500, text='{"detail":"Lean process broken pipe"}'))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is False and result.server_error is True
        assert "HTTP 500" in result.error
        assert classify_lean_result(result) == "lean_error"

    async def test_a_repl_startup_500_is_a_transport_failure(self, monkeypatch) -> None:
        """``manager.prep`` raises this one before any Lean code runs, model's or not.

        ``server/manager.py:197-202`` normalises a REPL process that would not
        start to ``ReplError("Failed to start REPL")``, which ``check.py:118``
        turns into ``HTTPException(500, str(e))`` and FastAPI serialises as
        ``{"detail": ...}``. Charging that to the model would score a proof Lean
        never saw. Unlike its sibling below, no submission can cause it.
        """
        _patch_request(monkeypatch, _FakeResponse(500, text=f'{{"detail":"{REPL_START_FAILURE_DETAIL}"}}'))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is True and result.server_error is False
        assert result.header_error is False
        assert classify_lean_result(result) == STATUS_SANDBOX_ERROR

    async def test_a_header_run_500_is_left_for_the_caller_to_attribute(self, monkeypatch) -> None:
        """The other ``manager.prep`` detail is not lifecycle: the header is the submission's.

        ``server/manager.py:203-215`` re-raises a header ``TimeoutError``
        unchanged but normalises every *other* header failure to
        ``ReplError("Failed to run header on REPL")``. Kimina's header is the
        submission's own leading ``import`` run (``server/split.py``), so
        ``import Foo`` from the model lands here. The client defaults it to the
        masked status and flags ``header_error`` so
        ``app.CombibenchVerifier.verify`` can charge a model-authored header,
        exactly as it already does for the header timeout.
        """
        _patch_request(monkeypatch, _FakeResponse(500, text=f'{{"detail":"{HEADER_RUN_FAILURE_DETAIL}"}}'))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.header_error is True
        assert result.transport_failure is False and result.server_error is False
        # Masked unless the caller says the header was the model's own.
        assert classify_lean_result(result) == STATUS_SANDBOX_ERROR

    async def test_the_two_prep_details_are_still_the_pinned_strings(self) -> None:
        """Both markers are pinned upstream text; a reworded one must fail loudly, not silently."""
        assert REPL_LIFECYCLE_DETAILS == (REPL_START_FAILURE_DETAIL, HEADER_RUN_FAILURE_DETAIL)
        assert is_header_run_failure(500, '{"detail":"Failed to run header on REPL"}') is True
        assert is_header_run_failure(500, '{"detail":"Failed to start REPL"}') is False
        assert is_header_run_failure(502, '{"detail":"Failed to run header on REPL"}') is False

    @pytest.mark.parametrize("status", [502, 504])
    async def test_a_gateway_5xx_is_a_transport_failure(self, monkeypatch, status) -> None:
        """No Kimina path emits 502/504; they come from a proxy in front of it, which the model cannot cause."""
        _patch_request(monkeypatch, _FakeResponse(status, text="<html>Bad Gateway</html>"))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is True and result.server_error is False
        assert f"HTTP {status}" in result.error
        assert classify_lean_result(result) == STATUS_SANDBOX_ERROR

    async def test_a_500_saying_timed_out_is_still_a_lean_error(self, monkeypatch) -> None:
        """Only the two REPL-lifecycle details are excused; other wordings stay charged."""
        _patch_request(monkeypatch, _FakeResponse(500, text='{"detail":"worker timed out"}'))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert classify_lean_result(result) == "lean_error"

    async def test_a_500_whose_body_is_not_json_is_still_a_lean_error(self, monkeypatch) -> None:
        """A body that is not FastAPI's ``{"detail": ...}`` cannot be excused, so it stays charged."""
        _patch_request(monkeypatch, _FakeResponse(500, text="Internal Server Error"))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.server_error is True
        assert classify_lean_result(result) == "lean_error"

    async def test_connection_error_is_a_transport_failure(self, monkeypatch) -> None:
        _patch_request(monkeypatch, exc=ConnectionError("refused"))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is True and "refused" in result.error

    async def test_invalid_json_is_a_transport_failure(self, monkeypatch) -> None:
        _patch_request(monkeypatch, _FakeResponse(200, ValueError("not json")))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is True

    async def test_server_side_timeout_is_a_lean_verdict(self, monkeypatch) -> None:
        body = {"results": [{"custom_id": "x", "error": "Lean REPL command timed out in 10 seconds"}]}
        _patch_request(monkeypatch, _FakeResponse(200, body))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is False and "timed out" in result.error

    async def test_a_timed_out_compile_is_not_retried(self, monkeypatch) -> None:
        """Three tries cost three REPL jobs and 3x the wall clock without changing the verdict."""
        calls = _patch_request(monkeypatch, _FakeResponse(200, {"results": [{"custom_id": "x", "response": {}}]}))
        await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert calls[0]["_max_connection_retries"] == 1


class TestSaturationIsRetried:
    """429/503 cost the server no REPL time, so the single-try rule does not apply to them."""

    @pytest.fixture
    def slept(self, monkeypatch) -> list[float]:
        recorded: list[float] = []

        async def fake_sleep(seconds: float) -> None:
            recorded.append(seconds)

        monkeypatch.setattr(lean_client.asyncio, "sleep", fake_sleep)
        return recorded

    @pytest.mark.parametrize("status", SATURATION_STATUSES)
    async def test_a_saturated_server_is_retried_and_then_succeeds(self, monkeypatch, slept, status) -> None:
        ok = {"results": [{"custom_id": "x", "response": {"messages": [], "time": 0.1}}]}
        replies = [_FakeResponse(status, text="busy"), _FakeResponse(200, ok)]
        calls: list[dict] = []

        async def fake_request(method, url, **kwargs):
            calls.append(kwargs)
            return replies[len(calls) - 1]

        monkeypatch.setattr(lean_client, "request", fake_request)
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        # Retried, so the rollout is scored instead of being masked out of the denominator.
        assert result.transport_failure is False and result.error is None
        assert len(calls) == 2 and slept == [SATURATION_BACKOFF_SECONDS]

    async def test_retries_are_bounded(self, monkeypatch, slept) -> None:
        calls = _patch_request(monkeypatch, _FakeResponse(503, text="busy"))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert len(calls) == MAX_SATURATION_ATTEMPTS
        assert slept == [SATURATION_BACKOFF_SECONDS, SATURATION_BACKOFF_SECONDS * 2]
        # Still masked: a server that never freed a REPL reached no verdict on this proof.
        assert result.transport_failure is True and result.server_error is False
        assert "HTTP 503" in result.error
        assert classify_lean_result(result) == STATUS_SANDBOX_ERROR

    async def test_a_saturation_reply_releases_its_connection(self, monkeypatch, slept) -> None:
        """An unread body holds a pooled connection until GC -- when the pool is already starved."""
        response = _FakeResponse(429, text="busy")
        _patch_request(monkeypatch, response)
        await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert response.released is True

    async def test_a_timeout_is_still_tried_once(self, monkeypatch, slept) -> None:
        """The timeout path is unchanged: it already cost a REPL its whole budget."""
        body = {"results": [{"custom_id": "x", "error": "Lean REPL command timed out in 10 seconds"}]}
        calls = _patch_request(monkeypatch, _FakeResponse(200, body))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert len(calls) == 1 and slept == []
        assert calls[0]["_max_connection_retries"] == 1
        assert "timed out" in result.error


class TestHttpBudget:
    """The client must not give up before the server has had its own worst case."""

    @pytest.mark.parametrize("timeout_seconds", [10, 60, 120])
    async def test_the_budget_covers_the_servers_worst_case(self, timeout_seconds: int) -> None:
        """``max_wait`` for a free REPL, then the header command, then the body command.

        Below that sum a non-terminating proof — the model's own output — is cut
        off by the client and masked as a ``sandbox_error`` instead of coming
        back as the server's timeout and being charged.
        """
        assert http_budget_seconds(timeout_seconds) > DEFAULT_LEAN_SERVER_MAX_WAIT_SECONDS + 2 * timeout_seconds

    async def test_the_budget_follows_the_configured_server_max_wait(self, monkeypatch) -> None:
        calls = _patch_request(monkeypatch, _FakeResponse(200, {"results": [{"custom_id": "x", "response": {}}]}))
        client = KiminaLeanClient("http://lean:8000", lean_server_max_wait=300)
        await client.verify("code", 60)
        assert calls[0]["timeout"].total == 300 + 2 * 60 + HTTP_TIMEOUT_MARGIN_SECONDS


class TestErrorPayloads:
    """The per-item ``response`` can itself be an error object, with no outer ``error``."""

    @pytest.mark.parametrize(
        "payload",
        [{"error": "boom"}, {"stderr": "cannot open shared object file"}],
        ids=["error", "stderr"],
    )
    async def test_an_error_or_stderr_payload_is_a_transport_failure(self, monkeypatch, payload: dict) -> None:
        """Neither key has a reading as a verdict, so neither is charged to the model."""
        body = {"results": [{"custom_id": "x", "response": {**payload, "time": 0.1}}]}
        _patch_request(monkeypatch, _FakeResponse(200, body))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is True and result.server_error is False
        assert classify_lean_result(result) == STATUS_SANDBOX_ERROR

    async def test_a_message_payload_is_charged_to_the_model(self, monkeypatch) -> None:
        """Kimina's own client reads ``{"message": ...}`` as a Lean error on the snippet.

        ``client/kimina_client/proof_utils.py::parse_error_message`` turns that
        payload into a single ``FinalMessage`` of severity ``"error"``, which
        ``parse_lean_response`` then treats like any other compiler diagnostic.
        So it is a verdict, not an infrastructure failure, and masking it would
        take a rollout the model can produce (a bad import in its own header)
        out of the denominator.
        """
        body = {"results": [{"custom_id": "x", "response": {"message": "unknown package 'Foo'", "time": 0.1}}]}
        _patch_request(monkeypatch, _FakeResponse(200, body))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is False and result.server_error is True
        assert "unknown package 'Foo'" in result.error
        assert classify_lean_result(result) == "lean_error"

    @pytest.mark.parametrize(
        "payload",
        [{"error": None}, {"stderr": ""}],
        ids=["error-null", "stderr-empty"],
    )
    async def test_a_falsy_error_key_still_fails_closed(self, monkeypatch, payload: dict) -> None:
        """Upstream's ``is_error`` tests ``"error" in feedback``, not its truth.

        A reply carrying the key with a falsy value is one upstream fails; a
        truthiness test here would score it a clean compile, which is the one
        outcome this guard exists to prevent.
        """
        body = {"results": [{"custom_id": "x", "response": {**payload, "time": 0.1}}]}
        _patch_request(monkeypatch, _FakeResponse(200, body))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is True
        assert classify_lean_result(result) == STATUS_SANDBOX_ERROR
        # The message still names the value rather than trailing off after the colon.
        assert result.error.endswith(repr(next(iter(payload.values()))))

    @pytest.mark.parametrize(
        "result",
        [{"custom_id": "x"}, {"custom_id": "x", "response": None}],
        ids=["no-keys", "response-null"],
    )
    async def test_a_result_with_neither_error_nor_response_fails_closed(self, monkeypatch, result: dict) -> None:
        """Both shapes used to read as a clean compile and score 1.0.

        Defence against a malformed or non-Kimina server rather than a shape the
        pinned server emits: ``ReplResponse``
        (``client/kimina_client/models.py``) has a ``@model_validator``
        ``require_error_or_response`` that raises unless exactly one of the two
        is set. ``/verify`` being declared ``response_model_exclude_none=True``
        (``server/routers/backward.py``) is what would serialise such an object
        to neither key if one ever existed. Kept because failing closed is the
        right default: same class as the error payload guard above, one level up.
        """
        _patch_request(monkeypatch, _FakeResponse(200, {"results": [result]}))
        parsed = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert parsed.transport_failure is True
        assert classify_lean_result(parsed) == STATUS_SANDBOX_ERROR

    async def test_an_outer_error_with_no_response_is_still_a_verdict(self, monkeypatch) -> None:
        """The server's own timeout carries ``error`` and no ``response``; it must stay charged."""
        body = {"results": [{"custom_id": "x", "error": "Lean REPL command timed out in 10 seconds"}]}
        _patch_request(monkeypatch, _FakeResponse(200, body))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is False
        assert classify_lean_result(result) == "timeout"

    async def test_a_command_response_is_still_a_verdict(self, monkeypatch) -> None:
        """The guard must not swallow ordinary compiler diagnostics, which live in ``messages``."""
        body = {
            "results": [
                {"custom_id": "x", "response": {"messages": [{"severity": "error", "data": "unknown id"}], "env": 0}}
            ]
        }
        _patch_request(monkeypatch, _FakeResponse(200, body))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is False
        assert classify_lean_result(result) == STATUS_COMPILE_ERROR


class TestConcurrencyBound:
    async def test_in_flight_requests_are_capped(self, monkeypatch) -> None:
        """Rollout fan-out is unbounded; the Lean server runs LEAN_SERVER_MAX_REPLS at a time."""
        in_flight = 0
        peak = 0

        async def fake_request(method, url, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0)
            in_flight -= 1
            return _FakeResponse(200, {"results": [{"custom_id": "x", "response": {}}]})

        monkeypatch.setattr(lean_client, "request", fake_request)
        client = KiminaLeanClient("http://lean:8000", max_concurrency=2)
        await asyncio.gather(*(client.verify("code", 10) for _ in range(8)))
        assert peak <= 2


class TestToolchainProbe:
    def _info(self, data: str) -> _FakeResponse:
        return _FakeResponse(200, {"results": [{"custom_id": "x", "response": {"messages": [{"data": data}]}}]})

    async def test_version_is_read_from_the_probe_and_cached(self, monkeypatch) -> None:
        calls = _patch_request(monkeypatch, self._info('"4.24.0"'))
        client = KiminaLeanClient("http://lean:8000")
        assert await client.toolchain_version() == "4.24.0"
        assert await client.toolchain_version() == "4.24.0"
        assert len(calls) == 1
        assert "Lean.versionString" in calls[0]["json"]["codes"][0]["proof"]

    async def test_a_dead_server_is_probed_a_bounded_number_of_times(self, monkeypatch) -> None:
        """Not once per rollout, but not once per run either: a failure is not an answer."""
        calls = _patch_request(monkeypatch, exc=ConnectionError("refused"))
        client = KiminaLeanClient("http://lean:8000")
        for _ in range(10):
            assert await client.toolchain_version() is None
        assert len(calls) == MAX_VERSION_PROBES

    async def test_a_transient_failure_does_not_disable_the_guard(self, monkeypatch) -> None:
        """Caching the first miss would switch the mismatch guard off for the whole run."""
        replies: list[Any] = [ConnectionError("refused"), self._info('"4.24.0"')]
        calls: list[dict] = []

        async def fake_request(method, url, **kwargs):
            calls.append(kwargs)
            reply = replies[len(calls) - 1]
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(lean_client, "request", fake_request)
        client = KiminaLeanClient("http://lean:8000")
        assert await client.toolchain_version() is None
        assert await client.toolchain_version() == "4.24.0"
        # ... and the hit is cached, so the cold import is not paid again.
        assert await client.toolchain_version() == "4.24.0"
        assert len(calls) == 2

    async def test_start_version_probe_returns_without_waiting_for_the_server(self, monkeypatch) -> None:
        """Nothing scored may wait on the probe, so starting it must not block."""
        never_answers = asyncio.Event()

        async def fake_request(method, url, **kwargs):
            await never_answers.wait()

        monkeypatch.setattr(lean_client, "request", fake_request)
        client = KiminaLeanClient("http://lean:8000")
        client.start_version_probe()
        assert client.lean_version is None
        # A second call does not stack a second probe on the hung first one.
        client.start_version_probe()
        await asyncio.sleep(0)
        assert client._version_probes == 1
        never_answers.set()

    async def test_the_probe_is_retried_and_cached_through_start(self, monkeypatch) -> None:
        """``start_version_probe`` keeps probe-once/cache/re-probe-a-failure."""
        replies: list[Any] = [ConnectionError("refused"), self._info('"4.24.0"')]
        calls: list[dict] = []

        async def fake_request(method, url, **kwargs):
            calls.append(kwargs)
            reply = replies[min(len(calls), len(replies)) - 1]
            if isinstance(reply, Exception):
                raise reply
            return reply

        monkeypatch.setattr(lean_client, "request", fake_request)
        client = KiminaLeanClient("http://lean:8000")
        for _ in range(5):
            client.start_version_probe()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
        assert client.lean_version == "4.24.0"
        # One failure, then one success that is cached: no probe per call.
        assert len(calls) == 2
