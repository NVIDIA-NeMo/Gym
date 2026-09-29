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
from resources_servers.combibench.lean_client import HTTP_TIMEOUT_MARGIN_SECONDS, KiminaLeanClient
from resources_servers.lean_proof.status import STATUS_COMPILE_ERROR, STATUS_SANDBOX_ERROR


class _FakeResponse:
    def __init__(self, status: int, body: Any = None, text: str = ""):
        self.status = status
        self._body = body
        self._text = text

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body

    async def text(self):
        return self._text


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
        assert call["timeout"].total == 45 + HTTP_TIMEOUT_MARGIN_SECONDS

    async def test_no_api_key_sends_no_authorization_header(self, monkeypatch) -> None:
        calls = _patch_request(monkeypatch, _FakeResponse(200, {"results": [{"custom_id": "x", "response": {}}]}))
        await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert "Authorization" not in calls[0]["headers"]

    @pytest.mark.parametrize("status", [401, 429, 500])
    async def test_non_200_is_a_transport_failure(self, monkeypatch, status) -> None:
        _patch_request(monkeypatch, _FakeResponse(status, text="nope"))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is True and f"HTTP {status}" in result.error

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


class TestErrorPayloads:
    """The per-item ``response`` can itself be an error object, with no outer ``error``."""

    @pytest.mark.parametrize(
        "payload",
        [{"message": "Failed to start REPL"}, {"error": "boom"}, {"stderr": "cannot open shared object file"}],
        ids=["message", "error", "stderr"],
    )
    async def test_an_error_payload_is_a_transport_failure(self, monkeypatch, payload: dict) -> None:
        body = {"results": [{"custom_id": "x", "response": {**payload, "time": 0.1}}]}
        _patch_request(monkeypatch, _FakeResponse(200, body))
        result = await KiminaLeanClient("http://lean:8000").verify("code", 10)
        assert result.transport_failure is True
        assert classify_lean_result(result) == STATUS_SANDBOX_ERROR

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

    async def test_a_dead_server_is_probed_once(self, monkeypatch) -> None:
        calls = _patch_request(monkeypatch, exc=ConnectionError("refused"))
        client = KiminaLeanClient("http://lean:8000")
        assert await client.toolchain_version() is None
        assert await client.toolchain_version() is None
        assert len(calls) == 1
