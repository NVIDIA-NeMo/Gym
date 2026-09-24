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

from typing import Any

import pytest

from resources_servers.combibench import lean_client
from resources_servers.combibench.lean_client import HTTP_TIMEOUT_MARGIN_SECONDS, KiminaLeanClient


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
