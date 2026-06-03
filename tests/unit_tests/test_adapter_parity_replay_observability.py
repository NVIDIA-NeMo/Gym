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
"""Parity replay test for the adapter middleware.

Loads every JSON fixture under ``adapter_fixtures/`` and asserts that the
recorded ``request`` → ``expected_response`` pair still holds when the
middleware is installed with the recorded ``interceptor_specs`` and the
(mocked) upstream returns the recorded ``upstream_response``. ``caching``
is covered by a dedicated round-trip test below since its parity behavior
is "second hit ≡ first response".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from nemo_gym.adapters import install_middleware


FIXTURE_DIR = Path(__file__).parent / "adapter_fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixtures() -> list[tuple[str, dict[str, Any]]]:
    """Return ``[(fixture_name, fixture_data), ...]`` sorted by name."""
    if not FIXTURE_DIR.exists():
        return []
    fixtures: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        with path.open() as f:
            fixtures.append((path.stem, json.load(f)))
    return fixtures


_FIXTURES = _load_fixtures()


def _build_replay_app(interceptor_specs: list[dict[str, Any]], upstream: dict[str, Any]) -> FastAPI:
    """FastAPI app whose chat-completions route returns ``upstream`` verbatim,
    with the adapter middleware installed on top."""
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def _chat(body: dict):
        return JSONResponse(
            content=upstream["body"],
            status_code=upstream["status_code"],
            headers=upstream.get("headers") or {},
        )

    install_middleware(app, interceptor_specs)
    return app


_VOLATILE_HEADERS = {"date", "server", "content-length"}


def _normalise_headers(headers: dict[str, str]) -> dict[str, str]:
    """Drop headers whose values are non-deterministic.

    Matches the normalisation in ``generate_adapter_fixtures.py``. Without this,
    replays would fail because ``content-length`` is recomputed per-response
    by Starlette and ``date`` / ``server`` vary across runs.
    """
    return {k: v for k, v in headers.items() if k.lower() not in _VOLATILE_HEADERS}


# ---------------------------------------------------------------------------
# Parametrised replay
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _FIXTURES, reason="no fixtures recorded yet (run generate_adapter_fixtures.py)")
@pytest.mark.parametrize("name,fixture", _FIXTURES, ids=[name for name, _ in _FIXTURES])
def test_parity_replay(name: str, fixture: dict[str, Any]) -> None:
    """Each fixture must replay byte-equal through the middleware chain."""
    request = fixture["request"]
    upstream = fixture["upstream_response"]
    expected = fixture["expected_response"]

    app = _build_replay_app(fixture["interceptor_specs"], upstream)
    with TestClient(app) as client:
        resp = client.post(request["path"], json=request["body"], headers=request["headers"])

    assert resp.status_code == expected["status_code"], (
        f"fixture {name!r}: status mismatch (got {resp.status_code}, expected {expected['status_code']})"
    )

    actual_body = resp.json()
    assert actual_body == expected["body"], (
        f"fixture {name!r}: body mismatch\n  got: {actual_body!r}\n  expected: {expected['body']!r}"
    )

    actual_headers = _normalise_headers(dict(resp.headers))
    expected_headers = _normalise_headers(expected["headers"])
    assert actual_headers == expected_headers, (
        f"fixture {name!r}: headers mismatch\n  got: {actual_headers!r}\n  expected: {expected_headers!r}"
    )
