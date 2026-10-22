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
"""The sec-api.io request this server sends and the reply it keeps.

Pinned against the upstream implementation by
resources_servers/finance_agent_v2/tests/test_live_edgar_conformance.py, which
runs where the finance_agent package is installed.
"""

import pytest

from resources_servers.sec_local_index.edgar_search_service import EdgarSearchService
from resources_servers.sec_local_index.live_edgar_search import (
    LiveEdgarSearch,
    build_payload,
    select_filings,
)
from resources_servers.sec_local_index.local_edgar_search import normalize_request


CUTOFF = "2025-04-07"


def _request(**kwargs):
    return normalize_request(max_end_date=CUTOFF, **kwargs)


def test_payload_carries_only_the_filters_that_were_set() -> None:
    payload = build_payload(_request(search_query="revenue"))

    assert payload == {
        "query": "revenue",
        "startDate": "1900-01-01",
        "endDate": CUTOFF,
        "page": 1,
    }


def test_payload_renders_every_filter() -> None:
    payload = build_payload(
        _request(
            search_query="revenue",
            start_date="2024-01-01",
            end_date="2024-06-30",
            form_types=["10-K", "10-Q"],
            ciks=["0000320193"],
            page=2,
        )
    )

    assert payload == {
        "query": "revenue",
        "startDate": "2024-01-01",
        "endDate": "2024-06-30",
        "page": 2,
        "formTypes": ["10-K", "10-Q"],
        "ciks": ["320193"],
    }


def test_dates_are_clamped_before_they_reach_the_wire() -> None:
    payload = build_payload(_request(search_query="revenue", end_date="2030-01-01"))

    assert payload["endDate"] == CUTOFF


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ({"filings": [{"a": 1}, {"a": 2}, {"a": 3}]}, [{"a": 1}, {"a": 2}]),
        ({"filings": []}, []),
        ({}, []),
        ("not a mapping", []),
    ],
)
def test_select_filings_trims_to_the_requested_count(body, expected) -> None:
    assert select_filings(body, top_n_results=2) == expected


def test_an_empty_key_is_refused() -> None:
    with pytest.raises(ValueError, match="sec-api.io key is required"):
        LiveEdgarSearch("")


class _FakeResponse:
    def __init__(self, body):
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._body


class _FakeSession:
    def __init__(self, body):
        self._body = body
        self.calls = []

    def post(self, url, json, headers, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(self._body)


@pytest.mark.asyncio
async def test_execute_async_sends_the_key_and_returns_the_filings() -> None:
    session = _FakeSession({"filings": [{"accessionNo": "0000320193-24-000001"}]})

    async def provider():
        return session

    backend = LiveEdgarSearch("secret-key", session_provider=provider)
    results = await backend.execute_async(_request(search_query="revenue", top_n_results=10))

    assert results == [{"accessionNo": "0000320193-24-000001"}]
    assert session.calls[0]["headers"]["Authorization"] == "secret-key"
    assert session.calls[0]["json"]["query"] == "revenue"


@pytest.mark.asyncio
async def test_the_service_serializes_a_live_failure_as_a_tool_error() -> None:
    class _Failing:
        async def execute_async(self, request):
            raise RuntimeError("sec-api.io unavailable")

    service = EdgarSearchService(_Failing(), max_end_date=CUTOFF)

    assert await service.run({"search_query": "revenue"}) == '{"error": "sec-api.io unavailable"}'
