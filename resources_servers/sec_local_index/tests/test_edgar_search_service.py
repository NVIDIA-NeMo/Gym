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
"""The decisions EdgarSearchService makes for every backend that plugs into it."""

from __future__ import annotations

import json
from typing import Any

import pytest

from resources_servers.sec_local_index.edgar_search_service import (
    EdgarSearchService,
    coerce_stringified_collection,
)
from resources_servers.sec_local_index.local_edgar_search import LocalEdgarRequest


CUTOFF = "2025-04-07"


class RecordingBackend:
    """Captures the normalized request and returns a fixed result."""

    def __init__(self, results: list[dict[str, Any]] | None = None) -> None:
        self.results = results if results is not None else [{"cik": "320193", "filedAt": "2024-11-01"}]
        self.request: LocalEdgarRequest | None = None

    async def execute_async(self, request: LocalEdgarRequest) -> list[dict[str, Any]]:
        self.request = request
        return self.results


def _service(backend: RecordingBackend, **kwargs: Any) -> EdgarSearchService:
    return EdgarSearchService(backend, max_end_date=CUTOFF, **kwargs)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('["10-K"]', ["10-K"]),
        ("['10-K']", ["10-K"]),
        ('["10-K", "10-Q"]', ["10-K", "10-Q"]),
        ('{"a": 1}', {"a": 1}),
        (["10-K"], ["10-K"]),
        (None, None),
        ("10-K", "10-K"),
        ("", ""),
    ],
)
def test_stringified_collections_are_deserialized(raw: Any, expected: Any) -> None:
    assert coerce_stringified_collection(raw) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("raw", ['["10-K"]', "['10-K']", ["10-K"]])
async def test_form_types_reach_the_backend_as_a_list(raw: Any) -> None:
    backend = RecordingBackend()

    await _service(backend).run({"search_query": "revenue", "form_types": raw})

    assert backend.request is not None
    assert backend.request.form_types == ("10-K",)


@pytest.mark.asyncio
async def test_omitted_end_date_becomes_the_cutoff() -> None:
    backend = RecordingBackend()

    await _service(backend).run({"search_query": "revenue"})

    assert backend.request is not None
    assert backend.request.end_date == CUTOFF


@pytest.mark.asyncio
async def test_end_date_beyond_the_cutoff_is_clamped() -> None:
    backend = RecordingBackend()

    await _service(backend).run({"search_query": "revenue", "end_date": "2030-01-01"})

    assert backend.request is not None
    assert backend.request.end_date == CUTOFF


@pytest.mark.asyncio
async def test_cutoff_is_per_service_not_global() -> None:
    backend = RecordingBackend()

    await EdgarSearchService(backend, max_end_date="2026-03-01").run(
        {"search_query": "revenue", "end_date": "2025-12-31"}
    )

    assert backend.request is not None
    assert backend.request.end_date == "2025-12-31"


@pytest.mark.asyncio
async def test_results_are_serialized_as_a_json_list() -> None:
    backend = RecordingBackend(results=[{"cik": "320193"}])

    output = await _service(backend).run({"search_query": "revenue"})

    assert json.loads(output) == [{"cik": "320193"}]


@pytest.mark.asyncio
async def test_a_rejected_request_serializes_its_message() -> None:
    backend = RecordingBackend()

    output = await _service(backend).run({"search_query": ""})

    assert "search_query is required" in json.loads(output)["error"]
    assert backend.request is None


@pytest.mark.asyncio
async def test_a_failing_backend_serializes_its_message() -> None:
    class Failing:
        async def execute_async(self, request: LocalEdgarRequest) -> list[dict[str, Any]]:
            raise RuntimeError("index is gone")

    output = await EdgarSearchService(Failing(), max_end_date=CUTOFF).run({"search_query": "revenue"})

    assert json.loads(output) == {"error": "index is gone"}


@pytest.mark.asyncio
async def test_on_results_sees_the_results_before_serialization() -> None:
    backend = RecordingBackend(results=[{"filingUrl": "https://example.com/a.htm"}])
    seen: list[list[dict[str, Any]]] = []

    async def record(results: list[dict[str, Any]]) -> None:
        seen.append(results)

    await _service(backend, on_results=record).run({"search_query": "revenue"})

    assert seen == [[{"filingUrl": "https://example.com/a.htm"}]]


@pytest.mark.asyncio
async def test_on_results_is_skipped_when_the_request_is_rejected() -> None:
    backend = RecordingBackend()
    seen: list[Any] = []

    async def record(results: list[dict[str, Any]]) -> None:
        seen.append(results)

    await _service(backend, on_results=record).run({"search_query": ""})

    assert seen == []
