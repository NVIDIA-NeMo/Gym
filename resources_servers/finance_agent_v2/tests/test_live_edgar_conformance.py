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
"""Drift guard between sec_local_index and the upstream tools it restates.

finance_sec_search cannot import finance_agent without also taking on Vals'
multi-provider model SDKs, so it reaches sec-api.io through its own backend and
reduces filing HTML with its own helper. Those are copies, and a copy goes stale
silently. These tests run both against the same inputs and fail when they part,
including when a bumped upstream pin is what moved.

This file lives here because this is the venv that has finance_agent installed.

On failure, change sec_local_index to match upstream rather than editing the
expectations below.
"""

import json
from typing import Any

import pytest
from finance_agent import tools as upstream_tools
from finance_agent.tools import MAX_END_DATE, EDGARSearch, ParseHtmlPage

from resources_servers.sec_local_index.html_text import html_to_text
from resources_servers.sec_local_index.live_edgar_search import build_payload, select_filings
from resources_servers.sec_local_index.local_edgar_search import normalize_request


# Spans the axes either implementation could diverge on: defaults, every
# optional filter, clamping, and paging.
SEARCH_MATRIX: tuple[dict[str, Any], ...] = (
    {"search_query": "revenue"},
    {"search_query": "net income", "start_date": "2024-01-01", "end_date": "2024-06-30"},
    {"search_query": "revenue", "end_date": "2030-01-01"},
    {"search_query": "revenue", "start_date": "2030-01-01"},
    {"search_query": "revenue", "form_types": ["10-K", "10-Q"]},
    {"search_query": "revenue", "ciks": ["320193", "789019"]},
    {"search_query": "revenue", "page": 3},
    {"search_query": "revenue", "top_n_results": 10},
    {
        "search_query": "climate risk",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31",
        "form_types": ["8-K"],
        "ciks": ["1318605"],
        "page": 2,
        "top_n_results": 25,
    },
)

HTML_SAMPLES = (
    "<html><body><p>Net income was $93.7 billion.</p></body></html>",
    "<html><head><style>p {color: red}</style></head><body><p>Revenue</p>"
    "<script>var x = 1;</script><p>grew</p></body></html>",
    "<html><body><table><tr><td>2024</td><td>$391.0B</td></tr></table></body></html>",
    "<html><body><div>  wide   gaps   between   phrases  </div></body></html>",
    "<html><body><p>Line one</p>\n\n<p>   </p>\n<p>Line two</p></body></html>",
    "<html><body>caf\u00e9 na\u00efve \u2014 em dash &amp; entity</body></html>",
    "<p>unclosed paragraph",
    "",
)


class _Response:
    def __init__(self, body: Any = None, text: str = ""):
        self._body = body
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    async def json(self):
        return self._body

    async def text(self):
        return self._text


class _RecordingSession:
    """Stands in for aiohttp inside finance_agent so upstream runs offline."""

    posted: list[dict[str, Any]] = []
    filings: list[dict[str, Any]] = []
    html: str = ""

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def post(self, url, json=None, headers=None, **kwargs):
        type(self).posted.append({"url": url, "json": json, "headers": headers})
        return _Response(body={"filings": list(type(self).filings)})

    def get(self, url, **kwargs):
        return _Response(text=type(self).html)


@pytest.fixture
def offline_upstream(monkeypatch):
    _RecordingSession.posted = []
    _RecordingSession.filings = []
    _RecordingSession.html = ""
    monkeypatch.setattr(upstream_tools.aiohttp, "ClientSession", _RecordingSession)
    return _RecordingSession


async def _upstream_payload(arguments: dict[str, Any]) -> dict[str, Any]:
    await EDGARSearch(sec_api_key="conformance-key")._execute_search(**arguments)
    return _RecordingSession.posted[-1]["json"]


@pytest.mark.asyncio
@pytest.mark.parametrize("arguments", SEARCH_MATRIX, ids=lambda a: a["search_query"] + str(sorted(a)))
async def test_the_request_sent_to_sec_api_matches_upstream(offline_upstream, arguments) -> None:
    ours = build_payload(normalize_request(**arguments, max_end_date=MAX_END_DATE))

    assert ours == await _upstream_payload(arguments)


@pytest.mark.asyncio
async def test_the_reply_is_trimmed_the_way_upstream_trims_it(offline_upstream) -> None:
    offline_upstream.filings = [{"accessionNo": str(index)} for index in range(100)]

    upstream = await EDGARSearch(sec_api_key="conformance-key")._execute_search(
        search_query="revenue", top_n_results=10
    )

    assert select_filings({"filings": list(offline_upstream.filings)}, top_n_results=10) == upstream


@pytest.mark.asyncio
@pytest.mark.parametrize("html", HTML_SAMPLES, ids=range(len(HTML_SAMPLES)))
async def test_filing_text_matches_upstreams_reduction(offline_upstream, html) -> None:
    offline_upstream.html = html

    upstream = await ParseHtmlPage()._parse_html_page("https://www.sec.gov/Archives/edgar/data/1/2/a.htm")

    assert html_to_text(html) == upstream


@pytest.mark.asyncio
async def test_upstream_still_clamps_to_its_own_cutoff(offline_upstream) -> None:
    """Pins the date our normalizer has to be given for this lineage. A bumped
    pin that moves MAX_END_DATE lands here."""
    payload = await _upstream_payload({"search_query": "revenue", "end_date": "2099-01-01"})

    assert payload["endDate"] == MAX_END_DATE == "2026-03-01"


def test_the_local_tool_advertises_the_upstream_schema() -> None:
    """edgar_search's declared parameters are what samples were written against,
    so local mode may not widen or narrow them."""
    from resources_servers.finance_agent_v2.local_tools import LocalEDGARSearch

    assert LocalEDGARSearch.parameters == EDGARSearch.parameters
    assert json.dumps(LocalEDGARSearch.parameters, sort_keys=True) == json.dumps(
        EDGARSearch.parameters, sort_keys=True
    )


async def _upstream_error(arguments: dict[str, Any]) -> str:
    with pytest.raises(ValueError) as raised:
        await EDGARSearch(sec_api_key="conformance-key")._execute_search(**arguments)
    return str(raised.value)


def _our_error(arguments: dict[str, Any]) -> str:
    with pytest.raises(ValueError) as raised:
        normalize_request(**arguments, max_end_date=MAX_END_DATE)
    return str(raised.value)


@pytest.mark.asyncio
async def test_the_out_of_order_date_error_matches_upstream(offline_upstream) -> None:
    """Reaches the model verbatim, so the wording is part of the contract."""
    arguments = {"search_query": "x", "start_date": "2022-01-01", "end_date": "2021-01-01"}

    assert _our_error(arguments) == await _upstream_error(arguments)


@pytest.mark.asyncio
async def test_the_malformed_date_error_is_worded_differently(offline_upstream) -> None:
    """A known, deliberate divergence. Adopting upstream's wording would change
    an error finance_sec_search has already trained against, so it is recorded
    here instead: if either side rewords, this fails and someone chooses.
    """
    arguments = {"search_query": "x", "start_date": "not-a-date"}

    assert await _upstream_error(arguments) == "Invalid start_date format: 'not-a-date'. Expected YYYY-MM-DD."
    assert _our_error(arguments) == "start_date 'not-a-date' is not in yyyy-mm-dd format"
