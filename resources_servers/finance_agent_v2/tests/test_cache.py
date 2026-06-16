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
"""Tests for the finance_agent_v2 disk cache layer.

The network seam of each upstream tool is monkeypatched, so these run offline.
The core assertion is *fidelity*: a cache hit must be byte-identical to a live
call, because the cached tool stores the raw upstream response and re-serializes
it with the untouched upstream code.
"""

import logging

import pytest

from finance_agent.tools import EDGARSearch, ParseHtmlPage, PriceHistory
from resources_servers.finance_agent_v2.cache import ToolCache
from resources_servers.finance_agent_v2.cached_tools import (
    CachedEDGARSearch,
    CachedParseHtmlPage,
    CachedPriceHistory,
    SecFilingSearch,
)

_LOG = logging.getLogger("test")

# A small immutable daily series (Tiingo-shaped records).
_RECORDS = [
    {"date": f"2024-01-{d:02d}T00:00:00.000Z", "open": 10.0 + d, "high": 11.0 + d, "low": 9.0 + d,
     "close": 10.5 + d, "adjOpen": 9.9 + d, "adjHigh": 10.9 + d, "adjLow": 8.9 + d, "adjClose": 10.4 + d,
     "volume": 1000 + d, "adjVolume": 1000 + d, "divCash": 0.0, "splitFactor": 1.0}
    for d in range(2, 11)  # 2024-01-02 .. 2024-01-10
]


def _fake_fetch_factory(counter: list[int]):
    async def _fake_fetch(self, endpoint, ticker, start_date, end_date):
        counter[0] += 1
        return [r for r in _RECORDS if start_date <= r["date"].split("T", 1)[0] <= end_date]

    return _fake_fetch


def _args(ticker="AAPL", start="2024-01-02", end="2024-01-10", asset_class="equity"):
    return {"ticker": ticker, "start_date": start, "end_date": end, "asset_class": asset_class}


# ============================================================================
# ToolCache policy
# ============================================================================


class TestToolCachePolicy:
    def test_disabled_when_use_cache_false(self, tmp_path):
        c = ToolCache(tmp_path, use_cache=False)
        assert not c.enabled

    def test_enabled_with_dir(self, tmp_path):
        c = ToolCache(tmp_path)
        assert c.enabled and c.root == tmp_path

    def test_defaults_to_home_when_no_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))
        c = ToolCache(None)
        assert c.enabled and str(tmp_path) in str(c.root)

    def test_jsonl_roundtrip(self, tmp_path):
        c = ToolCache(tmp_path)
        p = c.path("pricing", "equity", "AAPL.jsonl")
        c.write_jsonl(p, _RECORDS)
        assert c.read_jsonl(p) == _RECORDS


# ============================================================================
# price_history caching
# ============================================================================


class TestCachedPriceHistory:
    @pytest.mark.asyncio
    async def test_hit_is_byte_identical_to_live(self, tmp_path, monkeypatch):
        counter = [0]
        monkeypatch.setattr(PriceHistory, "_fetch", _fake_fetch_factory(counter))

        live = PriceHistory("x")
        live_out = (await live.execute(_args(), {}, _LOG)).output

        cache = ToolCache(tmp_path)
        cached = CachedPriceHistory("x", cache)
        first = (await cached.execute(_args(), {}, _LOG)).output
        calls_after_first = counter[0]
        second = (await cached.execute(_args(), {}, _LOG)).output

        assert first == live_out          # cache miss -> live -> identical
        assert second == live_out         # cache hit  -> identical
        assert counter[0] == calls_after_first  # second call hit disk, not network

    @pytest.mark.asyncio
    async def test_master_slice_matches_direct_subrange(self, tmp_path, monkeypatch):
        counter = [0]
        monkeypatch.setattr(PriceHistory, "_fetch", _fake_fetch_factory(counter))

        cache = ToolCache(tmp_path)
        cached = CachedPriceHistory("x", cache)
        # Warm the master with the full window.
        await cached.execute(_args(start="2024-01-02", end="2024-01-10"), {}, _LOG)
        calls = counter[0]

        # Sub-range served from the master (no new network call)...
        sub = (await cached.execute(_args(start="2024-01-04", end="2024-01-06"), {}, _LOG)).output
        assert counter[0] == calls

        # ...and equals a direct live sub-range fetch + serialize.
        live = PriceHistory("x")
        live_sub = (await live.execute(_args(start="2024-01-04", end="2024-01-06"), {}, _LOG)).output
        assert sub == live_sub

    @pytest.mark.asyncio
    async def test_ticker_normalization_paths(self, tmp_path, monkeypatch):
        monkeypatch.setattr(PriceHistory, "_fetch", _fake_fetch_factory([0]))
        cache = ToolCache(tmp_path)
        cached = CachedPriceHistory("x", cache)

        await cached.execute(_args(ticker="aapl", asset_class="equity"), {}, _LOG)
        await cached.execute(_args(ticker="BTCUSD", asset_class="crypto"), {}, _LOG)

        assert cache.path("pricing", "equity", "AAPL.jsonl").exists()   # equity uppercases
        assert cache.path("pricing", "crypto", "btcusd.jsonl").exists()  # crypto lowercases


# ============================================================================
# edgar_search caching
# ============================================================================


class TestCachedEDGARSearch:
    @pytest.mark.asyncio
    async def test_hit_and_topn_slicing(self, tmp_path, monkeypatch):
        filings = [{"accessionNo": f"000-{i}", "formType": "10-K"} for i in range(10)]
        counter = [0]

        async def _fake_search(self, search_query, start_date="1900-01-01", end_date="2026-03-01",
                               top_n_results=100, page=1, form_types=None, ciks=None):
            counter[0] += 1
            return filings[: int(top_n_results)]

        monkeypatch.setattr(EDGARSearch, "_execute_search", _fake_search)

        cache = ToolCache(tmp_path)
        cached = CachedEDGARSearch(sec_api_key="k", cache=cache)

        five = await cached._execute_search("nvidia revenue", top_n_results=5)
        assert five == filings[:5]
        assert counter[0] == 1  # one live fetch (of the full page)

        three = await cached._execute_search("nvidia revenue", top_n_results=3)
        assert three == filings[:3]
        assert counter[0] == 1  # served from cache; top_n applied locally

    @pytest.mark.asyncio
    async def test_cache_file_is_debuggable(self, tmp_path, monkeypatch):
        filings = [{"accessionNo": "000-1", "formType": "10-K"}]

        async def _fake_search(self, search_query, start_date="1900-01-01", end_date="2026-03-01",
                               top_n_results=100, page=1, form_types=None, ciks=None):
            return filings

        monkeypatch.setattr(EDGARSearch, "_execute_search", _fake_search)
        cache = ToolCache(tmp_path)
        cached = CachedEDGARSearch(sec_api_key="k", cache=cache)
        await cached._execute_search("NVIDIA Revenue Growth!", top_n_results=1)

        files = list(cache.path("edgar_search").glob("*.json"))
        assert len(files) == 1
        # Readable slug prefix, and the request is stored alongside the filings.
        assert files[0].name.startswith("nvidia-revenue-growth_")
        stored = cache.read_json(files[0])
        assert stored["request"]["query"] == "NVIDIA Revenue Growth!"
        assert stored["filings"] == filings


# ============================================================================
# parse_html_page caching (sec.gov docs only)
# ============================================================================


class TestCachedParseHtmlPage:
    def test_sec_doc_path_is_nested(self, tmp_path):
        cache = ToolCache(tmp_path)
        tool = CachedParseHtmlPage(cache)
        url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
        path = tool._doc_path(url)
        assert path is not None
        # sec_filings/<cik_padded>/<accession>/<filename>.txt
        assert path.parts[-3:] == ("0000320193", "000032019324000123", "aapl-20240928.htm.txt")

    def test_non_sec_url_not_cached(self, tmp_path):
        cache = ToolCache(tmp_path)
        tool = CachedParseHtmlPage(cache)
        assert tool._doc_path("https://example.com/report.html") is None

    @pytest.mark.asyncio
    async def test_sec_doc_cached_second_call_is_hit(self, tmp_path, monkeypatch):
        counter = [0]

        async def _fake_parse(self, url):
            counter[0] += 1
            return f"PARSED:{url}"

        monkeypatch.setattr(ParseHtmlPage, "_parse_html_page", _fake_parse)

        cache = ToolCache(tmp_path)
        tool = CachedParseHtmlPage(cache)
        url = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
        a = await tool._parse_html_page(url)
        b = await tool._parse_html_page(url)
        assert a == b == f"PARSED:{url}"
        assert counter[0] == 1  # second call served from disk

    @pytest.mark.asyncio
    async def test_non_sec_url_always_live(self, tmp_path, monkeypatch):
        counter = [0]

        async def _fake_parse(self, url):
            counter[0] += 1
            return "web"

        monkeypatch.setattr(ParseHtmlPage, "_parse_html_page", _fake_parse)
        tool = CachedParseHtmlPage(ToolCache(tmp_path))
        await tool._parse_html_page("https://example.com/a")
        await tool._parse_html_page("https://example.com/a")
        assert counter[0] == 2  # general web is never cached


# ============================================================================
# sec_filing_search
# ============================================================================


class TestSecFilingSearch:
    @pytest.mark.asyncio
    async def test_search_uses_cache_and_builds_sec_gov_urls(self, tmp_path, monkeypatch):
        tool = SecFilingSearch(cache=ToolCache(tmp_path), user_agent="ua")

        async def _fake_tickers(self):
            self._tickers = {"AAPL": {"cik": "0000320193", "name": "Apple Inc."}}
            return self._tickers

        submissions = {
            "0000320193": {
                "000032019324000123": {
                    "ticker": "AAPL", "cik": "0000320193", "form": "10-K",
                    "filing_date": "2024-11-01", "report_date": "2024-09-28",
                    "accession_number": "0000320193-24-000123", "primary_document": "aapl-20240928.htm",
                    "filing_url": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm",
                }
            }
        }

        async def _fake_filings(self, cik, ticker):
            return submissions[str(cik).zfill(10)]

        monkeypatch.setattr(SecFilingSearch, "_ensure_tickers", _fake_tickers)
        monkeypatch.setattr(SecFilingSearch, "_get_company_filings", _fake_filings)

        out = await tool.execute({"ticker": "AAPL"}, {}, _LOG)
        assert "sec.gov/Archives/edgar/data/320193" in out.output
        assert "10-K" in out.output

    @pytest.mark.asyncio
    async def test_unknown_ticker_errors(self, tmp_path, monkeypatch):
        tool = SecFilingSearch(cache=ToolCache(tmp_path), user_agent="ua")

        async def _fake_tickers(self):
            self._tickers = {}
            return self._tickers

        monkeypatch.setattr(SecFilingSearch, "_ensure_tickers", _fake_tickers)
        out = await tool.execute({"ticker": "ZZZZ"}, {}, _LOG)
        assert "No company found" in out.output
