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
"""Local-mode SEC and pricing tools: same contract as upstream, different source."""

import json
import logging
from pathlib import Path

import pytest
from finance_agent.tools import MAX_END_DATE, EDGARSearch, ParseHtmlPage, PriceHistory

from resources_servers.finance_agent_v2.cached_tools import CachedPriceHistory
from resources_servers.finance_agent_v2.local_tools import LocalEDGARSearch, LocalParseHtmlPage, LocalPriceHistory
from resources_servers.sec_local_index.cache import ToolCache
from resources_servers.sec_local_index.local_edgar_search import LocalEdgarSearch
from resources_servers.sec_local_index.tests.index_fixtures import build_index


FILING_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000001/aapl.htm"
UNINDEXED_FILING_URL = "https://www.sec.gov/Archives/edgar/data/1045810/000104581025000010/nvda.htm"


@pytest.fixture
def engine(tmp_path: Path) -> LocalEdgarSearch:
    return LocalEdgarSearch(build_index(tmp_path / "index.sqlite"), max_end_date=MAX_END_DATE)


def test_the_model_sees_the_upstream_tool_contract(engine) -> None:
    """Local mode may not change the tool a sample was written against."""
    local = LocalEDGARSearch(engine)

    assert local.name == EDGARSearch.name
    assert local.description == EDGARSearch.description
    assert local.parameters == EDGARSearch.parameters
    assert local.required == EDGARSearch.required


@pytest.mark.asyncio
async def test_search_runs_without_a_key_or_a_network_call(engine) -> None:
    local = LocalEDGARSearch(engine)

    output = await local.execute({"search_query": "quantum pineapple"}, {}, logging.getLogger(__name__))

    # Both filings are inside upstream's MAX_END_DATE, including the one past
    # the v1 benchmark cutoff.
    assert sorted(row["accessionNo"] for row in json.loads(output.output)) == [
        "0000320193-24-000001",
        "0000789019-25-000001",
    ]
    assert output.error is None


@pytest.mark.asyncio
async def test_a_search_outside_the_corpus_reports_the_span(engine) -> None:
    local = LocalEDGARSearch(engine)

    output = await local.execute(
        {"search_query": "quantum pineapple", "start_date": "2019-01-01", "end_date": "2019-12-31"},
        {},
        logging.getLogger(__name__),
    )

    assert "2024-11-01 through 2025-04-08" in output.error


def test_parse_html_page_keeps_the_upstream_contract(engine, tmp_path: Path) -> None:
    local = LocalParseHtmlPage(engine, tmp_path)

    assert local.name == ParseHtmlPage.name
    assert local.parameters == ParseHtmlPage.parameters


@pytest.mark.asyncio
async def test_a_filing_is_read_from_the_corpus(engine, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    document = corpus / "AAPL/10-K/2024/0000320193-24-000001/primary-document.html"
    document.parent.mkdir(parents=True)
    document.write_text("<html><body><p>Net income was $93.7 billion.</p></body></html>")

    text = await LocalParseHtmlPage(engine, corpus)._parse_html_page(FILING_URL)

    assert text == "Net income was $93.7 billion."


@pytest.mark.asyncio
async def test_a_url_the_corpus_lacks_falls_back_to_the_network(engine, tmp_path: Path, monkeypatch) -> None:
    fetched = []

    async def fake_fetch(self, url):
        fetched.append(url)
        return "from the network"

    monkeypatch.setattr(ParseHtmlPage, "_parse_html_page", fake_fetch)
    local = LocalParseHtmlPage(engine, tmp_path / "empty")

    assert await local._parse_html_page("https://example.com/page.htm") == "from the network"
    assert fetched == ["https://example.com/page.htm"]
    assert local.read_sources == {"live": 1}


@pytest.mark.asyncio
async def test_reads_are_counted_by_source(engine, tmp_path: Path, monkeypatch) -> None:
    """A corpus missing most of what is asked for otherwise shows up only as a
    slow run."""

    async def fake_fetch(self, url):
        return "from the network"

    monkeypatch.setattr(ParseHtmlPage, "_parse_html_page", fake_fetch)
    corpus = tmp_path / "corpus"
    document = corpus / "AAPL/10-K/2024/0000320193-24-000001/primary-document.html"
    document.parent.mkdir(parents=True)
    document.write_text("<html><body><p>Net income.</p></body></html>")
    local = LocalParseHtmlPage(engine, corpus)

    await local._parse_html_page(FILING_URL)
    await local._parse_html_page("https://example.com/other.htm")

    assert local.read_sources == {"sec-corpus": 1, "live": 1}


@pytest.mark.asyncio
async def test_a_filing_the_corpus_lacks_is_fetched_once_and_then_cached(engine, tmp_path: Path, monkeypatch) -> None:
    """A partial corpus otherwise pays the network on every rollout that asks
    for the same missing filing."""
    fetched = []

    async def fake_fetch(self, url):
        fetched.append(url)
        return "from the network"

    monkeypatch.setattr(ParseHtmlPage, "_parse_html_page", fake_fetch)
    local = LocalParseHtmlPage(engine, tmp_path / "empty", cache=ToolCache(tmp_path / "cache"))

    first = await local._parse_html_page(UNINDEXED_FILING_URL)
    second = await local._parse_html_page(UNINDEXED_FILING_URL)

    assert first == second == "from the network"
    assert fetched == [UNINDEXED_FILING_URL]
    assert local.read_sources == {"live": 1, "cache": 1}


@pytest.mark.asyncio
async def test_a_corpus_read_is_parsed_once_and_then_cached(engine, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    document = corpus / "AAPL/10-K/2024/0000320193-24-000001/primary-document.html"
    document.parent.mkdir(parents=True)
    document.write_text("<html><body><p>Net income.</p></body></html>")
    local = LocalParseHtmlPage(engine, corpus, cache=ToolCache(tmp_path / "cache"))

    first = await local._parse_html_page(FILING_URL)
    document.unlink()
    second = await local._parse_html_page(FILING_URL)

    assert first == second == "Net income."
    assert local.read_sources == {"sec-corpus": 1, "cache": 1}


@pytest.mark.asyncio
async def test_the_cache_is_read_before_the_corpus(engine, tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    document = corpus / "AAPL/10-K/2024/0000320193-24-000001/primary-document.html"
    document.parent.mkdir(parents=True)
    document.write_text("<html><body><p>From the corpus.</p></body></html>")
    cache = ToolCache(tmp_path / "cache")
    local = LocalParseHtmlPage(engine, corpus, cache=cache)
    cache.write_text(local._doc_path(FILING_URL), "From the cache.")

    assert await local._parse_html_page(FILING_URL) == "From the cache."
    assert local.read_sources == {"cache": 1}


def _write_prices(root: Path, ticker: str, rows: list[tuple[str, float]], *, with_volume: bool = False) -> None:
    path = root / "equity" / f"{ticker}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for day, close in rows:
        record = {"date": f"{day}T00:00:00.000Z", "close": close, "high": close + 1, "low": close - 1, "open": close}
        if with_volume:
            record["volume"] = 100
        records.append(json.dumps(record))
    path.write_text("\n".join(records) + "\n")


WEEK = [("2024-01-02", 10.0), ("2024-01-03", 11.0), ("2024-01-04", 12.0), ("2024-01-05", 13.0), ("2024-01-08", 14.0)]


def _price_args(ticker: str, start: str, end: str, asset_class: str = "equity") -> dict:
    return {"ticker": ticker, "start_date": start, "end_date": end, "asset_class": asset_class}


def test_price_history_keeps_the_upstream_contract(tmp_path: Path) -> None:
    (tmp_path / "equity").mkdir()
    local = LocalPriceHistory(tmp_path)

    assert local.name == PriceHistory.name
    assert local.description == PriceHistory.description
    assert local.parameters == PriceHistory.parameters
    assert local.required == PriceHistory.required


@pytest.mark.asyncio
async def test_price_history_serves_the_requested_slice_as_upstream_csv(tmp_path: Path) -> None:
    _write_prices(tmp_path, "AAA", WEEK)

    output = await LocalPriceHistory(tmp_path).execute(
        _price_args("aaa", "2024-01-03", "2024-01-06"), {}, logging.getLogger(__name__)
    )

    assert output.error is None
    assert output.output == (
        "date,open,high,low,close\n"
        "2024-01-03,11.0,12.0,10.0,11.0\n"
        "2024-01-04,12.0,13.0,11.0,12.0\n"
        "2024-01-05,13.0,14.0,12.0,13.0"
    )


@pytest.mark.asyncio
async def test_price_history_shows_volume_only_when_the_records_have_it(tmp_path: Path) -> None:
    _write_prices(tmp_path, "AAA", WEEK, with_volume=True)

    output = await LocalPriceHistory(tmp_path).execute(
        _price_args("AAA", "2024-01-02", "2024-01-02"), {}, logging.getLogger(__name__)
    )

    assert output.output.splitlines()[0] == "date,open,high,low,close,volume"


@pytest.mark.asyncio
async def test_price_history_etf_reads_the_equity_store(tmp_path: Path) -> None:
    _write_prices(tmp_path, "SPY", WEEK)

    output = await LocalPriceHistory(tmp_path).execute(
        _price_args("SPY", "2024-01-08", "2024-01-08", asset_class="etf"), {}, logging.getLogger(__name__)
    )

    assert output.output.splitlines()[1] == "2024-01-08,14.0,15.0,13.0,14.0"


@pytest.mark.asyncio
async def test_price_history_accepts_dash_share_classes_for_dot_files(tmp_path: Path) -> None:
    _write_prices(tmp_path, "BRK.B", WEEK)

    output = await LocalPriceHistory(tmp_path).execute(
        _price_args("BRK-B", "2024-01-02", "2024-01-02"), {}, logging.getLogger(__name__)
    )

    assert output.output.splitlines()[1].startswith("2024-01-02,")


@pytest.mark.asyncio
async def test_price_history_unknown_ticker_reports_no_data(tmp_path: Path) -> None:
    _write_prices(tmp_path, "AAA", WEEK)

    output = await LocalPriceHistory(tmp_path).execute(
        _price_args("ZZZZ", "2024-01-02", "2024-01-08"), {}, logging.getLogger(__name__)
    )

    assert output.output == "No pricing data returned for ZZZZ (equity) 2024-01-02..2024-01-08"
    assert output.error is None


@pytest.mark.asyncio
async def test_price_history_rejects_path_like_tickers(tmp_path: Path) -> None:
    _write_prices(tmp_path, "AAA", WEEK)

    output = await LocalPriceHistory(tmp_path).execute(
        _price_args("../equity/AAA", "2024-01-02", "2024-01-08"), {}, logging.getLogger(__name__)
    )

    assert output.output.startswith("No pricing data returned")


@pytest.mark.asyncio
async def test_price_history_crypto_is_unavailable_locally(tmp_path: Path) -> None:
    (tmp_path / "equity").mkdir()

    output = await LocalPriceHistory(tmp_path).execute(
        _price_args("btcusd", "2024-01-02", "2024-01-08", asset_class="crypto"), {}, logging.getLogger(__name__)
    )

    assert "not available from local pricing data" in output.error


@pytest.mark.asyncio
async def test_price_history_clamps_to_the_upstream_end_date(tmp_path: Path) -> None:
    _write_prices(tmp_path, "AAA", [("2026-02-27", 10.0), ("2026-03-02", 11.0)])

    output = await LocalPriceHistory(tmp_path).execute(
        _price_args("AAA", "2026-02-01", "2026-12-31"), {}, logging.getLogger(__name__)
    )

    assert MAX_END_DATE == "2026-03-01"
    assert output.output.splitlines()[1:] == ["2026-02-27,10.0,11.0,9.0,10.0"]


@pytest.mark.asyncio
async def test_local_output_matches_cached_live_output_for_the_same_records(tmp_path: Path, monkeypatch) -> None:
    live_rows = [
        {
            "date": f"{day}T00:00:00.000Z",
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100,
            "adjOpen": close / 2,
            "adjHigh": (close + 1) / 2,
            "adjLow": (close - 1) / 2,
            "adjClose": close / 2,
            "adjVolume": 200,
            "divCash": 0.0,
            "splitFactor": 1.0,
        }
        for day, close in WEEK
    ]

    async def fake_live(self, endpoint, ticker, start_date, end_date):
        return [row for row in live_rows if start_date <= row["date"][:10] <= end_date]

    monkeypatch.setattr(PriceHistory, "_fetch", fake_live)
    args = _price_args("AAA", "2024-01-03", "2024-01-05")
    cached = CachedPriceHistory("unused-key", ToolCache(tmp_path / "cache"))
    prefetch = await cached.execute(_price_args("AAA", "2024-01-01", "2024-01-31"), {}, logging.getLogger(__name__))
    assert prefetch.error is None
    expected = (await cached.execute(args, {}, logging.getLogger(__name__))).output

    local_output = await LocalPriceHistory(tmp_path / "cache" / "pricing").execute(
        args, {}, logging.getLogger(__name__)
    )

    assert local_output.output == expected
    assert "adjClose" in expected.splitlines()[0]


@pytest.mark.asyncio
async def test_price_history_never_opens_a_network_session(tmp_path: Path, monkeypatch) -> None:
    _write_prices(tmp_path, "AAA", WEEK)

    def refuse(*_args, **_kwargs):
        raise AssertionError("local price_history opened a network session")

    monkeypatch.setattr("aiohttp.ClientSession", refuse)

    output = await LocalPriceHistory(tmp_path).execute(
        _price_args("AAA", "2024-01-02", "2024-01-08"), {}, logging.getLogger(__name__)
    )

    assert output.error is None


@pytest.mark.asyncio
async def test_loaded_price_records_stay_within_the_budget(tmp_path: Path, monkeypatch) -> None:
    for ticker in ("AAA", "BBB", "CCC"):
        _write_prices(tmp_path, ticker, WEEK)
    monkeypatch.setattr(LocalPriceHistory, "MAX_LOADED_RECORDS", 2 * len(WEEK))
    local = LocalPriceHistory(tmp_path)

    for ticker in ("AAA", "BBB", "AAA", "CCC"):
        output = await local.execute(_price_args(ticker, "2024-01-02", "2024-01-08"), {}, logging.getLogger(__name__))
        assert output.error is None

    assert [path.stem for path in local._loaded] == ["AAA", "CCC"]
    assert local._loaded_records == 2 * len(WEEK)


@pytest.mark.asyncio
async def test_a_ticker_larger_than_the_budget_is_still_served(tmp_path: Path, monkeypatch) -> None:
    _write_prices(tmp_path, "AAA", WEEK)
    monkeypatch.setattr(LocalPriceHistory, "MAX_LOADED_RECORDS", 1)

    output = await LocalPriceHistory(tmp_path).execute(
        _price_args("AAA", "2024-01-02", "2024-01-08"), {}, logging.getLogger(__name__)
    )

    assert output.error is None
    assert output.output.count("\n") == len(WEEK)
