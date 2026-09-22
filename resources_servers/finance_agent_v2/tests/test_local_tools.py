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
"""Local-mode SEC tools: same contract as upstream, different source."""

import json
import logging
from pathlib import Path

import pytest
from finance_agent.tools import MAX_END_DATE, EDGARSearch, ParseHtmlPage

from resources_servers.finance_agent_v2.local_tools import LocalEDGARSearch, LocalParseHtmlPage
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
