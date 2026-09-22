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
import sqlite3
from pathlib import Path

import pytest
from finance_agent.tools import MAX_END_DATE, EDGARSearch, ParseHtmlPage

from resources_servers.finance_agent_v2.local_tools import LocalEDGARSearch, LocalParseHtmlPage
from resources_servers.sec_local_index.local_edgar_search import LocalEdgarSearch


DUMP_PREFIX = "/workspace/outputs/finance/demo/workflow-2-download-sec/step-0-download/data"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000001/aapl.htm"


def _index(path: Path) -> Path:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            accession_number TEXT NOT NULL,
            cik TEXT NOT NULL,
            company_name TEXT NOT NULL,
            ticker TEXT NOT NULL,
            description TEXT,
            form_type TEXT NOT NULL,
            document_type TEXT NOT NULL,
            filing_date TEXT NOT NULL,
            url TEXT NOT NULL,
            canonical_url_key TEXT NOT NULL,
            source_path TEXT NOT NULL,
            body TEXT NOT NULL
        );
        CREATE UNIQUE INDEX documents_url_key ON documents(canonical_url_key);
        CREATE VIRTUAL TABLE documents_fts USING fts5(
            body, content='documents', content_rowid='id'
        );
        """
    )
    row = (
        1,
        "0000320193-24-000001",
        "320193",
        "Apple Inc.",
        "AAPL",
        "10-K",
        "10-K",
        "10-K",
        "2024-11-01",
        FILING_URL,
        "320193:000032019324000001:aapl.htm",
        f"{DUMP_PREFIX}/AAPL/10-K/2024/0000320193-24-000001/primary-document.html",
        "quantum pineapple net income",
    )
    connection.execute(
        """
        INSERT INTO documents (
            id, accession_number, cik, company_name, ticker, description,
            form_type, document_type, filing_date, url, canonical_url_key,
            source_path, body
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        row,
    )
    connection.execute("INSERT INTO documents_fts(rowid, body) VALUES (?, ?)", (1, row[-1]))
    connection.commit()
    connection.close()
    return path


@pytest.fixture
def engine(tmp_path: Path) -> LocalEdgarSearch:
    return LocalEdgarSearch(_index(tmp_path / "index.sqlite"), max_end_date=MAX_END_DATE)


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

    assert json.loads(output.output)[0]["accessionNo"] == "0000320193-24-000001"
    assert output.error is None


@pytest.mark.asyncio
async def test_a_search_outside_the_corpus_reports_the_span(engine) -> None:
    local = LocalEDGARSearch(engine)

    output = await local.execute(
        {"search_query": "quantum pineapple", "start_date": "2019-01-01", "end_date": "2019-12-31"},
        {},
        logging.getLogger(__name__),
    )

    assert "2024-11-01 through 2024-11-01" in output.error


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
