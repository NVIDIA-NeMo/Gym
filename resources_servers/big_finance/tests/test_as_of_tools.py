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

import asyncio
import json
import socket
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from big_finance_harness.tools import ToolError

from resources_servers.big_finance.as_of_tools import AsOfEdgarSearchTool, AsOfFetchUrlTool
from resources_servers.big_finance.local_corpus import LocalSecCorpus, archive_accession
from resources_servers.big_finance.sec_cache import CachedFetchUrlTool, SecHttpCache


def _json_response(url: str, payload: dict) -> httpx.Response:
    return httpx.Response(200, json=payload, request=httpx.Request("GET", url))


def _submissions() -> dict[str, dict]:
    head = {
        "filings": {
            "recent": {
                "form": ["10-Q", "10-K", "8-K"],
                "filingDate": ["2024-02-02", "2023-11-03", "2023-08-03"],
                "accessionNumber": ["0000320193-24-000006", "0000320193-23-000106", "0000320193-23-000077"],
                "primaryDocument": ["q1.htm", "aapl-20230930.htm", "8k.htm"],
                "reportDate": ["2023-12-30", "2023-09-30", "2023-08-03"],
            },
            "files": [{"name": "CIK0000320193-submissions-001.json"}],
        }
    }
    page = {
        "form": ["10-K"],
        "filingDate": ["2020-10-30"],
        "accessionNumber": ["0000320193-20-000096"],
        "primaryDocument": ["aapl-20200926.htm"],
        "reportDate": ["2020-09-26"],
    }
    return {
        "https://data.sec.gov/submissions/CIK0000320193.json": head,
        "https://data.sec.gov/submissions/CIK0000320193-submissions-001.json": page,
    }


def _edgar_tool() -> AsOfEdgarSearchTool:
    tool = AsOfEdgarSearchTool(cache=SecHttpCache(cache_dir=None, use_cache=False), user_agent="Test test@example.com")
    pages = _submissions()

    async def fake_get(url: str) -> httpx.Response:
        return _json_response(url, pages[url])

    tool._get = fake_get
    tool._ticker_to_cik = AsyncMock(return_value="320193")
    return tool


def test_archive_accession_accepts_folder_and_dashed_forms() -> None:
    assert archive_accession(
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm"
    ) == (
        "320193",
        "0000320193-23-000106",
    )
    assert archive_accession("https://www.sec.gov/Archives/edgar/data/320193/0000320193-23-000106.txt") == (
        "320193",
        "0000320193-23-000106",
    )
    assert archive_accession("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany") is None


async def test_edgar_search_hides_later_filings_and_reads_older_pages() -> None:
    tool = _edgar_tool()
    result = json.loads(await tool.run({"ticker": "AAPL", "form_type": "10-K"}, as_of=date(2023, 11, 3)))
    assert result["filed_on_or_before"] == "2023-11-03"
    assert [row["accession"] for row in result["filings"]] == ["0000320193-23-000106", "0000320193-20-000096"]
    assert result["filings"][0]["report_date"] == "2023-09-30"
    assert result["filings"][1]["primary_document_url"].endswith("/000032019320000096/aapl-20200926.htm")

    unclamped = json.loads(await tool.run({"ticker": "AAPL", "limit": 1}))
    assert [row["accession"] for row in unclamped["filings"]] == ["0000320193-24-000006"]
    assert "filed_on_or_before" not in unclamped


async def test_fetch_url_refuses_archive_documents_filed_after_as_of() -> None:
    edgar = _edgar_tool()
    fetch = AsOfFetchUrlTool(
        cache=SecHttpCache(cache_dir=None, use_cache=False), history=edgar.history, sec_user_agent="Test t@e.com"
    )
    as_of = date(2023, 11, 3)
    later = "https://www.sec.gov/Archives/edgar/data/320193/000032019324000006/q1.htm"
    with pytest.raises(ToolError, match="filed on 2024-02-02"):
        await fetch.check_as_of(later, as_of)
    await fetch.check_as_of(
        "https://www.sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm", as_of
    )
    await fetch.check_as_of("https://www.sec.gov/Archives/edgar/data/320193/000032019320000096/a.htm", as_of)
    # Unknown accessions fall back to the year encoded in the accession number.
    with pytest.raises(ToolError, match="postdates"):
        await fetch.check_as_of("https://www.sec.gov/Archives/edgar/data/320193/000099999924000001/x.htm", as_of)
    await fetch.check_as_of("https://www.sec.gov/Archives/edgar/data/320193/000099999923000001/x.htm", as_of)
    await fetch.check_as_of(later, None)
    await fetch.check_as_of("https://data.sec.gov/api/xbrl/companyfacts/CIK0000320193.json", as_of)


_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/320193"


def _corpus(tmp_path: Path) -> LocalSecCorpus:
    download = tmp_path / "step-0-download"
    container = "/workspace/outputs/step-0-download"
    files = {
        "10-K/aapl.html": "<html><body><p>Net sales were 383,285.</p></body></html>",
        "10-K/exhibits/EX-99.1.html": "<html><body><p>Press release.</p></body></html>",
        "10-K/exhibits/EX-10.1.html": "<html><body><p>Whichever exhibit was written last.</p></body></html>",
        "10-Q/q1.html": "<html><body><p>Quarter.</p></body></html>",
        "4/form4.html": "<html><body><p>Rendered Form 4.</p></body></html>",
        "10-K/15.html": "<html><body><p>Item 15. Exhibits.</p></body></html>",
    }
    for name, text in files.items():
        (download / "data" / name).parent.mkdir(parents=True, exist_ok=True)
        (download / "data" / name).write_text(text, encoding="utf-8")

    def row(accession: str, form: str, filed: str, url: str, kind: str, name: str) -> dict:
        return {
            "accession_number": accession,
            "form_type": form,
            "report_date": "2023-09-30",
            "filing_date": date.fromisoformat(filed),
            "url": url,
            "file_type": kind,
            "file_location": f"{container}/data/{name}",
            "ticker": "AAPL",
        }

    k, q, f4 = "0000320193-23-000106", "0000320193-24-000006", "0000320193-23-000090"
    rows = [
        row(
            k,
            "10-K",
            "2023-11-03",
            f"{_ARCHIVE}/000032019323000106/aapl-20230930.htm",
            "primary_document",
            "10-K/aapl.html",
        ),
        row(
            k,
            "10-K",
            "2023-11-03",
            f"{_ARCHIVE}/000032019323000106/ex991.htm",
            "exhibit",
            "10-K/exhibits/EX-99.1.html",
        ),
        row(
            k,
            "10-K",
            "2023-11-03",
            f"{_ARCHIVE}/000032019323000106/ex101a.htm",
            "exhibit",
            "10-K/exhibits/EX-10.1.html",
        ),
        row(
            k,
            "10-K",
            "2023-11-03",
            f"{_ARCHIVE}/000032019323000106/ex101b.htm",
            "exhibit",
            "10-K/exhibits/EX-10.1.html",
        ),
        row(
            k,
            "10-K",
            "2023-11-03",
            f"{_ARCHIVE}/000032019323000106/aapl-20230930.htm",
            "section",
            "10-K/15.html",
        ),
        row(q, "10-Q", "2024-02-02", f"{_ARCHIVE}/000032019324000006/q1.htm", "primary_document", "10-Q/q1.html"),
        row(
            f4,
            "4",
            "2023-10-02",
            f"{_ARCHIVE}/000032019323000090/xslF345X05/form4.xml",
            "primary_document",
            "4/form4.html",
        ),
    ]
    pq.write_table(pa.Table.from_pylist(rows), download / "sec_metadata.parquet")
    return LocalSecCorpus(metadata_path=download / "sec_metadata.parquet")


def _offline_tools(tmp_path: Path) -> tuple[AsOfEdgarSearchTool, AsOfFetchUrlTool]:
    corpus = _corpus(tmp_path)
    cache = SecHttpCache(cache_dir=None, use_cache=False)
    edgar = AsOfEdgarSearchTool(cache=cache, user_agent="Test test@example.com", corpus=corpus)
    edgar._get = AsyncMock(side_effect=AssertionError("offline edgar_search requested SEC"))
    fetch = AsOfFetchUrlTool(cache=cache, history=edgar.history, sec_user_agent="Test t@e.com", corpus=corpus)
    return edgar, fetch


async def test_offline_edgar_search_lists_only_readable_filings(tmp_path: Path) -> None:
    edgar, _ = _offline_tools(tmp_path)
    result = json.loads(await edgar.run({"ticker": "aapl"}, as_of=date(2023, 11, 3)))
    assert [row["accession"] for row in result["filings"]] == ["0000320193-23-000106"]
    assert result["filings"][0]["primary_document_url"] == f"{_ARCHIVE}/000032019323000106/aapl-20230930.htm"
    unclamped = json.loads(await edgar.run({"ticker": "AAPL"}))
    assert [row["form"] for row in unclamped["filings"]] == ["10-Q", "10-K"]
    with pytest.raises(ToolError, match="unknown ticker"):
        await edgar.run({"ticker": "NOPE"})


async def test_offline_fetch_url_serves_the_corpus_and_refuses_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("big_finance_harness.tools.fetch_url._check_url_safe", lambda url: None)
    _, fetch = _offline_tools(tmp_path)
    as_of = date(2023, 11, 3)
    folder = f"{_ARCHIVE}/000032019323000106"

    bare_host = "http://sec.gov/Archives/edgar/data/320193/000032019323000106/aapl-20230930.htm#part-ii"
    assert "Net sales were 383,285." in await fetch.run({"url": bare_host}, as_of)
    assert "Press release." in await fetch.run({"url": f"{folder}/ex991.htm"}, as_of)
    for page in (f"{folder}/0000320193-23-000106-index.htm", f"{folder}/", folder):
        index = await fetch.run({"url": page}, as_of)
        assert f"{folder}/aapl-20230930.htm" in index and f"{folder}/ex991.htm" in index and "EX-99.1" in index
        assert "ex101a" not in index and index.count("aapl-20230930.htm") == 1
    assert "Item 15" not in await fetch.run({"url": f"{folder}/aapl-20230930.htm"}, as_of)
    for url in (
        f"{folder}/ex101a.htm",
        f"{_ARCHIVE}/000032019323000090/xslF345X05/form4.xml",
        f"{folder}/R4.htm",
        "https://efts.sec.gov/LATEST/search-index?q=revenue",
        "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=320193",
    ):
        with pytest.raises(ToolError, match="not in the offline SEC corpus"):
            await fetch.run({"url": url}, as_of)
    with pytest.raises(ToolError, match="filed on 2024-02-02"):
        await fetch.run({"url": f"{_ARCHIVE}/000032019324000006/q1.htm"}, as_of)


async def test_offline_corpus_reads_need_no_dns_while_other_fetches_keep_the_address_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    resolved: list[str] = []

    def no_dns(host, *args, **kwargs):
        resolved.append(host)
        raise socket.gaierror(socket.EAI_AGAIN, "Temporary failure in name resolution")

    monkeypatch.setattr(socket, "getaddrinfo", no_dns)
    _, fetch = _offline_tools(tmp_path)
    as_of = date(2023, 11, 3)
    sec_url = f"{_ARCHIVE}/000032019323000106/aapl-20230930.htm"

    assert "Net sales were 383,285." in await fetch.run({"url": sec_url}, as_of)
    assert resolved == []

    sec_result, web_result = await asyncio.gather(
        fetch.run({"url": sec_url}, as_of),
        fetch.run({"url": "https://example.com/news"}, as_of),
        return_exceptions=True,
    )
    assert "Net sales were 383,285." in sec_result
    assert isinstance(web_result, ToolError) and "could not resolve 'example.com'" in str(web_result)
    assert resolved == ["example.com"]

    live = CachedFetchUrlTool(cache=SecHttpCache(cache_dir=None, use_cache=False), sec_user_agent="Test t@e.com")
    with pytest.raises(ToolError, match="could not resolve 'www.sec.gov'"):
        await live.run({"url": sec_url})
