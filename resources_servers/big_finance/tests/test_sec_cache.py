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
import gzip
import json
from pathlib import Path
from unittest.mock import AsyncMock

import httpx
import pymupdf
import pytest
from big_finance_harness.tools import EdgarSearchTool, FetchUrlTool, ToolError
from big_finance_harness.tools import edgar_search as upstream_edgar
from big_finance_harness.tools import fetch_url as upstream_fetch

from resources_servers.big_finance import sec_cache
from resources_servers.big_finance.sec_cache import CachedEdgarSearchTool, CachedFetchUrlTool, SecHttpCache
from resources_servers.sec_local_index.cache import ToolCache


_USER_AGENT = "Gym cache tests cache-tests@example.com"
_DOCUMENT_URL = "https://www.sec.gov/Archives/edgar/data/123/annual.htm"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK0000000123.json"
_TICKERS = {"0": {"ticker": "EXAMPLE", "cik_str": 123, "title": "Example Corporation"}}
_SUBMISSIONS = {
    "filings": {
        "recent": {
            "form": ["10-Q", "10-K", "10-Q", "10-K"],
            "filingDate": ["2026-05-01", "2026-02-01", "2025-11-01", "2025-02-01"],
            "accessionNumber": ["0000123-26-000004", "0000123-26-000003", "0000123-25-000002", "0000123-25-000001"],
            "primaryDocument": ["q1-2026.htm", "annual-2025.htm", "q3-2025.htm", "annual-2024.htm"],
        },
        "files": [{"name": "CIK0000000123-submissions-001.json"}],
    }
}


@pytest.fixture(autouse=True)
def _isolate_upstream_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upstream_edgar, "_TICKER_CACHE", None)
    # Tests exercise cache and upstream text processing without DNS or external HTTP.
    monkeypatch.setattr(upstream_fetch, "_check_url_safe", lambda url: None)


def _response(url: str, text: str = "Annual revenue is $100 million.") -> httpx.Response:
    return httpx.Response(200, text=text, request=httpx.Request("GET", url))


def _fetch_tool(cache_dir: Path, *, use_cache: bool = True) -> CachedFetchUrlTool:
    return CachedFetchUrlTool(
        cache=SecHttpCache(cache_dir=str(cache_dir), use_cache=use_cache), sec_user_agent=_USER_AGENT
    )


def _cache_contents(root: Path) -> dict[str, bytes]:
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


async def test_edgar_caches_tickers_and_unfiltered_metadata_across_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def get_response(url: str) -> httpx.Response:
        payload = _TICKERS if url == upstream_edgar.TICKERS_URL else _SUBMISSIONS
        assert url in {upstream_edgar.TICKERS_URL, _SUBMISSIONS_URL}
        return httpx.Response(200, json=payload, request=httpx.Request("GET", url))

    transport = AsyncMock(side_effect=get_response)
    monkeypatch.setattr(EdgarSearchTool, "_get", transport)
    first = CachedEdgarSearchTool(cache=SecHttpCache(cache_dir=str(tmp_path), use_cache=True), user_agent=_USER_AGENT)
    newest = json.loads(await first.run({"ticker": "EXAMPLE", "form_type": "10-K", "limit": 1}))
    assert [filing["filing_date"] for filing in newest["filings"]] == ["2026-02-01"]
    assert transport.await_count == 2

    # Clear upstream's process-global ticker cache to prove both HTTP resources persist on disk.
    monkeypatch.setattr(upstream_edgar, "_TICKER_CACHE", None)
    second = CachedEdgarSearchTool(cache=SecHttpCache(cache_dir=str(tmp_path), use_cache=True), user_agent=_USER_AGENT)
    annuals = json.loads(await second.run({"ticker": "example", "form_type": "10-k", "limit": 20}))
    quarterlies = json.loads(await second.run({"ticker": "EXAMPLE", "form_type": "10-Q", "limit": 20}))
    assert [filing["filing_date"] for filing in annuals["filings"]] == ["2026-02-01", "2025-02-01"]
    assert [filing["filing_date"] for filing in quarterlies["filings"]] == ["2026-05-01", "2025-11-01"]
    assert annuals["filings"][0]["primary_document_url"].endswith("/000012326000003/annual-2025.htm")
    assert transport.await_count == 2


async def test_fetch_reuses_raw_document_for_different_queries_and_token_budgets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = (
        "<html><body><nav>Navigation text</nav>"
        + "".join(
            f"<p>{topic} " + (f"{detail} " * 400) + "</p><p></p>"
            for topic, detail in [("Revenue", "sales growth"), ("Debt", "interest obligations"), ("Cash", "liquidity")]
        )
        + "</body></html>"
    )

    def make_response(url: str) -> httpx.Response:
        return httpx.Response(200, text=html, headers={"content-type": "text/html"}, request=httpx.Request("GET", url))

    transport = AsyncMock(side_effect=make_response)
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)
    upstream = FetchUrlTool(sec_user_agent=_USER_AGENT)
    requests = [
        {"url": _DOCUMENT_URL, "max_tokens": 500},
        {"url": _DOCUMENT_URL, "max_tokens": 2000},
        {"url": _DOCUMENT_URL, "query": "debt interest obligations", "max_tokens": 500},
        {"url": _DOCUMENT_URL, "query": "cash liquidity", "max_tokens": 1000},
    ]
    expected = [await upstream.run(args) for args in requests]
    assert len(set(expected)) == len(requests)
    assert all("Navigation text" not in result for result in expected)
    transport.reset_mock()

    actual = [await _fetch_tool(tmp_path).run(args) for args in requests]
    assert actual == expected
    assert transport.await_count == 1


async def test_disabled_cache_ignores_existing_entries_and_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = AsyncMock(return_value=_response(_DOCUMENT_URL, "Previously downloaded report."))
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)
    assert await _fetch_tool(tmp_path).run({"url": _DOCUMENT_URL}) == "Previously downloaded report."
    saved = _cache_contents(tmp_path)
    assert saved

    transport.return_value = _response(_DOCUMENT_URL, "Current live report.")
    tool = _fetch_tool(tmp_path, use_cache=False)
    for _ in range(2):
        assert await tool.run({"url": _DOCUMENT_URL}) == "Current live report."
    assert transport.await_count == 3
    assert _cache_contents(tmp_path) == saved

    fresh_dir = tmp_path / "disabled"
    assert await _fetch_tool(fresh_dir, use_cache=False).run({"url": _DOCUMENT_URL}) == "Current live report."
    assert not _cache_contents(fresh_dir)


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/annual.htm",
        "https://sec.gov.evil.example/annual.htm",
        "https://notsec.gov/annual.htm",
        "https://example.com/sec.gov/annual.htm",
    ],
)
async def test_non_sec_documents_are_never_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str) -> None:
    transport = AsyncMock(side_effect=lambda requested: _response(requested))
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)
    tool = _fetch_tool(tmp_path)
    assert await tool.run({"url": url}) == await tool.run({"url": url})
    assert transport.await_count == 2
    assert not _cache_contents(tmp_path)


@pytest.mark.parametrize("url", ["https://sec.gov/annual.txt", "https://data.sec.gov/annual.txt", _DOCUMENT_URL])
async def test_sec_hosts_share_persistent_document_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    transport = AsyncMock(side_effect=lambda requested: _response(requested))
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)
    assert await _fetch_tool(tmp_path).run({"url": url}) == "Annual revenue is $100 million."
    assert await _fetch_tool(tmp_path).run({"url": url}) == "Annual revenue is $100 million."
    assert transport.await_count == 1


async def test_concurrent_requests_download_one_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def delayed_response(url: str) -> httpx.Response:
        entered.set()
        await release.wait()
        return _response(url)

    transport = AsyncMock(side_effect=delayed_response)
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)
    tool = _fetch_tool(tmp_path)
    tasks = [asyncio.create_task(tool.run({"url": _DOCUMENT_URL})) for _ in range(8)]
    try:
        await asyncio.wait_for(entered.wait(), timeout=5)
    finally:
        release.set()
    results = await asyncio.gather(*tasks)
    assert results == ["Annual revenue is $100 million."] * 8
    assert transport.await_count == 1


async def test_http_failure_is_not_cached_and_later_request_can_recover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = httpx.Request("GET", _DOCUMENT_URL)
    failure = httpx.HTTPStatusError("SEC temporarily unavailable", request=request, response=httpx.Response(503))
    transport = AsyncMock(side_effect=[failure, failure, _response(_DOCUMENT_URL)])
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)
    tool = _fetch_tool(tmp_path)
    for _ in range(2):
        with pytest.raises(ToolError, match="temporarily unavailable"):
            await tool.run({"url": _DOCUMENT_URL})
    assert not _cache_contents(tmp_path)
    assert await tool.run({"url": _DOCUMENT_URL}) == "Annual revenue is $100 million."
    assert await _fetch_tool(tmp_path).run({"url": _DOCUMENT_URL}) == "Annual revenue is $100 million."
    assert transport.await_count == 3


async def test_corrupt_cache_entry_is_refetched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transport = AsyncMock(return_value=_response(_DOCUMENT_URL, "Old report."))
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)
    assert await _fetch_tool(tmp_path).run({"url": _DOCUMENT_URL}) == "Old report."
    entries = list(tmp_path.rglob("*.json"))
    assert entries
    for entry in entries:
        entry.write_text("{invalid json", encoding="utf-8")

    transport.return_value = _response(_DOCUMENT_URL, "Fresh report.")
    assert await _fetch_tool(tmp_path).run({"url": _DOCUMENT_URL}) == "Fresh report."
    assert await _fetch_tool(tmp_path).run({"url": _DOCUMENT_URL}) == "Fresh report."
    assert transport.await_count == 2


@pytest.mark.parametrize("method", ["read_json", "write_json"])
async def test_cache_io_failure_does_not_fail_successful_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    transport = AsyncMock(side_effect=lambda requested: _response(requested))
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)

    def unavailable(*args: object, **kwargs: object) -> None:
        raise OSError("cache filesystem unavailable")

    monkeypatch.setattr(ToolCache, method, unavailable)
    tool = _fetch_tool(tmp_path)
    assert await tool.run({"url": _DOCUMENT_URL}) == "Annual revenue is $100 million."
    assert await tool.run({"url": _DOCUMENT_URL}) == "Annual revenue is $100 million."
    assert transport.await_count == 2


async def test_cached_url_still_obeys_upstream_blocklist(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transport = AsyncMock(side_effect=lambda requested: _response(requested))
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)
    assert await _fetch_tool(tmp_path).run({"url": _DOCUMENT_URL}) == "Annual revenue is $100 million."
    monkeypatch.setattr(upstream_fetch, "is_blocked_url", lambda url: url == _DOCUMENT_URL)
    with pytest.raises(ToolError, match="blocked"):
        await _fetch_tool(tmp_path).run({"url": _DOCUMENT_URL})
    assert transport.await_count == 1


async def test_redirect_to_blocked_host_is_not_cached(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transport = AsyncMock(return_value=_response("https://github.com/example/benchmark", "Blocked source."))
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)
    with pytest.raises(ToolError, match="blocked"):
        await _fetch_tool(tmp_path).run({"url": _DOCUMENT_URL})
    assert not _cache_contents(tmp_path)


@pytest.mark.parametrize("body", ["Not JSON", "[]", "null"])
async def test_metadata_success_status_with_invalid_payload_is_not_cached(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str
) -> None:
    transport = AsyncMock(return_value=_response(_SUBMISSIONS_URL, body))
    monkeypatch.setattr(EdgarSearchTool, "_get", transport)
    tool = CachedEdgarSearchTool(cache=SecHttpCache(cache_dir=str(tmp_path), use_cache=True), user_agent=_USER_AGENT)
    # The HTTP layer leaves validation errors to the unchanged upstream caller.
    assert (await tool._get(_SUBMISSIONS_URL)).text == body
    assert (await tool._get(_SUBMISSIONS_URL)).text == body
    assert transport.await_count == 2
    assert not _cache_contents(tmp_path)


async def test_warm_cache_does_not_bypass_required_sec_user_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with monkeypatch.context() as setup:
        setup.setattr(FetchUrlTool, "_fetch", AsyncMock(return_value=_response(_DOCUMENT_URL)))
        assert await _fetch_tool(tmp_path).run({"url": _DOCUMENT_URL}) == "Annual revenue is $100 million."
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    tool = CachedFetchUrlTool(cache=SecHttpCache(cache_dir=str(tmp_path), use_cache=True), sec_user_agent=None)
    with pytest.raises(ToolError, match="SEC_EDGAR_USER_AGENT is unset"):
        await tool.run({"url": _DOCUMENT_URL})


async def test_cached_redirect_history_is_checked_against_current_blocklist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    intermediate = "https://www.sec.gov/redirect/report"
    final = "https://www.sec.gov/Archives/final-report.txt"
    response = httpx.Response(
        200,
        text="Redirected annual report.",
        request=httpx.Request("GET", final),
        history=[
            httpx.Response(302, headers={"location": intermediate}, request=httpx.Request("GET", _DOCUMENT_URL)),
            httpx.Response(302, headers={"location": final}, request=httpx.Request("GET", intermediate)),
        ],
    )
    transport = AsyncMock(return_value=response)
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)
    assert await _fetch_tool(tmp_path).run({"url": _DOCUMENT_URL}) == "Redirected annual report."
    monkeypatch.setattr(sec_cache, "is_blocked_url", lambda url: url == intermediate)
    with pytest.raises(ToolError, match="blocked"):
        await _fetch_tool(tmp_path).run({"url": _DOCUMENT_URL})
    assert transport.await_count == 1


async def test_cached_pdf_bytes_use_upstream_pdf_extraction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pymupdf.open() as document:
        page = document.new_page()
        page.insert_text((72, 72), "Annual revenue was 100 million dollars.")
        pdf = document.tobytes()
    # Neither URL nor content type identifies a PDF; the preserved bytes must identify it.
    url = "https://www.sec.gov/Archives/report"
    response = httpx.Response(
        200, content=pdf, headers={"content-type": "application/octet-stream"}, request=httpx.Request("GET", url)
    )
    transport = AsyncMock(return_value=response)
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)
    expected = await FetchUrlTool(sec_user_agent=_USER_AGENT).run({"url": url})
    transport.reset_mock()
    assert "Annual revenue was 100 million dollars." in expected
    assert await _fetch_tool(tmp_path).run({"url": url}) == expected
    assert await _fetch_tool(tmp_path).run({"url": url}) == expected
    assert transport.await_count == 1


@pytest.mark.parametrize("compressed", [False, True])
async def test_cache_preserves_character_encoding_without_double_decompression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, compressed: bool
) -> None:
    url = "https://www.sec.gov/Archives/report.txt"
    expected = "Caf\u00e9 revenue: \u00a3123 million."
    raw = expected.encode("iso-8859-1")
    headers = {"content-type": "text/plain"}
    if compressed:
        raw = gzip.compress(raw)
        headers["content-encoding"] = "gzip"
    response = httpx.Response(200, content=raw, headers=headers, request=httpx.Request("GET", url))
    response.encoding = "iso-8859-1"
    transport = AsyncMock(return_value=response)
    monkeypatch.setattr(FetchUrlTool, "_fetch", transport)
    assert await _fetch_tool(tmp_path).run({"url": url}) == expected
    assert await _fetch_tool(tmp_path).run({"url": url}) == expected
    assert transport.await_count == 1
