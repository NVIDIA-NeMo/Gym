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
"""Cache-friendly wrappers around the upstream Vals finance-agent-v2 tools.

Fidelity principle: we never reimplement a tool's logic. Each ``Cached*`` class
subclasses the upstream tool and overrides *only its network method*, caching
the **raw upstream response** and letting the untouched upstream serializer
render it. A cache hit is therefore byte-identical to a live call, and the cache
survives an upstream formatting/SHA bump without a refetch (we re-serialize the
stored raw records with the new code).

Overridden seams (see upstream ``finance_agent/tools.py``):
  - ``PriceHistory._fetch``       -> per-(endpoint, ticker) master of raw Tiingo
                                     records; slice-on-read via the untouched
                                     ``_records_to_csv``.
  - ``EDGARSearch._execute_search`` -> cache the raw sec-api ``filings`` list,
                                     keyed by the normalized request payload.
  - ``ParseHtmlPage._parse_html_page`` -> cache parsed text for sec.gov filing
                                     URLs only; all other URLs pass through.

``SecFilingSearch`` is a *new* tool (not from Vals): a ticker->CIK submissions
lookup over ``data.sec.gov`` that returns sec.gov Archives filing URLs. It is
intended for training/SDG (cheaper, no sec-api key), and is NOT byte-parity with
Vals's ``edgar_search`` full-text search. Expose it only via ``enabled_sec_tools``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import aiohttp
from model_library.agent import Tool, ToolOutput

from finance_agent.tools import (
    MAX_END_DATE,
    EDGARSearch,
    ParseHtmlPage,
    PriceHistory,
    _validate_date_format,
)

# Support both package import (tests: resources_servers.finance_agent_v2.cached_tools)
# and flat script execution (the nemo-gym entrypoint runs app.py directly, so app.py
# imports this module flat as `cached_tools`, and a relative import would fail here).
try:
    from .cache import ToolCache
except ImportError:  # pragma: no cover - exercised only under flat entrypoint execution
    from cache import ToolCache

logger = logging.getLogger(__name__)


# ============================================================================
# price_history
# ============================================================================


class CachedPriceHistory(PriceHistory):
    """PriceHistory with a per-(endpoint, ticker) disk master of raw records.

    The upstream ``execute`` clamps dates, calls ``_fetch`` (which we override),
    then serializes via the classmethod ``_records_to_csv``. We return the
    requested date slice as raw records; the untouched serializer then renders
    byte-identical output (including its active-column drop logic, applied to the
    slice).
    """

    _NAMESPACE = "pricing"

    def __init__(self, api_key: Optional[str], cache: ToolCache) -> None:
        super().__init__(api_key)
        self._cache = cache

    @staticmethod
    def _norm_ticker(endpoint: str, ticker: str) -> str:
        # Mirror upstream URL construction: equity uppercases, crypto/fx lowercase.
        t = ticker.strip()
        return t.upper() if endpoint == "equity" else t.lower()

    def _master_paths(self, endpoint: str, ticker: str) -> tuple[Path, Path]:
        t = self._norm_ticker(endpoint, ticker)
        base = self._cache.path(self._NAMESPACE, endpoint)
        return base / f"{t}.jsonl", base / f"{t}.meta.json"

    @staticmethod
    def _rec_date(rec: dict[str, Any]) -> str:
        d = rec.get("date", "")
        if isinstance(d, str) and "T" in d:
            return d.split("T", 1)[0]
        return str(d)

    def _slice(self, records: list[dict[str, Any]], start_date: str, end_date: str) -> list[dict[str, Any]]:
        return [r for r in records if start_date <= self._rec_date(r) <= end_date]

    def _merge(self, old: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Keep already-cached rows on overlap (freezes the adjusted-as-of of the
        # first fetch for reproducibility); only genuinely new dates are added.
        by_date: dict[str, dict[str, Any]] = {self._rec_date(r): r for r in fresh}
        by_date.update({self._rec_date(r): r for r in old})
        return [by_date[d] for d in sorted(by_date)]

    async def _fetch(
        self, endpoint: str, ticker: str, start_date: str, end_date: str
    ) -> list[dict[str, Any]]:
        cache = self._cache
        if not cache.enabled:
            return await super()._fetch(endpoint, ticker, start_date, end_date)

        recs_path, meta_path = self._master_paths(endpoint, ticker)
        meta = cache.read_json(meta_path)
        records = cache.read_jsonl(recs_path)

        covered = (
            isinstance(meta, dict)
            and records is not None
            and meta.get("cov_start", "9999-99-99") <= start_date
            and meta.get("cov_end", "0000-00-00") >= end_date
        )
        if covered:
            return self._slice(records, start_date, end_date)

        # Fetch the union of the requested range and any existing coverage, so the
        # stored coverage stays a single contiguous interval.
        fetch_start, fetch_end = start_date, end_date
        if isinstance(meta, dict):
            fetch_start = min(fetch_start, meta.get("cov_start", start_date))
            fetch_end = max(fetch_end, meta.get("cov_end", end_date))

        fresh = await super()._fetch(endpoint, ticker, fetch_start, fetch_end)
        merged = self._merge(records or [], fresh)
        cache.write_jsonl(recs_path, merged)
        cache.write_json(
            meta_path,
            {
                "endpoint": endpoint,
                "ticker": self._norm_ticker(endpoint, ticker),
                "cov_start": fetch_start,
                "cov_end": fetch_end,
            },
        )
        return self._slice(merged, start_date, end_date)


# ============================================================================
# edgar_search
# ============================================================================


class CachedEDGARSearch(EDGARSearch):
    """EDGARSearch that caches the raw sec-api.io ``filings`` list.

    Keyed by the normalized request payload *excluding* ``top_n_results`` (the
    upstream slices by ``top_n_results`` after fetching a page): we cache the full
    page and apply the slice locally, so the cache is independent of ``top_n``.
    """

    _NAMESPACE = "edgar_search"

    def __init__(
        self,
        sec_api_key: Optional[str] = None,
        key_rotator: Any = None,
        cache: Optional[ToolCache] = None,
    ) -> None:
        super().__init__(sec_api_key=sec_api_key, key_rotator=key_rotator)
        self._cache = cache

    async def _execute_search(
        self,
        search_query: str,
        start_date: str = "1900-01-01",
        end_date: str = MAX_END_DATE,
        top_n_results: int = 100,
        page: int = 1,
        form_types: Any = None,
        ciks: Any = None,
    ) -> list:
        cache = self._cache
        if cache is None or not cache.enabled:
            return await super()._execute_search(
                search_query, start_date, end_date, top_n_results, page, form_types, ciks
            )

        # Mirror the upstream clamp/validation so the key matches what a live call
        # would have fetched (and invalid dates raise identically).
        _validate_date_format("start_date", start_date)
        _validate_date_format("end_date", end_date)
        k_start = min(start_date, MAX_END_DATE)
        k_end = min(end_date, MAX_END_DATE)
        request = {
            "query": search_query,
            "startDate": k_start,
            "endDate": k_end,
            "page": page,
            "formTypes": form_types,
            "ciks": ciks,
        }
        # A full-text search has no accession/filename to name the file by, so the
        # key is a hash of the request. Prefix it with a human-readable slug of the
        # query, and store the request alongside the results, so a stray cache file
        # is easy to identify and debug.
        key = cache.hash_key(request)
        path = cache.path(self._NAMESPACE, f"{self._slug(search_query)}_{key[:12]}.json")

        stored = cache.read_json(path)
        if isinstance(stored, dict) and "filings" in stored:
            full = stored["filings"]
        else:
            # Fetch the full page (top_n=100) so the stored entry is top_n-independent.
            full = await super()._execute_search(
                search_query, start_date, end_date, 100, page, form_types, ciks
            )
            cache.write_json(path, {"request": request, "filings": full})

        return full[: int(top_n_results)]

    @staticmethod
    def _slug(text: str, max_len: int = 48) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
        return (slug[:max_len].rstrip("-") or "query")


# ============================================================================
# parse_html_page (SEC documents only)
# ============================================================================


# Matches a SEC EDGAR Archives *document* URL:
#   https://www.sec.gov/Archives/edgar/data/<CIK>/<ACCESSION_NODASH>/<filename>
_SEC_DOC_RE = re.compile(r"sec\.gov/Archives/edgar/data/(\d+)/([0-9A-Za-z]+)/(.+)$")


class CachedParseHtmlPage(ParseHtmlPage):
    """ParseHtmlPage that caches parsed text for sec.gov filing URLs.

    Only SEC EDGAR Archives document URLs are cached (immutable per accession);
    every other URL falls through to the untouched upstream fetch+parse. The
    on-disk layout is the corrected nested form (the V1 server used a flat
    ``<accession>.txt`` filename, conflating a filing's multiple documents):
        sec_filings/<cik_padded>/<accession_nodash>/<primary-doc-filename>.txt
    """

    _NAMESPACE = "sec_filings"

    def __init__(self, cache: ToolCache) -> None:
        super().__init__()
        self._cache = cache

    def _doc_path(self, url: str) -> Optional[Path]:
        clean = url.split("?", 1)[0].split("#", 1)[0]
        m = _SEC_DOC_RE.search(clean)
        if not m:
            return None
        cik = m.group(1).zfill(10)
        accession = m.group(2)
        filename = m.group(3).strip("/").replace("/", "_")
        if not filename:
            return None
        return self._cache.path(self._NAMESPACE, cik, accession, f"{filename}.txt")

    async def _parse_html_page(self, url: str) -> str:
        cache = self._cache
        path = self._doc_path(url) if cache.enabled else None
        if path is None:
            # Non-SEC URL (or cache disabled): identical to upstream, uncached.
            return await super()._parse_html_page(url)

        cached = cache.read_text(path)
        if cached is not None:
            return cached

        text = await super()._parse_html_page(url)
        if text:
            cache.write_text(path, text)
        return text


# ============================================================================
# sec_filing_search (new; training/SDG)
# ============================================================================


class _RateLimiter:
    """Sliding-window rate limiter for polite sec.gov access."""

    def __init__(self, max_requests: int = 10, window_seconds: float = 1.0) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._requests and (now - self._requests[0]) >= self.window_seconds:
                    self._requests.popleft()
                if len(self._requests) < self.max_requests:
                    self._requests.append(now)
                    return
                sleep_time = self.window_seconds - (now - self._requests[0])
            await asyncio.sleep(max(sleep_time, 0.01))


class SecFilingSearch(Tool):
    """Ticker -> CIK -> filings lookup over data.sec.gov (cheap, no sec-api key).

    Returns filing metadata (form, dates, accession number, and a sec.gov
    Archives ``filing_url``) sorted newest first. The agent then reads a filing
    by calling ``parse_html_page`` on a ``filing_url`` (cached by
    ``CachedParseHtmlPage``). NOT byte-parity with Vals's ``edgar_search``.
    """

    name = "sec_filing_search"
    description = (
        "Search for SEC filings by ticker symbol. Returns a list of filing metadata "
        "(form type, filing date, report date, accession number, and a sec.gov filing_url), "
        "sorted newest first. This does not return filing text — call parse_html_page on a "
        "filing_url to read the document. Supports optional form_types, start_date, and end_date filters."
    )
    parameters: dict[str, Any] = {
        "ticker": {
            "type": "string",
            "description": "Stock ticker symbol as listed on a US exchange (e.g. 'AAPL', 'MSFT').",
        },
        "form_types": {
            "type": "array",
            "items": {"type": "string"},
            "description": "(optional) Limit to specific EDGAR form types, e.g. ['10-K', '10-Q', '8-K'].",
        },
        "start_date": {
            "type": "string",
            "description": "(optional) Earliest filing date, YYYY-MM-DD.",
        },
        "end_date": {
            "type": "string",
            "description": "(optional) Latest filing date, YYYY-MM-DD.",
        },
    }
    required: list[str] = ["ticker"]

    _NAMESPACE_META = "sec_submissions"
    _TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
    _SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    _SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"

    def __init__(
        self,
        cache: ToolCache,
        user_agent: str,
        max_filing_results: int = 50,
        requests_per_second: int = 10,
        max_retries: int = 5,
        request_timeout: float = 30.0,
    ) -> None:
        super().__init__()
        self._cache = cache
        self._user_agent = user_agent
        self._max_filing_results = max_filing_results
        self._max_retries = max_retries
        self._request_timeout = request_timeout
        self._rate_limiter = _RateLimiter(requests_per_second, 1.0)
        self._tickers: Optional[dict[str, dict[str, str]]] = None
        self._tickers_lock = asyncio.Lock()
        self._cik_locks: dict[str, asyncio.Lock] = {}

    # -- HTTP -----------------------------------------------------------------
    async def _fetch(self, url: str) -> Optional[str]:
        timeout = aiohttp.ClientTimeout(total=self._request_timeout)
        headers = {"User-Agent": self._user_agent}
        for attempt in range(self._max_retries):
            await self._rate_limiter.acquire()
            try:
                async with aiohttp.ClientSession(headers=headers) as session:
                    async with session.get(url, timeout=timeout) as response:
                        if response.status == 200:
                            return await response.text()
                        if response.status in (403, 429, 503):
                            logger.warning(
                                "sec.gov %d on attempt %d/%d for %s",
                                response.status, attempt + 1, self._max_retries, url,
                            )
                            await asyncio.sleep(2**attempt)
                            continue
                        logger.warning("sec.gov %d (non-retryable) for %s", response.status, url)
                        return None
            except (aiohttp.ClientError, asyncio.TimeoutError):
                logger.warning("Fetch error attempt %d/%d for %s", attempt + 1, self._max_retries, url, exc_info=True)
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(2**attempt)
        return None

    # -- ticker map -----------------------------------------------------------
    async def _ensure_tickers(self) -> dict[str, dict[str, str]]:
        if self._tickers is not None:
            return self._tickers
        async with self._tickers_lock:
            if self._tickers is not None:
                return self._tickers

            cache = self._cache
            path = cache.path(self._NAMESPACE_META, "company_tickers.json") if cache.enabled else None
            raw = cache.read_json(path) if path is not None else None

            if raw is None:
                data = await self._fetch(self._TICKERS_URL)
                if not data:
                    raise RuntimeError("Failed to download company_tickers.json from sec.gov.")
                raw = json.loads(data)
                if path is not None:
                    cache.write_json(path, raw)

            tickers: dict[str, dict[str, str]] = {}
            for item in raw.values():
                tickers[item["ticker"]] = {
                    "cik": str(item["cik_str"]).zfill(10),
                    "name": item["title"],
                }
            self._tickers = tickers
            logger.info("sec_filing_search: loaded %d ticker mappings", len(tickers))
            return tickers

    async def _resolve_ticker(self, ticker: str) -> Optional[dict[str, str]]:
        tickers = await self._ensure_tickers()
        info = tickers.get(ticker.strip().upper())
        if info is None:
            return None
        return {"cik": info["cik"], "ticker": ticker.strip().upper(), "name": info["name"]}

    # -- filings --------------------------------------------------------------
    @staticmethod
    def _parse_filings_columns(columns: dict[str, Any], cik: str, ticker: str) -> dict[str, dict[str, Any]]:
        acc_numbers = columns.get("accessionNumber", [])
        forms = columns.get("form", [])
        dates = columns.get("filingDate", [])
        report_dates = columns.get("reportDate", [])
        primary_docs = columns.get("primaryDocument", [])

        filings: dict[str, dict[str, Any]] = {}
        for acc, form, fdate, rdate, pdoc in zip(acc_numbers, forms, dates, report_dates, primary_docs):
            acc_nodash = acc.replace("-", "")
            filings[acc_nodash] = {
                "ticker": ticker,
                "cik": cik,
                "form": form,
                "filing_date": fdate,
                "report_date": rdate,
                "accession_number": acc,
                "primary_document": pdoc,
                "filing_url": f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{acc_nodash}/{pdoc}",
            }
        return filings

    async def _get_company_filings(self, cik: str, ticker: str) -> dict[str, dict[str, Any]]:
        cik_padded = str(cik).zfill(10)
        cache = self._cache

        path = cache.path(self._NAMESPACE_META, f"CIK{cik_padded}.json") if cache.enabled else None
        if path is not None:
            cached = cache.read_json(path)
            if isinstance(cached, dict):
                return cached

        lock = self._cik_locks.setdefault(cik_padded, asyncio.Lock())
        async with lock:
            if path is not None:
                cached = cache.read_json(path)
                if isinstance(cached, dict):
                    return cached

            data = await self._fetch(self._SUBMISSIONS_URL.format(cik=cik_padded))
            if not data:
                logger.warning("SEC submissions unavailable for CIK %s (%s)", cik_padded, ticker)
                return {}

            try:
                filings_data = json.loads(data).get("filings", {})
                recent = filings_data.get("recent", {})
                filings = self._parse_filings_columns(recent, cik_padded, ticker)

                for file_ref in filings_data.get("files", []):
                    name = file_ref.get("name", "")
                    if not name:
                        continue
                    extra_data = await self._fetch(self._SUBMISSIONS_FILE_URL.format(name=name))
                    if extra_data:
                        try:
                            extra = json.loads(extra_data)
                            filings.update(self._parse_filings_columns(extra, cik_padded, ticker))
                        except json.JSONDecodeError:
                            logger.warning("Failed to parse supplementary file %s for CIK %s", name, cik_padded)
            except (json.JSONDecodeError, KeyError):
                logger.warning("Failed to parse SEC submissions for CIK %s (%s)", cik_padded, ticker, exc_info=True)
                return {}

            if path is not None and filings:
                cache.write_json(path, filings)
            return filings

    # -- tool entry point -----------------------------------------------------
    async def execute(
        self, args: dict[str, Any], state: dict[str, Any], logger: logging.Logger
    ) -> ToolOutput:
        try:
            ticker = str(args.get("ticker", "")).strip()
            if not ticker:
                raise ValueError("ticker is required")

            form_types = args.get("form_types")
            if form_types is not None and not isinstance(form_types, list):
                raise ValueError(f"form_types must be a list if provided. Was {type(form_types).__name__}.")
            start_date = args.get("start_date")
            end_date = args.get("end_date")

            company = await self._resolve_ticker(ticker)
            if not company:
                return ToolOutput(
                    output=json.dumps(
                        {
                            "error": f"No company found for ticker '{ticker}'",
                            "suggestion": "Use the exact US-exchange ticker symbol (e.g. 'AAPL'). "
                            "Only companies in https://www.sec.gov/files/company_tickers.json are supported.",
                        }
                    )
                )

            filings = await self._get_company_filings(company["cik"], company["ticker"])

            results = []
            for filing in filings.values():
                if form_types and filing["form"] not in form_types:
                    continue
                results.append(
                    {
                        "ticker": company["ticker"],
                        "company_name": company["name"],
                        "form": filing["form"],
                        "filing_date": filing.get("filing_date", ""),
                        "report_date": filing.get("report_date", ""),
                        "accession_number": filing.get("accession_number", ""),
                        "filing_url": filing.get("filing_url", ""),
                    }
                )

            results.sort(key=lambda x: x["filing_date"], reverse=True)
            if start_date:
                results = [r for r in results if r["filing_date"] >= start_date]
            if end_date:
                results = [r for r in results if r["filing_date"] <= end_date]
            results = results[: self._max_filing_results]

            if not results:
                return ToolOutput(
                    output=json.dumps(
                        {
                            "error": f"No filings found for '{ticker}'",
                            "suggestion": "Try removing the form_types filter, widening the date range, or checking the ticker.",
                        }
                    )
                )

            return ToolOutput(output=json.dumps(results, indent=2))
        except Exception as e:  # noqa: BLE001 — surface as a tool error, never crash the agent
            error_msg = str(e)
            logger.warning("sec_filing_search failed: %s", error_msg)
            return ToolOutput(output=error_msg, error=error_msg)
