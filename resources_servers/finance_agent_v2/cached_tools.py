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

No tool logic is reimplemented. Each ``Cached*`` class subclasses the upstream tool
and overrides only its network method, storing the raw upstream response and letting
the untouched upstream serializer render it. A hit is therefore byte-identical to a
live call, and the cache survives an upstream formatting bump without a refetch.

Overridden seams (see upstream ``finance_agent/tools.py``):
  - ``PriceHistory._fetch`` -> per-(endpoint, ticker) master of raw Tiingo records,
    sliced on read by the untouched ``_records_to_csv``.
  - ``EDGARSearch._execute_search`` -> raw sec-api ``filings`` list, keyed by the
    normalized request payload.
  - ``ParseHtmlPage._parse_html_page`` -> parsed text for sec.gov filing URLs only.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

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
        # Both parts name a path component and both come from the tool call, so
        # both go through safe_name; a real ticker passes through unchanged.
        t = ToolCache.safe_name(self._norm_ticker(endpoint, ticker))
        endpoint_dir = ToolCache.safe_name(endpoint)
        return (
            self._cache.path(self._NAMESPACE, endpoint_dir, f"{t}.jsonl"),
            self._cache.path(self._NAMESPACE, endpoint_dir, f"{t}.meta.json"),
        )

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

    async def _fetch(self, endpoint: str, ticker: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
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
            full = await super()._execute_search(search_query, start_date, end_date, 100, page, form_types, ciks)
            cache.write_json(path, {"request": request, "filings": full})

        return full[: int(top_n_results)]

    @staticmethod
    def _slug(text: str, max_len: int = 48) -> str:
        slug = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower()).strip("-")
        return slug[:max_len].rstrip("-") or "query"


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
