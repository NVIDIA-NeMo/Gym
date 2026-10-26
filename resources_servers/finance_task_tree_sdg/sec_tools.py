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
"""SEC tools that only expose filings available on a task's as-of date."""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import date
from typing import Any, Optional

import httpx
from big_finance_harness.tools import ToolError

from resources_servers.big_finance.sec_cache import CachedEdgarSearchTool, CachedFetchUrlTool, SecHttpCache


SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
DEFAULT_LIMIT = 20

_ARCHIVE_FOLDER_RE = re.compile(r"/Archives/edgar/data/(\d+)/(\d{18})(?=[/.]|$)", re.IGNORECASE)
_ARCHIVE_DASHED_RE = re.compile(r"/Archives/edgar/data/(\d+)/(\d{10}-\d{2}-\d{6})(?=[/.]|$)", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class FilingRecord:
    form: str
    filing_date: str
    accession: str
    primary_document: str
    report_date: str


def _page_records(page: dict[str, Any]) -> list[FilingRecord]:
    columns = (
        page.get("form", []),
        page.get("filingDate", []),
        page.get("accessionNumber", []),
        page.get("primaryDocument", []),
        page.get("reportDate", []) or [""] * len(page.get("form", [])),
    )
    return [FilingRecord(*values) for values in zip(*columns)]


def parse_as_of(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(f"as_of_date must be an ISO date, got {value!r}") from exc


def archive_accession(url: str) -> Optional[tuple[str, str]]:
    """Return ``(cik, dashed accession)`` for an EDGAR archive URL."""
    if match := _ARCHIVE_DASHED_RE.search(url):
        return match.group(1), match.group(2)
    if match := _ARCHIVE_FOLDER_RE.search(url):
        digits = match.group(2)
        return match.group(1), f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"
    return None


class SecFilingHistory:
    """Read a company's full submission history, newest first, through the SEC cache."""

    def __init__(self, search_tool: CachedEdgarSearchTool) -> None:
        self._tool = search_tool
        self._pages: dict[str, list[list[FilingRecord]]] = {}
        self._page_names: dict[str, list[str]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def _get_json(self, url: str) -> dict[str, Any]:
        try:
            response = await self._tool._get(url)
        except httpx.HTTPError as exc:
            raise ToolError(f"edgar_search failed: {exc}") from exc
        payload = response.json()
        if not isinstance(payload, dict):
            raise ToolError(f"edgar_search failed: unexpected payload from {url}")
        return payload

    async def _ensure_head(self, cik: str) -> None:
        if cik in self._pages:
            return
        lock = self._locks.setdefault(cik, asyncio.Lock())
        async with lock:
            if cik in self._pages:
                return
            data = await self._get_json(SUBMISSIONS_URL.format(cik=cik))
            filings = data.get("filings", {})
            self._page_names[cik] = [item["name"] for item in filings.get("files", []) if item.get("name")]
            self._pages[cik] = [_page_records(filings.get("recent", {}))]

    async def _load_page(self, cik: str, index: int) -> Optional[list[FilingRecord]]:
        pages = self._pages[cik]
        if index < len(pages):
            return pages[index]
        names = self._page_names[cik]
        if index - 1 >= len(names):
            return None
        async with self._locks[cik]:
            if index < len(pages):
                return pages[index]
            page = await self._get_json(SUBMISSIONS_PAGE_URL.format(name=names[index - 1]))
            pages.append(_page_records(page))
            return pages[index]

    async def records(self, cik: str) -> AsyncIterator[FilingRecord]:
        await self._ensure_head(cik)
        index = 0
        while (page := await self._load_page(cik, index)) is not None:
            for record in page:
                yield record
            index += 1

    async def filing_date(self, cik: str, accession: str) -> Optional[str]:
        async for record in self.records(cik):
            if record.accession == accession:
                return record.filing_date
        return None


class AsOfEdgarSearchTool(CachedEdgarSearchTool):
    """List filings, newest first, filed on or before the task's as-of date."""

    description = (
        "List SEC EDGAR filings for a public company by ticker, newest first. Returns form "
        "type (10-K, 10-Q, 8-K, etc.), filing date, period of report, accession number, and "
        "the URL of the primary document. Only filings available on this task's as-of date "
        "are listed; older filings are included. Use `fetch_url` to read a filing's contents."
    )

    def __init__(self, *, cache: SecHttpCache, user_agent: str, timeout_s: float = 20.0) -> None:
        super().__init__(cache=cache, user_agent=user_agent, timeout_s=timeout_s)
        self.history = SecFilingHistory(self)

    async def run(self, args: dict[str, Any], as_of: Optional[date] = None) -> str:
        ticker = str(args.get("ticker") or "").strip()
        if not ticker:
            raise ToolError("ticker is required")
        form_type = args.get("form_type")
        limit = int(args.get("limit") or DEFAULT_LIMIT)
        try:
            cik = await self._ticker_to_cik(ticker)
        except httpx.HTTPError as exc:
            raise ToolError(f"edgar_search failed: {exc}") from exc

        cutoff = as_of.isoformat() if as_of else None
        out: list[dict[str, Any]] = []
        async for record in self.history.records(cik):
            if cutoff and record.filing_date > cutoff:
                continue
            if form_type and record.form.upper() != str(form_type).upper():
                continue
            folder = record.accession.replace("-", "")
            out.append(
                {
                    "form": record.form,
                    "filing_date": record.filing_date,
                    "report_date": record.report_date,
                    "accession": record.accession,
                    "primary_document_url": (
                        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{folder}/{record.primary_document}"
                    ),
                }
            )
            if len(out) >= limit:
                break
        payload: dict[str, Any] = {"ticker": ticker.upper(), "cik": cik, "filings": out}
        if cutoff:
            payload["filed_on_or_before"] = cutoff
        return json.dumps(payload, ensure_ascii=False)


class AsOfFetchUrlTool(CachedFetchUrlTool):
    """Refuse EDGAR archive documents filed after the task's as-of date."""

    def __init__(
        self,
        *,
        cache: SecHttpCache,
        history: SecFilingHistory,
        timeout_s: float = 30.0,
        sec_user_agent: str | None = None,
    ) -> None:
        super().__init__(cache=cache, timeout_s=timeout_s, sec_user_agent=sec_user_agent)
        self._history = history

    async def check_as_of(self, url: str, as_of: Optional[date]) -> None:
        if as_of is None or (parsed := archive_accession(url)) is None:
            return
        cik, accession = parsed
        cutoff = as_of.isoformat()
        filed = await self._history.filing_date(cik.lstrip("0") or "0", accession)
        if filed is not None:
            if filed > cutoff:
                raise ToolError(
                    f"fetch_url refuses {url!r}: filing {accession} was filed on {filed}, "
                    f"after this task's as-of date {cutoff}"
                )
            return
        # Filings missing from the company's history fall back to the accession year.
        if 2000 + int(accession[11:13]) > as_of.year:
            raise ToolError(f"fetch_url refuses {url!r}: filing {accession} postdates this task's as-of date {cutoff}")

    async def run(self, args: dict[str, Any], as_of: Optional[date] = None) -> str:
        await self.check_as_of(str(args.get("url") or "").strip(), as_of)
        return await super().run(args)
