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
"""Serve SEC filing lists and documents from a downloaded corpus, with no SEC request.

The corpus is NVFlow's ``download-sec`` output, whose ``sec_metadata.parquet`` maps
each downloaded filing document's URL to its saved file. Filing lists, ticker
lookup and filing index pages are all derived from it, so everything listed can
be read.
"""

from __future__ import annotations

import html
import re
from collections import Counter, defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

import httpx
import pyarrow.parquet as pq
from big_finance_harness.tools import ToolError


_ARCHIVE_FOLDER_RE = re.compile(r"/Archives/edgar/data/(\d+)/(\d{18})(?=[/.]|$)", re.IGNORECASE)
_ARCHIVE_DASHED_RE = re.compile(r"/Archives/edgar/data/(\d+)/(\d{10}-\d{2}-\d{6})(?=[/.]|$)", re.IGNORECASE)
_HTML_PATH_RE = re.compile(r"\.html?$", re.IGNORECASE)
_FILING_PAGE_RE = re.compile(r"^/?$|-index(?:-headers)?\.html?$|^/index\.html?$", re.IGNORECASE)
_COLUMNS = (
    "accession_number",
    "form_type",
    "report_date",
    "filing_date",
    "url",
    "file_type",
    "file_location",
    "ticker",
)


@dataclass(frozen=True, slots=True)
class FilingRecord:
    form: str
    filing_date: str
    accession: str
    primary_document: str
    report_date: str


@dataclass(frozen=True, slots=True)
class _Document:
    url: str
    path: Path
    file_type: str
    name: str


@dataclass(slots=True)
class _Filing:
    form: str
    filing_date: str
    documents: list[_Document]


def archive_accession(url: str) -> Optional[tuple[str, str]]:
    """Return ``(cik, dashed accession)`` for an EDGAR archive URL."""
    if match := _ARCHIVE_DASHED_RE.search(url):
        return match.group(1), match.group(2)
    if match := _ARCHIVE_FOLDER_RE.search(url):
        digits = match.group(2)
        return match.group(1), f"{digits[:10]}-{digits[10:12]}-{digits[12:]}"
    return None


def normalize_sec_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if host == "sec.gov":
        host = "www.sec.gov"
    return urlunsplit(("https", host, parts.path, "", ""))


def _resolve(root: Path, location: str) -> Path:
    # Locations are recorded as container paths; resolve them under the parquet's own directory.
    marker = f"/{root.name}/"
    return root / location.split(marker, 1)[1] if marker in location else Path(location)


class LocalSecCorpus:
    """Read-only view of a downloaded SEC corpus.

    Only HTML documents whose saved file belongs to exactly one URL are served: the
    downloader names exhibits by document type, so same-type exhibits of one filing
    overwrite each other on disk, and raw XML filings are stored as rendered HTML.
    Section rows are skipped: they are local per-item splits that reuse the primary
    document's URL, and EDGAR serves only the whole document at that URL.
    A filing is listed only when its primary document is served.
    """

    def __init__(self, *, metadata_path: str | Path) -> None:
        metadata_path = Path(metadata_path)
        rows = pq.read_table(metadata_path, columns=list(_COLUMNS)).to_pylist()
        owners = Counter(row["file_location"] for row in rows)
        self._documents: dict[str, _Document] = {}
        self._filings: dict[str, _Filing] = {}
        self._tickers: dict[str, str] = {}
        histories: defaultdict[str, list[FilingRecord]] = defaultdict(list)
        for row in rows:
            if row["file_type"] == "section":
                continue
            url = normalize_sec_url(row["url"])
            parsed = archive_accession(url)
            if parsed is None or owners[row["file_location"]] > 1 or not _HTML_PATH_RE.search(urlsplit(url).path):
                continue
            cik, accession = parsed
            filing_date = str(row["filing_date"])[:10]
            path = _resolve(metadata_path.parent, row["file_location"])
            is_primary = row["file_type"] == "primary_document"
            document = _Document(
                url=url, path=path, file_type=row["file_type"], name="primary document" if is_primary else path.stem
            )
            self._documents[url] = document
            self._filings.setdefault(accession, _Filing(row["form_type"], filing_date, [])).documents.append(document)
            if is_primary:
                self._tickers.setdefault(str(row["ticker"]).upper(), cik)
                histories[cik].append(
                    FilingRecord(
                        form=row["form_type"],
                        filing_date=filing_date,
                        accession=accession,
                        primary_document=url.split(f"/{accession.replace('-', '')}/", 1)[1],
                        report_date=str(row["report_date"] or ""),
                    )
                )
        self._histories = {
            cik: sorted(records, key=lambda record: record.filing_date, reverse=True)
            for cik, records in histories.items()
        }

    def ticker_to_cik(self, ticker: str) -> str:
        cik = self._tickers.get(ticker.strip().upper())
        if cik is None:
            raise ToolError(f"unknown ticker: {ticker} (no filings for it in the offline SEC corpus)")
        return cik

    def history(self, cik: str) -> list[FilingRecord]:
        """A company's readable filings, newest first."""
        return self._histories.get(str(int(cik)), [])

    def filing_date(self, accession: str) -> Optional[str]:
        filing = self._filings.get(accession)
        return filing.filing_date if filing else None

    def response(self, url: str) -> httpx.Response:
        """Serve a saved document or a filing's index page; refuse anything else."""
        key = normalize_sec_url(url)
        document = self._documents.get(key)
        if document is not None:
            try:
                content = document.path.read_bytes()
            except OSError as exc:
                raise ToolError(f"fetch_url failed to read the offline copy of {url!r}: {exc}") from exc
            return _html_response(key, content)
        page = self._filing_page(key)
        if page is not None:
            return _html_response(key, page.encode("utf-8"))
        raise ToolError(
            f"fetch_url: {url!r} is not in the offline SEC corpus. Only filings listed by edgar_search, "
            "their filing index pages and the exhibits shown there can be read; EDGAR full-text search, "
            "browse pages and XBRL viewer (R*.htm) pages are unavailable."
        )

    def _filing_page(self, url: str) -> Optional[str]:
        path = urlsplit(url).path
        match = _ARCHIVE_FOLDER_RE.search(path)
        if match is None or not _FILING_PAGE_RE.search(path[match.end() :]):
            return None
        parsed = archive_accession(path)
        filing = self._filings.get(parsed[1]) if parsed else None
        if parsed is None or filing is None:
            return None
        rows = "".join(
            f"<tr><td>{html.escape(document.name)}</td><td>{html.escape(document.url)}</td></tr>"
            for document in sorted(filing.documents, key=lambda document: document.file_type != "primary_document")
        )
        return (
            f"<html><head><title>Filing {parsed[1]}</title></head><body>"
            f"<h1>Filing {parsed[1]}</h1><p>Form {html.escape(filing.form)}, filed {filing.filing_date}</p>"
            f"<p>Documents available (pass a URL to fetch_url):</p><table>{rows}</table></body></html>"
        )


def _html_response(url: str, content: bytes) -> httpx.Response:
    return httpx.Response(
        200,
        content=content,
        headers={"content-type": "text/html; charset=utf-8"},
        request=httpx.Request("GET", url),
    )


class LocalFilingHistory:
    """The ``SecFilingHistory`` interface over the corpus."""

    def __init__(self, corpus: LocalSecCorpus) -> None:
        self._corpus = corpus

    async def records(self, cik: str) -> AsyncIterator[FilingRecord]:
        for record in self._corpus.history(cik):
            yield record

    async def filing_date(self, cik: str, accession: str) -> Optional[str]:
        return self._corpus.filing_date(accession)
