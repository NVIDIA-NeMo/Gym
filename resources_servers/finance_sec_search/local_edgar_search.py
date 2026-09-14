# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""SQLite-backed implementation of the agent-facing EDGAR search contract."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional
from urllib.parse import unquote


logger = logging.getLogger(__name__)

DEFAULT_START_DATE = "1900-01-01"
MAX_END_DATE = "2025-04-07"
PAGE_SIZE = 100
TOKEN_RE = re.compile(r'"(?:[^"]|"")*"|\S+')
BAREWORD_RE = re.compile(r"^[A-Za-z0-9_]+$")

SIDECAR_SCHEMA_VERSION = 1
SIDECAR_SUFFIX = ".metadata"
SIDECAR_ALIAS = "meta"
SIDECAR_TABLE = f"{SIDECAR_ALIAS}.documents_meta"
FINGERPRINT_SAMPLES = 64

# Above this size, serving result metadata out of a table that also stores filing
# text is slow enough that a rollout reads it as a hang, so it is refused rather
# than allowed to degrade quietly. Small indexes pay no meaningful penalty.
SLOW_METADATA_LIMIT_BYTES = 1 << 30

# Mirrored into the sidecar so that swapping the metadata source leaves every
# column name the search query references unchanged.
METADATA_COLUMNS = (
    "id",
    "accession_number",
    "cik",
    "ticker",
    "company_name",
    "form_type",
    "document_type",
    "description",
    "filing_date",
    "url",
)

# Not mirrored into the sidecar, so dump path lookups always read documents.
DUMP_PATH_COLUMNS = ("canonical_url_key", "source_path")

# Keeps the IN clause below SQLite's bound-parameter limit on any build.
DUMP_PATH_CHUNK_SIZE = 500

SEC_ARCHIVES_URL_RE = re.compile(r"sec\.gov/Archives/edgar/data/(\d+)/(\d+)/([^?#]*)")


def default_sidecar_path(index_path: str | Path) -> Path:
    return Path(str(index_path) + SIDECAR_SUFFIX)


def canonical_url_key(url: str) -> str | None:
    """Return the index's document key for an SEC Archives URL, or None.

    Must stay byte-identical to the key the index builder writes: unpadded CIK,
    dashless accession, lowercased filename.
    """
    match = SEC_ARCHIVES_URL_RE.search(url)
    if not match:
        return None
    filename = unquote(match.group(3)).strip("/").rsplit("/", 1)[-1].lower()
    if not filename:
        return None
    return f"{int(match.group(1))}:{match.group(2)}:{filename}"


def _relative_source_path(source_path: str) -> str | None:
    """Return the part of an indexed path below the download's data directory.

    Indexed paths are container-absolute, so only the fragment below data/ is
    portable to whatever the reader has mounted.
    """
    _, separator, relative = source_path.replace("\\", "/").partition("/data/")
    return relative if separator and relative else None


def fingerprint_source_index(connection: sqlite3.Connection) -> str:
    """Fingerprint the row identity of an index.

    The sidecar is joined to the full-text index on rowid, so a rebuilt index
    that reassigns rowids would silently pair filings with the wrong metadata.
    Sampling by primary key keeps this cheap enough to verify on every startup.
    """
    highest = connection.execute("SELECT MAX(id) FROM documents").fetchone()[0]
    if not highest:
        return "empty"
    stride = max(1, highest // FINGERPRINT_SAMPLES)
    sampled = [1 + offset * stride for offset in range(FINGERPRINT_SAMPLES)]
    placeholders = ",".join("?" for _ in sampled)
    rows = connection.execute(
        f"SELECT id, accession_number, filing_date FROM documents WHERE id IN ({placeholders}) ORDER BY id",
        sampled,
    ).fetchall()
    digest = hashlib.sha256()
    digest.update(f"{highest}\x1e".encode())
    for row in rows:
        digest.update("\x1f".join(str(value) for value in row).encode())
        digest.update(b"\x1e")
    return digest.hexdigest()


@dataclass(frozen=True)
class LocalEdgarRequest:
    search_query: str
    form_types: tuple[str, ...] | None
    ciks: tuple[str, ...] | None
    start_date: str
    end_date: str
    page: int
    top_n_results: int


def _quote_fts(value: str) -> str:
    return f'"{value.replace(chr(34), chr(34) * 2)}"'


def _translate_term(token: str) -> str:
    prefix = token.endswith("*")
    value = token[:-1] if prefix else token
    if not value:
        raise ValueError("Wildcard requires a non-empty prefix")
    if "*" in value:
        raise ValueError("Wildcards are supported only at the end of a term")
    if token.startswith('"'):
        if prefix:
            raise ValueError("Wildcards are not supported on quoted phrases")
        if not token.endswith('"') or len(token) < 2:
            raise ValueError("Unterminated quoted phrase")
        return _quote_fts(token[1:-1].replace('""', '"'))
    translated = value if BAREWORD_RE.fullmatch(value) else _quote_fts(value)
    return f"{translated}*" if prefix else translated


def translate_query(query: str) -> str:
    if not query.strip():
        raise ValueError("Query must not be empty")
    if any(character in query for character in "()"):
        raise ValueError("Parentheses are not supported")

    groups: list[list[str]] = [[]]
    exclusions: list[str] = []
    negate_next = False
    for raw in TOKEN_RE.findall(query):
        if raw == "OR":
            if not groups[-1]:
                raise ValueError("OR must follow a search term")
            groups.append([])
            negate_next = False
            continue
        if raw == "AND":
            raise ValueError("Explicit AND is unsupported; use spaces for implicit AND")
        if raw == "NOT":
            negate_next = True
            continue

        excluded = negate_next or raw.startswith("-")
        negate_next = False
        term = _translate_term(raw[1:] if raw.startswith("-") else raw)
        (exclusions if excluded else groups[-1]).append(term)

    if negate_next:
        raise ValueError("NOT must be followed by a search term")
    if not groups[-1]:
        raise ValueError("Query must end with a positive search term")
    expressions = [" AND ".join(group) for group in groups]
    positive = expressions[0] if len(expressions) == 1 else f"({' OR '.join(expressions)})"
    if not exclusions:
        return positive
    negative = " OR ".join(exclusions)
    return f"{positive} NOT ({negative})" if len(exclusions) > 1 else f"{positive} NOT {negative}"


def _date_value(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string in yyyy-mm-dd format")
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError as error:
        raise ValueError(f"{name} '{value}' is not in yyyy-mm-dd format") from error


def _optional_strings(name: str, value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError(f"The parameter {name} must be a list if provided. Was of type {type(value)}")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"The parameter {name} must contain only strings")
    return tuple(value) or None


def normalize_request(
    search_query: str,
    start_date: str = DEFAULT_START_DATE,
    end_date: str = MAX_END_DATE,
    top_n_results: int = PAGE_SIZE,
    page: int = 1,
    form_types: Optional[list[str]] = None,
    ciks: Optional[list[str]] = None,
    *,
    max_end_date: str = MAX_END_DATE,
) -> LocalEdgarRequest:
    if not isinstance(search_query, str) or not search_query.strip():
        raise ValueError(
            "search_query is required and cannot be empty. Provide a search term "
            "to search the contents of SEC filings."
        )

    maximum = _date_value("max_end_date", max_end_date)
    start = min(_date_value("start_date", start_date or DEFAULT_START_DATE), maximum)
    end = min(_date_value("end_date", end_date or maximum), maximum)
    if start > end:
        raise ValueError(f"Parameter start_date '{start}' was set to a date that is later than end_date '{end}'")
    if isinstance(page, bool) or not isinstance(page, int) or page < 1:
        raise ValueError("page must be an integer greater than or equal to 1")
    if isinstance(top_n_results, bool) or not isinstance(top_n_results, int) or not 1 <= top_n_results <= PAGE_SIZE:
        raise ValueError("top_n_results must be an integer between 1 and 100")

    forms = _optional_strings("form_types", form_types)
    raw_ciks = _optional_strings("ciks", ciks)
    try:
        normalized_ciks = tuple(str(int(cik)) for cik in raw_ciks) if raw_ciks else None
    except ValueError as error:
        raise ValueError("The parameter ciks must contain numeric strings") from error

    return LocalEdgarRequest(
        search_query=search_query,
        form_types=forms,
        ciks=normalized_ciks,
        start_date=start,
        end_date=end,
        page=page,
        top_n_results=top_n_results,
    )


class LocalEdgarSearch:
    def __init__(
        self,
        index_path: str | Path,
        *,
        max_end_date: str = MAX_END_DATE,
        metrics_dir: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ):
        self.index_path = Path(index_path)
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Local EDGAR index not found: {self.index_path}")
        self._document_columns = self._validate_index()
        self.metadata_path = self._resolve_metadata_path(metadata_path)
        self._require_usable_metadata_source()
        self.max_end_date = _date_value("max_end_date", max_end_date)
        self.metrics_path: Path | None = None
        self._metrics_lock = threading.Lock()
        self._local = threading.local()
        if metrics_dir:
            destination = Path(metrics_dir)
            destination.mkdir(parents=True, exist_ok=True)
            identity = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
            self.metrics_path = destination / f"search-{identity}-{os.getpid()}.jsonl"

    @property
    def uses_metadata_sidecar(self) -> bool:
        return self.metadata_path is not None

    def _resolve_metadata_path(self, metadata_path: str | Path | None) -> Path | None:
        """Locate the metadata sidecar, requiring it to match the index if present.

        A configured-but-unusable sidecar is an error rather than a downgrade:
        searches would still answer correctly without it, but orders of magnitude
        more slowly, and that is worth failing loudly for.
        """
        if metadata_path is None:
            candidate = default_sidecar_path(self.index_path)
            if not candidate.is_file():
                logger.info(
                    "No metadata sidecar at %s — searches will read filing metadata from the index's documents table",
                    candidate,
                )
                return None
        else:
            candidate = Path(metadata_path)
            if not candidate.is_file():
                raise FileNotFoundError(f"Local EDGAR metadata sidecar not found: {candidate}")

        self._validate_metadata_sidecar(candidate)
        logger.info("Local EDGAR metadata sidecar initialized from %s", candidate)
        return candidate

    def _validate_metadata_sidecar(self, candidate: Path) -> None:
        sidecar = sqlite3.connect(f"file:{candidate}?mode=ro", uri=True)
        try:
            recorded = {str(row[0]): str(row[1]) for row in sidecar.execute("SELECT key, value FROM sidecar_metadata")}
        except sqlite3.DatabaseError as error:
            raise ValueError(f"Metadata sidecar {candidate} is not readable: {error}") from error
        finally:
            sidecar.close()

        version = recorded.get("schema_version")
        if version != str(SIDECAR_SCHEMA_VERSION):
            raise ValueError(
                f"Metadata sidecar {candidate} has schema version {version!r}, "
                f"expected {SIDECAR_SCHEMA_VERSION}. Rebuild it with "
                f"scripts/build_local_edgar_metadata.py."
            )

        connection = self._connect()
        try:
            documents = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            fingerprint = fingerprint_source_index(connection)
        finally:
            connection.close()

        if recorded.get("document_count") != str(documents):
            raise ValueError(
                f"Metadata sidecar {candidate} covers {recorded.get('document_count')} documents "
                f"but the index holds {documents}. Rebuild it with "
                f"scripts/build_local_edgar_metadata.py."
            )
        if recorded.get("source_fingerprint") != fingerprint:
            raise ValueError(
                f"Metadata sidecar {candidate} was built from a different index. Rebuild it with "
                f"scripts/build_local_edgar_metadata.py."
            )

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.index_path}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _session(self) -> sqlite3.Connection:
        """Return this thread's connection, reusing it across searches.

        Reuse matters more than it looks: a fresh connection starts with an empty
        page cache, so every search would re-read the index pages it just read.
        """
        connection: sqlite3.Connection | None = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._connect()
            if self.metadata_path is not None:
                connection.execute(
                    "ATTACH DATABASE ? AS " + SIDECAR_ALIAS,
                    (f"file:{self.metadata_path}?mode=ro",),
                )
            self._local.connection = connection
        return connection

    def close(self) -> None:
        connection: sqlite3.Connection | None = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def _validate_index(self) -> frozenset[str]:
        connection = self._connect()
        try:
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
            }
            required = {"documents", "documents_fts"}
            missing = sorted(required - tables)
            if missing:
                raise ValueError("Local EDGAR index is missing required tables: " + ", ".join(missing))
            columns = frozenset(str(row[1]) for row in connection.execute("PRAGMA table_info(documents)"))
        finally:
            connection.close()

        # Checked here rather than left to the first search, which would surface
        # a missing column as a failed rollout instead of a failed startup.
        absent = sorted(column for column in METADATA_COLUMNS if column not in columns)
        if absent:
            raise ValueError(
                f"Local EDGAR index {self.index_path} table documents is missing required columns: {', '.join(absent)}"
            )
        return columns

    def _require_usable_metadata_source(self) -> None:
        if self.metadata_path is not None or "body" not in self._document_columns:
            return
        if self.index_path.stat().st_size < SLOW_METADATA_LIMIT_BYTES:
            return
        raise ValueError(
            f"Local EDGAR index {self.index_path} stores filing text in documents and has no "
            f"metadata sidecar, so every search would read filing text to return metadata. "
            f"Build one with 'python resources_servers/finance_sec_search/scripts/"
            f"build_local_edgar_metadata.py --index {self.index_path}', or point "
            f"local_edgar_metadata_path at it if it is stored elsewhere."
        )

    def search(
        self,
        search_query: str,
        start_date: str = DEFAULT_START_DATE,
        end_date: str = MAX_END_DATE,
        top_n_results: int = PAGE_SIZE,
        page: int = 1,
        form_types: Optional[list[str]] = None,
        ciks: Optional[list[str]] = None,
    ) -> list[dict[str, Any]]:
        started = time.perf_counter()
        request = normalize_request(
            search_query,
            start_date,
            end_date,
            top_n_results,
            page,
            form_types,
            ciks,
            max_end_date=self.max_end_date,
        )
        match_all = request.search_query.strip() == "*"
        results = self._execute(request, match_all=match_all)
        filter_browse_fallback = not results and not match_all and bool(request.ciks)
        if filter_browse_fallback:
            results = self._execute(request, match_all=True)
        self._assert_invariants(results, request)
        self._record_metrics(
            request,
            result_count=len(results),
            latency_ms=(time.perf_counter() - started) * 1000,
            filter_browse_fallback=filter_browse_fallback,
        )
        return results

    def _execute(
        self,
        request: LocalEdgarRequest,
        *,
        match_all: bool,
    ) -> list[dict[str, Any]]:
        conditions = ["d.filing_date >= ?", "d.filing_date <= ?"]
        parameters: list[Any] = [request.start_date, request.end_date]
        if not match_all:
            conditions.insert(0, "documents_fts MATCH ?")
            parameters.insert(0, translate_query(request.search_query))
        if request.form_types:
            conditions.append(f"d.form_type IN ({','.join('?' for _ in request.form_types)})")
            parameters.extend(request.form_types)
        if request.ciks:
            conditions.append(f"d.cik IN ({','.join('?' for _ in request.ciks)})")
            parameters.extend(request.ciks)
        parameters.extend([request.top_n_results, (request.page - 1) * PAGE_SIZE])

        # The metadata source mirrors the index's column names, so switching it
        # leaves the filters, ordering and paging below unchanged and the results
        # identical.
        table = SIDECAR_TABLE if self.metadata_path is not None else "documents"
        if match_all:
            # Browsing has no full-text term to rank, so the filter columns are
            # the only way in and their index is what makes it quick.
            source = f"{table} AS d"
        else:
            # NOT INDEXED forces the full-text match to drive the join and the
            # metadata to be fetched by rowid. Left to its own judgement the
            # planner sometimes inverts this, scanning every filing that matches
            # the form and date filters and probing the full-text index once per
            # row, which costs hundreds of times more.
            source = f"documents_fts JOIN {table} AS d NOT INDEXED ON d.id = documents_fts.rowid"
        ordering = "d.filing_date DESC, d.url" if match_all else "bm25(documents_fts), d.filing_date DESC, d.url"
        statement = f"""
            SELECT
                d.accession_number AS accessionNo,
                d.cik,
                d.company_name AS companyNameLong,
                NULLIF(d.ticker, '') AS ticker,
                d.description,
                d.form_type AS formType,
                d.document_type AS type,
                d.url AS filingUrl,
                d.filing_date AS filedAt
            FROM {source}
            WHERE {" AND ".join(conditions)}
            ORDER BY {ordering}
            LIMIT ? OFFSET ?
        """
        connection = self._session()
        return [dict(row) for row in connection.execute(statement, parameters)]

    async def search_async(self, **arguments: Any) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.search, **arguments)

    @property
    def supports_dump_paths(self) -> bool:
        return all(column in self._document_columns for column in DUMP_PATH_COLUMNS)

    def dump_paths_for_urls(self, urls: Iterable[str]) -> dict[str, str]:
        """Map SEC URLs to where the index recorded each document on disk.

        Returned keys are canonical document keys, so callers look up with
        canonical_url_key() rather than the URL itself.
        """
        if not self.supports_dump_paths:
            return {}
        keys = sorted({key for key in (canonical_url_key(url) for url in urls) if key})
        if not keys:
            return {}

        connection = self._session()
        resolved: dict[str, str] = {}
        for start in range(0, len(keys), DUMP_PATH_CHUNK_SIZE):
            chunk = keys[start : start + DUMP_PATH_CHUNK_SIZE]
            placeholders = ",".join("?" for _ in chunk)
            rows = connection.execute(
                f"SELECT canonical_url_key, source_path FROM documents WHERE canonical_url_key IN ({placeholders})",
                chunk,
            )
            for row in rows:
                relative = _relative_source_path(str(row["source_path"]))
                if relative:
                    resolved[str(row["canonical_url_key"])] = relative
        return resolved

    async def dump_paths_for_urls_async(self, urls: Iterable[str]) -> dict[str, str]:
        return await asyncio.to_thread(self.dump_paths_for_urls, list(urls))

    @staticmethod
    def _assert_invariants(
        results: list[dict[str, Any]],
        request: LocalEdgarRequest,
    ) -> None:
        forms = set(request.form_types or ())
        ciks = set(request.ciks or ())
        for result in results:
            if not request.start_date <= result["filedAt"] <= request.end_date:
                raise RuntimeError("Local search returned a filing outside the date range")
            if forms and result["formType"] not in forms:
                raise RuntimeError("Local search returned an unrequested form type")
            if ciks and str(int(result["cik"])) not in ciks:
                raise RuntimeError("Local search returned an unrequested CIK")

    def _record_metrics(
        self,
        request: LocalEdgarRequest,
        *,
        result_count: int,
        latency_ms: float,
        filter_browse_fallback: bool,
    ) -> None:
        if self.metrics_path is None:
            return
        record = {
            "search_query": request.search_query,
            "form_types": request.form_types,
            "ciks": request.ciks,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "page": request.page,
            "top_n_results": request.top_n_results,
            "result_count": result_count,
            "latency_ms": latency_ms,
            "filter_browse_fallback": filter_browse_fallback,
            "completed_at_unix_seconds": time.time(),
        }
        with self._metrics_lock, self.metrics_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
