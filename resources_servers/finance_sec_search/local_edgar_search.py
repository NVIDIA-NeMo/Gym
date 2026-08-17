"""SQLite-backed implementation of the agent-facing EDGAR search contract."""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional


DEFAULT_START_DATE = "1900-01-01"
MAX_END_DATE = "2025-04-07"
PAGE_SIZE = 100
TOKEN_RE = re.compile(r'"(?:[^"]|"")*"|\S+')
BAREWORD_RE = re.compile(r"^[A-Za-z0-9_]+$")


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
    ):
        self.index_path = Path(index_path)
        if not self.index_path.is_file():
            raise FileNotFoundError(f"Local EDGAR index not found: {self.index_path}")
        self._validate_index()
        self.max_end_date = _date_value("max_end_date", max_end_date)
        self.metrics_path: Path | None = None
        self._metrics_lock = threading.Lock()
        if metrics_dir:
            destination = Path(metrics_dir)
            destination.mkdir(parents=True, exist_ok=True)
            identity = os.environ.get("SLURM_JOB_ID") or str(os.getpid())
            self.metrics_path = destination / f"search-{identity}-{os.getpid()}.jsonl"

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.index_path}?mode=ro&immutable=1"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        return connection

    def _validate_index(self) -> None:
        connection = self._connect()
        try:
            tables = {
                str(row[0])
                for row in connection.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
            }
        finally:
            connection.close()
        required = {"documents", "documents_fts"}
        missing = sorted(required - tables)
        if missing:
            raise ValueError("Local EDGAR index is missing required tables: " + ", ".join(missing))

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

        source = "documents AS d" if match_all else "documents_fts JOIN documents AS d ON d.id = documents_fts.rowid"
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
        connection = self._connect()
        try:
            results = [dict(row) for row in connection.execute(statement, parameters)]
        finally:
            connection.close()
        return results

    async def search_async(self, **arguments: Any) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.search, **arguments)

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
