# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Thread-safe local Tantivy retrieval for LongFact SAFE."""

from __future__ import annotations

import asyncio
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class TantivyProviderError(RuntimeError):
    """Raised when the local index or corpus cannot serve a search request."""


@dataclass(frozen=True)
class TantivyEvidence:
    title: str
    url: str
    snippet: str
    search_score: float


class TantivyProvider:
    """Query Tantivy documents and hydrate snippets from SQLite."""

    def __init__(
        self,
        *,
        index_dir: str,
        corpus_db_path: str,
        max_results: int,
        candidate_pool: int,
        max_result_chars: int,
    ) -> None:
        try:
            import tantivy
        except ImportError as exc:  # pragma: no cover - depends on resource venv.
            raise TantivyProviderError("tantivy is required for search_backend=tantivy") from exc
        if not Path(index_dir).is_dir():
            raise TantivyProviderError(f"Tantivy index directory is not readable: {index_dir}")
        if not Path(corpus_db_path).is_file():
            raise TantivyProviderError(f"Tantivy corpus database is not readable: {corpus_db_path}")
        self._max_results = max_results
        self._candidate_pool = max(max_results, candidate_pool)
        self._max_result_chars = max_result_chars
        self._index = tantivy.Index.open(index_dir)
        self._index.reload()
        self._searcher = self._index.searcher()
        self._search_lock = threading.Lock()
        self._db_lock = threading.Lock()
        self._db = sqlite3.connect(
            f"file:{corpus_db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )
        self._db.execute("PRAGMA query_only=ON")

    async def search(self, query: str) -> list[TantivyEvidence]:
        return await asyncio.to_thread(self._search, query)

    @staticmethod
    def _value(document: dict[str, Any], key: str) -> str:
        value = document.get(key)
        if isinstance(value, list):
            return str(value[0]) if value else ""
        return str(value) if value is not None else ""

    def _search(self, query: str) -> list[TantivyEvidence]:
        query = query.strip()
        if not query:
            return []
        try:
            parsed, _ = self._index.parse_query_lenient(query, ["text", "url"])
            with self._search_lock:
                result = self._searcher.search(parsed, self._candidate_pool)
                documents = [(float(score), self._searcher.doc(address).to_dict()) for score, address in result.hits]
            docids = [self._value(document, "docid") for _, document in documents]
            rows = self._fetch_rows(docids)
        except Exception as exc:
            raise TantivyProviderError(f"Tantivy search failed for {query!r}: {exc}") from exc

        evidence: list[TantivyEvidence] = []
        for score, document in documents:
            row = rows.get(self._value(document, "docid"))
            if row is None:
                continue
            url, text = row
            evidence.append(
                TantivyEvidence(
                    title=self._value(document, "title") or url,
                    url=self._value(document, "url") or url,
                    snippet=" ".join(text.split()),
                    search_score=score,
                )
            )
            if len(evidence) == self._max_results:
                break
        return evidence

    def _fetch_rows(self, docids: list[str]) -> dict[str, tuple[str, str]]:
        docids = [docid for docid in docids if docid]
        if not docids:
            return {}
        placeholders = ",".join("?" for _ in docids)
        with self._db_lock:
            rows = self._db.execute(
                f"SELECT docid, url, substr(text, 1, ?) FROM corpus WHERE docid IN ({placeholders})",
                [self._max_result_chars, *docids],
            ).fetchall()
        return {str(docid): (str(url), str(text)) for docid, url, text in rows}
