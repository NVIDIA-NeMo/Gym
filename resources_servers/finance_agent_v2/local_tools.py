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
"""Upstream SEC tools answered from a local corpus instead of the network.

Subclasses rather than replacements: the name, description and parameter
schema the model sees stay whatever upstream declares, so a sample written
against the live benchmark runs unchanged. Only the fetch is swapped, which
keeps training throughput off sec-api.io and, for filings the corpus holds,
off sec.gov.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from finance_agent.tools import MAX_END_DATE, EDGARSearch

from resources_servers.finance_agent_v2.cached_tools import CachedParseHtmlPage
from resources_servers.sec_local_index.cache import ToolCache
from resources_servers.sec_local_index.edgar_search_service import EdgarSearchService
from resources_servers.sec_local_index.html_text import html_to_text
from resources_servers.sec_local_index.local_edgar_search import LocalEdgarSearch, canonical_url_key


logger = logging.getLogger(__name__)


class LocalEDGARSearch(EDGARSearch):
    """edgar_search served from a local SQLite full-text index."""

    def __init__(self, engine: LocalEdgarSearch, *, max_end_date: str = MAX_END_DATE):
        self._engine = engine
        self._service = EdgarSearchService(engine, max_end_date=max_end_date)
        self.sec_api_url = ""

    async def _execute_search(
        self,
        search_query: str,
        start_date: str = "1900-01-01",
        end_date: str = MAX_END_DATE,
        top_n_results: int = 100,
        page: int = 1,
        form_types: Any = None,
        ciks: Any = None,
    ) -> list[dict[str, Any]]:
        request = self._service.normalize(
            {
                "search_query": search_query,
                "start_date": start_date,
                "end_date": end_date,
                "top_n_results": top_n_results,
                "page": page,
                "form_types": form_types,
                "ciks": ciks,
            }
        )
        return await self._engine.execute_async(request)


class LocalParseHtmlPage(CachedParseHtmlPage):
    """parse_html_page that reads SEC filings from the downloaded corpus.

    Reads go cache, then corpus, then network. Parsed corpus text is written to
    the cache so a large filing is only parsed once; anything the corpus does
    not hold goes through the same cached fetch live mode uses.
    """

    # Reads are frequent, so the tally is logged periodically rather than per call.
    LOG_EVERY = 100

    def __init__(self, engine: LocalEdgarSearch, corpus_root: str | Path, cache: Optional[ToolCache] = None):
        super().__init__(cache if cache is not None else ToolCache(None, use_cache=False))
        self._engine = engine
        self._corpus_root = Path(corpus_root)

    def local_path_for(self, url: str) -> Optional[Path]:
        key = canonical_url_key(url)
        if key is None:
            return None
        relative = self._engine.dump_paths_for_urls([url]).get(key)
        if not relative:
            return None
        candidate = self._corpus_root / relative
        return candidate if candidate.is_file() else None

    async def _parse_html_page(self, url: str) -> str:
        cache_path = self._doc_path(url) if self._cache.enabled else None
        if cache_path is not None:
            cached = self._cache.read_text(cache_path)
            if cached is not None:
                self._record_read("cache")
                return cached

        corpus_path = self.local_path_for(url)
        if corpus_path is None:
            return await super()._parse_html_page(url)
        text = html_to_text(corpus_path.read_text(encoding="utf-8", errors="replace"))
        self._record_read("sec-corpus")
        if cache_path is not None and text:
            self._cache.write_text(cache_path, text)
        return text

    def _record_read(self, source: str) -> None:
        """Surfaces a corpus that is quietly missing most of what is asked for,
        which otherwise shows up only as a slow run."""
        super()._record_read(source)
        if sum(self.read_sources.values()) % self.LOG_EVERY == 0:
            logger.info(
                "SEC filing reads by source: %s",
                " ".join(f"{name}={count}" for name, count in sorted(self.read_sources.items())),
            )
