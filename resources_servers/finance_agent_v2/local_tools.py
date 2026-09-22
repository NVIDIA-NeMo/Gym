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
against the live benchmark runs unchanged. Only the fetch is swapped, which is
what makes training throughput independent of sec-api.io and sec.gov.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from finance_agent.tools import MAX_END_DATE, EDGARSearch, ParseHtmlPage

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


class LocalParseHtmlPage(ParseHtmlPage):
    """parse_html_page served from the downloaded filing corpus.

    Falls back to upstream's fetch for anything the corpus does not hold, so a
    non-SEC URL still resolves.
    """

    def __init__(self, engine: LocalEdgarSearch, corpus_root: str | Path):
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
        path = self.local_path_for(url)
        if path is None:
            return await super()._parse_html_page(url)
        return html_to_text(path.read_text(encoding="utf-8", errors="replace"))
