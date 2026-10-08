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
"""Live ``edgar_search`` backend, answered by sec-api.io full-text search.

Reproduces the request body and response slicing of
``finance_agent.tools.EDGARSearch``. finance_agent_v2 calls that class directly;
finance_sec_search cannot import it without also taking on Vals' multi-provider
model SDK stack, so the call is restated here. The two are pinned together by a
conformance test that runs in the finance_agent_v2 venv, where upstream is
installed, and fails when either side moves.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable, Optional

import aiohttp

from resources_servers.sec_local_index.local_edgar_search import LocalEdgarRequest


logger = logging.getLogger(__name__)

SEC_API_URL = "https://api.sec-api.io/full-text-search"
RETRY_STATUSES = frozenset({429, 503})


def build_payload(request: LocalEdgarRequest) -> dict[str, Any]:
    """Render a normalized request as a sec-api.io full-text-search body."""
    payload: dict[str, Any] = {
        "query": request.search_query,
        "startDate": request.start_date,
        "endDate": request.end_date,
    }
    if request.page:
        payload["page"] = request.page
    if request.form_types:
        payload["formTypes"] = list(request.form_types)
    if request.ciks:
        payload["ciks"] = list(request.ciks)
    return payload


def select_filings(body: Any, top_n_results: int) -> list[dict[str, Any]]:
    """Take the page of filings sec-api.io returned, trimmed to the asked-for count."""
    filings = body.get("filings", []) if isinstance(body, dict) else []
    return list(filings[:top_n_results])


class LiveEdgarSearch:
    """Answers normalized requests from sec-api.io."""

    def __init__(
        self,
        api_key: str,
        *,
        session_provider: Optional[Callable[[], Awaitable[aiohttp.ClientSession]]] = None,
        max_retries: int = 5,
        request_timeout: float = 60.0,
        url: str = SEC_API_URL,
    ) -> None:
        if not api_key:
            raise ValueError("A sec-api.io key is required to run edgar_search in live mode.")
        self._api_key = api_key
        self._session_provider = session_provider
        self._max_retries = max_retries
        self._request_timeout = request_timeout
        self._url = url

    async def execute_async(self, request: LocalEdgarRequest) -> list[dict[str, Any]]:
        payload = build_payload(request)
        headers = {"Content-Type": "application/json", "Authorization": self._api_key}
        timeout = aiohttp.ClientTimeout(total=self._request_timeout)

        for attempt in range(self._max_retries):
            try:
                body = await self._post(payload, headers, timeout)
            except aiohttp.ClientResponseError as error:
                if error.status not in RETRY_STATUSES or attempt == self._max_retries - 1:
                    raise
                logger.warning("sec-api.io %d on attempt %d/%d", error.status, attempt + 1, self._max_retries)
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == self._max_retries - 1:
                    raise
                logger.warning("sec-api.io request failed on attempt %d/%d", attempt + 1, self._max_retries)
            else:
                return select_filings(body, request.top_n_results)
            await asyncio.sleep(2**attempt)

        raise RuntimeError("unreachable: the final attempt either returns or raises")

    async def _post(self, payload: dict[str, Any], headers: dict[str, str], timeout: aiohttp.ClientTimeout) -> Any:
        if self._session_provider is not None:
            session = await self._session_provider()
            async with session.post(self._url, json=payload, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                return await response.json()

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(self._url, json=payload, headers=headers) as response:
                response.raise_for_status()
                return await response.json()
