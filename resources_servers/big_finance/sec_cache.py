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
"""Optional SEC response caching beneath the pinned BigFinance tool serializers."""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit
from weakref import WeakValueDictionary

import httpx
from big_finance_harness.tools import EdgarSearchTool, FetchUrlTool, ToolError
from big_finance_harness.tools.blocked_urls import is_blocked_url
from pydantic import BaseModel, Field, ValidationError

from resources_servers.sec_local_index.cache import ToolCache


logger = logging.getLogger(__name__)


def _is_sec_url(url: str) -> bool:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    return parsed.scheme in ("http", "https") and (host == "sec.gov" or host.endswith(".sec.gov"))


class _StoredResponse(BaseModel):
    version: Literal[1]
    status_code: int = Field(ge=200, lt=300)
    url: str
    redirect_urls: list[str]
    content_type: str
    encoding: str
    body_base64: str


class SecHttpCache:
    """Cache successful SEC downloads; leave tool arguments and rendering to upstream.

    Entries are snapshots with no expiry. Use a new directory to refresh a run.
    Per-URL locks coalesce concurrent misses within this server process.
    """

    def __init__(self, *, cache_dir: str | None, use_cache: bool) -> None:
        self._store = ToolCache(cache_dir, use_cache, app_name="big_finance")
        self._locks: WeakValueDictionary[str, asyncio.Lock] = WeakValueDictionary()

    def _read(self, path: Path, url: str) -> httpx.Response | None:
        try:
            raw = self._store.read_json(path)
            if raw is None:
                return None
            saved = _StoredResponse.model_validate(raw)
            body = base64.b64decode(saved.body_base64, validate=True)
            # HTTPX has already decompressed response.content. Rebuild only the
            # headers used by upstream parsing, never stale Content-Encoding.
            response = httpx.Response(
                saved.status_code,
                content=body,
                headers={"content-type": saved.content_type},
                request=httpx.Request("GET", saved.url),
                default_encoding=saved.encoding,
            )
            response.encoding = saved.encoding
        except (ValueError, LookupError, ValidationError):
            logger.warning("Ignoring invalid SEC cache entry: %s", path)
            return None
        # Re-evaluate the current leakage denylist, including redirect targets
        # from an older download. The upstream run() still checks the input URL.
        for target in (url, saved.url, *saved.redirect_urls):
            if is_blocked_url(target):
                raise ToolError(f"fetch_url refuses blocked URL {target!r}")
        return response

    async def get(
        self,
        url: str,
        download: Callable[[], Awaitable[httpx.Response]],
        *,
        expect_json: bool = False,
    ) -> httpx.Response:
        """Read or download a response; cache failures never replace a successful download."""
        if not self._store.enabled or not _is_sec_url(url):
            return await download()
        key = ToolCache.hash_key(url)
        path = self._store.path("sec_http_v1", f"{key}.json")
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            try:
                response = await asyncio.to_thread(self._read, path, url)
            except OSError:
                logger.warning("Unable to read SEC cache entry: %s", path, exc_info=True)
                response = None
            if response is not None:
                if expect_json:
                    try:
                        if not isinstance(response.json(), dict):
                            response = None
                    except ValueError:
                        response = None
                if response is not None:
                    logger.info("SEC cache hit: %s", url)
                    return response
            response = await download()
            if not response.is_success:
                return response
            for target in (str(response.url), *(str(item.url) for item in response.history)):
                if is_blocked_url(target):
                    raise ToolError(f"fetch_url refuses blocked URL {target!r}")
            if expect_json:
                try:
                    if not isinstance(response.json(), dict):
                        return response
                except ValueError:
                    return response
            saved = _StoredResponse(
                version=1,
                status_code=response.status_code,
                url=str(response.url),
                redirect_urls=[str(item.url) for item in response.history],
                content_type=response.headers.get("content-type", ""),
                encoding=response.encoding or "utf-8",
                body_base64=base64.b64encode(response.content).decode("ascii"),
            )
            try:
                await asyncio.to_thread(self._store.write_json, path, saved.model_dump())
            except OSError:
                logger.warning("Unable to write SEC cache entry: %s", path, exc_info=True)
            logger.info("SEC cache miss: %s", url)
            return response


class CachedEdgarSearchTool(EdgarSearchTool):
    """Retain upstream recent-only search while caching SEC JSON responses."""

    def __init__(self, *, cache: SecHttpCache, user_agent: str, timeout_s: float = 20.0) -> None:
        super().__init__(user_agent=user_agent, timeout_s=timeout_s)
        self._cache = cache

    async def _get(self, url: str) -> httpx.Response:
        return await self._cache.get(url, lambda: super(CachedEdgarSearchTool, self)._get(url), expect_json=True)


class CachedFetchUrlTool(FetchUrlTool):
    """Retain upstream URL checks, extraction, retrieval, and token truncation."""

    def __init__(self, *, cache: SecHttpCache, timeout_s: float = 30.0, sec_user_agent: str | None = None) -> None:
        super().__init__(timeout_s=timeout_s, sec_user_agent=sec_user_agent)
        self._cache = cache

    async def _fetch(self, url: str) -> httpx.Response:
        # Preserve upstream's missing-SEC-User-Agent error even with a warm cache.
        if not self.sec_user_agent:
            return await super()._fetch(url)
        return await self._cache.get(url, lambda: super(CachedFetchUrlTool, self)._fetch(url))
