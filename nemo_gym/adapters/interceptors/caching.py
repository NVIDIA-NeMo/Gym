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
from __future__ import annotations

import logging

from nemo_gym.adapters.cache.disk_cache import DiskCache
from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    RequestToResponseInterceptor,
    ResponseInterceptor,
)


logger = logging.getLogger(__name__)


class Interceptor(RequestToResponseInterceptor, ResponseInterceptor):
    stream_safe = False

    def __init__(self, cache_dir: str, *, bypass: bool = False) -> None:
        self._bypass = bypass
        self._cache = DiskCache(cache_dir)

    async def intercept_request(
        self,
        req: AdapterRequest,
    ) -> AdapterRequest | AdapterResponse:
        if self._bypass:
            return req
        session_prefix = req.ctx.extra.get("session_id", "")
        key = DiskCache.cache_key(req.body, session_prefix=session_prefix)
        hit = await self._cache.get(key)
        if hit is not None:
            logger.debug("cache hit key=%s", key[:16])
            req.ctx.extra["cache_hit"] = True
            return AdapterResponse(
                status_code=200,
                headers={},
                body=hit,
                latency_ms=0.0,
                ctx=req.ctx,
            )
        req.ctx.extra["cache_key"] = key
        return req

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        key = resp.ctx.extra.get("cache_key")
        if key is None or not resp.ok:
            return resp
        if isinstance(resp.body, dict):
            await self._cache.set(key, resp.body)
        resp.ctx.extra.pop("cache_key", None)
        return resp
