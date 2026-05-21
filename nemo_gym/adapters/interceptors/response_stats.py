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

import asyncio
import logging

from nemo_gym.adapters.types import AdapterResponse, ResponseInterceptor


logger = logging.getLogger(__name__)


class Interceptor(ResponseInterceptor):
    stream_safe = False
    best_effort = True

    def __init__(self, *, every: int = 100) -> None:
        self._every = max(every, 1)
        self._n = 0
        self._tokens = 0
        self._latency_ms = 0.0
        self._lock = asyncio.Lock()

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        body = resp.body
        tokens = 0
        if isinstance(body, dict):
            u = body.get("usage")
            if isinstance(u, dict):
                tokens = int(u.get("total_tokens") or 0)
        async with self._lock:
            self._n += 1
            self._tokens += tokens
            self._latency_ms += resp.latency_ms
            n = self._n
            tot_tok = self._tokens
            tot_lat = self._latency_ms
        if n % self._every == 0:
            logger.info(
                "response_stats requests=%d total_tokens=%d total_latency_ms=%.2f",
                n,
                tot_tok,
                tot_lat,
            )
        return resp
