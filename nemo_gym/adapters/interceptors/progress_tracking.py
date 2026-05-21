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

import aiohttp

from nemo_gym.adapters.types import AdapterResponse, ResponseInterceptor
from nemo_gym.server_utils import request as global_request


logger = logging.getLogger(__name__)


class Interceptor(ResponseInterceptor):
    best_effort = True

    def __init__(
        self,
        *,
        webhook_url: str | None = None,
        every: int = 10,
    ) -> None:
        self._webhook_url = webhook_url
        self._every = max(every, 1)
        self._completed = 0
        self._lock = asyncio.Lock()

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        async with self._lock:
            self._completed += 1
            n = self._completed
        if n % self._every == 0:
            logger.info("progress completed=%d", n)
            if self._webhook_url:
                try:
                    webhook_resp = await global_request(
                        method="POST",
                        url=self._webhook_url,
                        json={"completed": n},
                        timeout=aiohttp.ClientTimeout(total=30),
                    )
                    async with webhook_resp:
                        webhook_resp.raise_for_status()
                except Exception:
                    logger.warning("progress webhook failed", exc_info=True)
        return resp
