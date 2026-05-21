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

import json
import logging

from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    RequestInterceptor,
    ResponseInterceptor,
)


logger = logging.getLogger(__name__)

_MAX = 512


def _trunc_preview(obj: object) -> str:
    if isinstance(obj, bytes):
        text = obj.decode(errors="replace")
    elif isinstance(obj, dict):
        text = json.dumps(obj, default=str, ensure_ascii=False)
    else:
        text = str(obj)
    if len(text) <= _MAX:
        return text
    return text[:_MAX] + "..."


class Interceptor(RequestInterceptor, ResponseInterceptor):
    best_effort = True

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        keys = list(req.body.keys()) if isinstance(req.body, dict) else []
        logger.info(
            "request %s %s body_keys=%s body_preview=%s",
            req.method,
            req.path,
            keys,
            _trunc_preview(req.body),
        )
        return req

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        logger.info(
            "response status=%d latency_ms=%.2f body_preview=%s",
            resp.status_code,
            resp.latency_ms,
            _trunc_preview(resp.body),
        )
        return resp
