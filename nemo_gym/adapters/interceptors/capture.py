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
"""Capture interceptor — records model exchanges (with token-ids) per session.

The model-bound ``endpoint``/``logging`` interceptors forward and log; neither
persists what would be needed to rebuild a trajectory. This interceptor is meant
to run inside a *sandbox-bound* proxy that the harness starts per rollout: it
tags every model call with that rollout's ``session_id`` and writes the full
request + response to a durable :class:`CaptureStore`.

It also optionally injects request-level fields (``inject_extra_body``) so the
policy returns the extra information the trajectory needs — e.g. flags that turn
on token-id reporting, or ``chat_template_kwargs`` that keep reasoning history
intact for on-policy RL.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from nemo_gym.adapters.capture_store import CaptureStore
from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    RequestInterceptor,
    ResponseInterceptor,
)


logger = logging.getLogger(__name__)

# Stashed on the per-request context so the response phase can pair the
# captured response with the request body that produced it.
_REQUEST_KEY = "_capture_request_body"
_SESSION_KEY = "session_id"


class Interceptor(RequestInterceptor, ResponseInterceptor):
    """Record each model exchange to a session-keyed store.

    Best-effort: a capture failure logs and is swallowed by the pipeline rather
    than breaking the rollout (the agent's model call still succeeds).
    """

    best_effort = True

    def __init__(
        self,
        *,
        store_dir: str,
        session_id: str | None = None,
        inject_extra_body: dict[str, Any] | None = None,
        upstream_api_key: str | None = None,
    ) -> None:
        self._store = CaptureStore(store_dir)
        self._session_id = session_id
        self._inject_extra_body = inject_extra_body or {}
        # When set, the real upstream key is stamped onto the *forwarded* request
        # so the in-box agent only ever holds a dummy. It is applied to the
        # headers (which are NOT persisted), never to the recorded body.
        self._upstream_api_key = upstream_api_key

    def _session(self, ctx_extra: dict[str, Any]) -> str:
        return self._session_id or ctx_extra.get(_SESSION_KEY) or "session"

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        if isinstance(req.body, dict):
            for key, value in self._inject_extra_body.items():
                req.body.setdefault(key, value)
            req.ctx.extra[_REQUEST_KEY] = req.body
        if self._upstream_api_key:
            req.headers["Authorization"] = f"Bearer {self._upstream_api_key}"
        return req

    async def intercept_response(self, resp: AdapterResponse) -> AdapterResponse:
        session_id = self._session(resp.ctx.extra)
        exchange = {
            "request_id": resp.ctx.request_id,
            "session_id": session_id,
            "ts": time.time(),
            "status": resp.status_code,
            "latency_ms": resp.latency_ms,
            "request": resp.ctx.extra.get(_REQUEST_KEY),
            "response": resp.body if isinstance(resp.body, dict) else None,
        }
        self._store.record(session_id, exchange)
        return resp
