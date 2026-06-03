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
import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from nemo_gym.adapters.types import AdapterRequest, RequestInterceptor


logger = logging.getLogger(__name__)

_WARN_THRESHOLD = 0.80
_URGENT_THRESHOLD = 0.95
_STALE_SESSION_SEC = 900.0
_GC_INTERVAL_SEC = 300.0


def _session_key_from_body(body: dict[str, Any]) -> str:
    """Fallback: derive a session key from the first non-system message."""
    messages = body.get("messages") or []
    for msg in messages:
        if msg.get("role") == "system":
            continue
        content = msg.get("content", "")
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if content:
            return hashlib.sha256(content.encode()).hexdigest()[:8]
    return "unknown"


@dataclass
class _Session:
    count: int = 0
    last_time: float = field(default_factory=time.monotonic)


class Interceptor(RequestInterceptor):
    def __init__(
        self,
        *,
        every: int = 1,
        max_turns: int | None = None,
    ) -> None:
        self._every = max(every, 1)
        self._max = max_turns
        self._sessions: dict[str, _Session] = {}
        self._lock = asyncio.Lock()
        self._last_gc = time.monotonic()

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        key = req.ctx.extra.get("session_id")
        if not key:
            key = _session_key_from_body(req.body)
            logger.warning("no session_id in context — falling back to body-hash key %s", key)

        async with self._lock:
            now = time.monotonic()
            if now - self._last_gc > _GC_INTERVAL_SEC:
                self._gc(now)
                self._last_gc = now

            sess = self._sessions.setdefault(key, _Session())
            sess.count += 1
            n = sess.count
            dt = now - sess.last_time if sess.count > 1 else 0.0
            sess.last_time = now
            active = len(self._sessions)

        if n % self._every == 0 or n == 1:
            cap = f"/{self._max}" if self._max else ""
            elapsed = f" (+{dt:.1f}s)" if dt > 0 else ""
            logger.info(
                "task %s turn %d%s%s (%d active)",
                key,
                n,
                cap,
                elapsed,
                active,
            )

        if self._max is None:
            return req

        if n > self._max:
            logger.warning(
                "task %s REJECTED turn %d (max_turns=%d exceeded)",
                key,
                n,
                self._max,
            )
            from nemo_gym.adapters.types import GracefulError

            raise GracefulError(
                f"Turn budget exhausted: {n}/{self._max} turns used. "
                f"The evaluation framework has terminated this agent session."
            )

        remaining = self._max - n
        messages = req.body.get("messages")
        if not isinstance(messages, list):
            return req

        ratio = n / self._max
        if ratio >= _URGENT_THRESHOLD:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"[SYSTEM] URGENT: Turn {n}/{self._max} — only {remaining} turn(s) left. "
                        f"You MUST provide your final answer NOW. Do not start new work."
                    ),
                }
            )
        elif ratio >= _WARN_THRESHOLD:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        f"[SYSTEM] Turn {n}/{self._max} — {remaining} turns remaining. "
                        f"Begin wrapping up: finish current work and prepare your final answer."
                    ),
                }
            )

        return req

    def _gc(self, now: float) -> None:
        """Remove sessions idle longer than ``_STALE_SESSION_SEC``."""
        stale = [k for k, s in self._sessions.items() if now - s.last_time > _STALE_SESSION_SEC]
        for k in stale:
            del self._sessions[k]
        if stale:
            logger.debug("turn_counter GC: removed %d stale sessions, %d remaining", len(stale), len(self._sessions))
