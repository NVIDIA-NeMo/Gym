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
"""Host-side shared inference allowance for one frozen benchmark trial.

The adapter releases the prompt only after setup and observation are ready.
All provider requests, including retries and delegated work, pass through one
instance. Callers must use the yielded payload and disable transport retries.
This object does not infer session identity from agent-controlled HTTP headers.
"""

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from copy import deepcopy
from typing import Literal

from environments.skill_trigger.scoring import EvidenceError, Request


class BudgetReached(Exception):
    """A valid model-behavior boundary, not a provider failure to retry."""

    def __init__(self, outcome: Literal["timeout", "routing_horizon"]):
        self.outcome = outcome
        super().__init__(outcome)


class TrialBudget:
    """Enforce three shared requests, 120 seconds, and 8,192 output tokens.

    Register sessions only after the adapter proves their lineage. A forwarded
    request remains counted if its connection fails or its coroutine is cancelled.
    Rate-limit admission may happen before ``forward``; the deadline is checked
    again here, immediately before the transport is invoked.
    """

    def __init__(
        self, *, model: str, root_session_id: str, preserve_payload: bool, clock: Callable[[], float] = time.monotonic
    ):
        if not model or not root_session_id:
            raise ValueError("A trial requires a model and root session")
        self.model = model
        self.preserve_payload = preserve_payload
        self.sessions = {root_session_id}
        self.requests: list[Request] = []
        self._clock = clock
        self._released_at: float | None = None
        self._lock = asyncio.Lock()
        self._active = 0
        self._closed = False
        self._settled = asyncio.Event()
        self._settled.set()

    def release(self) -> None:
        """Anchor the deadline to prompt release, never to the first request."""
        if self._released_at is not None or self._closed:
            raise EvidenceError("Prompt release must happen exactly once on an open trial")
        self._released_at = self._clock()

    def add_session(self, *, session_id: str, parent_id: str) -> None:
        """Register an adapter-verified child with this same shared allowance."""
        if self._closed or parent_id not in self.sessions or not session_id or session_id in self.sessions:
            raise EvidenceError("Cannot register this child session with the trial")
        self.sessions.add(session_id)

    def elapsed(self) -> float:
        if self._released_at is None:
            raise EvidenceError("Inference attempted before prompt release")
        return self._clock() - self._released_at

    def close(self) -> None:
        """Prevent new requests; the adapter must also stop active agent work."""
        self._closed = True

    async def wait_settled(self) -> None:
        """Wait until every admitted transport call has exited its context."""
        await self._settled.wait()

    def _payload(self, endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        if endpoint not in {"/responses", "/chat/completions"}:
            raise EvidenceError("Only Responses and Chat Completions are allowed")
        if payload.get("model") != self.model:
            raise EvidenceError("Model substitution is forbidden")
        result = deepcopy(payload)
        if self.preserve_payload:
            # The original Pi SDK sets its own API-specific reasoning fields.
            # Validate its cap without translating its request or instructions.
            caps = [
                result[key] for key in ("max_output_tokens", "max_tokens", "max_completion_tokens") if key in result
            ]
            if not caps or any(type(cap) is not int or cap != 8192 for cap in caps):
                raise EvidenceError("Pi request differs from the frozen output-token cap")
        elif endpoint == "/responses":
            reasoning = result.setdefault("reasoning", {})
            if not isinstance(reasoning, dict):
                raise EvidenceError("Responses reasoning configuration must be an object")
            reasoning["effort"] = "medium"
            result["max_output_tokens"] = 8192
            result["store"] = False
        else:
            result.pop("max_completion_tokens", None)
            result["max_tokens"] = 8192
            result["reasoning_effort"] = "medium"
        return result

    @asynccontextmanager
    async def forward(
        self, *, session_id: str, endpoint: str, payload: dict[str, object]
    ) -> AsyncIterator[dict[str, object]]:
        """Wrap one transport attempt; preserve its payload and request record.

        Deadline expiration cancels the transport context, including a stream
        that keeps producing chunks. The caller closes the upstream connection
        in its own ``async with``/``finally``. Provider errors propagate unchanged.
        """
        body = self._payload(endpoint, payload)
        async with self._lock:
            if self._closed or session_id not in self.sessions:
                raise EvidenceError("Request is for a stopped trial or an unproven session")
            elapsed = self.elapsed()
            if elapsed >= 120:
                raise BudgetReached("timeout")
            if len(self.requests) >= 3:
                raise BudgetReached("routing_horizon")
            self.requests.append(
                Request(session_id=session_id, model=self.model, sent_at=elapsed, max_output_tokens=8192)
            )
            self._active += 1
            self._settled.clear()
        deadline = asyncio.timeout(120 - elapsed)
        try:
            try:
                async with deadline:
                    yield body
            except TimeoutError:
                # A transport's own timeout is a provider error unless the
                # absolute trial timer actually fired.
                if deadline.expired():
                    raise BudgetReached("timeout") from None
                raise
        finally:
            self._active -= 1
            if self._active == 0:
                self._settled.set()
