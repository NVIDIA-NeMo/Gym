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
"""A semaphore that times its own acquisition, under ``GymSpanGroup.CONCURRENCY``.

Gym gates concurrency at ~60 call sites across the rollout driver, agent servers, model
servers, resources-server judges, and sandbox providers — all `asyncio.Semaphore`. None
of them distinguish "waiting for a slot" from "doing the work", so a run that is actually
bottlenecked on concurrency looks identical, in every existing metric, to one that is not.

A subclass, rather than a second context manager callers must remember to wrap
`async with sem:` in, keeps every one of those `async with` lines untouched. Only the
**construction** site changes, from `asyncio.Semaphore(n)` to
`TimedSemaphore(n, site="...")` — the smallest diff for full coverage, and forgetting to
opt in is a visibly different constructor call, not a silently-skipped wrapper.
"""

import asyncio
import time
from typing import Optional

from nemo_gym.telemetry._fallbacks import is_span_group_enabled


class TimedSemaphore(asyncio.Semaphore):
    """``asyncio.Semaphore`` that records ``gym.concurrency.queue_wait_duration_ms``.

    Args:
        value: Initial semaphore value, same as ``asyncio.Semaphore``.
        site: Stable label identifying this semaphore, e.g. ``"rollout_driver"``,
            ``"model.openai_model"``, ``"resources.math_with_judge"``,
            ``"sandbox.daytona"``. Required as a keyword so every call site names itself.
    """

    def __init__(self, value: int = 1, *, site: str) -> None:
        super().__init__(value)
        self._gym_site = site

    async def acquire(self) -> bool:
        # Gate first, with nothing above it — kb/knowledge/conventions/hot-path-overhead.md.
        # A disabled group costs one frozenset membership test, same as every other site.
        from nemo_gym.telemetry.span_groups import GymSpanGroup

        if not is_span_group_enabled(GymSpanGroup.CONCURRENCY):
            return await super().acquire()

        from nemo_gym.telemetry.gym_metrics import record_queue_wait

        started = time.perf_counter()
        result = await super().acquire()
        record_queue_wait((time.perf_counter() - started) * 1000.0, site=self._gym_site)
        return result


def timed_semaphore_or_null(value: Optional[int], *, site: str):
    """``TimedSemaphore(value, site=site)`` if ``value`` is set, else ``contextlib.nullcontext()``.

    Convenience for the common ``Semaphore(mc) if mc is not None else nullcontext()``
    pattern scattered across resources-server judges — there is nothing to time when
    there is no configured concurrency limit.
    """
    from contextlib import nullcontext

    if value is None:
        return nullcontext()
    return TimedSemaphore(value, site=site)
