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
    """``asyncio.Semaphore`` that records ``gym.concurrency.queue_wait_duration_ms`` plus
    the live-state gauges (``gym.concurrency.limit``/``.active``/``.available``/
    ``.queue_depth``) a "concurrency limit: 100, active: 100, queued: 75" dashboard panel
    needs -- queue-wait duration alone shows work waited, not the state that caused the
    wait.

    Args:
        value: Initial semaphore value, same as ``asyncio.Semaphore``.
        site: Stable label identifying this semaphore, e.g. ``"rollout_driver"``,
            ``"model.openai_model"``, ``"resources.math_with_judge"``,
            ``"sandbox.daytona"``. Required as a keyword so every call site names itself.
    """

    def __init__(self, value: int = 1, *, site: str) -> None:
        super().__init__(value)
        self._gym_site = site
        self._gym_limit = value
        self._gym_active = 0
        self._gym_waiting = 0
        # The limit never changes for this semaphore's lifetime, so one `.set()` here
        # (unlike active/queue_depth, which change per acquire/release) is enough -- see
        # `record_concurrency_limit`'s docstring on why a synchronous gauge does not need
        # resampling to stay correct across export ticks.
        from nemo_gym.telemetry.span_groups import GymSpanGroup

        if is_span_group_enabled(GymSpanGroup.CONCURRENCY):
            from nemo_gym.telemetry.gym_metrics import record_concurrency_limit

            record_concurrency_limit(value, site=site)
            self._record_active_and_available()

    async def acquire(self) -> bool:
        # Gate first, with nothing above it — kb/knowledge/conventions/hot-path-overhead.md.
        # A disabled group costs one frozenset membership test, same as every other site.
        from nemo_gym.telemetry.span_groups import GymSpanGroup

        if not is_span_group_enabled(GymSpanGroup.CONCURRENCY):
            return await super().acquire()

        from nemo_gym.telemetry.gym_metrics import record_concurrency_admission_cancelled, record_queue_wait

        self._gym_waiting += 1
        self._record_queue_depth()
        started = time.perf_counter()
        try:
            result = await super().acquire()
        except asyncio.CancelledError:
            # The one abnormal-admission signal Gym's semaphores actually have: no
            # reject-on-full policy exists to distinguish "rejected" from "timed out"
            # (see `record_concurrency_admission_cancelled`'s docstring), so a caller's
            # own timeout cancelling this await is the whole story here.
            record_concurrency_admission_cancelled(site=self._gym_site)
            raise
        finally:
            self._gym_waiting -= 1
            self._record_queue_depth()
        record_queue_wait((time.perf_counter() - started) * 1000.0, site=self._gym_site)
        self._gym_active += 1
        self._record_active_and_available()
        return result

    def release(self) -> None:
        super().release()
        if self._gym_active > 0:
            self._gym_active -= 1
        from nemo_gym.telemetry.span_groups import GymSpanGroup

        if is_span_group_enabled(GymSpanGroup.CONCURRENCY):
            self._record_active_and_available()

    def _record_active_and_available(self) -> None:
        from nemo_gym.telemetry.gym_metrics import record_concurrency_active, record_concurrency_available

        record_concurrency_active(self._gym_active, site=self._gym_site)
        record_concurrency_available(max(0, self._gym_limit - self._gym_active), site=self._gym_site)

    def _record_queue_depth(self) -> None:
        from nemo_gym.telemetry.gym_metrics import record_concurrency_queue_depth

        record_concurrency_queue_depth(self._gym_waiting, site=self._gym_site)


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
