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
"""``TimedSemaphore`` — every `async with` call site is untouched; only construction
changes. These tests cover both the disabled (no-op-cost) and enabled (recorded) paths."""

import asyncio
from contextlib import nullcontext

import pytest

from nemo_gym.telemetry.concurrency import TimedSemaphore, timed_semaphore_or_null
from tests.unit_tests.telemetry.conftest import requires_lens


pytestmark = requires_lens


async def test_disabled_group_behaves_like_a_plain_semaphore():
    """`GymSpanGroup.CONCURRENCY` is off by default, so a normal run must see ordinary
    semaphore semantics -- no telemetry side effect required to work correctly."""
    sem = TimedSemaphore(1, site="test")
    async with sem:
        assert sem.locked()
    assert not sem.locked()


async def test_two_holders_serialize_through_the_semaphore():
    sem = TimedSemaphore(1, site="test")
    order = []

    async def holder(name):
        async with sem:
            order.append(f"{name}-enter")
            await asyncio.sleep(0)
            order.append(f"{name}-exit")

    await asyncio.gather(holder("a"), holder("b"))
    # Whichever ran first, it must fully exit before the other enters -- interleaved
    # enter/enter would mean the semaphore did not serialize them.
    assert order in (["a-enter", "a-exit", "b-enter", "b-exit"], ["b-enter", "b-exit", "a-enter", "a-exit"])


async def test_enabled_group_records_queue_wait(monkeypatch):
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    from nemo_gym.telemetry import gym_metrics
    from nemo_gym.telemetry import setup as telemetry_setup
    from nemo_gym.telemetry.span_groups import GymSpanGroup

    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])

    class _Handle:
        is_exporting = True
        meter = provider.get_meter("test")

    monkeypatch.setattr(telemetry_setup, "_TELEMETRY_HANDLE", _Handle())
    gym_metrics._reset_for_testing()
    monkeypatch.setattr(
        "nemo_gym.telemetry.concurrency.is_span_group_enabled",
        lambda group: group == GymSpanGroup.CONCURRENCY,
    )

    sem = TimedSemaphore(1, site="test_site")
    async with sem:
        pass

    data = reader.get_metrics_data()
    points = [
        p
        for rm in data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
        if m.name == "gym.concurrency.queue_wait_duration_ms"
        for p in m.data.data_points
    ]
    assert points, "queue-wait histogram must record when CONCURRENCY is enabled"
    assert dict(points[0].attributes)["nemo.gym.concurrency.site"] == "test_site"


def test_timed_semaphore_or_null_with_value_is_a_timed_semaphore():
    result = timed_semaphore_or_null(4, site="resources.example")
    assert isinstance(result, TimedSemaphore)


def test_timed_semaphore_or_null_without_value_is_nullcontext():
    result = timed_semaphore_or_null(None, site="resources.example")
    assert isinstance(result, type(nullcontext()))


@pytest.mark.parametrize("n", [1, 8])
async def test_semaphore_value_is_respected(n):
    sem = TimedSemaphore(n, site="test")
    concurrent = 0
    peak = 0

    async def holder():
        nonlocal concurrent, peak
        async with sem:
            concurrent += 1
            peak = max(peak, concurrent)
            await asyncio.sleep(0)
            concurrent -= 1

    await asyncio.gather(*(holder() for _ in range(n * 3)))
    assert peak <= n
