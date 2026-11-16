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
import asyncio
import time

from nemo_gym.rollouts.rollout_collection import _DispatchReservations


async def test_first_reservation_is_written_on_the_next_loop_turn():
    batches = []
    reservations = _DispatchReservations(batches.append, interval_s=60.0)

    await asyncio.wait_for(reservations.reserve({"id": 0}), timeout=1.0)

    assert batches == [[{"id": 0}]]


async def test_reservations_within_the_interval_share_one_write():
    batches = []
    reservations = _DispatchReservations(batches.append, interval_s=0.2)
    await reservations.reserve({"id": 0})

    started = time.monotonic()
    waiters = [asyncio.create_task(reservations.reserve({"id": i})) for i in (1, 2, 3)]
    await asyncio.sleep(0)
    assert batches == [[{"id": 0}]], "a request must not start before its reservation is written"
    await asyncio.gather(*waiters)

    assert batches == [[{"id": 0}], [{"id": 1}, {"id": 2}, {"id": 3}]]
    assert time.monotonic() - started >= 0.15


async def test_reservation_after_a_quiet_interval_is_not_delayed():
    batches = []
    reservations = _DispatchReservations(batches.append, interval_s=0.05)
    await reservations.reserve({"id": 0})
    await asyncio.sleep(0.1)

    waiter = asyncio.create_task(reservations.reserve({"id": 1}))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert batches == [[{"id": 0}], [{"id": 1}]]
    await waiter


async def test_cancelled_waiter_is_not_reserved():
    batches = []
    reservations = _DispatchReservations(batches.append, interval_s=0.1)
    await reservations.reserve({"id": 0})

    cancelled = asyncio.create_task(reservations.reserve({"id": 1}))
    kept = asyncio.create_task(reservations.reserve({"id": 2}))
    await asyncio.sleep(0)
    cancelled.cancel()
    await asyncio.wait_for(kept, timeout=1.0)

    assert batches == [[{"id": 0}], [{"id": 2}]]
    assert cancelled.cancelled()


async def test_only_cancelled_waiters_skip_the_write():
    batches = []
    reservations = _DispatchReservations(batches.append, interval_s=0.05)
    await reservations.reserve({"id": 0})

    cancelled = asyncio.create_task(reservations.reserve({"id": 1}))
    await asyncio.sleep(0)
    cancelled.cancel()
    await asyncio.sleep(0.1)

    assert batches == [[{"id": 0}]]
    await reservations.reserve({"id": 2})
    assert batches == [[{"id": 0}], [{"id": 2}]]


async def test_write_failure_reaches_every_waiter_in_the_batch():
    def fail(rows):
        raise OSError("disk full")

    reservations = _DispatchReservations(fail, interval_s=0.0)
    waiters = [asyncio.create_task(reservations.reserve({"id": i})) for i in (0, 1)]

    results = await asyncio.wait_for(asyncio.gather(*waiters, return_exceptions=True), timeout=1.0)

    assert [type(result) for result in results] == [OSError, OSError]


async def test_a_batch_the_budget_ran_out_on_is_drained_and_not_written():
    batches = []
    drained = {"drained": True}
    reservations = _DispatchReservations(batches.append, interval_s=0.0, drained=lambda: drained)

    results = await asyncio.gather(reservations.reserve({"id": 0}), reservations.reserve({"id": 1}))

    assert batches == []
    assert results == [drained, drained]
    assert results[0] is not results[1] and results[0] is not drained, "each waiter gets its own result"


async def test_an_eligible_batch_is_reserved_and_returns_none():
    batches = []
    reservations = _DispatchReservations(batches.append, interval_s=0.0, drained=lambda: None)

    assert await reservations.reserve({"id": 0}) is None
    assert batches == [[{"id": 0}]]
