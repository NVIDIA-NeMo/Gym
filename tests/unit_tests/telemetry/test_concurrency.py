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
"""Concurrency-limit state: limit, slots in use, waiters, and failed admissions, read at export."""

import asyncio
import gc
from typing import NamedTuple

import pytest

from nemo_gym.telemetry import concurrency, observer
from nemo_gym.telemetry import setup as telemetry_setup
from nemo_gym.telemetry.concurrency import MeteredSemaphore
from tests.unit_tests.telemetry.test_observer import collected  # noqa: F401 - fixture


pytest.importorskip("opentelemetry.sdk.metrics")

STATE_INSTRUMENTS = (
    concurrency.LIMIT_INSTRUMENT,
    concurrency.ACTIVE_INSTRUMENT,
    concurrency.AVAILABLE_INSTRUMENT,
    concurrency.QUEUE_DEPTH_INSTRUMENT,
)
DRIVER_ATTRIBUTES = {
    concurrency.COMPONENT_ATTRIBUTE: "rollout_driver",
    concurrency.LIMIT_NAME_ATTRIBUTE: "rollouts",
}
MODEL_ATTRIBUTES = {
    concurrency.COMPONENT_ATTRIBUTE: "model_server",
    concurrency.LIMIT_NAME_ATTRIBUTE: "upstream_requests",
    "nemo.gym.server.name": "policy_model",
}


@pytest.fixture(autouse=True)
def _fresh_registry():
    concurrency._reset_for_testing()
    yield
    concurrency._reset_for_testing()


def _driver_limit(limit: int, **kwargs) -> MeteredSemaphore:
    return MeteredSemaphore(limit, component="rollout_driver", limit_name="rollouts", **kwargs)


def _model_limit(limit: int, **kwargs) -> MeteredSemaphore:
    return MeteredSemaphore(
        limit, component="model_server", limit_name="upstream_requests", server_name="policy_model", **kwargs
    )


def _values(exported: dict, name: str) -> dict[tuple, int]:
    return {tuple(sorted(point.attributes.items())): point.value for point in exported.get(name, [])}


def _key(attributes: dict) -> tuple:
    return tuple(sorted(attributes.items()))


async def _hold(semaphore: MeteredSemaphore, release: asyncio.Event) -> None:
    async with semaphore:
        await release.wait()


async def _export_while(collected, semaphore: MeteredSemaphore, holders: int, waiters: int):  # noqa: F811
    """Start ``holders + waiters`` tasks on ``semaphore``, export while they hold or wait, then let them finish."""
    async with observer.observe_event_loop("driver"):
        release = asyncio.Event()
        tasks = [asyncio.create_task(_hold(semaphore, release)) for _ in range(holders + waiters)]
        await asyncio.sleep(0.01)
        exported = collected()
        release.set()
        await asyncio.gather(*tasks)
        return exported


class AtCapacityCase(NamedTuple):
    limit: int
    holders: int
    waiters: int
    expected_active: int
    expected_queued: int


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(AtCapacityCase(4, 0, 0, 0, 0), id="idle limit reports its slots, none in use"),
        pytest.param(AtCapacityCase(4, 3, 0, 3, 0), id="under capacity has no queue"),
        pytest.param(AtCapacityCase(4, 4, 0, 4, 0), id="exactly full has no queue"),
        pytest.param(AtCapacityCase(4, 4, 3, 4, 3), id="full with waiters reports the queue"),
        pytest.param(AtCapacityCase(1, 1, 5, 1, 5), id="limit of one with a long queue"),
    ],
)
def test_the_exported_state_matches_the_slots_in_use_and_the_waiters(collected, case):  # noqa: F811
    semaphore = _driver_limit(case.limit)

    exported = asyncio.run(_export_while(collected, semaphore, case.holders, case.waiters))

    key = _key(DRIVER_ATTRIBUTES)
    assert _values(exported, concurrency.LIMIT_INSTRUMENT) == {key: case.limit}
    assert _values(exported, concurrency.ACTIVE_INSTRUMENT) == {key: case.expected_active}
    assert _values(exported, concurrency.AVAILABLE_INSTRUMENT) == {key: case.limit - case.expected_active}
    assert _values(exported, concurrency.QUEUE_DEPTH_INSTRUMENT) == {key: case.expected_queued}
    assert (semaphore.active, semaphore.queued) == (0, 0), "every slot and wait is returned once the work finishes"


class AdmissionCase(NamedTuple):
    admission_timeout_s: float | None
    cancel_waiter: bool
    expected_error: type[BaseException] | None
    expected_timeouts: int | None
    expected_cancelled: int


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(
            AdmissionCase(None, False, None, None, 0), id="admitted wait without a timeout reports no timeouts"
        ),
        pytest.param(AdmissionCase(5.0, False, None, 0, 0), id="admitted wait with a timeout counts zero timeouts"),
        pytest.param(AdmissionCase(0.01, False, TimeoutError, 1, 0), id="timed out wait counts one timeout only"),
        pytest.param(AdmissionCase(None, True, asyncio.CancelledError, None, 1), id="cancelled wait counts once"),
        pytest.param(
            AdmissionCase(5.0, True, asyncio.CancelledError, 0, 1), id="cancelled wait is not counted as a timeout"
        ),
    ],
)
def test_failed_admissions_count_once_by_outcome(collected, case):  # noqa: F811
    semaphore = _model_limit(1, admission_timeout_s=case.admission_timeout_s)

    async def body():
        async with observer.observe_event_loop("server"):
            release = asyncio.Event()
            holder = asyncio.create_task(_hold(semaphore, release))
            await asyncio.sleep(0)
            waiter = asyncio.create_task(semaphore.acquire())
            await asyncio.sleep(0.05 if case.expected_error is TimeoutError else 0)
            if case.cancel_waiter:
                waiter.cancel()
            release.set()
            outcome = (await asyncio.gather(waiter, return_exceptions=True))[0]
            if outcome is True:
                semaphore.release()
            await holder
            return outcome, collected()

    outcome, exported = asyncio.run(body())

    if case.expected_error is None:
        assert outcome is True
    else:
        assert isinstance(outcome, case.expected_error)
    key = _key(MODEL_ATTRIBUTES)
    expected_timeouts = {} if case.expected_timeouts is None else {key: case.expected_timeouts}
    assert _values(exported, concurrency.ADMISSION_TIMEOUTS_INSTRUMENT) == expected_timeouts
    assert _values(exported, concurrency.ADMISSION_CANCELLED_INSTRUMENT) == {key: case.expected_cancelled}
    assert (semaphore.active, semaphore.queued) == (0, 0)
    assert not semaphore.locked(), "a failed wait never keeps a slot"


def test_limits_with_the_same_attributes_are_summed_and_others_kept_apart(collected):  # noqa: F811
    busy, idle_same_attributes, idle_other_server = _driver_limit(2), _driver_limit(3), _model_limit(5)

    exported = asyncio.run(_export_while(collected, busy, holders=2, waiters=1))

    assert (idle_same_attributes.active, idle_other_server.active) == (0, 0)
    assert _values(exported, concurrency.LIMIT_INSTRUMENT) == {_key(DRIVER_ATTRIBUTES): 5, _key(MODEL_ATTRIBUTES): 5}
    assert _values(exported, concurrency.ACTIVE_INSTRUMENT) == {_key(DRIVER_ATTRIBUTES): 2, _key(MODEL_ATTRIBUTES): 0}
    assert _values(exported, concurrency.QUEUE_DEPTH_INSTRUMENT) == {
        _key(DRIVER_ATTRIBUTES): 1,
        _key(MODEL_ATTRIBUTES): 0,
    }


def test_a_garbage_collected_limit_stops_reporting_state_but_keeps_its_totals(collected):  # noqa: F811
    semaphore = _model_limit(1)
    del semaphore
    gc.collect()

    async def body():
        async with observer.observe_event_loop("server"):
            return collected()

    exported = asyncio.run(body())

    for name in STATE_INSTRUMENTS:
        assert name not in exported, f"{name} must not report a limit that no longer exists"
    assert _values(exported, concurrency.ADMISSION_CANCELLED_INSTRUMENT) == {_key(MODEL_ATTRIBUTES): 0}


def test_nothing_is_exported_while_metrics_are_off(collected, monkeypatch):  # noqa: F811
    monkeypatch.setattr(telemetry_setup, "_METRICS_EXPORTING", False)
    semaphore = _driver_limit(2)

    exported = asyncio.run(_export_while(collected, semaphore, holders=2, waiters=1))

    for name in (*STATE_INSTRUMENTS, concurrency.ADMISSION_CANCELLED_INSTRUMENT):
        assert name not in exported


def test_a_component_without_a_limit_reports_no_series(collected):  # noqa: F811
    async def body():
        async with observer.observe_event_loop("server"):
            return collected()

    exported = asyncio.run(body())

    for name in (*STATE_INSTRUMENTS, concurrency.ADMISSION_TIMEOUTS_INSTRUMENT):
        assert name not in exported, f"{name} must be missing, not zero, when no limit is configured"


@pytest.mark.parametrize(
    ("limit", "admission_timeout_s", "message"),
    [
        pytest.param(0, None, "at least one slot", id="zero slots"),
        pytest.param(2, 0.0, "must be positive", id="zero admission timeout"),
    ],
)
def test_invalid_limits_are_rejected(limit, admission_timeout_s, message):
    with pytest.raises(ValueError, match=message):
        _driver_limit(limit, admission_timeout_s=admission_timeout_s)


def test_releasing_a_slot_that_is_not_held_raises():
    semaphore = _driver_limit(1)

    with pytest.raises(ValueError):
        semaphore.release()
    assert semaphore.active == 0


def test_a_second_release_while_a_slot_is_handed_to_a_waiter_raises():
    """asyncio hands a released slot straight to the next waiter, so the base class's bound check misses this."""
    semaphore = _driver_limit(1)

    async def body():
        await semaphore.acquire()
        waiter = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0)
        semaphore.release()
        with pytest.raises(ValueError):
            semaphore.release()
        await waiter
        state = (semaphore.active, semaphore.queued, semaphore.locked())
        semaphore.release()
        return state

    assert asyncio.run(body()) == (1, 0, True)
    assert (semaphore.active, semaphore.queued, semaphore.locked()) == (0, 0, False)


def test_a_waiter_cancelled_after_its_slot_was_handed_over_passes_it_on():
    semaphore = _driver_limit(1)

    async def body():
        await semaphore.acquire()
        first = asyncio.create_task(semaphore.acquire())
        second = asyncio.create_task(semaphore.acquire())
        await asyncio.sleep(0)
        semaphore.release()  # hands the slot to `first`, which has not resumed yet
        first.cancel()
        outcomes = await asyncio.gather(first, second, return_exceptions=True)
        state = (semaphore.active, semaphore.queued)
        semaphore.release()
        return outcomes, state

    (first, second), state = asyncio.run(body())
    assert isinstance(first, asyncio.CancelledError) and second is True
    assert state == (1, 0)
    assert (semaphore.active, semaphore.queued, semaphore.locked()) == (0, 0, False)


def test_an_error_inside_the_guarded_block_returns_the_slot():
    semaphore = _driver_limit(1)

    async def body():
        with pytest.raises(RuntimeError):
            async with semaphore:
                raise RuntimeError("tool failed")

    asyncio.run(body())
    assert (semaphore.active, semaphore.queued, semaphore.locked()) == (0, 0, False)
