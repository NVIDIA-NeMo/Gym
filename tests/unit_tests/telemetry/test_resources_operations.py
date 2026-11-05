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
"""The shared resources-operation contract: queue wait, execution time, and outcome of bounded work."""

import asyncio

import pytest

from nemo_gym.telemetry import observer  # noqa: E402
from nemo_gym.telemetry import resources_operations as ops
from tests.unit_tests.telemetry.test_observer import collected  # noqa: F401 - fixture


pytest.importorskip("opentelemetry.sdk.metrics")


def _run(collected, body):  # noqa: F811
    async def run():
        async with observer.observe_event_loop("server"):
            await body()
            return collected()

    return asyncio.run(run())


def test_queue_wait_and_execution_are_recorded_by_operation_and_executor(collected):  # noqa: F811
    semaphore = asyncio.Semaphore(1)
    snapshots = []

    async def work():
        async with ops.resources_operation(semaphore, operation="verify", executor="subprocess"):
            snapshots.append(dict(ops._metrics().active))
            await asyncio.sleep(0.05)

    async def body():
        await asyncio.gather(work(), work(), work())

    exported = _run(collected, body)

    assert snapshots == [{("verify", "subprocess"): 1}] * 3, "the limit of one runs one operation at a time"
    (queue,) = exported[ops.QUEUE_DURATION_INSTRUMENT]
    assert queue.attributes == {ops.OPERATION_ATTRIBUTE: "verify", ops.EXECUTOR_ATTRIBUTE: "subprocess"}
    assert queue.count == 3 and queue.max >= 90, "the third operation waited for the two ahead of it, in milliseconds"
    (duration,) = exported[ops.DURATION_INSTRUMENT]
    assert duration.attributes[ops.OUTCOME_ATTRIBUTE] == "ok" and duration.count == 3
    assert {point.value for point in exported[ops.ACTIVE_INSTRUMENT]} == {0}
    assert {point.value for point in exported[ops.QUEUED_INSTRUMENT]} == {0}


def test_the_block_can_report_a_timeout(collected):  # noqa: F811
    async def body():
        async with ops.resources_operation(asyncio.Semaphore(1), operation="verify", executor="subprocess") as op:
            op.outcome = "timeout"

    (duration,) = _run(collected, body)[ops.DURATION_INSTRUMENT]
    assert duration.attributes[ops.OUTCOME_ATTRIBUTE] == "timeout"


@pytest.mark.parametrize(
    ("error", "outcome"), [(RuntimeError("boom"), "error"), (asyncio.CancelledError(), "cancelled")]
)
def test_an_exception_sets_the_outcome_and_still_releases_the_limit(collected, error, outcome):  # noqa: F811
    semaphore = asyncio.Semaphore(1)

    async def body():
        with pytest.raises(type(error)):
            async with ops.resources_operation(semaphore, operation="tool", executor="thread"):
                raise error
        assert not semaphore.locked()

    (duration,) = _run(collected, body)[ops.DURATION_INSTRUMENT]
    assert duration.attributes[ops.OUTCOME_ATTRIBUTE] == outcome


def test_unknown_labels_are_folded_into_other(collected):  # noqa: F811
    async def body():
        async with ops.resources_operation(asyncio.Semaphore(1), operation="sandbox-123", executor="gpu"):
            pass

    (duration,) = _run(collected, body)[ops.DURATION_INSTRUMENT]
    assert (duration.attributes[ops.OPERATION_ATTRIBUTE], duration.attributes[ops.EXECUTOR_ATTRIBUTE]) == (
        "other",
        "other",
    )


def test_without_metrics_it_only_enters_the_limit():
    semaphore = asyncio.Semaphore(1)

    async def body():
        async with ops.resources_operation(semaphore, operation="verify", executor="inline"):
            assert semaphore.locked()

    asyncio.run(body())
    assert not semaphore.locked()
