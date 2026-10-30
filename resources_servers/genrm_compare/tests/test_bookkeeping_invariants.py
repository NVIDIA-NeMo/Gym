# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protected bookkeeping must finish before yielding to another request."""

import asyncio
import time

import pytest
from fastapi import HTTPException

import resources_servers.genrm_compare.app as genrm
from nemo_gym.core import EpisodeId
from nemo_gym.server_runtime._checkpoint.control import CheckpointRequest
from nemo_gym.server_runtime._checkpoint.resources import ResourcesParticipant
from nemo_gym.verification import JudgeError
from resources_servers.genrm_compare.tests.support import request_with_session
from resources_servers.genrm_compare.tests.test_cohort_lifecycle import collected_comparisons, member


def record_lock_yields(server, monkeypatch) -> list[str]:
    """Give the server locks that record each time the event loop advances while one of them is held."""
    violations = []

    class TurnLock(asyncio.Lock):
        async def __aenter__(self):
            await super().__aenter__()
            self.turn = token = object()
            asyncio.get_running_loop().call_soon(self.check, token)
            return self

        def check(self, token):
            if self.turn is token:
                violations.append("event loop advanced while bookkeeping lock was held")

        async def __aexit__(self, *args):
            self.turn = None
            return await super().__aexit__(*args)

    cohort_class = genrm._CohortState

    def cohort(**kwargs):
        return cohort_class(**kwargs, lock=TurnLock())

    monkeypatch.setattr(genrm, "_CohortState", cohort)
    server._cohort_registry_lock = TurnLock()
    return violations


async def test_bookkeeping_does_not_yield_while_holding_locks(server, monkeypatch):
    violations = record_lock_yields(server, monkeypatch)

    async def compare(**kwargs):
        await asyncio.sleep(0)
        return collected_comparisons([3.0, 3.0])

    server._collect_comparisons = compare
    # Independent groups, duplicate arrivals, completion and completed replay.
    await asyncio.gather(
        *(server.verify(request_with_session(), member(i, group=g)) for g in ("a", "b", None) for i in (0, 1, 0))
    )
    await server.verify(request_with_session(), member(0, group="a"))
    # Replacement of an incomplete attempt and a disconnected duplicate.
    old = asyncio.create_task(server.verify(request_with_session(), member(0, group="replace")))
    await asyncio.sleep(0)
    replacement = asyncio.create_task(server.verify(request_with_session(), member(0, group="replace", attempt=1)))
    await asyncio.sleep(0)
    with pytest.raises(HTTPException):
        await old
    duplicate = asyncio.create_task(server.verify(request_with_session(), member(0, group="replace", attempt=1)))
    await asyncio.sleep(0)
    duplicate.cancel()
    await asyncio.gather(duplicate, return_exceptions=True)
    await server.verify(request_with_session(), member(1, group="replace", attempt=1))
    await replacement
    # Deadline failure, judge failure and shutdown also publish under the locks.
    server.config.cohort_collection_timeout_s = 0.01
    with pytest.raises(HTTPException):
        await server.verify(request_with_session(), member(0, group="timeout"))

    async def fail(**kwargs):
        await asyncio.sleep(0)
        raise ValueError("judge failed")

    server._collect_comparisons = fail
    outcomes = await asyncio.gather(
        *(server.verify(request_with_session(), member(i, group="failure")) for i in range(2)), return_exceptions=True
    )
    assert all(isinstance(result, HTTPException) for result in outcomes)
    pending = asyncio.create_task(server.verify(request_with_session(), member(0, group="shutdown")))
    await asyncio.sleep(0)
    await server.aclose()
    with pytest.raises(HTTPException):
        await pending
    await asyncio.sleep(0)
    assert not violations


@pytest.mark.parametrize("judge_fails", [False, True], ids=["reward", "judge_failure"])
async def test_outcomes_held_for_a_checkpoint_do_not_yield_while_holding_locks(server, monkeypatch, judge_fails):
    """With a checkpoint participant, publication and judge failures wait for admission and then take the lock."""
    violations = record_lock_yields(server, monkeypatch)
    participant = ResourcesParticipant(server, "exported", verify_step_type="replay")
    server._checkpoint = participant

    async def compare(**kwargs):
        await asyncio.sleep(0)
        if judge_fails:
            raise JudgeError("judge is down")
        return collected_comparisons([3.0, 3.0])

    server._collect_comparisons = compare
    for index in range(2):
        participant.seeded(f"s{index}", EpisodeId(rollout_id=f"r{index}"))
    await participant.close_admission(CheckpointRequest(checkpoint_id="c", deadline_ts=time.time() + 60))
    members = [
        asyncio.create_task(server.verify(request_with_session(f"s{index}"), member(index, group="held")))
        for index in range(2)
    ]
    for _ in range(20):
        await asyncio.sleep(0)
    assert not any(task.done() for task in members)
    await participant.open_admission()
    outcomes = await asyncio.gather(*members, return_exceptions=True)
    if judge_fails:
        assert all(isinstance(outcome, JudgeError) for outcome in outcomes)
    else:
        assert [outcome.reward for outcome in outcomes] == [3.0, 3.0]
    await asyncio.sleep(0)
    assert not violations
