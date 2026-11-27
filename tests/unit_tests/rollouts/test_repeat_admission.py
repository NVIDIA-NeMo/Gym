# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from unittest.mock import AsyncMock

import pytest

from nemo_gym.config.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollouts.repeat_admission import RepeatAdmission
from nemo_gym.rollouts.repeat_stopping import STOP_ON_SUCCESS_THRESHOLD_KEY, repeat_group
from nemo_gym.rollouts.rollout_collection import RolloutCollectionHelper
from tests.unit_tests.rollouts.test_rollout_collection import FakeResponse, install_fake_server_client


@pytest.mark.parametrize("window", [None, 2])
async def test_fast_task_advances_while_another_is_waiting(monkeypatch, window):
    # The input is grouped by task and out of repeat order. A fixed window over these rows would
    # initially admit only repeats of the blocked task; a round barrier would block the fast task.
    rows = [
        {
            AGENT_REF_KEY_NAME: {"name": "my_agent"},
            TASK_INDEX_KEY_NAME: task,
            ROLLOUT_INDEX_KEY_NAME: repeat,
            STOP_ON_SUCCESS_THRESHOLD_KEY: 1.0,
        }
        for task in (0, 1)
        for repeat in (2, 0, 1)
    ]
    admission = RepeatAdmission(rows, repeat_group)
    release_slow = asyncio.Event()
    posted = []

    def identity(row):
        return row[TASK_INDEX_KEY_NAME], row[ROLLOUT_INDEX_KEY_NAME]

    async def post(**kwargs):
        assert STOP_ON_SUCCESS_THRESHOLD_KEY not in kwargs["json"]
        task, repeat = identity(kwargs["json"])
        posted.append((task, repeat))
        if task == 0:
            await release_slow.wait()
        return FakeResponse(200, {"reward": float(task == 0 or repeat == 1)})

    install_fake_server_client(monkeypatch, AsyncMock(side_effect=post))
    reserved = []
    completions = RolloutCollectionHelper()._run_examples_with_metadata(
        rows,
        semaphore=asyncio.Semaphore(2),
        max_resident_tasks=window,
        repeat_policy=admission,
        on_dispatch_batch=reserved.extend,
    )
    skipped = []
    try:
        async with asyncio.timeout(5):
            for future in completions:
                result = await future
                skipped.extend(admission.settle(result.row, solved=result.result["reward"] >= 1))
                if identity(result.row) == (1, 1):
                    release_slow.set()
    finally:
        await completions.aclose()

    assert posted == [(0, 0), (1, 0), (1, 1)]
    assert [identity(row) for row in reserved] == posted
    assert sorted(identity(row) for row in skipped) == [(0, 1), (0, 2), (1, 2)]


def test_solving_one_agent_keeps_another_agents_repeats():
    # Fan-out assigns rollout indices across agents, so an agent's first pending index need not be zero.
    rows = [
        {AGENT_REF_KEY_NAME: {"name": agent}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: repeat}
        for repeat, agent in enumerate(("a", "b", "a", "b"))
    ]
    admission = RepeatAdmission(rows, repeat_group)
    assert admission.take() == rows[0]
    assert admission.take() == rows[1]
    assert admission.take() is None
    assert admission.settle(rows[0], solved=True) == [rows[2]]
    assert admission.settle(rows[1], solved=False) == []
    assert admission.take() == rows[3]
    assert admission.settle(rows[3], solved=False) == []
    assert admission.take() is None


async def test_empty_resume_finishes_without_a_request(monkeypatch):
    post = AsyncMock()
    install_fake_server_client(monkeypatch, post)
    completions = RolloutCollectionHelper()._run_examples_with_metadata(
        [], repeat_policy=RepeatAdmission([], repeat_group)
    )
    assert next(completions, None) is None
    await completions.aclose()
    post.assert_not_awaited()
