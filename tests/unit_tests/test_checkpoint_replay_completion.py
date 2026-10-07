# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Replay completion regression from Gym PR #3882, adapted to the older steps API."""

import asyncio

import pytest

from nemo_gym._checkpoint.steps import EpisodeSteps


async def _no_notify() -> None:
    return None


@pytest.mark.parametrize("resume_before_boundary", [False, True])
async def test_replay_completion_preserves_cut_until_next_boundary(resume_before_boundary: bool) -> None:
    steps = EpisodeSteps(_no_notify)
    entered = asyncio.Event()
    finish_step = asyncio.Event()
    stepped = asyncio.Event()
    record_boundary = asyncio.Event()

    async def episode() -> None:
        steps.begin("r")
        try:
            await steps.boundary("r", {"next": "verify"})
            async with steps.step("r", "replay"):
                entered.set()
                await finish_step.wait()
            stepped.set()
            await record_boundary.wait()
            await steps.boundary("r", {"next": "close"})
        finally:
            await steps.end("r")

    task = asyncio.create_task(episode())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        steps.close()
        assert steps.blockers() == []
        finish_step.set()
        await asyncio.wait_for(stepped.wait(), 1)
        assert steps.blockers() == []
        assert steps.exported() == {"r": {"next": "verify"}}
        if resume_before_boundary:
            steps.open()
        record_boundary.set()
        if not resume_before_boundary:
            await asyncio.sleep(0)
            assert steps.blockers() == []
            assert steps.exported() == {"r": {"next": "close"}}
            steps.open()
        await asyncio.wait_for(task, 1)
    finally:
        steps.open()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_boundary_after_resume_restores_running_state() -> None:
    steps = EpisodeSteps(_no_notify)
    steps.begin("r")
    await steps.boundary("r", {"next": "verify"})
    async with steps.step("r", "replay"):
        steps.close()
    assert steps.blockers() == []
    steps.open()
    await steps.boundary("r", {"next": "close"})
    assert steps.blockers() == ["r"]
    assert steps.exported() == {}
    await steps.end("r")


@pytest.mark.parametrize("mode,closed", [("wait", True), ("wait", False), ("replay", False)])
async def test_other_step_completions_still_require_a_boundary(mode: str, closed: bool) -> None:
    steps = EpisodeSteps(_no_notify)
    steps.begin("r")
    await steps.boundary("r", {"next": "verify"})
    async with steps.step("r", mode):
        if closed:
            steps.close()
    assert steps.blockers() == ["r"]
    assert steps.exported() == {}
    steps.open()
    await steps.end("r")
