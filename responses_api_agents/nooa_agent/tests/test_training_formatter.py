# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

from nooa.context_blocks.renderers.cached import CachedBlockFormatter
from nooa.events import PythonOutput, ResultStatus, Task

from responses_api_agents.nooa_agent.training_formatter import TrainingBlockFormatter


def output(value, **kwargs):
    return PythonOutput(
        tool_call_id="cell", execution_count=1, execution_status=ResultStatus.COMPLETE, value=value, **kwargs
    )


async def test_live_task_keeps_first_rendering_and_remains_awaitable():
    gate = asyncio.Event()

    async def work():
        await gate.wait()
        return 7

    task = asyncio.create_task(work())
    event = output(task)
    stock = CachedBlockFormatter()
    formatter = TrainingBlockFormatter(stock)
    before = formatter.format_event(event)
    assert "Task pending" in before
    gate.set()
    assert await event.value == 7
    assert "Task finished" in stock.format_event(event)
    assert stock.format_event(event) != before  # The pinned upstream mechanism reproduces the defect.
    assert formatter.format_event(event) == before
    assert event.value is task


def test_mutable_value_is_snapshotted_per_event_and_per_agent():
    values = []
    event = output(values)
    formatter = TrainingBlockFormatter(CachedBlockFormatter())
    before = formatter.format_event(event)
    values.append("later")
    assert formatter.format_event(event) == before
    assert formatter.format_event(event.model_copy()) == before
    assert "later" in formatter.format_event(output(values))
    assert "later" in TrainingBlockFormatter(CachedBlockFormatter()).format_event(event)
    assert event.value is values


def test_other_events_are_not_frozen():
    event = Task(prompt="first")
    formatter = TrainingBlockFormatter(CachedBlockFormatter())
    before = formatter.format_event(event)
    event.prompt = "changed"
    assert formatter.format_event(event) != before
    assert formatter.format_type == CachedBlockFormatter().format_type
    assert formatter.format_description() == CachedBlockFormatter().format_description()
