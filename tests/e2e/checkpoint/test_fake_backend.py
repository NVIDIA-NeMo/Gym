# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The fake backend's hold and release controls, served in process. These need no Gym servers, so they always run."""

import asyncio
from collections.abc import AsyncIterator

import fake_backend
import httpx
import pytest


@pytest.fixture
async def backend() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=fake_backend.app), base_url="http://b") as client:
        yield client
        await client.post("/_ctl/release")
    fake_backend.STATE["calls"].clear()


CHAT = {"messages": [{"role": "user", "content": "hi"}]}


async def until_received(calls: int) -> None:
    async with asyncio.timeout(5):
        while len(fake_backend.STATE["calls"]) < calls:
            await asyncio.sleep(0.01)


async def test_a_second_hold_before_a_release_keeps_the_calls_the_first_one_held(backend: httpx.AsyncClient) -> None:
    await backend.post("/_ctl/hold", json={"after_calls": 0})
    first = asyncio.create_task(backend.post("/v1/chat/completions", json=CHAT))
    await until_received(1)
    await backend.post("/_ctl/hold", json={"after_calls": 0})
    second = asyncio.create_task(backend.post("/v1/chat/completions", json=CHAT))
    await until_received(2)
    await backend.post("/_ctl/release")

    # One release frees both calls; neither waits on an event nothing will set.
    replies = await asyncio.wait_for(asyncio.gather(first, second), 5)
    assert [reply.status_code for reply in replies] == [200, 200]
