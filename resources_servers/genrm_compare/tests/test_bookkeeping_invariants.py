# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Protected bookkeeping must finish before yielding to another request."""

import asyncio

import pytest
from fastapi import HTTPException

import resources_servers.genrm_compare.app as genrm
from resources_servers.genrm_compare.tests.test_cohort_lifecycle import member


async def test_bookkeeping_does_not_yield_while_holding_locks(server, monkeypatch):
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

    async def compare(**kwargs):
        await asyncio.sleep(0)
        return [3.0, 3.0], {}, [], []

    server._run_compare = compare
    # Independent groups, duplicate arrivals, completion and completed replay.
    await asyncio.gather(*(server.verify(member(i, group=g)) for g in ("a", "b", None) for i in (0, 1, 0)))
    await server.verify(member(0, group="a"))
    # Replacement of an incomplete attempt and a disconnected duplicate.
    old = asyncio.create_task(server.verify(member(0, group="replace")))
    await asyncio.sleep(0)
    replacement = asyncio.create_task(server.verify(member(0, group="replace", attempt=1)))
    await asyncio.sleep(0)
    with pytest.raises(HTTPException):
        await old
    duplicate = asyncio.create_task(server.verify(member(0, group="replace", attempt=1)))
    await asyncio.sleep(0)
    duplicate.cancel()
    await asyncio.gather(duplicate, return_exceptions=True)
    await server.verify(member(1, group="replace", attempt=1))
    await replacement
    # Deadline failure, judge failure and shutdown also publish under the locks.
    server.config.cohort_collection_timeout_s = 0.01
    with pytest.raises(HTTPException):
        await server.verify(member(0, group="timeout"))

    async def fail(**kwargs):
        await asyncio.sleep(0)
        raise ValueError("judge failed")

    server._run_compare = fail
    outcomes = await asyncio.gather(
        *(server.verify(member(i, group="failure")) for i in range(2)), return_exceptions=True
    )
    assert all(isinstance(result, HTTPException) for result in outcomes)
    pending = asyncio.create_task(server.verify(member(0, group="shutdown")))
    await asyncio.sleep(0)
    await server.aclose()
    with pytest.raises(HTTPException):
        await pending
    await asyncio.sleep(0)
    assert not violations
