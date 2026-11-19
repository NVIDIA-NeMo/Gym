# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from responses_api_agents.osworld_agent.fork_groups import ForkClaim, ForkGroupRegistry, fork_claim_from_request


def _extra(**overrides: Any) -> dict[str, Any]:
    return {
        "_ng_group_id": "g1",
        "_ng_group_attempt": 0,
        "_ng_rollout_index": 1,
        "_ng_active_generation_indices": [0, 1, 2],
        **overrides,
    }


def _claim(index: int, active: tuple[int, ...] = (0, 1, 2), key: str = "g1:0:task") -> ForkClaim:
    return ForkClaim(key=key, active_indices=active, rollout_index=index)


class _Guests:
    """Records every prepare and release; each prepare returns one descriptor per rollout."""

    def __init__(self, *, fail: BaseException | None = None) -> None:
        self.prepared: list[int] = []
        self.released: list[list[dict[str, Any]]] = []
        self.gate = asyncio.Event()
        self.gate.set()
        self.fail = fail

    async def prepare(self, count: int) -> list[dict[str, Any]]:
        self.prepared.append(count)
        await self.gate.wait()
        if self.fail is not None:
            raise self.fail
        batch = len(self.prepared)
        return [{"sandbox_id": f"p{batch}-child{index}"} for index in range(count)]

    async def release(self, descriptors: list[dict[str, Any]]) -> None:
        self.released.append(descriptors)


def test_claim_from_request_builds_the_group_key() -> None:
    claim = fork_claim_from_request(_extra(_ng_group_attempt=2, _ng_active_generation_indices=[2, 0, 1]), "task")
    assert claim == ForkClaim(key="g1:2:task", active_indices=(0, 1, 2), rollout_index=1)


@pytest.mark.parametrize("missing", ["_ng_group_id", "_ng_active_generation_indices"])
def test_claim_from_request_without_group_identity_is_none(missing: str) -> None:
    extra = _extra()
    extra.pop(missing)
    assert fork_claim_from_request(extra, "task") is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"_ng_rollout_index": None},
        {"_ng_rollout_index": True},
        {"_ng_group_attempt": -1},
        {"_ng_active_generation_indices": []},
        {"_ng_active_generation_indices": [0, 0, 1]},
        {"_ng_active_generation_indices": [0, 2]},
        {"_ng_active_generation_indices": "0,1,2"},
    ],
)
def test_claim_from_request_rejects_malformed_identity(overrides: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        fork_claim_from_request(_extra(**overrides), "task")


async def test_one_preparation_serves_the_whole_group() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    guests = _Guests()
    guests.gate.clear()

    claims = [
        asyncio.create_task(registry.claim(_claim(index), prepare=guests.prepare, release=guests.release))
        for index in (2, 0, 1)
    ]
    await asyncio.sleep(0)
    guests.gate.set()
    descriptors = await asyncio.gather(*claims)

    assert guests.prepared == [3]
    assert descriptors == [{"sandbox_id": "p1-child2"}, {"sandbox_id": "p1-child0"}, {"sandbox_id": "p1-child1"}]
    assert guests.released == []


async def test_subset_dispatch_maps_children_to_the_active_indices() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    guests = _Guests()

    first = await registry.claim(_claim(3, active=(1, 3)), prepare=guests.prepare, release=guests.release)
    second = await registry.claim(_claim(1, active=(1, 3)), prepare=guests.prepare, release=guests.release)

    assert guests.prepared == [2]
    assert (first, second) == ({"sandbox_id": "p1-child1"}, {"sandbox_id": "p1-child0"})


async def test_a_resent_row_after_the_group_is_claimed_gets_a_fresh_guest() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    guests = _Guests()
    for index in (0, 1, 2):
        await registry.claim(_claim(index), prepare=guests.prepare, release=guests.release)

    assert await registry.claim(_claim(1), prepare=guests.prepare, release=guests.release) is None
    assert guests.prepared == [3]


async def test_a_duplicate_claim_while_siblings_wait_gets_a_fresh_guest() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    guests = _Guests()

    assert await registry.claim(_claim(0), prepare=guests.prepare, release=guests.release) is not None
    assert await registry.claim(_claim(0), prepare=guests.prepare, release=guests.release) is None
    assert await registry.claim(_claim(1), prepare=guests.prepare, release=guests.release) is not None
    assert guests.prepared == [3]


async def test_groups_with_different_keys_prepare_separately() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    guests = _Guests()

    await registry.claim(_claim(0, active=(0,), key="g1:0:task"), prepare=guests.prepare, release=guests.release)
    await registry.claim(_claim(0, active=(0,), key="g1:1:task"), prepare=guests.prepare, release=guests.release)

    assert guests.prepared == [1, 1]


async def test_a_changed_active_set_is_rejected() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    guests = _Guests()
    await registry.claim(_claim(0), prepare=guests.prepare, release=guests.release)

    with pytest.raises(ValueError, match="changed its active rollouts"):
        await registry.claim(_claim(1, active=(1, 2)), prepare=guests.prepare, release=guests.release)


async def test_a_failed_preparation_fails_every_waiter_and_can_be_retried() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    guests = _Guests(fail=RuntimeError("fork refused"))
    guests.gate.clear()
    claims = [
        asyncio.create_task(registry.claim(_claim(index), prepare=guests.prepare, release=guests.release))
        for index in (0, 1)
    ]
    await asyncio.sleep(0)
    guests.gate.set()
    results = await asyncio.gather(*claims, return_exceptions=True)

    assert [str(result) for result in results] == ["fork refused", "fork refused"]
    assert guests.prepared == [3]

    guests.fail = None
    assert await registry.claim(_claim(2), prepare=guests.prepare, release=guests.release) == {
        "sandbox_id": "p2-child2"
    }
    assert guests.prepared == [3, 3]


async def test_a_wrong_number_of_guests_is_an_error() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)

    async def prepare(count: int) -> list[dict[str, Any]]:
        return [{"sandbox_id": "only"}]

    async def release(descriptors: list[dict[str, Any]]) -> None:
        return None

    with pytest.raises(RuntimeError, match="prepared 1 guests for 3 rollouts"):
        await registry.claim(_claim(0), prepare=prepare, release=release)


async def test_unclaimed_guests_are_released_after_the_claim_timeout() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=0.01)
    guests = _Guests()

    await registry.claim(_claim(0), prepare=guests.prepare, release=guests.release)
    await asyncio.sleep(0.05)

    assert guests.released == [[{"sandbox_id": "p1-child1"}, {"sandbox_id": "p1-child2"}]]
    assert await registry.claim(_claim(1), prepare=guests.prepare, release=guests.release) is None
    assert guests.prepared == [3]


async def test_a_fully_claimed_group_releases_nothing() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=0.01)
    guests = _Guests()

    await registry.claim(_claim(0, active=(0, 1)), prepare=guests.prepare, release=guests.release)
    await registry.claim(_claim(1, active=(0, 1)), prepare=guests.prepare, release=guests.release)
    await asyncio.sleep(0.05)

    assert guests.released == []


async def test_a_cancelled_claimant_does_not_cancel_the_preparation() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    guests = _Guests()
    guests.gate.clear()
    first = asyncio.create_task(registry.claim(_claim(0), prepare=guests.prepare, release=guests.release))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    guests.gate.set()
    assert await registry.claim(_claim(1), prepare=guests.prepare, release=guests.release) == {
        "sandbox_id": "p1-child1"
    }
    assert guests.prepared == [3]
