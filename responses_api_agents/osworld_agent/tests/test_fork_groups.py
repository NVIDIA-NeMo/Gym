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


class _Snapshots:
    """Records every prepare and release; prepare ``n`` returns snapshot ``s<n>``."""

    def __init__(self, *, fail: BaseException | None = None) -> None:
        self.prepared = 0
        self.released: list[str] = []
        self.gate = asyncio.Event()
        self.gate.set()
        self.fail = fail

    async def prepare(self) -> str:
        self.prepared += 1
        await self.gate.wait()
        if self.fail is not None:
            raise self.fail
        return f"s{self.prepared}"

    async def release(self, snapshot_id: str) -> None:
        self.released.append(snapshot_id)

    async def claim(self, registry: ForkGroupRegistry, claim: ForkClaim) -> str | None:
        return await registry.claim(claim, prepare=self.prepare, release=self.release)


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
    snapshots = _Snapshots()
    snapshots.gate.clear()

    claims = [asyncio.create_task(snapshots.claim(registry, _claim(index))) for index in (2, 0, 1)]
    await asyncio.sleep(0)
    snapshots.gate.set()

    assert await asyncio.gather(*claims) == ["s1", "s1", "s1"]
    assert snapshots.prepared == 1
    assert snapshots.released == []


async def test_the_snapshot_is_deleted_after_the_last_rollout_finishes() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    snapshots = _Snapshots()
    for index in (0, 1, 2):
        await snapshots.claim(registry, _claim(index))

    await registry.finish(_claim(0))
    await registry.finish(_claim(2))
    assert snapshots.released == []
    await registry.finish(_claim(1))
    assert snapshots.released == ["s1"]


async def test_a_rollout_that_finishes_before_its_siblings_claim_keeps_the_snapshot() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    snapshots = _Snapshots()

    await snapshots.claim(registry, _claim(0, active=(0, 1)))
    await registry.finish(_claim(0, active=(0, 1)))
    assert snapshots.released == []

    assert await snapshots.claim(registry, _claim(1, active=(0, 1))) == "s1"
    await registry.finish(_claim(1, active=(0, 1)))
    assert snapshots.released == ["s1"]


async def test_a_resent_row_after_the_group_closed_gets_a_fresh_guest() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    snapshots = _Snapshots()
    for index in (0, 1, 2):
        await snapshots.claim(registry, _claim(index))

    assert await snapshots.claim(registry, _claim(1)) is None
    assert snapshots.prepared == 1


async def test_a_duplicate_claim_while_siblings_wait_gets_a_fresh_guest() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    snapshots = _Snapshots()

    assert await snapshots.claim(registry, _claim(0)) == "s1"
    assert await snapshots.claim(registry, _claim(0)) is None
    assert await snapshots.claim(registry, _claim(1)) == "s1"
    assert snapshots.prepared == 1


async def test_groups_with_different_keys_prepare_separately() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    snapshots = _Snapshots()

    assert await snapshots.claim(registry, _claim(0, active=(0,), key="g1:0:task")) == "s1"
    assert await snapshots.claim(registry, _claim(0, active=(0,), key="g1:1:task")) == "s2"


async def test_a_changed_active_set_is_rejected() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    snapshots = _Snapshots()
    await snapshots.claim(registry, _claim(0))

    with pytest.raises(ValueError, match="changed its active rollouts"):
        await snapshots.claim(registry, _claim(1, active=(1, 2)))


async def test_a_failed_preparation_fails_every_waiter_and_can_be_retried() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    snapshots = _Snapshots(fail=RuntimeError("snapshot refused"))
    snapshots.gate.clear()
    claims = [asyncio.create_task(snapshots.claim(registry, _claim(index))) for index in (0, 1)]
    await asyncio.sleep(0)
    snapshots.gate.set()
    results = await asyncio.gather(*claims, return_exceptions=True)

    assert [str(result) for result in results] == ["snapshot refused", "snapshot refused"]
    assert snapshots.released == []

    snapshots.fail = None
    assert await snapshots.claim(registry, _claim(2)) == "s2"


async def test_the_claim_timeout_closes_the_group_and_waits_for_running_rollouts() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=0.01)
    snapshots = _Snapshots()

    await snapshots.claim(registry, _claim(0))
    await asyncio.sleep(0.05)
    assert snapshots.released == []
    assert await snapshots.claim(registry, _claim(1)) is None

    await registry.finish(_claim(0))
    assert snapshots.released == ["s1"]


async def test_the_claim_timeout_deletes_a_snapshot_nobody_runs_from() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=0.01)
    snapshots = _Snapshots()

    await snapshots.claim(registry, _claim(0))
    await registry.finish(_claim(0))
    await asyncio.sleep(0.05)

    assert snapshots.released == ["s1"]


async def test_a_fully_claimed_group_is_not_closed_again_by_the_timeout() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=0.01)
    snapshots = _Snapshots()
    for index in (0, 1):
        await snapshots.claim(registry, _claim(index, active=(0, 1)))
    await asyncio.sleep(0.05)
    assert snapshots.released == []

    for index in (0, 1):
        await registry.finish(_claim(index, active=(0, 1)))
    assert snapshots.released == ["s1"]


async def test_a_cancelled_claimant_does_not_cancel_the_preparation() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)
    snapshots = _Snapshots()
    snapshots.gate.clear()
    first = asyncio.create_task(snapshots.claim(registry, _claim(0, active=(0, 1))))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    snapshots.gate.set()
    assert await snapshots.claim(registry, _claim(1, active=(0, 1))) == "s1"
    assert snapshots.prepared == 1

    await registry.finish(_claim(1, active=(0, 1)))
    assert snapshots.released == ["s1"]


async def test_a_group_whose_every_claimant_was_cancelled_still_deletes_its_snapshot() -> None:
    registry = ForkGroupRegistry(claim_timeout_s=0.01)
    snapshots = _Snapshots()
    snapshots.gate.clear()
    first = asyncio.create_task(snapshots.claim(registry, _claim(0, active=(0, 1))))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first

    snapshots.gate.set()
    await asyncio.sleep(0.05)
    assert snapshots.released == ["s1"]


async def test_a_failed_delete_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    registry = ForkGroupRegistry(claim_timeout_s=60)

    async def prepare() -> str:
        return "s1"

    async def release(snapshot_id: str) -> None:
        raise RuntimeError("delete refused")

    await registry.claim(_claim(0, active=(0,)), prepare=prepare, release=release)
    await registry.finish(_claim(0, active=(0,)))

    assert "Could not delete the snapshot s1" in caplog.text
