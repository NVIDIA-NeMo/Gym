# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Share one task setup across a prompt group through a sandbox snapshot.

The first rollout of a group to arrive prepares the group: one guest runs the task
setup and is snapshotted, memory and running processes included. Every rollout the
group dispatched together then starts its own guest from that snapshot, so all of
them start from the same desktop state.

The snapshot is deleted once the group is closed and every rollout that started from
it has finished. A group closes when all of its rollouts have claimed the snapshot,
or ``claim_timeout_s`` after the snapshot is ready, whichever comes first; a rollout
that arrives after that starts from a fresh guest.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Set, Tuple


LOG = logging.getLogger(__name__)

Prepare = Callable[[], Awaitable[str]]
Release = Callable[[str], Awaitable[None]]

# Keys of closed groups, kept so a row re-sent after its group closed gets a fresh
# guest instead of preparing the whole group again.
_MAX_CONSUMED_KEYS = 100_000


@dataclass(frozen=True)
class ForkClaim:
    key: str
    active_indices: Tuple[int, ...]
    rollout_index: int


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _succeeded(task: "asyncio.Task[Any]") -> bool:
    return task.done() and not task.cancelled() and task.exception() is None


def fork_claim_from_request(request_extra: Mapping[str, Any], task_id: str) -> Optional[ForkClaim]:
    """Read the group identity the trainer stamps on each row; None when the row has none."""
    group_id = request_extra.get("_ng_group_id")
    active = request_extra.get("_ng_active_generation_indices")
    if not group_id or active is None:
        return None
    attempt = request_extra.get("_ng_group_attempt", 0)
    rollout_index = request_extra.get("_ng_rollout_index")
    if not isinstance(group_id, str) or not _non_negative_int(attempt) or not _non_negative_int(rollout_index):
        raise ValueError("fork_rollout_groups needs a string _ng_group_id and non-negative attempt and rollout index")
    if not isinstance(active, list) or not active or not all(_non_negative_int(index) for index in active):
        raise ValueError("_ng_active_generation_indices must be a non-empty list of non-negative integers")
    active_indices = tuple(sorted(set(active)))
    if len(active_indices) != len(active) or rollout_index not in active_indices:
        raise ValueError("_ng_active_generation_indices must be unique and contain _ng_rollout_index")
    return ForkClaim(f"{group_id}:{attempt}:{task_id}", active_indices, rollout_index)


@dataclass
class _Group:
    active_indices: Tuple[int, ...]
    preparation: "asyncio.Task[str]"
    release: Release
    claimed: Set[int] = field(default_factory=set)
    running: int = 0
    closed: bool = False
    expiry: Optional["asyncio.Task[None]"] = None


class ForkGroupRegistry:
    """Prepare each group's snapshot once and delete it after the group's last rollout."""

    def __init__(self, *, claim_timeout_s: float) -> None:
        self._claim_timeout_s = claim_timeout_s
        self._groups: Dict[str, _Group] = {}
        self._consumed: "OrderedDict[str, None]" = OrderedDict()
        self._lock = asyncio.Lock()

    async def claim(self, claim: ForkClaim, *, prepare: Prepare, release: Release) -> Optional[str]:
        """Return the snapshot this rollout starts from, or None when it should start a fresh guest.

        ``prepare()`` runs once per group and returns a ready snapshot id; ``release`` deletes
        it. A rollout that gets a snapshot id must call :meth:`finish` once its guest is
        stopped. A failed preparation raises to every waiting rollout and lets a re-sent
        row prepare the group again.
        """
        async with self._lock:
            if claim.key in self._consumed:
                return None
            group = self._groups.get(claim.key)
            if group is None:
                group = _Group(
                    active_indices=claim.active_indices,
                    preparation=asyncio.create_task(prepare()),
                    release=release,
                )
                self._groups[claim.key] = group
                group.preparation.add_done_callback(
                    lambda _task, key=claim.key, group=group: self._prepared(key, group)
                )
            elif group.active_indices != claim.active_indices:
                raise ValueError(f"fork group {claim.key} changed its active rollouts")
            if claim.rollout_index in group.claimed:
                return None
            group.claimed.add(claim.rollout_index)
            group.running += 1

        try:
            snapshot_id = await asyncio.shield(group.preparation)
        except BaseException:
            group.running -= 1
            if _succeeded(group.preparation):
                # This rollout was cancelled; it will not start from the snapshot.
                asyncio.ensure_future(self._release(claim.key, group, self._take_releasable(claim.key, group)))
            raise

        async with self._lock:
            if len(group.claimed) == len(group.active_indices):
                self._close(claim.key, group)
        return snapshot_id

    def _prepared(self, key: str, group: _Group) -> None:
        if not _succeeded(group.preparation):
            if self._groups.get(key) is group:
                self._groups.pop(key)
        elif not group.closed:
            group.expiry = asyncio.ensure_future(self._expire(key, group))

    async def finish(self, claim: ForkClaim) -> None:
        """Mark a rollout that started from its group's snapshot as done."""
        async with self._lock:
            group = self._groups.get(claim.key)
            if group is None:
                return
            group.running -= 1
            snapshot_id = self._take_releasable(claim.key, group)
        await self._release(claim.key, group, snapshot_id)

    def _close(self, key: str, group: _Group) -> None:
        group.closed = True
        self._consumed[key] = None
        while len(self._consumed) > _MAX_CONSUMED_KEYS:
            self._consumed.popitem(last=False)
        if group.expiry is not None and group.expiry is not asyncio.current_task():
            group.expiry.cancel()

    def _take_releasable(self, key: str, group: _Group) -> Optional[str]:
        """Drop a closed group nobody runs from any more and return its snapshot id."""
        if not group.closed or group.running > 0 or self._groups.get(key) is not group:
            return None
        self._groups.pop(key)
        return group.preparation.result()

    async def _release(self, key: str, group: _Group, snapshot_id: Optional[str]) -> None:
        if snapshot_id is None:
            return
        try:
            await group.release(snapshot_id)
        except Exception:  # noqa: BLE001 - a leaked snapshot is listed by its name prefix
            LOG.exception("Could not delete the snapshot %s of OSWorld group %s", snapshot_id, key)

    async def _expire(self, key: str, group: _Group) -> None:
        await asyncio.sleep(self._claim_timeout_s)
        async with self._lock:
            if group.closed:
                return
            unclaimed = len(group.active_indices) - len(group.claimed)
            LOG.warning("OSWorld group %s closed with %d rollout(s) that never claimed its snapshot", key, unclaimed)
            self._close(key, group)
            snapshot_id = self._take_releasable(key, group)
        await self._release(key, group, snapshot_id)
