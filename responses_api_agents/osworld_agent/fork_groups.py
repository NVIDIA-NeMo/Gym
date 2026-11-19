# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Share one task setup across a prompt group by forking its guest.

The first rollout of a group to arrive prepares the group: one guest runs the task
setup and is forked once per rollout the group dispatched together. Every rollout of
the group then claims its own child, so all of them start from the same desktop state.
"""

from __future__ import annotations

import asyncio
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional, Tuple


LOG = logging.getLogger(__name__)

Descriptor = Dict[str, Any]
Prepare = Callable[[int], Awaitable[List[Descriptor]]]
Release = Callable[[List[Descriptor]], Awaitable[None]]

# Keys of fully claimed or expired groups, kept so a row re-sent after its group
# was claimed gets a fresh guest instead of preparing the whole group again.
_MAX_CONSUMED_KEYS = 100_000


@dataclass(frozen=True)
class ForkClaim:
    key: str
    active_indices: Tuple[int, ...]
    rollout_index: int


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def fork_claim_from_request(request_extra: Mapping[str, Any], task_id: str) -> Optional[ForkClaim]:
    """Read the group identity NeMo RL stamps on each row; None when the row has none."""
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
    preparation: "asyncio.Task[List[Descriptor]]"
    release: Release
    unclaimed: Optional[Dict[int, Descriptor]] = None
    expiry: Optional["asyncio.Task[None]"] = field(default=None)


class ForkGroupRegistry:
    """Prepare each group once and hand one forked guest to each of its rollouts."""

    def __init__(self, *, claim_timeout_s: float) -> None:
        self._claim_timeout_s = claim_timeout_s
        self._groups: Dict[str, _Group] = {}
        self._consumed: "OrderedDict[str, None]" = OrderedDict()
        self._lock = asyncio.Lock()

    def _consume(self, key: str) -> None:
        self._groups.pop(key, None)
        self._consumed[key] = None
        while len(self._consumed) > _MAX_CONSUMED_KEYS:
            self._consumed.popitem(last=False)

    async def claim(self, claim: ForkClaim, *, prepare: Prepare, release: Release) -> Optional[Descriptor]:
        """Return this rollout's forked guest, or None when it should start a fresh one.

        ``prepare(count)`` runs once per group and returns ``count`` descriptors;
        ``release`` stops guests nobody claimed. A failed preparation raises to every
        waiting rollout and lets a re-sent row prepare the group again.
        """
        async with self._lock:
            if claim.key in self._consumed:
                return None
            group = self._groups.get(claim.key)
            if group is None:
                group = _Group(
                    active_indices=claim.active_indices,
                    preparation=asyncio.create_task(prepare(len(claim.active_indices))),
                    release=release,
                )
                self._groups[claim.key] = group
            elif group.active_indices != claim.active_indices:
                raise ValueError(f"fork group {claim.key} changed its active rollouts")

        try:
            descriptors = await asyncio.shield(group.preparation)
        except Exception:
            async with self._lock:
                if self._groups.get(claim.key) is group:
                    self._groups.pop(claim.key)
            raise

        async with self._lock:
            if group.unclaimed is None:
                if len(descriptors) != len(group.active_indices):
                    raise RuntimeError(
                        f"fork group {claim.key} prepared {len(descriptors)} guests for "
                        f"{len(group.active_indices)} rollouts"
                    )
                group.unclaimed = dict(zip(group.active_indices, descriptors))
            descriptor = group.unclaimed.pop(claim.rollout_index, None)
            if not group.unclaimed:
                if self._groups.get(claim.key) is group:
                    self._consume(claim.key)
                if group.expiry is not None:
                    group.expiry.cancel()
            elif group.expiry is None:
                group.expiry = asyncio.create_task(self._expire(claim.key, group))
        return descriptor

    async def _expire(self, key: str, group: _Group) -> None:
        await asyncio.sleep(self._claim_timeout_s)
        async with self._lock:
            unclaimed = list((group.unclaimed or {}).values())
            group.unclaimed = {}
            if self._groups.get(key) is group:
                self._consume(key)
        if unclaimed:
            LOG.warning("Stopping %d unclaimed forked OSWorld guests of group %s", len(unclaimed), key)
            try:
                await group.release(unclaimed)
            except Exception:  # noqa: BLE001 - the guests' TTL is the backstop
                LOG.exception("Could not stop unclaimed forked OSWorld guests of group %s", key)
