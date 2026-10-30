# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Cohort bookkeeping for the DWRL pairwise resources server.

The reward of a rollout depends on the other rollouts of its prompt: the pairwise judgments of all of them are fitted
together (see `aggregate_scores`). verify() is called once per rollout, so rollouts of the same prompt are grouped
into a cohort of `size` members, and rewards are computed once every comparison of the cohort has been judged.

A comparison is a "slot" `(i, j, judge_idx)`: response `i` against response `j`, judged by pass `judge_idx`. Members
are numbered in arrival order. A comparison is judged by the member that arrives last of the two responses in it, so
each comparison is judged exactly once, as early as it can be, and by a request that is already waiting for the
cohort. A comparison against the reference answer (index `size`, which is not a member and is there from the start) is
judged by the member it involves.

Judging takes a long and highly variable time, so members finish in an order that has nothing to do with the order
in which they arrived. The cohort is therefore completed by whichever member finishes judging last, not by whichever
member arrived last.

A request can be delivered twice (for example a client that retries after a dropped connection).
When a request carries an `identity` that an active cohort already has a member for, it does not
take a second slot: it is handed that member's future and waits for the same reward. A cohort is
active from the moment it is created until `finish()` is called for it, so this also covers a
repeat that arrives after the cohort has filled up but before it has been finalized. A repeat that
arrives after the cohort was finalized is not recognized.

This module is deliberately free of heavy imports so the logic can be unit tested on its own.
All methods are synchronous and are meant to be called from a single asyncio event loop: there is
no await between reading and updating the state, so no further locking is needed.
"""

from __future__ import annotations

import asyncio
from typing import Any, Collection, Dict, Hashable, Iterable, List, Mapping, NamedTuple, Optional, Tuple


# (index of response 1, index of response 2, judge pass)
Slot = Tuple[int, int, int]


class MemberJudgmentFailed(RuntimeError):
    """Raised for a request that was waiting on a rollout whose own judgment failed or was cancelled."""


def comparison_slots(pairs: Iterable[Tuple[int, int]], num_judges: int) -> List[Slot]:
    """Every (pair, judge pass) combination, ordered by judge pass and then by pair."""
    pairs = list(pairs)
    return [(i, j, judge_idx) for judge_idx in range(num_judges) for i, j in pairs]


def slot_owner(slot: Slot, size: int) -> int:
    """Index of the member that judges a slot: the one that arrives last of the members whose responses it involves."""
    i, j, _ = slot
    return max(idx for idx in (i, j) if idx < size)


class Cohort:
    """The rollouts of one prompt, the judge results gathered for them, and their reward futures."""

    def __init__(self, size: int, slots: Collection[Slot], key: Optional[str] = None) -> None:
        self.size = size
        self.slots = frozenset(slots)
        self.key = key
        self.payloads: List[Any] = []
        self.futures: List[asyncio.Future] = []
        # slot -> judge result. A slot is written at most once.
        self.results: Dict[Slot, Any] = {}
        # identity of a member (see CohortRegistry.join) -> member_idx
        self.member_by_identity: Dict[Hashable, int] = {}
        # member_idx -> why the judgment of that member failed. The slots that member was to judge were filled with
        # neutral results, so the cohort can complete, but its future is failed instead of being given a reward.
        self.failures: Dict[int, BaseException] = {}

        self._owned: Dict[int, List[Slot]] = {member_idx: [] for member_idx in range(size)}
        for slot in sorted(self.slots, key=_slot_order):
            self._owned[slot_owner(slot, size)].append(slot)

    @property
    def is_full(self) -> bool:
        """All members have arrived."""
        return len(self.payloads) >= self.size

    @property
    def is_complete(self) -> bool:
        """All members have arrived and every slot has a result."""
        return self.is_full and len(self.results) >= len(self.slots)

    def owned_slots(self, member_idx: int) -> List[Slot]:
        """The slots that the member judges, ordered by judge pass and then by pair."""
        return list(self._owned[member_idx])

    def record(self, results: Mapping[Slot, Any]) -> None:
        """Store judge results by slot. A slot that already has a result keeps it."""
        for slot in results:
            if slot not in self.slots:
                raise ValueError(f"Unexpected comparison slot {slot}")
        for slot, result in results.items():
            self.results.setdefault(slot, result)

    def record_failure(self, member_idx: int, error: BaseException, neutral_results: Mapping[Slot, Any]) -> None:
        """Fill the slots of a member whose judgment failed, and remember that it failed."""
        self.failures.setdefault(member_idx, error)
        self.record(neutral_results)

    def ordered(self) -> Tuple[List[Any], List[Slot]]:
        """Results and their slots, ordered by (judge_idx, i, j)."""
        slots = sorted(self.results, key=_slot_order)
        return [self.results[slot] for slot in slots], slots


def _slot_order(slot: Slot) -> Tuple[int, int, int]:
    i, j, judge_idx = slot
    return judge_idx, i, j


class Joined(NamedTuple):
    """What CohortRegistry.join returns."""

    cohort: Cohort
    member_idx: int  # The member's index in the cohort (its arrival order)
    future: asyncio.Future  # Resolves to the member's reward
    is_new: bool  # False if the request repeats a member that the cohort already has


class CohortRegistry:
    """Groups arrivals by key into cohorts."""

    def __init__(self) -> None:
        # Cohorts that are still waiting for members, by key.
        self._open: Dict[str, Cohort] = {}
        # Cohorts that have not been finalized yet (so including the ones that are still filling), by key.
        self._active: Dict[str, List[Cohort]] = {}

    @property
    def num_open(self) -> int:
        """Cohorts that have not received all of their members yet."""
        return len(self._open)

    @property
    def num_active(self) -> int:
        """Cohorts that have not been finalized yet, whether they are still filling or being judged."""
        return sum(len(cohorts) for cohorts in self._active.values())

    def join(
        self,
        key: str,
        payload: Any,
        size: int,
        slots: Collection[Slot],
        identity: Optional[Hashable] = None,
    ) -> Joined:
        """Add one arrival to the open cohort for `key`, opening a new one if needed.

        `slots` are the comparisons that the cohort has to judge; they are only used when a new cohort is opened.

        `identity` identifies the rollout that the request is for. If an active cohort for `key` already has a
        member with the same identity, the request is a repeat: no member is added, and the existing member's future
        is returned with `is_new=False`. Without an identity a request is never treated as a repeat.

        Once a cohort is full it stops accepting members, so later arrivals with the same key (e.g. a prompt that
        occurs twice in a batch, or the same prompt in a later epoch) start a new cohort instead of overflowing this
        one. The cohort object stays alive for as long as its members hold a reference to it.
        """
        if identity is not None:
            for cohort in self._active.get(key, ()):
                existing_idx = cohort.member_by_identity.get(identity)
                if existing_idx is not None:
                    return Joined(cohort, existing_idx, cohort.futures[existing_idx], False)

        cohort = self._open.get(key)
        if cohort is None:
            cohort = Cohort(size, slots, key)
            self._open[key] = cohort
            self._active.setdefault(key, []).append(cohort)

        member_idx = len(cohort.payloads)
        future = asyncio.get_running_loop().create_future()
        cohort.payloads.append(payload)
        cohort.futures.append(future)
        if identity is not None:
            cohort.member_by_identity[identity] = member_idx

        if cohort.is_full:
            del self._open[key]

        return Joined(cohort, member_idx, future, True)

    def finish(self, cohort: Cohort) -> None:
        """The cohort has been finalized: repeats of its members are no longer recognized. Safe to call twice."""
        cohorts = self._active.get(cohort.key)
        if cohorts is None or cohort not in cohorts:
            return
        cohorts.remove(cohort)
        if not cohorts:
            del self._active[cohort.key]
