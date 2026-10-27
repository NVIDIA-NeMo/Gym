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
Cohort bookkeeping for the DWRL pointwise resources server.

verify() is called once per rollout. Rollouts of the same prompt are grouped into a cohort of
`size` members, each member is judged independently, and rewards are computed once every member
has been judged. Judging takes a long and highly variable time, so members finish in an order
that has nothing to do with the order in which they arrived. The cohort is therefore completed
by whichever member finishes judging last, not by whichever member arrived last.

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
from typing import Any, Dict, Hashable, List, NamedTuple, Optional, Tuple


class MemberJudgmentFailed(RuntimeError):
    """Raised for a request that was waiting on a rollout whose own judgment failed or was cancelled."""


class Cohort:
    """The rollouts of one prompt, the judge results gathered for them, and their reward futures."""

    def __init__(self, size: int, num_judges: int, key: Optional[str] = None) -> None:
        self.size = size
        self.num_judges = num_judges
        self.key = key
        self.payloads: List[Any] = []
        self.futures: List[asyncio.Future] = []
        # (member_idx, judge_idx) -> judge result. A slot is written at most once.
        self.results: Dict[Tuple[int, int], Any] = {}
        # identity of a member (see CohortRegistry.join) -> member_idx
        self.member_by_identity: Dict[Hashable, int] = {}
        # member_idx -> why the judgment of that member failed. Such a member still has (neutral) results, so the
        # cohort can complete, but its future is failed instead of being given a reward.
        self.failures: Dict[int, BaseException] = {}

    @property
    def is_full(self) -> bool:
        """All members have arrived."""
        return len(self.payloads) >= self.size

    @property
    def is_complete(self) -> bool:
        """All members have arrived and every (member, judge) result is in."""
        return self.is_full and len(self.results) >= self.size * self.num_judges

    def record(self, member_idx: int, judge_results: List[Any]) -> None:
        """Store the `num_judges` results for one member. A repeated call for a member is ignored."""
        if len(judge_results) != self.num_judges:
            raise ValueError(f"Expected {self.num_judges} judge results, got {len(judge_results)}")
        for judge_idx, result in enumerate(judge_results):
            self.results.setdefault((member_idx, judge_idx), result)

    def record_failure(self, member_idx: int, error: BaseException, neutral_results: List[Any]) -> None:
        """Fill the slots of a member whose judgment failed, and remember that it failed."""
        self.failures.setdefault(member_idx, error)
        self.record(member_idx, neutral_results)

    def ordered(self) -> Tuple[List[Any], List[Tuple[int, int]]]:
        """Results and their (member_idx, judge_idx) metadata, ordered by (judge_idx, member_idx)."""
        keys = sorted(self.results, key=lambda key: (key[1], key[0]))
        return [self.results[key] for key in keys], keys


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
        num_judges: int,
        identity: Optional[Hashable] = None,
    ) -> Joined:
        """Add one arrival to the open cohort for `key`, opening a new one if needed.

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
            cohort = Cohort(size, num_judges, key)
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
