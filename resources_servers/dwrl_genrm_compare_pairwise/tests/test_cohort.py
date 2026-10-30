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
"""Tests for the cohort bookkeeping used by the DWRL pairwise resources server."""

import asyncio
import itertools

import pytest

from resources_servers.dwrl_genrm_compare_pairwise.cohort import (
    Cohort,
    CohortRegistry,
    MemberJudgmentFailed,
    comparison_slots,
    slot_owner,
)


def _circular(n):
    return [(i, (i + 1) % n) for i in range(n)]


def _all_pairs(n):
    return list(itertools.combinations(range(n), 2))


def _with_reference(pairs, n):
    return pairs + [(i, n) for i in range(n)]


def _slots(size, num_judges=1, strategy=_circular, reference=False):
    pairs = strategy(size)
    if reference:
        pairs = _with_reference(pairs, size)
    return comparison_slots(pairs, num_judges)


def _join_all(registry, key, count, size=4, slots=None):
    slots = slots if slots is not None else _slots(size)

    async def run():
        return [registry.join(key, f"payload_{i}", size=size, slots=slots) for i in range(count)]

    return asyncio.run(run())


def _join_with_identities(registry, key, identities, size=4, slots=None):
    slots = slots if slots is not None else _slots(size)

    async def run():
        return [
            registry.join(key, f"payload_{i}", size=size, slots=slots, identity=identity)
            for i, identity in enumerate(identities)
        ]

    return asyncio.run(run())


class TestComparisonSlots:
    def test_one_slot_per_pair_and_judge_pass(self):
        assert comparison_slots([(0, 1), (1, 2)], num_judges=2) == [(0, 1, 0), (1, 2, 0), (0, 1, 1), (1, 2, 1)]

    def test_no_pairs_no_slots(self):
        assert comparison_slots([], num_judges=3) == []


class TestSlotOwner:
    def test_the_member_that_arrives_last_of_the_two_judges_the_comparison(self):
        assert slot_owner((0, 1, 0), size=4) == 1
        assert slot_owner((2, 1, 0), size=4) == 2

    def test_the_wrap_around_pair_of_a_circle_is_judged_by_the_last_member(self):
        assert slot_owner((3, 0, 0), size=4) == 3

    def test_a_comparison_against_the_reference_is_judged_by_the_rollout_in_it(self):
        # The reference answer has index `size` and is not a member.
        assert slot_owner((2, 4, 0), size=4) == 2
        assert slot_owner((4, 2, 0), size=4) == 2

    def test_the_judge_pass_does_not_matter(self):
        assert slot_owner((0, 1, 5), size=4) == slot_owner((0, 1, 0), size=4)


class TestCohortOwnedSlots:
    @pytest.mark.parametrize("size", [2, 3, 4, 8])
    @pytest.mark.parametrize("strategy", [_circular, _all_pairs])
    @pytest.mark.parametrize("num_judges", [1, 3])
    @pytest.mark.parametrize("reference", [False, True])
    def test_every_slot_is_owned_by_exactly_one_member(self, size, strategy, num_judges, reference):
        slots = _slots(size, num_judges, strategy, reference)
        cohort = Cohort(size, slots)

        owned = [slot for member_idx in range(size) for slot in cohort.owned_slots(member_idx)]

        assert sorted(owned) == sorted(slots)

    @pytest.mark.parametrize("size", [2, 3, 4, 8])
    @pytest.mark.parametrize("strategy", [_circular, _all_pairs])
    @pytest.mark.parametrize("reference", [False, True])
    def test_an_owner_has_arrived_by_the_time_of_every_response_in_its_slots(self, size, strategy, reference):
        # Members are numbered in arrival order, so this is what lets a request judge its comparisons at once.
        cohort = Cohort(size, _slots(size, 1, strategy, reference))

        for member_idx in range(size):
            for i, j, _ in cohort.owned_slots(member_idx):
                assert all(idx <= member_idx or idx == size for idx in (i, j))

    def test_circular_cohort_of_four(self):
        cohort = Cohort(4, _slots(4))

        assert cohort.owned_slots(0) == []
        assert cohort.owned_slots(1) == [(0, 1, 0)]
        assert cohort.owned_slots(2) == [(1, 2, 0)]
        assert cohort.owned_slots(3) == [(2, 3, 0), (3, 0, 0)]

    def test_all_pairs_cohort_of_three_with_two_judges(self):
        cohort = Cohort(3, _slots(3, num_judges=2, strategy=_all_pairs))

        assert cohort.owned_slots(1) == [(0, 1, 0), (0, 1, 1)]
        assert cohort.owned_slots(2) == [(0, 2, 0), (1, 2, 0), (0, 2, 1), (1, 2, 1)]

    def test_reference_comparisons_belong_to_each_member(self):
        cohort = Cohort(3, _slots(3, reference=True))

        assert (0, 3, 0) in cohort.owned_slots(0)
        assert (1, 3, 0) in cohort.owned_slots(1)
        assert (2, 3, 0) in cohort.owned_slots(2)
        assert cohort.owned_slots(0) == [(0, 3, 0)]

    def test_owned_slots_returns_a_copy(self):
        cohort = Cohort(4, _slots(4))

        cohort.owned_slots(3).clear()

        assert len(cohort.owned_slots(3)) == 2


class TestCohort:
    def test_not_complete_until_full_and_all_results_in(self):
        cohort = Cohort(size=2, slots=[(0, 1, 0), (1, 0, 0)])
        assert not cohort.is_full
        assert not cohort.is_complete

        cohort.payloads.extend(["a", "b"])
        assert cohort.is_full
        assert not cohort.is_complete

        cohort.record({(0, 1, 0): "r01"})
        assert not cohort.is_complete
        cohort.record({(1, 0, 0): "r10"})
        assert cohort.is_complete

    def test_results_alone_do_not_complete_a_cohort_that_is_not_full(self):
        # Every comparison between the arrived members is judged, but more members are still to come.
        cohort = Cohort(size=3, slots=_slots(3))
        cohort.payloads.extend(["a", "b"])
        cohort.record({(0, 1, 0): "r01"})

        assert not cohort.is_complete

    def test_record_rejects_a_slot_that_is_not_part_of_the_cohort(self):
        cohort = Cohort(size=2, slots=[(0, 1, 0)])

        with pytest.raises(ValueError):
            cohort.record({(1, 0, 0): "r"})

    def test_a_rejected_record_stores_nothing(self):
        cohort = Cohort(size=2, slots=[(0, 1, 0)])

        with pytest.raises(ValueError):
            cohort.record({(0, 1, 0): "fine", (1, 0, 0): "unexpected"})

        assert cohort.results == {}

    def test_record_is_write_once_per_slot(self):
        cohort = Cohort(size=2, slots=[(0, 1, 0)])
        cohort.record({(0, 1, 0): "first"})
        cohort.record({(0, 1, 0): "second"})

        assert cohort.results[(0, 1, 0)] == "first"

    def test_recording_nothing_is_fine(self):
        # The first member of a circular cohort has no comparison to judge when it arrives.
        cohort = Cohort(size=2, slots=[(0, 1, 0)])
        cohort.record({})

        assert cohort.results == {}

    def test_ordered_is_by_judge_then_pair(self):
        cohort = Cohort(size=3, slots=_slots(3, num_judges=2))
        # Recorded in an order that has nothing to do with the sort order.
        cohort.record({(2, 0, 1): "c1", (0, 1, 1): "a1"})
        cohort.record({(1, 2, 0): "b0", (0, 1, 0): "a0", (1, 2, 1): "b1", (2, 0, 0): "c0"})

        results, slots = cohort.ordered()

        assert slots == [(0, 1, 0), (1, 2, 0), (2, 0, 0), (0, 1, 1), (1, 2, 1), (2, 0, 1)]
        assert results == ["a0", "b0", "c0", "a1", "b1", "c1"]

    def test_record_failure_fills_the_slots_and_remembers_the_error(self):
        cohort = Cohort(size=2, slots=[(0, 1, 0), (1, 0, 0)])
        error = RuntimeError("judge exploded")
        cohort.payloads.extend(["a", "b"])
        cohort.record({(0, 1, 0): "r01"})

        cohort.record_failure(1, error, {(1, 0, 0): "neutral"})

        assert cohort.results == {(0, 1, 0): "r01", (1, 0, 0): "neutral"}
        assert cohort.failures == {1: error}
        assert cohort.is_complete

    def test_a_member_that_did_not_fail_is_not_in_failures(self):
        cohort = Cohort(size=2, slots=[(0, 1, 0), (1, 0, 0)])
        cohort.record({(0, 1, 0): "r01"})
        cohort.record_failure(1, RuntimeError("x"), {(1, 0, 0): "neutral"})

        assert list(cohort.failures) == [1]

    def test_the_first_failure_of_a_member_is_kept(self):
        cohort = Cohort(size=2, slots=[(0, 1, 0)])
        first, second = RuntimeError("first"), RuntimeError("second")

        cohort.record_failure(1, first, {(0, 1, 0): "neutral"})
        cohort.record_failure(1, second, {})

        assert cohort.failures == {1: first}


class TestCohortRegistry:
    def test_members_get_arrival_order_indices(self):
        registry = CohortRegistry()
        joined = _join_all(registry, "prompt", count=3, size=4)

        assert [j.member_idx for j in joined] == [0, 1, 2]
        assert len({id(j.cohort) for j in joined}) == 1
        assert joined[0].cohort.payloads == ["payload_0", "payload_1", "payload_2"]
        assert all(j.is_new for j in joined)
        assert registry.num_open == 1

    def test_the_cohort_is_opened_with_the_given_slots(self):
        registry = CohortRegistry()
        slots = _slots(4, num_judges=2)

        cohort = _join_all(registry, "prompt", count=1, size=4, slots=slots)[0].cohort

        assert cohort.size == 4
        assert cohort.slots == frozenset(slots)

    def test_full_cohort_is_dropped_from_the_open_cohorts(self):
        registry = CohortRegistry()
        joined = _join_all(registry, "prompt", count=4, size=4)

        assert joined[0].cohort.is_full
        assert registry.num_open == 0

    def test_arrival_after_full_starts_a_new_cohort(self):
        # The same prompt can occur twice in a batch, or again in a later epoch.
        registry = CohortRegistry()
        joined = _join_all(registry, "prompt", count=6, size=4)

        first = joined[0].cohort
        second = joined[4].cohort
        assert first is not second
        assert len(first.payloads) == 4
        assert len(second.payloads) == 2
        assert [j.member_idx for j in joined[4:]] == [0, 1]
        assert registry.num_open == 1

    def test_different_keys_do_not_share_a_cohort(self):
        registry = CohortRegistry()

        async def run():
            a = registry.join("a", "pa", size=2, slots=_slots(2))
            b = registry.join("b", "pb", size=2, slots=_slots(2))
            return a.cohort, b.cohort

        a, b = asyncio.run(run())
        assert a is not b
        assert registry.num_open == 2

    def test_each_member_gets_its_own_pending_future(self):
        registry = CohortRegistry()
        joined = _join_all(registry, "prompt", count=2, size=4)

        futures = [j.future for j in joined]
        assert futures[0] is not futures[1]
        assert not any(f.done() for f in futures)
        assert joined[0].cohort.futures == futures

    def test_cohort_completes_when_the_last_judge_finishes_not_the_last_arrival(self):
        # The regression this module exists for: the last member to arrive finishes judging before an
        # earlier one. The cohort must not be treated as complete until the slow member is recorded.
        registry = CohortRegistry()
        cohort = _join_all(registry, "prompt", count=3, size=3)[0].cohort

        for member_idx in (2, 0):  # The last arrival finishes first, then the first one
            cohort.record({slot: "r" for slot in cohort.owned_slots(member_idx)})
            assert not cohort.is_complete
        cohort.record({slot: "r" for slot in cohort.owned_slots(1)})  # The slowest member completes the cohort
        assert cohort.is_complete

    def test_cohort_remembers_its_key(self):
        registry = CohortRegistry()

        assert _join_all(registry, "prompt", count=1)[0].cohort.key == "prompt"


class TestRepeatedRequests:
    """A request for a rollout that an active cohort already has a member for is not given a second slot."""

    def test_a_repeat_takes_no_slot_and_gets_the_same_future(self):
        registry = CohortRegistry()
        original, repeat = _join_with_identities(registry, "prompt", [(3, "digest"), (3, "digest")])

        assert original.is_new
        assert not repeat.is_new
        assert repeat.cohort is original.cohort
        assert repeat.member_idx == original.member_idx
        assert repeat.future is original.future
        assert original.cohort.payloads == ["payload_0"]  # The repeat's payload was not added.
        assert len(original.cohort.futures) == 1

    def test_a_repeat_is_recognized_after_the_cohort_has_filled_up(self):
        # All members have arrived, so the cohort no longer accepts members, but it is still being judged.
        registry = CohortRegistry()
        joined = _join_with_identities(registry, "prompt", [(i, f"d{i}") for i in range(4)], size=4)
        assert registry.num_open == 0

        repeat = _join_with_identities(registry, "prompt", [(2, "d2")], size=4)[0]

        assert not repeat.is_new
        assert repeat.cohort is joined[0].cohort
        assert repeat.member_idx == 2
        assert repeat.future is joined[2].future
        assert registry.num_open == 0  # No new cohort was started for it.

    def test_a_repeat_does_not_use_up_a_slot_so_the_real_last_member_still_fills_the_cohort(self):
        # The failure this prevents: 7 members, a repeat of one of them, then the 8th member. If the repeat took a
        # slot, the cohort would be full without the 8th member, which would then wait in a cohort of its own.
        registry = CohortRegistry()
        identities = [(i, f"d{i}") for i in range(3)] + [(1, "d1")] + [(3, "d3")]  # 0, 1, 2, repeat of 1, 3
        joined = _join_with_identities(registry, "prompt", identities, size=4)

        assert [j.is_new for j in joined] == [True, True, True, False, True]
        assert joined[4].cohort is joined[0].cohort
        assert joined[4].cohort.is_full
        assert registry.num_open == 0

    def test_same_index_with_a_different_response_is_a_different_member(self):
        registry = CohortRegistry()
        first, second = _join_with_identities(registry, "prompt", [(3, "digest a"), (3, "digest b")])

        assert first.is_new and second.is_new
        assert second.member_idx == 1
        assert second.future is not first.future

    def test_same_response_with_a_different_index_is_a_different_member(self):
        registry = CohortRegistry()
        first, second = _join_with_identities(registry, "prompt", [(3, "digest"), (4, "digest")])

        assert first.is_new and second.is_new

    def test_without_an_identity_nothing_is_a_repeat(self):
        registry = CohortRegistry()
        joined = _join_with_identities(registry, "prompt", [None, None, None])

        assert all(j.is_new for j in joined)
        assert [j.member_idx for j in joined] == [0, 1, 2]

    def test_the_same_identity_under_a_different_key_is_a_different_member(self):
        registry = CohortRegistry()

        async def run():
            a = registry.join("task a", "pa", size=4, slots=_slots(4), identity=(3, "digest"))
            b = registry.join("task b", "pb", size=4, slots=_slots(4), identity=(3, "digest"))
            return a, b

        a, b = asyncio.run(run())
        assert a.is_new and b.is_new
        assert a.cohort is not b.cohort

    def test_repeats_are_matched_to_the_right_cohort_when_a_key_has_several_active_ones(self):
        # The same prompt twice in a batch: two cohorts with the same key are active at the same time.
        registry = CohortRegistry()
        first_round = _join_with_identities(registry, "prompt", [(i, f"first{i}") for i in range(4)], size=4)
        second_round = _join_with_identities(registry, "prompt", [(i, f"second{i}") for i in range(2)], size=4)
        assert registry.num_active == 2

        repeat_of_first = _join_with_identities(registry, "prompt", [(1, "first1")], size=4)[0]
        repeat_of_second = _join_with_identities(registry, "prompt", [(1, "second1")], size=4)[0]

        assert repeat_of_first.cohort is first_round[0].cohort
        assert repeat_of_first.future is first_round[1].future
        assert repeat_of_second.cohort is second_round[0].cohort
        assert repeat_of_second.future is second_round[1].future

    def test_a_repeat_after_the_cohort_was_finished_is_not_recognized(self):
        # Known limitation: once a cohort is finalized its members are forgotten, so a late repeat starts a new
        # cohort. It is only recognized while the original is still being handled.
        registry = CohortRegistry()
        slots = _slots(2)
        original = _join_with_identities(registry, "prompt", [(0, "d0"), (1, "d1")], size=2, slots=slots)[0]

        registry.finish(original.cohort)
        late = _join_with_identities(registry, "prompt", [(1, "d1")], size=2, slots=slots)[0]

        assert late.is_new
        assert late.cohort is not original.cohort


class TestFinish:
    def test_finish_removes_the_cohort_from_the_active_ones(self):
        registry = CohortRegistry()
        cohort = _join_all(registry, "prompt", count=4, size=4)[0].cohort
        assert registry.num_active == 1

        registry.finish(cohort)

        assert registry.num_active == 0

    def test_finish_is_safe_to_call_twice(self):
        registry = CohortRegistry()
        cohort = _join_all(registry, "prompt", count=4, size=4)[0].cohort

        registry.finish(cohort)
        registry.finish(cohort)

        assert registry.num_active == 0

    def test_finish_only_removes_the_given_cohort(self):
        registry = CohortRegistry()
        joined = _join_all(registry, "prompt", count=6, size=4)  # Two cohorts with the same key
        assert registry.num_active == 2

        registry.finish(joined[0].cohort)

        assert registry.num_active == 1
        assert registry.num_open == 1  # The second cohort is still filling.


class TestMemberJudgmentFailed:
    def test_it_is_a_runtime_error(self):
        assert issubclass(MemberJudgmentFailed, RuntimeError)
