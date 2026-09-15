# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""sink_window selection, the interval contract, and the token-budget clamp."""

from __future__ import annotations

import pytest

from responses_api_agents.osworld_agent.history_policy import (
    HistoryPolicySpec,
    HistoryPolicyState,
    plan_history,
)


def _walk(spec: HistoryPolicySpec, turns: int, *, max_images: int | None = None):
    """Run a whole trajectory and return one record per turn."""

    state = HistoryPolicyState()
    seen = []
    for completed in range(turns):
        plan = plan_history(spec, state, completed_turns=completed, max_images=max_images)
        seen.append(plan)
        state = plan.next_state
    return seen


def _one_based(plan) -> list[int]:
    return [index + 1 for index in plan.image_turns]


def test_sink_window_reproduces_the_requested_shape() -> None:
    """[1, ..., 7,8,9] then [1, ..., 8,9,10] with a one-turn sink."""

    spec = HistoryPolicySpec.sink_window(sink=1, low_water=4, high_water=4)
    plans = _walk(spec, 12)

    assert _one_based(plans[8]) == [1, 7, 8, 9]
    assert _one_based(plans[9]) == [1, 8, 9, 10]
    assert plans[8].image_intervals == ((0, 1), (6, 9))
    assert plans[9].image_intervals == ((0, 1), (7, 10))
    # The sink turn is never folded into text, at any depth.
    assert all(0 in plan.image_turns for plan in plans)


def test_sink_window_degenerates_to_one_interval_before_the_gap_opens() -> None:
    spec = HistoryPolicySpec.sink_window(sink=1, low_water=4, high_water=4)
    plans = _walk(spec, 12)

    for plan in plans[:4]:
        assert len(plan.image_intervals) == 1
        assert plan.image_window_start == 0
    for plan in plans[4:]:
        assert len(plan.image_intervals) == 2


def test_image_window_start_fails_closed_on_a_non_contiguous_set() -> None:
    """A consumer still reading the scalar must crash, not silently misread."""

    spec = HistoryPolicySpec.sink_window(sink=1, low_water=4, high_water=4)
    plan = _walk(spec, 10)[-1]

    assert len(plan.image_intervals) == 2
    with pytest.raises(ValueError, match="image_window_start is undefined"):
        _ = plan.image_window_start
    # The trailing-window start stays available for telemetry.
    assert plan.recent_window_start == plan.image_intervals[-1][0]


def test_low_and_high_water_count_total_images_including_the_sink() -> None:
    spec = HistoryPolicySpec.sink_window(sink=1, low_water=3, high_water=10)
    plans = _walk(spec, 40)
    counts = [len(plan.image_turns) for plan in plans]

    assert max(counts) == 10, "peak must equal high_water, with no hidden sink surcharge"
    compacted = [len(plan.image_turns) for plan in plans if plan.compaction_triggered]
    assert compacted and set(compacted) == {3}, "every compaction must land exactly on low_water"


@pytest.mark.parametrize(
    ("sink", "low", "high"),
    [(1, 4, 4), (1, 3, 10), (2, 4, 12), (3, 5, 5), (1, 2, 3)],
)
def test_sink_window_invariants_hold_for_a_long_trajectory(sink: int, low: int, high: int) -> None:
    spec = HistoryPolicySpec.sink_window(sink=sink, low_water=low, high_water=high)
    for plan in _walk(spec, 200):
        images = plan.image_turns
        assert len(images) <= high
        assert len(images) == len(set(images))
        assert list(images) == sorted(images)
        assert set(range(min(sink, plan.completed_turns + 1))) <= set(images)
        assert plan.completed_turns in images, "the current observation is always live"
        assert set(images) | set(plan.text_turns) == set(range(plan.completed_turns + 1))
        assert not plan.dropped_turns
        if plan.compaction_triggered and not plan.image_budget_clamped:
            assert len(images) == min(low, plan.completed_turns + 1)


def test_sink_window_rejects_a_low_water_that_leaves_no_recent_image() -> None:
    with pytest.raises(ValueError, match="low_water must exceed sink"):
        HistoryPolicySpec.sink_window(sink=3, low_water=3, high_water=10)
    with pytest.raises(ValueError, match="does not accept a sink"):
        HistoryPolicySpec(name="fixed", low_water=3, high_water=3, sink=1)


def test_sink_window_round_trips_through_config_and_rejects_stray_params() -> None:
    spec = HistoryPolicySpec.sink_window(sink=1, low_water=3, high_water=10)
    config = spec.to_config()

    assert config == {
        "schema_version": 1,
        "name": "sink_window",
        "params": {"sink": 1, "low_water": 3, "high_water": 10},
    }
    assert HistoryPolicySpec.from_mapping(config) == spec
    with pytest.raises(ValueError, match="must contain only sink, low_water and high_water"):
        HistoryPolicySpec.from_mapping(
            {"schema_version": 1, "name": "sink_window", "params": {"sink": 1, "low_water": 3}}
        )


def test_append_stability_follows_the_window_not_the_policy_name() -> None:
    """Only a hysteretic recent window appends between compactions."""

    sliding = HistoryPolicySpec.sink_window(sink=1, low_water=4, high_water=4)
    hysteretic = HistoryPolicySpec.sink_window(sink=1, low_water=3, high_water=10)

    assert sliding.to_contract()["supports_append_stable_intervals"] is False
    assert hysteretic.to_contract()["supports_append_stable_intervals"] is True
    # Unchanged for the two policies that shipped before sink_window.
    assert HistoryPolicySpec.fixed(3).to_contract()["supports_append_stable_intervals"] is False
    assert (
        HistoryPolicySpec.hysteresis(low_water=3, high_water=10).to_contract()[
            "supports_append_stable_intervals"
        ]
        is True
    )


def test_budget_clamp_shrinks_the_window_and_is_reported() -> None:
    spec = HistoryPolicySpec.hysteresis(low_water=3, high_water=10)
    state = HistoryPolicyState(image_window_start=0, compaction_epoch=0)

    unclamped = plan_history(spec, state, completed_turns=9)
    assert len(unclamped.image_turns) == 10
    assert unclamped.image_budget_clamped is False

    clamped = plan_history(spec, state, completed_turns=9, max_images=2)
    assert len(clamped.image_turns) == 2
    assert clamped.image_turns == (8, 9), "clamping keeps the most recent turns"
    assert clamped.image_budget_clamped is True
    assert clamped.telemetry()["snapshot_image_budget_clamped"] is True
    assert set(clamped.text_turns) == set(range(8))


def test_budget_clamp_preserves_the_sink() -> None:
    """A budget shrinks the recent window; it must not silently drop the sink."""

    spec = HistoryPolicySpec.sink_window(sink=2, low_water=4, high_water=12)
    state = HistoryPolicyState(image_window_start=0, compaction_epoch=0)

    clamped = plan_history(spec, state, completed_turns=19, max_images=3)
    assert clamped.image_turns == (0, 1, 19)
    assert clamped.image_budget_clamped is True


def test_budget_clamp_never_drops_the_current_observation() -> None:
    spec = HistoryPolicySpec.sink_window(sink=5, low_water=6, high_water=8)
    state = HistoryPolicyState(image_window_start=0, compaction_epoch=0)

    clamped = plan_history(spec, state, completed_turns=30, max_images=1)
    assert 30 in clamped.image_turns
    assert clamped.image_budget_clamped is True


def test_budget_clamp_is_a_no_op_when_the_plan_already_fits() -> None:
    spec = HistoryPolicySpec.fixed(3)
    state = HistoryPolicyState(image_window_start=0, compaction_epoch=0)

    generous = plan_history(spec, state, completed_turns=9, max_images=99)
    plain = plan_history(spec, state, completed_turns=9)
    assert generous.image_turns == plain.image_turns
    assert generous.image_budget_clamped is False
    assert generous.compaction_triggered == plain.compaction_triggered


def test_plan_history_rejects_a_non_positive_budget() -> None:
    spec = HistoryPolicySpec.fixed(3)
    with pytest.raises(ValueError, match="max_images must be a positive integer"):
        plan_history(spec, HistoryPolicyState(), completed_turns=3, max_images=0)


def test_randomised_policies_never_violate_the_selection_invariants() -> None:
    """Sweep the parameter space a fixed table cannot cover.

    Seeded so a failure is reproducible from the printed case rather than
    flaky. Every invariant here is one a wrong plan would break silently:
    a duplicated turn inflates the image count past the budget, a missing
    current observation blinds the model, and a gap in the text/image union
    would drop a turn the docstring promises is never dropped.
    """

    import random

    rng = random.Random(20260904)
    for _ in range(300):
        kind = rng.choice(["fixed", "hysteresis", "sink_window"])
        if kind == "fixed":
            keep = rng.randint(1, 12)
            spec = HistoryPolicySpec.fixed(keep)
        elif kind == "hysteresis":
            low = rng.randint(1, 8)
            spec = HistoryPolicySpec.hysteresis(low_water=low, high_water=low + rng.randint(1, 12))
        else:
            sink = rng.randint(1, 4)
            low = sink + rng.randint(1, 6)
            spec = HistoryPolicySpec.sink_window(
                sink=sink, low_water=low, high_water=low + rng.choice([0, 1, 5, 12])
            )
        budget = rng.choice([None, None, 1, 2, 3, 7])
        turns = rng.randint(1, 120)

        state = HistoryPolicyState()
        for completed in range(turns):
            plan = plan_history(spec, state, completed_turns=completed, max_images=budget)
            case = f"{spec.to_config()} budget={budget} n={completed}"
            images = plan.image_turns
            assert images == tuple(sorted(set(images))), case
            assert completed in images, f"current observation missing: {case}"
            assert set(images) | set(plan.text_turns) == set(range(completed + 1)), case
            assert not plan.dropped_turns, case
            if budget is not None:
                assert len(images) <= max(budget, spec.sink + 1), case
            else:
                assert len(images) <= spec.high_water, case
            if spec.name == "sink_window":
                assert set(range(min(spec.sink, completed + 1))) <= set(images), case
            else:
                assert len(plan.image_intervals) == 1, case
            for (_, previous_high), (next_low, _) in zip(plan.image_intervals, plan.image_intervals[1:]):
                assert previous_high < next_low, f"intervals must be disjoint and ordered: {case}"
            state = plan.next_state
