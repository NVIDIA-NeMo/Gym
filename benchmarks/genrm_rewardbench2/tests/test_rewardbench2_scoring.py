# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""REWARDBENCH2 scoring tests."""

import math

import pytest

from benchmarks.genrm_rewardbench2.metrics import SUBSETS, compute_metrics, metric_name, score


def rollout(ranking: float | None, *, repeat: int = 0, **metadata: object) -> dict:
    return {"verifier_metadata": metadata, "predicted_ranking": ranking, "_ng_rollout_index": repeat}


def comparison(pair_id: str, orders: tuple, label: str, shared: dict) -> list[list[dict]]:
    """Both candidate orders of one comparison, as (first-response ranking, flipped ranking)."""
    forward, flipped = orders
    flipped_label = "B>A" if label == "A>B" else label
    return [
        [rollout(forward, pair_id=pair_id, label=label, flipped=False, **shared)],
        [rollout(flipped, pair_id=f"{pair_id}-flipped", label=flipped_label, flipped=True, **shared)],
    ]


def best_of_four(example: str, subset: str, rankings: list[tuple]) -> list[list[dict]]:
    """A best-of-4 record: one chosen response against each distractor, no correct-answer pairs."""
    shared = {"example_id": example, "subset": subset, "comparisons": len(rankings), "correct_pairs": 0}
    return [
        task
        for index, orders in enumerate(rankings)
        for task in comparison(f"{example}:0:{index}", orders, "A>B", shared)
    ]


def ties_record(
    example: str, rejected: list[tuple], correct: list[tuple], *, expected_correct: int | None = None
) -> list[list[dict]]:
    """A Ties record: chosen-versus-rejected comparisons plus chosen-versus-chosen ones.

    `expected_correct` declares more correct-answer pairs than are supplied, to model a record
    whose chosen-versus-chosen rollouts went missing.
    """
    shared = {
        "example_id": example,
        "subset": "Ties",
        "comparisons": len(rejected),
        "correct_pairs": len(correct) if expected_correct is None else expected_correct,
    }
    tasks = [
        task
        for index, orders in enumerate(rejected)
        for task in comparison(f"{example}:0:{index}", orders, "A>B", shared)
    ]
    for index, orders in enumerate(correct):
        tasks += comparison(f"{example}:c0:c{index + 1}", orders, "A=B", shared)
    return tasks


WIN = (1, 6)  # mirrored mean 1.0, margin +2.5
TIE = (1, 1)  # mirrored mean 3.5, margin 0.0
LOSS = (6, 1)  # mirrored mean 6.0, margin -2.5
NARROW = (3, 4)  # mirrored mean 3.0, margin +0.5


def test_record_needs_every_comparison_to_win() -> None:
    metrics = compute_metrics(
        best_of_four("Math:1", "Math", [WIN, WIN, WIN])
        + best_of_four("Math:2", "Math", [WIN, WIN, LOSS])
        + best_of_four("Safety:1", "Safety", [WIN, WIN, WIN])
    )
    assert metrics["metrics"]["math"] == 0.5
    assert metrics["metrics"]["safety"] == 1.0
    assert metrics["counts"]["math"] == {"total": 2, "correct": 1, "partial": 0}
    assert metrics["incomplete_records"] == 0
    assert metrics["tied_comparisons"] == 0


@pytest.mark.parametrize(
    ("ties", "expected"),
    # "two-way ties as partial credit of 0.5, three-way ties as 0.33, and four-way ties as 0.25".
    [(0, 1.0), (1, 1 / 2), (2, 1 / 3), (3, 1 / 4)],
)
def test_tie_ladder_matches_the_paper(ties: int, expected: float) -> None:
    metrics = compute_metrics(best_of_four("Math:1", "Math", [TIE] * ties + [WIN] * (3 - ties)))
    assert metrics["metrics"]["math"] == pytest.approx(expected)
    assert metrics["tied_comparisons"] == ties
    assert metrics["counts"]["math"] == {"total": 1, "correct": int(ties == 0), "partial": int(ties > 0)}


def test_a_loss_forfeits_the_record_even_alongside_ties() -> None:
    metrics = compute_metrics(best_of_four("Math:1", "Math", [TIE, TIE, LOSS]))
    assert metrics["metrics"]["math"] == 0.0
    assert metrics["tied_comparisons"] == 2


@pytest.mark.parametrize(
    ("orders", "expected"),
    [
        # Both orders agree that the chosen response is better.
        ((1, 6), 1.0),
        # One order unparseable: the other decides, and a lone integer never hits the midpoint.
        ((1, None), 1.0),
        ((None, 6), 1.0),
        ((6, None), 0.0),
        ((None, 1), 0.0),
        ((None, None), 0.0),
        # The same ranking both ways is pure position bias and no net preference, so a tie.
        ((1, 1), 0.5),
        ((3, 3), 0.5),
        ((6, 6), 0.5),
        # Directionally inconsistent, but the magnitude nets out for the chosen response.
        ((1, 2), 1.0),
        ((2, 3), 1.0),
        # ...and against it.
        ((3, 2), 0.0),
        ((6, 5), 0.0),
        # Both orders agree that the rejected response is better.
        ((6, 1), 0.0),
    ],
)
def test_mirrored_average_combination_rule(orders, expected) -> None:
    metrics = compute_metrics(best_of_four("Math:1", "Math", [orders]))
    assert metrics["metrics"]["math"] == pytest.approx(expected)
    assert metrics["no_answer"] == (orders[0] is None) + (orders[1] is None)


def ties_pair(rejected: list[tuple], correct: list[tuple]) -> dict:
    """One reference record (a single correct answer) alongside its tied record."""
    return compute_metrics(ties_record("Ties:ref:0", rejected, []) + ties_record("Ties:tied:0", rejected, correct))


def test_ties_rewards_a_correctness_margin_wider_than_the_style_margin() -> None:
    # Confident about correct-versus-incorrect (2.5), barely separates the correct answers (0.5).
    metrics = ties_pair([WIN, WIN], [NARROW])["metrics"]
    assert metrics["ties_reference_accuracy"] == 1.0
    assert metrics["ties_tied_accuracy"] == 1.0
    assert metrics["ties_correctness_preferred"] == 1.0
    assert metrics["ties_correctness_preferred_hard"] == 1.0
    assert metrics["ties_margin_score"] == pytest.approx(math.tanh(2.5 / 0.5 - 1))
    # Upstream's weights total 1.01, so a flawless judge tops out just above 1.
    assert metrics["ties"] == pytest.approx(1.0 + 0.01 * math.tanh(4))


def test_ties_penalises_separating_correct_answers_more_than_incorrect_ones() -> None:
    # Accurate, but it ranks the two correct answers further apart (2.5) than correct from
    # incorrect (0.5). Accuracy alone would call this a perfect record.
    metrics = ties_pair([NARROW, NARROW], [WIN])["metrics"]
    assert metrics["ties_tied_accuracy"] == 1.0
    assert metrics["ties_correctness_preferred"] == 0.0
    assert metrics["ties_correctness_preferred_hard"] == 0.0
    assert metrics["ties"] == pytest.approx(0.60 + 0.01 * math.tanh(0.5 / 2.5 - 1))


def test_ties_accuracy_fails_when_a_correct_answer_loses() -> None:
    metrics = ties_pair([WIN, LOSS], [NARROW])["metrics"]
    assert metrics["ties_reference_accuracy"] == 0.0
    assert metrics["ties_tied_accuracy"] == 0.0
    assert metrics["ties_correctness_preferred"] == 0.0
    assert metrics["ties"] == pytest.approx(0.01 * math.tanh(-2.5 / 0.5 - 1))


def test_ties_without_correct_answer_pairs_loses_the_margin_terms() -> None:
    # The chosen-versus-chosen rollouts went missing, so the correct-answer spread is unmeasured
    # and only the accuracy half of the score survives.
    result = compute_metrics(
        ties_record("Ties:ref:0", [WIN, WIN], []) + ties_record("Ties:tied:0", [WIN, WIN], [], expected_correct=1)
    )
    metrics = result["metrics"]
    assert metrics["ties_tied_accuracy"] == 1.0
    assert metrics["ties_correctness_preferred"] == 0.0
    assert metrics["ties_correctness_preferred_hard"] == 0.0
    assert metrics["ties"] == pytest.approx(0.60)
    assert result["incomplete_records"] == 1


def test_ties_counts_report_accuracy() -> None:
    counts = ties_pair([WIN, WIN], [NARROW])["counts"]["ties"]
    assert counts == {"total": 2, "correct": 2, "partial": 0}


def test_overall_is_the_unweighted_average_over_the_six_domains() -> None:
    metrics = compute_metrics(
        # Three Math records, two of them correct, and no other domain present.
        best_of_four("Math:1", "Math", [WIN, WIN, WIN])
        + best_of_four("Math:2", "Math", [WIN, WIN, WIN])
        + best_of_four("Math:3", "Math", [WIN, WIN, LOSS])
    )
    assert metrics["metrics"]["math"] == pytest.approx(2 / 3)
    # Per domain, then averaged over six: absent domains contribute zero.
    assert metrics["metrics"]["overall"] == pytest.approx((2 / 3) / 6)
    assert metrics["metrics"]["subset_coverage"] == pytest.approx(1 / 6)


def test_full_coverage_averages_the_six_domains() -> None:
    tasks = [
        task for subset in SUBSETS if subset != "Ties" for task in best_of_four(f"{subset}:1", subset, [WIN, WIN, WIN])
    ]
    metrics = compute_metrics(
        tasks + ties_record("Ties:ref:0", [WIN], []) + ties_record("Ties:tied:0", [WIN], [NARROW])
    )
    assert metrics["metrics"]["subset_coverage"] == 1.0
    assert metrics["metrics"]["overall"] == pytest.approx((5 + 1.0 + 0.01 * math.tanh(4)) / 6)


def test_missing_order_forfeits_the_record() -> None:
    tasks = best_of_four("Math:1", "Math", [WIN, WIN, WIN])
    del tasks[1]  # Drop the flipped order of the first comparison.
    metrics = compute_metrics(tasks)
    assert metrics["metrics"]["math"] == 0.0
    assert metrics["incomplete_comparisons"] == 1
    assert metrics["incomplete_records"] == 0


def test_missing_comparison_forfeits_the_record() -> None:
    tasks = best_of_four("Math:1", "Math", [WIN, WIN, WIN])
    metrics = compute_metrics(tasks[:4])
    assert metrics["metrics"]["math"] == 0.0
    assert metrics["counts"]["math"] == {"total": 1, "correct": 0, "partial": 0}
    assert metrics["incomplete_records"] == 1
    assert metrics["incomplete_comparisons"] == 0


def test_repeats_are_scored_independently() -> None:
    shared = {"example_id": "Math:1", "subset": "Math", "comparisons": 1, "correct_pairs": 0}
    tasks = [
        [
            rollout(ranking, repeat=repeat, pair_id=pair_id, label=label, flipped=flipped, **shared)
            for repeat, ranking in enumerate(rankings)
        ]
        for pair_id, label, flipped, rankings in (
            ("Math:1:0:0", "A>B", False, (1, 6)),
            ("Math:1:0:0-flipped", "B>A", True, (6, 1)),
        )
    ]
    metrics = compute_metrics(tasks)
    assert metrics["metrics"]["math"] == 0.5
    assert metrics["counts"]["math"] == {"total": 2, "correct": 1, "partial": 0}


def test_ties_repeats_are_paired_within_their_own_repeat() -> None:
    """A `tied:N` record pairs with the `ref:N` of its own repeat, never across repeats.

    Both repeats hold the same zero spread, and one wins its chosen-versus-rejected comparison
    while the other loses it, so every component averages the two instead of taking either alone.
    """
    tasks = []
    for example, correct_pairs in (("Ties:ref:0", 0), ("Ties:tied:0", 1)):
        shared = {"example_id": example, "subset": "Ties", "comparisons": 1, "correct_pairs": correct_pairs}
        comparisons = [(f"{example}:0:0", "A>B", (WIN, LOSS))]
        if correct_pairs:
            comparisons.append((f"{example}:c0:c1", "A=B", (TIE, TIE)))
        for pair_id, label, repeats in comparisons:
            flipped_label = "B>A" if label == "A>B" else label
            tasks.append(
                [
                    rollout(orders[0], repeat=repeat, pair_id=pair_id, label=label, flipped=False, **shared)
                    for repeat, orders in enumerate(repeats)
                ]
            )
            tasks.append(
                [
                    rollout(
                        orders[1],
                        repeat=repeat,
                        pair_id=f"{pair_id}-flipped",
                        label=flipped_label,
                        flipped=True,
                        **shared,
                    )
                    for repeat, orders in enumerate(repeats)
                ]
            )

    result = compute_metrics(tasks)
    assert result["metrics"]["ties_reference_accuracy"] == 0.5
    assert result["metrics"]["ties_tied_accuracy"] == 0.5
    assert result["metrics"]["ties_correctness_preferred"] == 0.5
    assert result["metrics"]["ties_correctness_preferred_hard"] == 0.5
    # The two repeats saturate tanh in opposite directions, so the tie-break cancels.
    assert result["metrics"]["ties_margin_score"] == 0.0
    assert result["metrics"]["ties"] == pytest.approx(0.5)
    assert result["counts"]["ties"] == {"total": 4, "correct": 2, "partial": 0}


def test_duplicate_orientations_are_rejected() -> None:
    row = rollout(1, pair_id="Math:1:0:0", label="A>B", flipped=False, example_id="Math:1", subset="Math")
    with pytest.raises(ValueError, match="Duplicate"):
        compute_metrics([[row], [row]])


def test_same_order_twice_is_rejected() -> None:
    shared = {"example_id": "Math:1", "subset": "Math"}
    tasks = [
        [rollout(1, pair_id="Math:1:0:0", label="A>B", flipped=False, **shared)],
        [rollout(1, pair_id="Math:1:0:0-flipped", label="A>B", flipped=False, **shared)],
    ]
    with pytest.raises(ValueError, match="both candidate orders"):
        compute_metrics(tasks)


@pytest.mark.parametrize(
    ("metadata", "ranking", "expected"),
    [
        ({"label": "A>B"}, 3, 1),
        ({"label": "B>A"}, 3, 0),
        ({"label": "A>B"}, 4, 0),
        ({"label": "B>A"}, 4, 1),
        # Two equally correct answers: the reward goes to declining to separate them.
        ({"label": "A=B"}, 3, 1),
        ({"label": "A=B"}, 4, 1),
        ({"label": "A=B"}, 1, 0),
        ({"label": "A=B"}, 6, 0),
    ],
)
def test_per_row_reward(metadata, ranking, expected) -> None:
    assert score(metadata, ranking) == expected


def test_unparseable_ranking_earns_no_reward() -> None:
    assert score({"label": "A>B"}, None) == 0.0
    assert score({"label": "A=B"}, None) == 0.0


def test_metric_names_are_key_safe() -> None:
    assert [metric_name(subset) for subset in SUBSETS] == [
        "factuality",
        "focus",
        "math",
        "precise_if",
        "safety",
        "ties",
    ]
