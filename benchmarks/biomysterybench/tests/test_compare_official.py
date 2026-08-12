# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from benchmarks.biomysterybench.compare_official import ComparisonError, compare_rollouts, main, render_markdown


REVISION = "test-revision"


def _expected() -> list[dict]:
    return [
        {"id": "easy-1", "human_solvable": "yes", "dataset_revision": REVISION},
        {"id": "easy-2", "human_solvable": "yes", "dataset_revision": REVISION},
        {"id": "hard-1", "human_solvable": "no", "dataset_revision": REVISION},
    ]


def _rollouts() -> list[dict]:
    scores = {"easy-1": [1, 1], "easy-2": [1, 0], "hard-1": [0, 1]}
    rows = []
    for task_index, expected in enumerate(_expected()):
        for rollout_index, reward in enumerate(scores[expected["id"]]):
            rows.append(
                {
                    **expected,
                    "_ng_task_index": task_index,
                    "_ng_rollout_index": rollout_index,
                    "reward": reward,
                    "mask_sample": False,
                    "sandbox_failed": False,
                    "container_timed_out": False,
                    "agent_timed_out": False,
                    "agent_failed": False,
                    "invalid_judge_response": False,
                    "response": {"model": "candidate-model"},
                }
            )
    return rows


def test_computes_subset_accuracy_and_consistency() -> None:
    rows = _rollouts()
    rows[0]["_ng_policy_evidence_normalizations"] = ["legacy_anyterminal_missing_agent_failed"]
    comparison = compare_rollouts(rows, _expected(), repeats=2)

    easy = comparison["subsets"]["human-solvable"]
    hard = comparison["subsets"]["human-difficult"]
    assert easy["correct_rollouts"] == 3
    assert easy["accuracy_pct"] == 75.0
    assert easy["correct_rollouts_per_task_histogram"] == {"0": 0, "1": 1, "2": 1}
    assert easy["tasks_solved_at_least_once"] == 2
    assert easy["pass_at_5_pct"] == 100.0
    assert hard["accuracy_pct"] == 50.0
    assert comparison["policy_models"] == ["candidate-model"]
    assert comparison["overall_accuracy_pct"] == pytest.approx(400 / 6)
    assert comparison["policy_evidence_normalizations"] == {"legacy_anyterminal_missing_agent_failed": 1}
    assert [row["model"] for row in comparison["published_comparisons"]] == [
        "Claude Haiku 4.5",
        "Claude Sonnet 4.6",
        "Claude Opus 4.6",
        "Claude Opus 4.7",
        "Claude Mythos Preview",
        "Claude Opus 4.8",
        "Claude Mythos 5",
    ]
    assert comparison["published_comparisons"][2]["human_solvable_delta_percentage_points"] == pytest.approx(-2.4)
    assert "Anthropic" in render_markdown(comparison)
    assert "Claude Mythos 5" in render_markdown(comparison)
    assert "legacy_anyterminal_missing_agent_failed" in render_markdown(comparison)


def test_official_denominators_reproduce_reported_rounding() -> None:
    expected = []
    rows = []
    split_specs = (("yes", 76, 294), ("no", 23, 27))
    task_index = 0
    for split, task_count, correct_rollouts in split_specs:
        for split_task_index in range(task_count):
            task_id = f"{split}-{split_task_index}"
            expected_row = {"id": task_id, "human_solvable": split, "dataset_revision": REVISION}
            expected.append(expected_row)
            for rollout_index in range(5):
                split_rollout_index = split_task_index * 5 + rollout_index
                rows.append(
                    {
                        **expected_row,
                        "_ng_task_index": task_index,
                        "_ng_rollout_index": rollout_index,
                        "reward": int(split_rollout_index < correct_rollouts),
                        "mask_sample": False,
                        "sandbox_failed": False,
                        "container_timed_out": False,
                        "agent_timed_out": False,
                        "agent_failed": False,
                        "invalid_judge_response": False,
                    }
                )
            task_index += 1

    comparison = compare_rollouts(rows, expected)
    assert comparison["subsets"]["human-solvable"]["accuracy_pct_rounded_1dp"] == 77.4
    assert comparison["subsets"]["human-solvable"]["rounded_score_matches_official"] is True
    assert comparison["subsets"]["human-difficult"]["accuracy_pct_rounded_1dp"] == 23.5
    assert comparison["subsets"]["human-difficult"]["rounded_score_matches_official"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda rows: rows.pop(), "missing 1 rollout keys"),
        (lambda rows: rows.append(deepcopy(rows[0])), "duplicate rollout key"),
        (lambda rows: rows[0].update(mask_sample=True), "mask_sample must be explicitly false"),
        (lambda rows: rows[0].update(invalid_judge_response=True), "invalid_judge_response must be explicitly false"),
        (
            lambda rows: rows[0].update(agent_metrics={"agent_failed": True}),
            "agent_metrics.agent_failed must be explicitly false",
        ),
        (
            lambda rows: rows[0].update(_ng_policy_evidence_normalizations="not-a-list"),
            "_ng_policy_evidence_normalizations must be a list of strings",
        ),
        (lambda rows: rows[0].update(dataset_revision="wrong"), "dataset_revision='wrong'"),
        (lambda rows: rows[0].update(reward=0.5), "reward must be binary"),
    ],
)
def test_rejects_invalid_official_evidence(mutation, message: str) -> None:
    rows = _rollouts()
    mutation(rows)
    with pytest.raises(ComparisonError, match=message):
        compare_rollouts(rows, _expected(), repeats=2)


def test_rejects_empty_rollout_file() -> None:
    with pytest.raises(ComparisonError, match="no completed rows"):
        compare_rollouts([], _expected(), repeats=2)


def test_cli_reports_validation_failure_without_traceback(tmp_path, capsys) -> None:
    expected_path = tmp_path / "expected.jsonl"
    rollout_path = tmp_path / "rollouts.jsonl"
    expected_path.write_text("{}\n")
    rollout_path.write_text("")

    with pytest.raises(SystemExit) as error:
        main([str(rollout_path), "--expected", str(expected_path)])

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert stderr.startswith("error: ")
    assert "Traceback" not in stderr
