# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a complete BioMysteryBench run and compare it with published scores."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Iterable

from benchmarks.biomysterybench.prepare import BENCHMARK_DIR, RELEASES


OFFICIAL_RELEASE = RELEASES["official-99"]
OFFICIAL_REPEATS = 5
OFFICIAL_TARGETS = {
    "yes": {
        "name": "human-solvable",
        "task_count": 76,
        "reported_accuracy_pct": 77.4,
    },
    "no": {
        "name": "human-difficult",
        "task_count": 23,
        "reported_accuracy_pct": 23.5,
    },
}
OFFICIAL_SOURCE = "https://www.anthropic.com/research/Evaluating-Claude-For-Bioinformatics-With-BioMysteryBench"
SYSTEM_CARD_SOURCE = (
    "https://www-cdn.anthropic.com/2f9323abbcc4abe219577539efe19a623c9ca2bd/"
    "Claude%20Fable%205%20%26%20Claude%20Mythos%205%20System%20Card.pdf"
)
PUBLISHED_BASELINES = [
    {
        "model": "Claude Haiku 4.5",
        "human_solvable_pct": 36.8,
        "human_difficult_pct": 5.2,
        "source": OFFICIAL_SOURCE,
    },
    {
        "model": "Claude Sonnet 4.6",
        "human_solvable_pct": 71.8,
        "human_difficult_pct": 19.1,
        "source": OFFICIAL_SOURCE,
    },
    {
        "model": "Claude Opus 4.6",
        "human_solvable_pct": 77.4,
        "human_difficult_pct": 23.5,
        "source": OFFICIAL_SOURCE,
    },
    {
        "model": "Claude Opus 4.7",
        "human_solvable_pct": 78.9,
        "human_difficult_pct": 27.0,
        "source": OFFICIAL_SOURCE,
    },
    {
        "model": "Claude Mythos Preview",
        "human_solvable_pct": 82.6,
        "human_difficult_pct": 29.6,
        "source": OFFICIAL_SOURCE,
    },
    {
        "model": "Claude Opus 4.8",
        "human_solvable_pct": 80.4,
        "human_difficult_pct": 40.0,
        "source": SYSTEM_CARD_SOURCE,
    },
    {
        "model": "Claude Mythos 5",
        "human_solvable_pct": 83.9,
        "human_difficult_pct": 46.1,
        "source": SYSTEM_CARD_SOURCE,
    },
]
DEFAULT_EXPECTED = BENCHMARK_DIR / "data" / OFFICIAL_RELEASE.output_filename
REQUIRED_FALSE_FIELDS = ("mask_sample", "sandbox_failed", "container_timed_out", "agent_timed_out", "agent_failed")


class ComparisonError(ValueError):
    """Raised when rollout evidence is not a complete, valid official run."""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ComparisonError(f"{path}:{line_number}: invalid JSON: {error}") from error
            if not isinstance(row, dict):
                raise ComparisonError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def _one_decimal(value: float) -> float:
    return float(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def _expected_by_id(expected_rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(expected_rows):
        task_id = row.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise ComparisonError(f"expected dataset row {index} has no valid id")
        if task_id in expected:
            raise ComparisonError(f"expected dataset contains duplicate id {task_id!r}")
        expected[task_id] = {
            "task_index": index,
            "human_solvable": row.get("human_solvable"),
            "dataset_revision": row.get("dataset_revision"),
        }
    return expected


def validate_official_expected_dataset(expected_rows: list[dict[str, Any]]) -> None:
    """Prove the reference JSONL is the exact 99-task release used in the article."""

    expected = _expected_by_id(expected_rows)
    split_counts = Counter(row["human_solvable"] for row in expected.values())
    revisions = {row["dataset_revision"] for row in expected.values()}
    if len(expected) != OFFICIAL_RELEASE.expected_task_count:
        raise ComparisonError(
            f"expected dataset has {len(expected)} tasks; official release requires {OFFICIAL_RELEASE.expected_task_count}"
        )
    if dict(split_counts) != OFFICIAL_RELEASE.expected_split_counts:
        raise ComparisonError(
            f"expected dataset has split counts {dict(split_counts)}; "
            f"official release requires {OFFICIAL_RELEASE.expected_split_counts}"
        )
    if revisions != {OFFICIAL_RELEASE.revision}:
        raise ComparisonError(
            f"expected dataset revisions are {sorted(str(value) for value in revisions)}; "
            f"official release requires {OFFICIAL_RELEASE.revision}"
        )


def compare_rollouts(
    rollout_rows: list[dict[str, Any]],
    expected_rows: list[dict[str, Any]],
    *,
    repeats: int = OFFICIAL_REPEATS,
) -> dict[str, Any]:
    """Validate exact task/repeat coverage and compute article-compatible subset accuracy."""

    if repeats < 1:
        raise ComparisonError("repeats must be at least 1")
    expected = _expected_by_id(expected_rows)
    if not rollout_rows:
        raise ComparisonError("rollout file contains no completed rows")

    errors: list[str] = []
    seen: set[tuple[str, int]] = set()
    rewards: dict[str, list[float]] = defaultdict(list)
    policy_models: set[str] = set()
    policy_evidence_normalizations: Counter[str] = Counter()

    for line_number, row in enumerate(rollout_rows, start=1):
        label = f"rollout row {line_number}"
        task_id = row.get("id")
        if task_id not in expected:
            errors.append(f"{label}: unknown task id {task_id!r}")
            continue
        expected_task = expected[task_id]

        rollout_index = row.get("_ng_rollout_index")
        if isinstance(rollout_index, bool) or not isinstance(rollout_index, int):
            errors.append(f"{label} ({task_id}): invalid _ng_rollout_index {rollout_index!r}")
            continue
        if not 0 <= rollout_index < repeats:
            errors.append(f"{label} ({task_id}): rollout index {rollout_index} is outside 0..{repeats - 1}")
        key = (task_id, rollout_index)
        if key in seen:
            errors.append(f"{label}: duplicate rollout key {key!r}")
        seen.add(key)

        for field in ("human_solvable", "dataset_revision"):
            if row.get(field) != expected_task[field]:
                errors.append(f"{label} ({task_id}): {field}={row.get(field)!r}; expected {expected_task[field]!r}")
        if row.get("_ng_task_index") != expected_task["task_index"]:
            errors.append(
                f"{label} ({task_id}): _ng_task_index={row.get('_ng_task_index')!r}; "
                f"expected {expected_task['task_index']}"
            )

        for field in REQUIRED_FALSE_FIELDS:
            if row.get(field) is not False:
                errors.append(f"{label} ({task_id}): {field} must be explicitly false, got {row.get(field)!r}")
        agent_metrics = row.get("agent_metrics")
        if isinstance(agent_metrics, dict):
            for field in REQUIRED_FALSE_FIELDS:
                if agent_metrics.get(field) is not False:
                    errors.append(
                        f"{label} ({task_id}): agent_metrics.{field} must be explicitly false, "
                        f"got {agent_metrics.get(field)!r}"
                    )
        if row.get("invalid_judge_response") is not False:
            errors.append(
                f"{label} ({task_id}): invalid_judge_response must be explicitly false, "
                f"got {row.get('invalid_judge_response')!r}"
            )

        normalizations = row.get("_ng_policy_evidence_normalizations", [])
        if not isinstance(normalizations, list) or not all(isinstance(value, str) for value in normalizations):
            errors.append(
                f"{label} ({task_id}): _ng_policy_evidence_normalizations must be a list of strings, "
                f"got {normalizations!r}"
            )
        else:
            policy_evidence_normalizations.update(normalizations)

        reward = row.get("reward")
        if isinstance(reward, bool) or reward not in (0, 1, 0.0, 1.0):
            errors.append(f"{label} ({task_id}): reward must be binary, got {reward!r}")
        else:
            rewards[task_id].append(float(reward))

        response = row.get("response")
        if isinstance(response, dict):
            policy_model = response.get("model")
            if isinstance(policy_model, str) and policy_model:
                policy_models.add(policy_model)

    expected_keys = {(task_id, rollout_index) for task_id in expected for rollout_index in range(repeats)}
    missing_keys = sorted(expected_keys - seen)
    extra_keys = sorted(seen - expected_keys)
    if missing_keys:
        errors.append(f"missing {len(missing_keys)} rollout keys; first: {missing_keys[:5]}")
    if extra_keys:
        errors.append(f"found {len(extra_keys)} unexpected rollout keys; first: {extra_keys[:5]}")
    expected_total = len(expected) * repeats
    if len(rollout_rows) != expected_total:
        errors.append(f"found {len(rollout_rows)} rollout rows; expected exactly {expected_total}")

    if errors:
        detail = "\n- ".join(errors[:20])
        suffix = f"\n- ... and {len(errors) - 20} more" if len(errors) > 20 else ""
        raise ComparisonError(f"run is not valid evidence for an official comparison:\n- {detail}{suffix}")

    subset_task_ids: dict[str, list[str]] = defaultdict(list)
    for task_id, task in expected.items():
        subset_task_ids[task["human_solvable"]].append(task_id)

    subsets: dict[str, dict[str, Any]] = {}
    for split, task_ids in sorted(subset_task_ids.items()):
        target = OFFICIAL_TARGETS.get(split, {})
        task_correct = {task_id: int(sum(rewards[task_id])) for task_id in task_ids}
        rollout_count = len(task_ids) * repeats
        correct_rollouts = sum(task_correct.values())
        tasks_solved_at_least_once = sum(correct > 0 for correct in task_correct.values())
        accuracy_pct = 100.0 * correct_rollouts / rollout_count
        reported_target = target.get("reported_accuracy_pct")
        subsets[target.get("name", split)] = {
            "split_value": split,
            "task_count": len(task_ids),
            "rollout_count": rollout_count,
            "correct_rollouts": correct_rollouts,
            "accuracy_pct": accuracy_pct,
            "accuracy_pct_rounded_1dp": _one_decimal(accuracy_pct),
            "tasks_solved_at_least_once": tasks_solved_at_least_once,
            "pass_at_5_pct": 100.0 * tasks_solved_at_least_once / len(task_ids),
            "official_reported_accuracy_pct": reported_target,
            "delta_percentage_points": accuracy_pct - reported_target if reported_target is not None else None,
            "rounded_score_matches_official": (
                _one_decimal(accuracy_pct) == reported_target if reported_target is not None else None
            ),
            "correct_rollouts_per_task_histogram": {
                str(correct): sum(value == correct for value in task_correct.values())
                for correct in range(repeats + 1)
            },
        }

    correct_total = sum(sum(task_rewards) for task_rewards in rewards.values())
    candidate_by_subset = {
        "human_solvable_pct": subsets["human-solvable"]["accuracy_pct"],
        "human_difficult_pct": subsets["human-difficult"]["accuracy_pct"],
    }
    published_comparisons = [
        {
            **baseline,
            "human_solvable_delta_percentage_points": (
                candidate_by_subset["human_solvable_pct"] - baseline["human_solvable_pct"]
            ),
            "human_difficult_delta_percentage_points": (
                candidate_by_subset["human_difficult_pct"] - baseline["human_difficult_pct"]
            ),
        }
        for baseline in PUBLISHED_BASELINES
    ]

    return {
        "benchmark": "Anthropic/BioMysteryBench-full",
        "dataset_revision": next(iter(expected.values()))["dataset_revision"],
        "task_count": len(expected),
        "repeats_per_task": repeats,
        "policy_models": sorted(policy_models),
        "rollout_count": len(rollout_rows),
        "correct_rollouts": int(correct_total),
        "overall_accuracy_pct": 100.0 * correct_total / len(rollout_rows),
        "policy_evidence_normalizations": dict(sorted(policy_evidence_normalizations.items())),
        "subsets": subsets,
        "published_comparisons": published_comparisons,
        "official_source": OFFICIAL_SOURCE,
    }


def render_markdown(comparison: dict[str, Any]) -> str:
    lines = [
        "# BioMysteryBench official-score comparison",
        "",
        f"- Dataset revision: `{comparison['dataset_revision']}`",
        f"- Candidate policy model(s): "
        f"{', '.join(f'`{model}`' for model in comparison['policy_models']) or 'not recorded'}",
        "- Published reference: Anthropic Claude Opus 4.6",
        f"- Coverage: {comparison['task_count']} tasks × {comparison['repeats_per_task']} repeats "
        f"= {comparison['rollout_count']} valid rollouts",
        f"- Overall accuracy: {comparison['overall_accuracy_pct']:.3f}% "
        f"({comparison['correct_rollouts']}/{comparison['rollout_count']})",
        "- Policy-evidence normalizations: "
        + (
            ", ".join(
                f"`{name}` × {count}" for name, count in comparison.get("policy_evidence_normalizations", {}).items()
            )
            or "none"
        ),
        f"- Official source: {comparison['official_source']}",
        "",
        "| Subset | Tasks | Correct / rollouts | Candidate | Anthropic | Delta | Pass@5 | Rounded match |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for name, subset in comparison["subsets"].items():
        target = subset["official_reported_accuracy_pct"]
        delta = subset["delta_percentage_points"]
        lines.append(
            f"| {name} | {subset['task_count']} | {subset['correct_rollouts']} / {subset['rollout_count']} "
            f"| {subset['accuracy_pct']:.3f}% | {target:.1f}% | {delta:+.3f} pp "
            f"| {subset['pass_at_5_pct']:.3f}% "
            f"| {'yes' if subset['rounded_score_matches_official'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Published model comparisons",
            "",
            "Kimi-K3 is the candidate policy model. Published Claude scores are reference baselines; "
            "they are not alternate judges for this run.",
            "",
            "| Published model | Human-solvable | Kimi delta | Human-difficult | Kimi delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for baseline in comparison["published_comparisons"]:
        lines.append(
            f"| {baseline['model']} | {baseline['human_solvable_pct']:.1f}% "
            f"| {baseline['human_solvable_delta_percentage_points']:+.3f} pp "
            f"| {baseline['human_difficult_pct']:.1f}% "
            f"| {baseline['human_difficult_delta_percentage_points']:+.3f} pp |"
        )
    lines.extend(
        [
            "",
            "## Per-problem solve consistency",
            "",
            "Counts below show how many problems were solved 0, 1, 2, 3, 4, or 5 times.",
            "",
        ]
    )
    for name, subset in comparison["subsets"].items():
        histogram = subset["correct_rollouts_per_task_histogram"]
        counts = ", ".join(f"{correct}/5: {count}" for correct, count in histogram.items())
        lines.append(f"- {name}: {counts}")
    return "\n".join(lines) + "\n"


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollouts", type=Path, help="Completed NeMo Gym rollout JSONL")
    parser.add_argument("--expected", type=Path, default=DEFAULT_EXPECTED, help="Prepared official task JSONL")
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--markdown-output", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        expected_rows = _read_jsonl(args.expected)
        validate_official_expected_dataset(expected_rows)
        comparison = compare_rollouts(_read_jsonl(args.rollouts), expected_rows)
    except (ComparisonError, OSError) as error:
        parser.exit(2, f"error: {error}\n")
    json_text = json.dumps(comparison, indent=2, sort_keys=True) + "\n"
    print(json_text, end="")
    if args.json_output:
        _write_text(args.json_output, json_text)
    if args.markdown_output:
        _write_text(args.markdown_output, render_markdown(comparison))


if __name__ == "__main__":
    main()
