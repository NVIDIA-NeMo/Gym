# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Convert pinned chemeval source records into model inputs and verifier metadata."""

from collections import Counter
from typing import Any

from benchmarks.chemeval.utils import (
    ANSWER_FORMAT_INSTRUCTIONS,
    MUZZLE_MARKERS,
    SUB_BENCHMARKS,
    TASK_NAMES_EN,
    TASK_SLUGS,
    TASKS,
    extract_number,
    strip_muzzle,
    task_key,
)


def answer_format_key(family: str, rubric: str | None) -> str:
    """Which trailing answer-format sentence a task gets.

    Every question except the five L1 ones already states its own output format, so the sentence
    only pins that format to the last line. The L1 subjective tasks (fill-in-the-blank, short
    answer, calculation) are graded by a judge that reads the whole response, so they get none.
    """
    if family in ("mcq", "true_false"):
        return family
    if family == "judged" and rubric in ("fill_in_the_blank", "short_answer", "calculation"):
        return "judged_open"
    return "formatted"


def answer_format(family: str, rubric: str | None) -> str:
    instruction = ANSWER_FORMAT_INSTRUCTIONS[answer_format_key(family, rubric)]
    return f"\n\n{instruction}" if instruction else ""


def build_judge_fields(rubric_name: str, problem: str) -> dict[str, str]:
    """Keep judge-only context separate from generation messages."""
    return {
        "judge_protocol": "english_v2",
        "judge_question": problem,
        "judge_scale": "0-1" if rubric_name == "fill_in_the_blank" else "1-5",
        "judge_rubric": rubric_name,
    }


def gold_spans(records: list[dict[str, str]]) -> dict[str, float]:
    """Spread of the gold values of each regression task, for the normalized per-question score.

    Upstream reports regression with corpus statistics (RMSE, RAE, R^2 and an NRMSE normalized by
    exactly this spread), which a per-question harness cannot produce for a subset. The evaluator
    instead scores each answer `max(0, 1 - |error| / span)`, which is on the same 0-1 scale as
    every other family and averages to `1 - NMAE`. The raw numbers stay in the output rows, so the
    corpus statistics remain one groupby away.
    """
    values = {}
    for record in records:
        key = task_key(record["filename"])
        if TASKS[key][2] != "regression":
            continue
        number = extract_number(record["target"])
        assert number is not None, f"{key}: gold {record['target']!r} has no number"
        values.setdefault(key, []).append(number)
    spans = {}
    for key, numbers in values.items():
        span = max(numbers) - min(numbers)
        assert span > 0, f"{key}: all gold values are identical, cannot normalize"
        spans[key] = span
    return spans


def format_entry(record: dict[str, str], spans: dict[str, float]) -> tuple[str, dict[str, Any]]:
    key = task_key(record["filename"])
    level, dimension, family, rubric = TASKS[key]
    problem = strip_muzzle(record["query"])
    expected_answer = record["target"]

    entry = {
        "problem": problem,
        "original_problem": record["query"],
        "expected_answer": expected_answer,
        "task": key,
        "task_en": TASK_NAMES_EN[key],
        "level": level,
        "dimension": dimension,
        "family": family,
        # carries its own leading blank line, so that the three tasks which get no answer format
        # do not end up with a trailing one when prepare.py joins the problem and format
        "answer_format": answer_format(family, rubric),
        # Preserve task and level labels for filtering. The server computes aggregate
        # scores from task metadata, so these labels do not duplicate observations.
        "subset_for_metrics": [TASK_SLUGS[key], level],
    }
    if family == "regression":
        entry["gold_span"] = spans[key]
    if family == "judged":
        entry.update(build_judge_fields(rubric, problem))
    return SUB_BENCHMARKS[family], entry


def check_task_coverage(records: list[dict[str, str]]) -> None:
    """Fail loudly if the release adds or renames a task, rather than silently dropping it."""
    unknown = Counter(task_key(r["filename"]) for r in records if task_key(r["filename"]) not in TASKS)
    if unknown:
        raise ValueError(
            "No routing entry for these task keys. Add them to TASKS in utils.py:\n"
            + "\n".join(f"  ({n}x) {key!r}" for key, n in unknown.most_common())
        )
    missing = set(TASKS) - {task_key(r["filename"]) for r in records}
    if missing:
        raise ValueError(f"TASKS lists tasks that are not in the release: {sorted(missing)}")


def check_muzzle_stripping(records: list[dict[str, str]]) -> None:
    """Fail loudly if the release adds an instruction whose muzzle we do not know how to remove."""
    leftovers = Counter()
    for record in records:
        stripped = strip_muzzle(record["query"]).lower()
        for marker in MUZZLE_MARKERS:
            if marker in stripped:
                leftovers[marker] += 1
    if leftovers:
        raise ValueError(
            "Questions still contain an answer-suppressing directive after stripping. Add the "
            "exact phrase to MUZZLE_PHRASES in utils.py:\n"
            + "\n".join(f"  ({n}x) matched {marker!r}" for marker, n in leftovers.most_common())
        )


def build_split(records: list[dict[str, str]], spans: dict[str, float]) -> dict[str, list[dict[str, Any]]]:
    by_sub_benchmark = {name: [] for name in set(SUB_BENCHMARKS.values())}
    for record in records:
        sub_benchmark, entry = format_entry(record, spans)
        by_sub_benchmark[sub_benchmark].append(entry)
    return by_sub_benchmark
