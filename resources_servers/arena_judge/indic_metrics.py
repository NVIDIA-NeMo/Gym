# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Indic Arena aggregation shared by Gym metrics and saved-rollout reports."""

import hashlib
import json
import math
from collections import Counter
from collections.abc import Callable
from pathlib import Path


INDIC_LANGUAGES = ("bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "ta", "te", "ur")
GROUPS = {
    "Information & advice": "info_advice",
    "Work writing & text tasks": "work_writing",
    "Creative writing / role-play": "creative_roleplay",
    "Planning & brainstorming": "planning",
    "Coding / technical": "coding",
    "Math / reasoning": "math",
    "Casual chat / unclear asks": "casual",
}
SLICES = ("all", *GROUPS.values(), "single_turn", "multi_turn")
VERDICTS = {"A>>B", "A>B", "A=B", "B>A", "B>>A"}
TASK_FIELDS = (
    "language",
    "group",
    "num_user_turns",
    "question",
    "baseline_answer",
    "category",
    "messages_sha256",
    "baseline_model",
)
AggregateScores = Callable[[list[list[str | None]]], dict]
HEADLINE_METRIC = "arena_elo/indic/score"


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                rows.append(row)
    return rows


def _flatten(row: dict) -> dict:
    """Accept flat rows and the materialized task / episode reply containers."""
    fields = {key: value for key, value in row.items() if key not in {"task_input", "task_data", "result"}}
    for container in ("task_input", "task_data", "result"):
        child = row.get(container)
        if child is None:
            continue
        if not isinstance(child, dict):
            raise ValueError(f"{container} must be a JSON object")
        for key, value in _flatten(child).items():
            if key in fields and fields[key] != value:
                raise ValueError(f"Conflicting field {key!r} in {container}")
            fields[key] = value
    return fields


def _identity(row: dict) -> str:
    uid = row.get("uid")
    if not isinstance(uid, str) or not uid:
        raise ValueError("Every prepared input and judged rollout must carry its nonempty uid")
    return uid


def _slices(row: dict) -> tuple[str, str, str]:
    group = row.get("group")
    if group not in GROUPS:
        raise ValueError(f"{_identity(row)}: unknown chat400 group {group!r}")
    turns = row.get("num_user_turns")
    if isinstance(turns, bool) or not isinstance(turns, int) or turns < 1:
        raise ValueError(f"{_identity(row)}: num_user_turns must be a positive integer")
    return "all", GROUPS[group], "single_turn" if turns == 1 else "multi_turn"


def _failed(row: dict) -> bool:
    return bool(
        row.get("_ng_failure_class")
        or row.get("failure")
        or row.get("failure_kind")
        or row.get("mask_sample")
        or row.get("_ng_no_persist")
        or row.get("verification_skipped")
        or row.get("evaluation_completed") is False
    )


def _finite_metrics(metrics: dict) -> dict:
    return {
        key: None if isinstance(value, float) and not math.isfinite(value) else value for key, value in metrics.items()
    }


def _validate_verdicts(row: dict) -> None:
    for suffix in ("gen_base", "base_gen"):
        key = f"verdict_{suffix}"
        if key not in row or row[key] is not None and row[key] not in VERDICTS:
            raise ValueError(f"{_identity(row)}: missing or unsupported {key}; pass judged rollouts")
        if f"invalid_{suffix}" in row and row[f"invalid_{suffix}"] != (row[key] is None):
            raise ValueError(f"{_identity(row)}: inconsistent invalid_{suffix} flag")


def _check_recovered_identity(success: dict, failure: dict) -> None:
    # Separate reverification outputs retain historical failures. Gym's resume and
    # aggregation treat their main-file success as authoritative for that saved answer.
    for key in (
        "_ng_task_index",
        "_ng_rollout_index",
        "_ng_attempt_index",
        "_ng_reverification_source_run_id",
        "_ng_reverification_source_answer",
    ):
        if success.get(key) != failure.get(key):
            raise ValueError(f"{_identity(success)}: success and failure name different saved rollouts ({key})")
    if "response" in success and "response" in failure and success["response"] != failure["response"]:
        raise ValueError(f"{_identity(success)}: success and failure contain different saved policy answers")
    binding = success.get("_ng_reverification_source_answer")
    has_binding = (
        isinstance(binding, dict) and isinstance(binding.get("answer_sha256"), str) and bool(binding["answer_sha256"])
    )
    has_answer = bool(success.get("response")) and success.get("response") == failure.get("response")
    if not (has_binding or has_answer):
        raise ValueError(f"{_identity(success)}: cannot establish that the success recovers the same saved rollout")


def _score_slices(aggregate_scores: AggregateScores, records: list[dict], *, complete: bool) -> dict:
    slices = {}
    for name in SLICES:
        subset = [row for row in records if name in _slices(row)]
        scores = [[row["verdict_gen_base"], row["verdict_base_gen"]] for row in subset]
        aggregate = _finite_metrics(aggregate_scores(scores))
        if not complete:
            aggregate.update(score=None, ci_lower=None, ci_upper=None)
        slices[name] = aggregate | {
            "verdicts_gen_base": dict(Counter(row["verdict_gen_base"] or "invalid" for row in subset))
        }
    return slices


def _compare_reference(
    paths: list[Path],
    *,
    expected: dict[str, dict],
    judged: dict[str, dict],
    aggregate_scores: AggregateScores,
    metrics: dict,
) -> dict:
    reference, artifacts = {}, []
    for path in paths:
        for row in _read_jsonl(path):
            uid = _identity(row)
            if uid not in expected:
                raise ValueError(f"Unexpected reference uid {uid!r}; supply exactly the selected task population")
            if uid in reference:
                raise ValueError(f"Duplicate reference uid {uid!r}")
            for key in (*TASK_FIELDS, "row_id"):
                required = key in ("language", "group", "num_user_turns", "row_id")
                if key in expected[uid] and (required or key in row) and row.get(key) != expected[uid][key]:
                    raise ValueError(f"{uid}: reference {key} disagrees with prepared inputs")
            template = "creative" if expected[uid]["group"] == "Creative writing / role-play" else "default"
            if row.get("template") != template:
                raise ValueError(f"{uid}: reference template disagrees with prepared inputs")
            if _failed(row):
                raise ValueError(f"{uid}: reference contains a failed or unverified judgment")
            _validate_verdicts(row)
            reference[uid] = row
        artifacts.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    if reference.keys() != expected.keys():
        raise ValueError(f"Reference verdicts are missing {len(expected.keys() - reference.keys())} selected tasks")
    selected = sorted({row["language"] for row in expected.values()})
    reference_metrics = _aggregate_population(
        expected=expected,
        judged=reference,
        failed=set(),
        selected=selected,
        population_scope="full_benchmark" if metrics["_avg"]["complete"] else "subset",
        failure_records=0,
        aggregate_scores=aggregate_scores,
    )
    by_language = {language: {name: reference_metrics[language][name] for name in SLICES} for language in selected}
    reference_average = reference_metrics["_avg"]["all"]
    candidate_average = metrics["_avg"]["all"]
    return {
        "artifacts": artifacts,
        "bootstrap_order": "prepared_inputs",
        "scope": "saved_verdicts_only",
        "ci_note": "Reference CIs are recomputed in prepared input order, not historical asynchronous completion order.",
        "verdict_pairs_compared": len(expected),
        "mismatched_verdict_pairs": sum(
            any(reference[uid][key] != judged[uid][key] for key in ("verdict_gen_base", "verdict_base_gen"))
            for uid in expected
        ),
        "reference_metrics": by_language,
        "indic_average": {
            "candidate": candidate_average,
            "reference": reference_average,
            "candidate_minus_reference": (
                candidate_average - reference_average
                if candidate_average is not None and reference_average is not None
                else None
            ),
        },
    }


def _population_scope(inputs: Path, expected: dict[str, dict]) -> str:
    report_path = inputs.parent / "preparation_report.json"
    if not report_path.exists():
        return "unverified_inputs"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("prepared_sha256") != hashlib.sha256(inputs.read_bytes()).hexdigest():
        raise ValueError("Prepared inputs do not match preparation_report.json; prepare the dataset again")
    counts = Counter(row["language"] for row in expected.values())
    coverage = report.get("coverage", {})
    for language, count in counts.items():
        entry = coverage.get(language, {})
        if entry.get("prepared_rows") != count:
            raise ValueError(f"{language}: prepared row count disagrees with preparation_report.json")
        excluded = entry.get("excluded_untranslated", [])
        if (
            entry.get("source_rows") != 400
            or entry.get("eligible_rows") != (399 if language == "ta" else 400)
            or entry.get("eligible_rows") != 400 - len(set(excluded))
        ):
            raise ValueError(f"{language}: invalid source coverage in preparation_report.json")
    if report.get("limit_per_language") is not None or any(
        coverage[language]["prepared_rows"] != coverage[language]["eligible_rows"] for language in counts
    ):
        return "subset"
    return "full_benchmark"


def _aggregate_population(
    *,
    expected: dict[str, dict],
    judged: dict[str, dict],
    failed: set[str],
    selected: list[str],
    population_scope: str,
    failure_records: int,
    aggregate_scores: AggregateScores,
) -> dict:
    metrics = {}
    for language in selected:
        population = {uid: row for uid, row in expected.items() if row["language"] == language}
        records = [judged[uid] for uid in population if uid in judged and uid not in failed]
        missing = sorted(set(population) - {row["uid"] for row in records})
        complete = not missing
        entry = {
            "rows": len(population),
            "judged": len(records),
            "complete": complete,
            "missing_or_failed_uids": missing,
            "failed": len(failed & population.keys()),
        }
        entry.update(_score_slices(aggregate_scores, records, complete=complete))
        metrics[language] = entry

    indic_complete = population_scope == "full_benchmark" and all(
        language in metrics and metrics[language]["complete"] for language in INDIC_LANGUAGES
    )
    averages = {}
    for name in SLICES:
        values = [metrics[language][name]["score"] for language in INDIC_LANGUAGES if language in metrics]
        averages[name] = (
            sum(values) / len(INDIC_LANGUAGES)
            if indic_complete and all(value is not None for value in values)
            else None
        )
    metrics["_avg"] = averages | {"languages": len(INDIC_LANGUAGES), "complete": indic_complete}
    metrics["_coverage"] = {
        "complete": all(metrics[language]["complete"] for language in selected),
        "languages": selected,
        "population_scope": population_scope,
        "expected": len(expected),
        "failure_records": failure_records,
        "headline_available": averages["all"] is not None,
        "english_reported_separately": "en" in metrics,
    }
    return metrics


def compute_indic_metrics(
    results: list[dict],
    *,
    inputs: Path | str,
    aggregate_scores: AggregateScores,
    failures: list[dict] | None = None,
    reconcile_failures: bool = False,
    languages: list[str] | None = None,
    reference_verdicts: list[Path] | None = None,
) -> dict:
    """Compute the original 100-resample Arena-Elo protocol without a partial headline.

    Inputs define the exact eligible population after translation exclusions. Each task
    must have one judged rollout; repeats and duplicate identities are rejected. Managed
    Gym runs use their selected outcomes, so superseded attempts cannot inflate scores.
    A missing or failed task suppresses its language's scores and the 12-language mean.
    The headline also requires a matching preparation report proving the population
    was not limited to a smoke subset. Unverified inputs still get language scores.
    Successfully returned but unparseable verdicts retain the Arena scorer's invalid
    verdict accounting; they are not silently converted into losses.
    Optional reference artifacts compare saved verdicts in the same prepared input
    order; they cannot establish that checkpoints or generation settings matched.
    """
    inputs = Path(inputs)
    expected = {}
    for raw in _read_jsonl(inputs):
        row = _flatten(raw)
        uid = _identity(row)
        if uid in expected:
            raise ValueError(f"Duplicate prepared uid {uid!r}; this protocol requires one rollout per task")
        if row.get("language") not in ("en", *INDIC_LANGUAGES):
            raise ValueError(f"{uid}: unsupported language {row.get('language')!r}")
        _slices(row)
        expected[uid] = row
    if not expected:
        raise ValueError(f"No prepared tasks in {inputs}")
    population_scope = _population_scope(inputs, expected)
    all_expected = expected
    available = {row["language"] for row in expected.values()}
    selected = languages if languages is not None else sorted(available)
    if not selected or len(set(selected)) != len(selected) or set(selected) - available:
        raise ValueError(f"languages must be unique and present in prepared inputs: {sorted(available)}")
    expected = {uid: row for uid, row in expected.items() if row["language"] in selected}

    failures = failures or []

    judged, failed = {}, set()
    repeats = set()
    for raw in results:
        row = _flatten(raw)
        uid = _identity(row)
        if row.get("language") not in selected and row.get("language") in available:
            continue
        if uid not in expected:
            raise ValueError(f"Unexpected rollout uid {uid!r}; use the matching prepared inputs")
        if uid in judged or uid in failed:
            raise ValueError(f"Duplicate rollout uid {uid!r}; this protocol requires one rollout per task")
        repeats.add(row.get("_ng_rollout_index", 0))
        for key in TASK_FIELDS:
            if key not in expected[uid]:
                continue
            if row.get(key) != expected[uid][key]:
                raise ValueError(f"{uid}: rollout {key} disagrees with prepared inputs")
        if _failed(row):
            failed.add(uid)
            continue
        _validate_verdicts(row)
        judged[uid] = row
    if len(repeats) > 1:
        raise ValueError("Indic Arena metrics require one repeat per aggregate; do not mix repeat populations")
    for raw in failures:
        row = _flatten(raw)
        if row.get("uid") not in all_expected:
            raise ValueError("A failure sidecar row lacks a known uid; resolve failures before publishing metrics")
        if reconcile_failures and row["uid"] in judged:
            _check_recovered_identity(judged[row["uid"]], row)
            continue
        if row.get("uid") in expected:
            failed.add(row["uid"])

    metrics = _aggregate_population(
        expected=expected,
        judged=judged,
        failed=failed,
        selected=selected,
        population_scope=population_scope,
        failure_records=len(failures),
        aggregate_scores=aggregate_scores,
    )
    if reference_verdicts is not None:
        if not reference_verdicts:
            raise ValueError("Supply at least one reference verdict artifact")
        if not metrics["_coverage"]["complete"]:
            raise ValueError("Reference comparison requires completed judgments for every selected task")
        metrics["_reference_comparison"] = _compare_reference(
            reference_verdicts, expected=expected, judged=judged, aggregate_scores=aggregate_scores, metrics=metrics
        )
    return metrics


def flatten_indic_metrics(report: dict) -> dict:
    """Expose finite scalar metrics to Gym exporters; unavailable scores stay absent."""
    flat = {}
    for language in report["_coverage"]["languages"]:
        entry = report[language]
        prefix = f"arena_elo/language/{language}"
        for name in ("rows", "judged", "complete", "failed"):
            flat[f"{prefix}/{name}"] = entry[name]
        for name in SLICES:
            for statistic, value in entry[name].items():
                if type(value) in (int, float):
                    flat[f"{prefix}/{name}/{statistic}"] = value
    for name in SLICES:
        value = report["_avg"][name]
        if value is not None:
            key = HEADLINE_METRIC if name == "all" else f"arena_elo/indic/{name}/score"
            flat[key] = value
    for name, value in report["_coverage"].items():
        if type(value) in (bool, int, str):
            flat[f"arena_elo/coverage/{name}"] = value
    flat["arena_elo/coverage/indic_complete"] = report["_avg"]["complete"]
    return flat
