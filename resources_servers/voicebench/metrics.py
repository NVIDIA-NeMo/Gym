# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed aggregation of VoiceBench scores, sample coverage, and judge diagnostics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

from pydantic import JsonValue

from resources_servers.voicebench.serialization import json_safe
from resources_servers.voicebench.task_data import ScoringType


MetricValue = float | int | bool | str | None


class SampleMetadataData(TypedDict, total=False):
    sample_id: str


class MetricRowData(TypedDict, total=False):
    """Fields consumed from full, stripped, or failed Gym verify responses."""

    sample_id: str | None
    verifier_metadata: SampleMetadataData | None
    score_valid: bool
    judge_valid: bool
    candidate_valid: bool
    scoring_metrics: dict[str, float | str] | None
    reward: float
    status_safe_reward: float
    voicebench_score_sum: float
    voicebench_score_count: int
    judge_actual_n: int
    judge_choice_count_mismatch: bool
    judge_peak_inflight: int
    legacy_scoring_error: str


@dataclass(frozen=True)
class MetricRow:
    """Aggregation fields with defaults for missing failure-row diagnostics."""

    sample_id: str | None
    score_valid: bool
    candidate_valid: bool
    scoring_metrics: dict[str, float]
    reward: float
    status_safe_reward: float
    legacy_score_sum: float
    legacy_score_count: int
    judge_choices: int
    judge_choice_count_mismatch: bool
    judge_peak_inflight: int

    @classmethod
    def from_data(cls, data: Mapping[str, object]) -> MetricRow:
        # Gym supplies unvalidated dictionaries; describe only the fields consumed
        # here without requiring complete VoiceBenchVerifyResponse objects.
        values = cast(MetricRowData, data)
        metadata = values.get("verifier_metadata") or {}
        return cls(
            sample_id=values.get("sample_id") or metadata.get("sample_id"),
            score_valid=values.get("score_valid", values.get("judge_valid", False)),
            candidate_valid=values.get("candidate_valid", False),
            scoring_metrics={key: float(value) for key, value in (values.get("scoring_metrics") or {}).items()},
            reward=values.get("reward", 0.0),
            status_safe_reward=values.get("status_safe_reward", 0.0),
            legacy_score_sum=float(values.get("voicebench_score_sum", 0.0)),
            legacy_score_count=values.get("voicebench_score_count", 0),
            judge_choices=values.get("judge_actual_n", 0),
            judge_choice_count_mismatch=values.get("judge_choice_count_mismatch", False),
            judge_peak_inflight=values.get("judge_peak_inflight", 0),
        )


def _sum_scoring_metrics(rows: list[MetricRow]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for row in rows:
        for key, value in row.scoring_metrics.items():
            totals[key] = totals.get(key, 0.0) + value
    return totals


@dataclass(frozen=True)
class MetricBatch:
    """Rows, validity partitions, sample identities, and sufficient statistics."""

    rows: list[MetricRow]
    valid_rows: list[MetricRow]
    completed_rows: list[MetricRow]
    observed_ids: set[str]
    totals: dict[str, float]

    @classmethod
    def from_tasks(cls, tasks: Sequence[Sequence[Mapping[str, object]]]) -> MetricBatch:
        rows = [MetricRow.from_data(data) for task in tasks for data in task]
        valid_rows = [row for row in rows if row.score_valid]
        return cls(
            rows=rows,
            valid_rows=valid_rows,
            completed_rows=[row for row in valid_rows if row.candidate_valid],
            observed_ids={row.sample_id for row in rows if row.sample_id is not None},
            totals=_sum_scoring_metrics(rows),
        )

    @property
    def count(self) -> int:
        return len(self.rows)


@dataclass(frozen=True)
class InstructionRates:
    """IFEval rates retain separate prompt and instruction denominators."""

    strict_prompt: float | None
    loose_prompt: float | None
    strict_instruction: float | None
    loose_instruction: float | None

    @property
    def mean(self) -> float | None:
        rates = (self.strict_prompt, self.loose_prompt, self.strict_instruction, self.loose_instruction)
        available = [rate for rate in rates if rate is not None]
        return sum(available) / 4 if len(available) == 4 else None

    def legacy_metrics(self) -> dict[str, MetricValue]:
        return {
            "strict-prompt": self.strict_prompt,
            "loose-prompt": self.loose_prompt,
            "strict-instruction": self.strict_instruction,
            "loose-instruction": self.loose_instruction,
        }


def _ifeval_rates(numerators: dict[str, float], denominators: dict[str, float]) -> InstructionRates:
    def ratio(correct: str, total: str) -> float | None:
        denominator = denominators.get(total, 0.0)
        return numerators.get(correct, 0.0) / denominator if denominator else None

    return InstructionRates(
        strict_prompt=ratio("strict_prompt_correct", "prompt_total"),
        loose_prompt=ratio("loose_prompt_correct", "prompt_total"),
        strict_instruction=ratio("strict_instruction_correct", "instruction_total"),
        loose_instruction=ratio("loose_instruction_correct", "instruction_total"),
    )


@dataclass(frozen=True)
class AggregateScore:
    """Subset score in source units, normalized units, and legacy metric names."""

    raw: float | None
    normalized: float | None
    legacy_metrics: dict[str, MetricValue]


def _compute_legacy_score(batch: MetricBatch, scoring_type: ScoringType) -> AggregateScore:
    totals = batch.totals
    if scoring_type == "open":
        score_sum = totals.get("score_sum", sum(row.legacy_score_sum for row in batch.valid_rows))
        choices = totals.get("score_count", sum(row.legacy_score_count for row in batch.valid_rows))
        raw = score_sum / choices if choices else float("nan") if batch.valid_rows else None
        normalized = raw * 20 if raw is not None else None
        return AggregateScore(raw, normalized, {"gpt": raw, "normalized_score": normalized})
    if scoring_type == "ifeval":
        rates = _ifeval_rates(totals, totals)
        raw = rates.mean
        normalized = raw * 100 if raw is not None else None
        legacy = rates.legacy_metrics() | {"final": raw, "normalized_score": normalized}
        return AggregateScore(raw, normalized, legacy)

    numerator = "refused" if scoring_type == "refusal" else "correct"
    total = totals.get("total", 0.0)
    raw = totals.get(numerator, 0.0) / total if total else None
    normalized = raw * 100 if raw is not None else None
    legacy = {"normalized_score": normalized}
    if scoring_type == "refusal":
        legacy["refusal_rate"] = raw
    elif scoring_type == "reference_qa":
        legacy |= {"gpt": normalized, "panda": -100.0}
    else:
        legacy["acc"] = normalized
        if scoring_type == "mcq":
            legacy["fail"] = 100 * totals.get("parse_failure", 0.0) / total if total else None
    return AggregateScore(raw, normalized, legacy)


def _compute_status_safe_score(batch: MetricBatch, scoring_type: ScoringType) -> float | None:
    if scoring_type == "ifeval":
        completed_totals = _sum_scoring_metrics(batch.completed_rows)
        # Failed/incomplete candidates contribute no correct answers, but keep the
        # original denominators so candidate status cannot inflate the safe score.
        mean = _ifeval_rates(completed_totals, batch.totals).mean
        return mean * 100 if mean is not None else None
    return 100 * sum(row.status_safe_reward for row in batch.rows) / batch.count if batch.count else None


def _compute_coverage_metrics(
    batch: MetricBatch, *, expected_num_rows: int, expected_sample_ids: set[str] | None
) -> dict[str, MetricValue]:
    missing = expected_sample_ids - batch.observed_ids if expected_sample_ids is not None else set()
    unexpected = batch.observed_ids - expected_sample_ids if expected_sample_ids is not None else set()
    distinct = len(batch.observed_ids)
    completed = len(batch.completed_rows)
    complete = batch.count == expected_num_rows and distinct == batch.count and not missing and not unexpected
    return {
        "count": batch.count,
        "distinct_sample_ids": distinct,
        "expected_rows": expected_num_rows,
        "missing_rows": len(missing) if expected_sample_ids is not None else max(0, expected_num_rows - distinct),
        "unexpected_sample_ids": len(unexpected),
        "duplicate_rows": batch.count - distinct,
        "valid_scores": len(batch.valid_rows),
        "invalid_scores": batch.count - len(batch.valid_rows),
        "completed_candidates": sum(bool(row.candidate_valid) for row in batch.rows),
        "completed_and_judged": completed,
        "completed_score_coverage": completed / batch.count if batch.count else 0.0,
        "complete": complete and completed == batch.count,
        "sample_complete": bool(batch.rows) and completed == batch.count and distinct == batch.count,
    }


def _compute_judge_metrics(
    batch: MetricBatch, *, judge_enabled: bool, judge_max_concurrency: int, judge_peak_inflight: int
) -> dict[str, MetricValue]:
    peak = max((row.judge_peak_inflight for row in batch.rows), default=0)
    if judge_enabled:
        peak = max(peak, judge_peak_inflight)
    return {
        "valid_judges": len(batch.valid_rows) if judge_enabled else 0,
        "invalid_judges": batch.count - len(batch.valid_rows) if judge_enabled else 0,
        "valid_judge_choices": sum(row.judge_choices for row in batch.valid_rows),
        "requested_n_mismatch_rows": sum(bool(row.judge_choice_count_mismatch) for row in batch.rows),
        "judge_coverage": len(batch.valid_rows) / batch.count if batch.count else 0.0,
        "judge_configured_concurrency": judge_max_concurrency if judge_enabled else 0,
        "judge_peak_inflight": peak,
    }


def compute_voicebench_metrics(
    tasks: Sequence[Sequence[Mapping[str, object]]],
    *,
    scoring_type: ScoringType,
    expected_num_rows: int,
    expected_sample_ids: set[str] | None,
    judge_enabled: bool,
    judge_max_concurrency: int,
    judge_peak_inflight: int,
) -> dict[str, MetricValue]:
    """Compute existing VoiceBench metrics from Gym's task-grouped verify rows."""
    for task in tasks:
        for row in task:
            if row.get("legacy_scoring_error"):
                raise ValueError(f"legacy VoiceBench evaluation failed: {row['legacy_scoring_error']}")
    batch = MetricBatch.from_tasks(tasks)
    score = _compute_legacy_score(batch, scoring_type)
    metrics = _compute_coverage_metrics(
        batch, expected_num_rows=expected_num_rows, expected_sample_ids=expected_sample_ids
    ) | _compute_judge_metrics(
        batch,
        judge_enabled=judge_enabled,
        judge_max_concurrency=judge_max_concurrency,
        judge_peak_inflight=judge_peak_inflight,
    )
    metrics |= {
        "raw_mean_valid": score.raw,
        "normalized_mean_valid": score.normalized,
        "normalized_score_all_rows": 100 * sum(row.reward for row in batch.rows) / batch.count
        if batch.count
        else None,
        "status_safe_normalized_score": _compute_status_safe_score(batch, scoring_type),
    }
    exported = {f"voicebench/{key}": value for key, value in metrics.items()} | {
        f"voicebench/legacy/{key}": value for key, value in score.legacy_metrics.items()
    }
    return cast(dict[str, MetricValue], json_safe(cast(dict[str, JsonValue], exported)))
