# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normalize a FACTS Parametric ``gym eval run`` directory into the shared ``NormalizedRun`` contract."""

from __future__ import annotations

import collections
from typing import Any

from benchmarks.facts_parametric import prepare as prepare_module

from .common import RunArtifacts, excerpt, failure_summaries, reconcile
from .schema import (
    AnchorFact,
    BladeMapping,
    CalibrationSummary,
    MetricValue,
    NormalizedRun,
    ReferenceComparison,
    RolloutSummary,
)


BENCHMARK = {
    "id": "facts_parametric",
    "display_name": "FACTS Parametric (public set)",
    "protocol": "closed-book factoid QA, 1,052 public questions, three Gemini 2.5 Pro grades per answer",
    "unit_of_analysis": "one question answered once; three sampled grades per answer",
    "dataset": {
        "source": prepare_module.KAGGLE_DATASET_URL,
        "revision": f"version {prepare_module.KAGGLE_DATASET_VERSION}",
        "file": prepare_module.CSV_MEMBER,
        "sha256": prepare_module.CSV_SHA256,
        "count": prepare_module.EXPECTED_ROWS,
        "license": prepare_module.LICENSE,
        "cohort": "the 1,052-question public half of FACTS Parametric; the 1,052-question private half is held by Kaggle",
    },
    "paper": {
        "citation": "Cheng et al., The FACTS Leaderboard: A Comprehensive Benchmark for Large Language Model Factuality, arXiv:2512.10791v1 (2025)",
        "url": "https://arxiv.org/abs/2512.10791v1",
        "sha256": "db046e76cc1877880843d0e7fd4898422f1064d47f8990b04f3c230229ede6be",
    },
    "upstream": {
        "reference_implementation": "Kaggle starter notebook yulongt/facts-parametric-benchmark-starter-code (v10)",
        "grader_prompt_sha256": "9d6a61f9ce3b875b5f97d25305abe6c073911f97e98297780483dd84fd9b548c",
    },
}

REFERENCES = [
    ReferenceComparison(
        label="Paper Table 6, Gemini 3 Pro accuracy",
        value="76.4% accuracy, 1.4% hedging, 77.6% attempted accuracy, F1 77.0",
        source="arXiv:2512.10791v1 Table 6",
        comparability="paper numbers pool the public and private halves; this run scores the public half only",
    ),
    ReferenceComparison(
        label="Paper Table 6, GPT-5 accuracy",
        value="55.7% accuracy, 13.3% hedging, 64.3% attempted accuracy, F1 59.7",
        source="arXiv:2512.10791v1 Table 6",
        comparability="public+private pooled; different model family; same grader protocol",
    ),
]


def _task_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("_ng_task_index"))


def receipts_for(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    """One receipt per grader sample, keyed ``<task>/grade-<sample>``."""
    task = _task_id(rollout)
    rollout_id = f"{rollout.get('_ng_task_index', 0)}-{rollout.get('_ng_rollout_index', 0)}"
    return [
        {
            "receipt_id": f"{task}/grade-{receipt['sample']}",
            "task_id": task,
            "rollout_id": rollout_id,
            **receipt,
        }
        for receipt in rollout.get("judge_receipts", [])
    ]


def _metric(
    metric_id: str,
    name: str,
    value: float | None,
    *,
    numerator: float | None = None,
    denominator: int | None = None,
    unit: str = "rate",
    direction: str = "higher_is_better",
    kind: str = "component",
    definition: str = "",
    slice_: dict[str, str] | None = None,
    ci95=None,
) -> MetricValue:
    return MetricValue(
        id=metric_id,
        name=name,
        value=value,
        numerator=numerator,
        denominator=denominator,
        unit=unit,
        direction=direction,
        kind=kind,
        definition=definition,
        slice=slice_,
        ci95=ci95,
    )


def normalize(
    artifacts: RunArtifacts, *, run_info: dict[str, Any], calibration: CalibrationSummary | None, num_repeats: int = 1
) -> NormalizedRun:
    outcomes, scored, unresolved = reconcile(artifacts, task_id_of=_task_id, num_repeats=num_repeats)
    rows = [scored[key] for key in sorted(scored)]
    agg = artifacts.aggregate
    grades = [label for row in rows for label in row.get("judge_labels", [])]
    n_grades = len(grades)
    counts = collections.Counter(grades)
    n_rows = len(rows)
    correct = counts.get("correct", 0)
    not_attempted = counts.get("not_attempted", 0)
    attempted = n_grades - not_attempted
    accuracy = correct / n_grades if n_grades else None
    hedging = not_attempted / n_grades if n_grades else None
    attempted_accuracy = correct / attempted if attempted else None
    f1 = (
        (2 * accuracy * attempted_accuracy / (accuracy + attempted_accuracy))
        if accuracy and attempted_accuracy
        else 0.0
    )
    ci = (
        (agg.get("accuracy_ci95_low"), agg.get("accuracy_ci95_high"))
        if agg.get("accuracy_ci95_low") is not None
        else None
    )
    strict = sum(float(row.get("all_correct", 0.0)) for row in rows)
    parse_agree_rows = sum(1 for row in rows if row.get("judge_parse_agreement", 1.0) == 1.0)
    empty_grades = sum(int(row.get("judge_empty_samples", 0)) for row in rows)
    truncated = sum(1 for row in rows if row.get("generation_truncated"))
    empty_gen = sum(1 for row in rows if row.get("generation_empty"))
    metrics = [
        _metric(
            "accuracy",
            "Accuracy (mean of three grades)",
            accuracy,
            numerator=correct,
            denominator=n_grades,
            kind="primary",
            definition="Share of grader samples labelled CORRECT; the paper's primary metric averaged over three sampled grades.",
            ci95=ci,
        ),
        _metric(
            "hedging_rate",
            "Hedging rate",
            hedging,
            numerator=not_attempted,
            denominator=n_grades,
            direction="neutral",
            definition="Share of grades labelled NOT_ATTEMPTED (the answer did not supply the fact).",
        ),
        _metric(
            "attempted_accuracy",
            "Attempted accuracy",
            attempted_accuracy,
            numerator=correct,
            denominator=attempted,
            definition="CORRECT grades over grades that were not NOT_ATTEMPTED (accuracy / (1 - hedging)).",
        ),
        _metric(
            "f1",
            "F1 (accuracy, attempted accuracy)",
            f1,
            unit="score",
            definition="Harmonic mean of accuracy and attempted accuracy, as in the paper.",
        ),
        _metric(
            "mistake_rate",
            "Mistake rate",
            counts.get("incorrect", 0) / n_grades if n_grades else None,
            numerator=counts.get("incorrect", 0),
            denominator=n_grades,
            direction="lower_is_better",
            definition="Share of grades labelled MISTAKE/INCORRECT.",
        ),
        _metric(
            "unknown_rate",
            "Grader-unknown rate",
            counts.get("unknown", 0) / n_grades if n_grades else None,
            numerator=counts.get("unknown", 0),
            denominator=n_grades,
            direction="neutral",
            kind="diagnostic",
            definition="Share of grades labelled UNKNOWN (grader could not decide, or reply unparseable).",
        ),
        _metric(
            "strict_all_correct_rate",
            "All-three-grades-correct rate (starter score)",
            strict / n_rows if n_rows else None,
            numerator=strict,
            denominator=n_rows,
            definition="Share of answers whose three grades were all CORRECT; the starter notebook's per-example score.",
        ),
        _metric(
            "judge_parse_agreement_rate",
            "Grader parse agreement (starter vs closing line)",
            parse_agree_rows / n_rows if n_rows else None,
            numerator=parse_agree_rows,
            denominator=n_rows,
            kind="diagnostic",
            definition="Answers whose starter-rule labels equal the labels parsed from the grader's closing Output line on all three grades.",
        ),
        _metric(
            "judge_empty_grade_rate",
            "Empty grader replies",
            empty_grades / n_grades if n_grades else None,
            numerator=empty_grades,
            denominator=n_grades,
            direction="lower_is_better",
            kind="diagnostic",
            definition="Grader replies with no content (counted as UNKNOWN, as the starter does).",
        ),
        _metric(
            "generation_truncated_rate",
            "Truncated answers",
            truncated / n_rows if n_rows else None,
            numerator=truncated,
            denominator=n_rows,
            direction="lower_is_better",
            kind="operational",
            definition="Answers cut off by the output-token cap (reasoning tokens count against it on this endpoint).",
        ),
        _metric(
            "generation_empty_rate",
            "Empty answers",
            empty_gen / n_rows if n_rows else None,
            numerator=empty_gen,
            denominator=n_rows,
            direction="lower_is_better",
            kind="operational",
            definition="Answers with no visible text after reasoning.",
        ),
        _metric(
            "mean_output_tokens",
            "Mean output tokens (reasoning + answer)",
            agg.get("mean/output_tokens"),
            unit="tokens",
            direction="neutral",
            kind="operational",
            definition="Policy output tokens per answer as reported by the endpoint.",
        ),
    ]
    by_topic: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_topic[str(row.get("topic") or "unknown")].append(row)
    for topic, topic_rows in sorted(by_topic.items(), key=lambda item: (-len(item[1]), item[0]))[:8]:
        topic_grades = [label for row in topic_rows for label in row.get("judge_labels", [])]
        correct_t = sum(1 for label in topic_grades if label == "correct")
        metrics.append(
            _metric(
                f"accuracy/topic/{topic}",
                f"Accuracy · topic {topic}",
                correct_t / len(topic_grades) if topic_grades else None,
                numerator=correct_t,
                denominator=len(topic_grades),
                kind="slice",
                slice_={"topic": topic},
                definition="Accuracy restricted to this answer-type topic label from the public CSV.",
            )
        )

    rollouts: list[RolloutSummary] = []
    for row in rows:
        task = _task_id(row)
        rollouts.append(
            RolloutSummary(
                task_id=task,
                rollout_id=f"{row.get('_ng_task_index', 0)}-{row.get('_ng_rollout_index', 0)}",
                receipt_ids=[f"{task}/grade-{receipt['sample']}" for receipt in row.get("judge_receipts", [])],
                slices={"topic": str(row.get("topic") or "unknown")},
                outcome_class="scored",
                reward=float(row.get("reward", 0.0)),
                components={
                    "judge_labels": row.get("judge_labels"),
                    "judge_labels_output_line": row.get("judge_labels_output_line"),
                    "all_correct": row.get("all_correct"),
                    "generation_truncated": row.get("generation_truncated"),
                    "generation_empty": row.get("generation_empty"),
                    "expected_answer": row.get("expected_answer"),
                },
                prompt_excerpt=excerpt(row.get("question")),
                response_excerpt=excerpt(row.get("generation")),
                judge_excerpt=excerpt((row.get("judge_receipts") or [{}])[0].get("content")),
            )
        )
    rollouts += failure_summaries(unresolved, task_id_of=_task_id)

    anchors: list[AnchorFact] = []
    anchors.append(
        AnchorFact(
            id="fact-accuracy",
            category="outcome",
            fact=f"{correct} of {n_grades} grader samples were CORRECT (accuracy {accuracy:.1%}); {not_attempted} were NOT_ATTEMPTED (hedging {hedging:.1%}); {counts.get('incorrect', 0)} were MISTAKE; {counts.get('unknown', 0)} UNKNOWN.",
            evidence=[r.rollout_id for r in rollouts[:1]] + [f"{run_info['run_id']}"]
            if False
            else [rollouts[0].rollout_id]
            if rollouts
            else ["run"],
        )
    )
    anchors.append(
        AnchorFact(
            id="fact-coverage",
            category="coverage",
            fact=f"{outcomes.scored_rollouts} of {outcomes.expected_rollouts} expected answers were graded; {outcomes.judge_failed} grader failures, {outcomes.infrastructure_failed} infrastructure failures, {outcomes.missing_rollouts} missing, {outcomes.replaced_attempts} sidecar attempts superseded by a later scored attempt.",
            evidence=[rollouts[0].rollout_id] if rollouts else ["run"],
        )
    )
    anchors.append(
        AnchorFact(
            id="fact-strict",
            category="outcome",
            fact=f"{int(strict)} of {n_rows} answers were graded CORRECT by all three samples (the starter's strict score, {strict / n_rows:.1%}); the paper-style averaged accuracy is {accuracy:.1%}.",
            evidence=[rollouts[0].rollout_id] if rollouts else ["run"],
        )
    )
    anchors.append(
        AnchorFact(
            id="fact-parse",
            category="invalid",
            fact=f"The starter's substring parse and the closing Output-line parse agreed on all three grades for {parse_agree_rows} of {n_rows} answers; {empty_grades} of {n_grades} grader replies were empty.",
            evidence=[
                r.rollout_id
                for r in rollouts
                if r.components.get("judge_labels") != r.components.get("judge_labels_output_line")
            ][:3]
            or ([rollouts[0].rollout_id] if rollouts else ["run"]),
        )
    )
    anchors.append(
        AnchorFact(
            id="fact-truncation",
            category="infrastructure",
            fact=f"{truncated} of {n_rows} answers hit the output-token cap and {empty_gen} were empty; both still received grades.",
            evidence=[r.rollout_id for r in rollouts if r.components.get("generation_truncated")][:3]
            or ([rollouts[0].rollout_id] if rollouts else ["run"]),
        )
    )
    if len(by_topic) > 1:
        topic_stats = []
        for topic, topic_rows in by_topic.items():
            topic_grades = [label for row in topic_rows for label in row.get("judge_labels", [])]
            if len(topic_rows) >= 20:
                topic_stats.append(
                    (
                        topic,
                        sum(1 for label in topic_grades if label == "correct") / len(topic_grades),
                        len(topic_rows),
                    )
                )
        if topic_stats:
            weakest = min(topic_stats, key=lambda item: item[1])
            strongest = max(topic_stats, key=lambda item: item[1])
            anchors.append(
                AnchorFact(
                    id="fact-slices",
                    category="slice",
                    fact=f"Among topics with at least 20 questions, accuracy was lowest for '{weakest[0]}' ({weakest[1]:.1%}, {weakest[2]} questions) and highest for '{strongest[0]}' ({strongest[1]:.1%}, {strongest[2]} questions).",
                    evidence=[r.rollout_id for r in rollouts if r.slices.get("topic") == weakest[0]][:2]
                    + [r.rollout_id for r in rollouts if r.slices.get("topic") == strongest[0]][:1],
                )
            )

    # trace-linked examples: a correct answer, a mistake, a hedge, and a grader disagreement
    def _first(predicate, label):
        for row in rows:
            if predicate(row):
                task = _task_id(row)
                grades = row.get("judge_labels", [])
                return AnchorFact(
                    id=f"example-{label}",
                    category="example",
                    fact=f"Question {task} (topic {row.get('topic')}): expected '{excerpt(row.get('expected_answer'), 80)}'; the model answered '{excerpt(row.get('generation'), 160)}'; the three grades were {grades}.",
                    evidence=[task, f"{row.get('_ng_task_index', 0)}-{row.get('_ng_rollout_index', 0)}"]
                    + [f"{task}/grade-{receipt['sample']}" for receipt in row.get("judge_receipts", [])],
                    excerpt=excerpt((row.get("judge_receipts") or [{}])[0].get("content"), 300),
                )
        return None

    for predicate, label in (
        (lambda row: row.get("reward") == 1.0, "correct"),
        (lambda row: row.get("judge_labels", []).count("incorrect") == 3, "mistake"),
        (lambda row: "not_attempted" in row.get("judge_labels", []), "hedge"),
        (lambda row: 0 < row.get("reward", 0) < 1, "mixed-grades"),
    ):
        fact = _first(predicate, label)
        if fact is not None:
            anchors.append(fact)
    if calibration is not None:
        anchors.append(
            AnchorFact(
                id="fact-calibration",
                category="calibration",
                fact=f"Calibration ({calibration.method}): {calibration.agreement} of {calibration.cases} cases agreed with the upstream control; {calibration.disagreements} disagreed.",
                evidence=[str(case) for case in calibration.details.get("case_ids", [])[:3]] or ["calibration"],
            )
        )

    limitations = [
        "Only the 1,052-question public half is scored; the leaderboard also uses a 1,052-question private half held by Kaggle, so this number is not a leaderboard submission.",
        "The grader (Gemini 2.5 Pro) samples; the three grades are averaged as the paper does, and the calibration re-judge shows how stable the labels are.",
        "The published starter's substring label parse fires on any mention of a label word inside the grader's explanation; the closing-line parse is reported alongside it.",
        "Reasoning tokens count against the output cap on this endpoint, so truncated or empty answers are graded as given (usually NOT_ATTEMPTED or MISTAKE).",
        "Kimi K3 answers may cite facts newer than the frozen gold answers; the grader is told to trust the gold answer.",
    ]
    blade = BladeMapping(
        d1_metrics={
            "pass_at_1": "N/A: the official metric is the mean of three graded samples, not a binary pass; see metric:accuracy",
            "primary": "metric:accuracy",
            "consistency": "N/A: one answer per question",
            "oracle_ceiling": "N/A: no repeats",
        },
        d2_anchor_categories=sorted({fact.category for fact in anchors}),
        d3_notes=[
            "Single-turn closed-book QA: no tool calls, no multi-turn funnel, no task root-cause taxonomy; judge validity and truncation are reported instead."
        ],
        not_applicable={
            "pass@k": "single rollout per question",
            "tool_call_funnel": "no tools",
            "task_root_cause": "no repeats or trajectories to compare",
        },
    )
    run = dict(run_info)
    run.setdefault("repeats", num_repeats)
    return NormalizedRun(
        benchmark=BENCHMARK,
        run=run,
        outcomes=outcomes,
        metrics=metrics,
        rollouts=rollouts,
        anchor_facts=anchors,
        calibration=calibration,
        limitations=limitations,
        reference_comparisons=REFERENCES,
        blade=blade,
        reward_semantics="reward = fraction of the three grader samples labelled CORRECT (0, 1/3, 2/3, 1); accuracy is its mean over the graded answers.",
    )
