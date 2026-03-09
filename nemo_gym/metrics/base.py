# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


# Sentinel value for get_answer(): extraction was attempted but failed.
# Distinct from None (not applicable — metric doesn't support answer extraction).
NOT_FOUND = "__NOT_FOUND__"


class MetricsOutput(BaseModel):
    """Structured output from metrics computation."""

    # Aggregate metrics keyed by aggregation mode.
    # Statistics are fused into pass@1[avg-of-*] entries with {score}_{stat} keys.
    # Derived metrics (e.g. precision/recall from _add_derived_metrics) also live here.
    # e.g. {"pass@1[avg-of-5]": {"accuracy": 85.0, "accuracy_std_dev_across_runs": 1.58},
    #        "pass@5": {"accuracy": 95.0},
    #        "majority@5": {"accuracy": 88.0}}
    aggregate: Dict[str, Dict[str, Any]]

    # Per-sample pass@1 values: element i = pass@1 score using only rollout i from each task.
    # e.g. {"accuracy": [82.0, 84.0, 83.0]} — mean of this list = pass@1[avg-of-k].
    # Auto-statistics (std_dev, std_err) are computed from these values and fused into aggregate.
    per_sample_aggregate: Dict[str, List[float]]

    # Per-task metrics (one entry per task with per-rollout scores and task-level aggregations)
    per_task: List[Dict[str, Any]]

    # Usage stats with mean and std_dev per metric. Nested dicts (e.g. input_token_details)
    # are flattened with dot-separated keys.
    # e.g. {"prompt_tokens": {"mean": 110.0, "std_dev": 14.1},
    #        "prompt_tokens_details.cached_tokens": {"mean": 0.0, "std_dev": 0.0}}
    usage: Dict[str, Dict[str, float]]


def compute_statistics(
    per_sample_agg: Dict[str, List[float]],
) -> Dict[str, Dict[str, float]]:
    """Compute variance statistics for every key in per_sample_aggregate.

    Anything in per_sample_agg gets statistics. This is the universal contract:
    scores from get_score_dict(), no_answer, and derived metrics (precision/recall/F1)
    all get the same treatment as long as they're in per_sample_agg.

    Args:
        per_sample_agg: {metric_name: [value_using_rollout_0, value_using_rollout_1, ...]}

    Returns:
        {metric_name: {"std_dev_across_runs": ..., "std_err_across_runs": ...}}
    """
    result = {}

    for name, sample_values in per_sample_agg.items():
        if len(sample_values) < 2:
            continue

        mean = sum(sample_values) / len(sample_values)
        variance = sum((x - mean) ** 2 for x in sample_values) / (len(sample_values) - 1)
        std_dev = math.sqrt(variance)
        std_err = std_dev / math.sqrt(len(sample_values))

        result[name] = {
            "std_dev_across_runs": std_dev,
            "std_err_across_runs": std_err,
        }

    return result


class BaseMetrics(ABC):
    """Base class for computing metrics from rollout results.

    Three-tier usage:

    Tier 1 (simple): Override get_score_dict() only. The standard pipeline handles
    pass@k, avg-of-k, majority@k, per_sample_aggregate, statistics, and usage automatically.
    Optionally override get_answer() to enable majority@k voting.
    Examples: RewardMetrics, CodeGenMetrics, MathMetrics.

    Tier 2 (derived aggregates): Also override _add_derived_metrics() to add
    per-sample-decomposable metrics (e.g. precision/recall/F1) to per_sample_aggregate.
    For sample i, compute precision from TP/FP across all N tasks using only rollout i.
    These get auto-statistics like any other per_sample_aggregate key.
    Can also add per-aggregate-key values (e.g. FP/FN counts) to aggregate directly.
    Example: AnswerJudgementMetrics.

    Tier 3 (fully custom): Override compute() entirely to bypass the standard pipeline.
    Return a custom MetricsOutput with domain-specific structure. If you populate
    per_sample_aggregate, auto-statistics will be computed by the consumer.
    Example: ArenaMetrics.
    """

    @abstractmethod
    def get_score_dict(self, result: dict) -> Dict[str, Union[float, bool]]:
        """Extract named scores from a single verify result.

        Returns a dict mapping score names to values.
        Boolean values are treated as binary (0/1) for aggregation.
        """
        ...

    def get_answer(self, result: dict) -> Optional[str]:
        """Extract answer for majority@k voting.

        Returns:
            str: Answer found — participates in majority voting.
            NOT_FOUND: Extraction attempted but failed — excluded from voting,
                       counted toward the no_answer metric.
            None: Not applicable — this metric doesn't support answer extraction.
                  majority@k and no_answer are skipped entirely.
        """
        return None

    @property
    def no_answer_label(self) -> str:
        """Label for the no-answer metric in aggregate output.

        Override to rename (e.g. AnswerJudgementMetrics returns "invalid_judgements").
        """
        return "no_answer"

    @property
    def primary_metrics(self) -> Optional[List[str]]:
        """Score names to highlight in user-facing display. None = show all.

        Example: AnswerJudgementMetrics returns ["correct_judgements", "precision", "recall", "f1"].
        """
        return None

    @property
    def primary_evaluations(self) -> Optional[List[str]]:
        """Aggregate keys to highlight in user-facing display. None = show all.

        Example: For k=5, return ["pass@1[avg-of-5]", "majority@5", "pass@5"]
        to skip intermediate k values.
        """
        return None

    def _add_derived_metrics(
        self,
        per_sample_aggregate: Dict[str, List[float]],
        aggregate: Dict[str, Dict[str, Any]],
        task_results: List[List[dict]],
        all_score_dicts: List[List[Dict[str, float]]],
        all_answers: List[List[Optional[str]]],
    ) -> None:
        """Hook for subclasses to add derived metrics.

        Called BEFORE statistics are computed. Subclasses should:
        - Add per-sample-decomposable metrics (e.g. precision/recall/F1) to
          per_sample_aggregate. For sample i, compute the metric across all N tasks
          using only rollout i. These get auto-statistics.
        - Optionally add per-aggregate-key values (e.g. FP/FN counts) to aggregate
          directly. These won't get auto-statistics.
        """
        pass

    def compute(self, task_results: List[List[dict]]) -> MetricsOutput:
        """Compute all metrics from grouped task results.

        Args:
            task_results: task_results[i] = list of k results for task i (sorted by rollout index).

        Returns:
            MetricsOutput with all computed metrics.
        """
        if not task_results:
            return MetricsOutput(aggregate={}, per_sample_aggregate={}, per_task=[], usage={})

        k = max(len(results) for results in task_results)

        # Extract score dicts for all tasks and rollouts
        all_score_dicts: List[List[Dict[str, float]]] = []
        all_answers: List[List[Optional[str]]] = []
        for results in task_results:
            task_scores = []
            task_answers = []
            for result in results:
                raw_scores = self.get_score_dict(result)
                # Convert booleans to int
                scores = {name: int(v) if isinstance(v, bool) else v for name, v in raw_scores.items()}
                task_scores.append(scores)
                task_answers.append(self.get_answer(result))
            all_score_dicts.append(task_scores)
            all_answers.append(task_answers)

        # Collect all score names
        score_names = set()
        for task_scores in all_score_dicts:
            for scores in task_scores:
                score_names.update(scores.keys())
        score_names = sorted(score_names)

        # Compute aggregate metrics
        aggregate: Dict[str, Dict[str, float]] = {}

        # pass@k and avg-of-k for each k value
        for k_val in range(1, k + 1):
            pass_at_k = self._compute_pass_at_k(all_score_dicts, score_names, k_val)
            if pass_at_k:
                aggregate[f"pass@{k_val}"] = pass_at_k

            avg_of_k = self._compute_avg_of_k(all_score_dicts, score_names, k_val)
            if avg_of_k:
                aggregate[f"pass@1[avg-of-{k_val}]"] = avg_of_k

        # majority@k for each k value.
        # Only if get_answer() is overridden (returns non-None for some results).
        # NOT_FOUND answers are excluded from voting but counted in no_answer.
        has_answers = any(
            any(a is not None and a is not NOT_FOUND for a in task_answers) for task_answers in all_answers
        )
        if has_answers:
            for k_val in range(1, k + 1):
                majority_at_k = self._compute_majority_at_k(all_score_dicts, all_answers, score_names, k_val)
                if majority_at_k:
                    aggregate[f"majority@{k_val}"] = majority_at_k

        # Per-sample aggregate: scores from get_score_dict()
        per_sample_aggregate = self._compute_per_sample_aggregate(all_score_dicts, score_names, k)

        # Per-sample no_answer: for sample i, fraction of tasks where rollout i is NOT_FOUND
        if has_answers:
            no_answer_label = self.no_answer_label
            per_sample_no_answer = []
            for sample_idx in range(k):
                not_found_count = sum(
                    1
                    for task_answers in all_answers
                    if sample_idx < len(task_answers) and task_answers[sample_idx] is NOT_FOUND
                )
                total = sum(1 for task_answers in all_answers if sample_idx < len(task_answers))
                per_sample_no_answer.append(100.0 * not_found_count / total if total else 0.0)
            if any(v > 0 for v in per_sample_no_answer):
                per_sample_aggregate[no_answer_label] = per_sample_no_answer

        # Derived metrics hook — runs BEFORE statistics so subclasses can add
        # per-sample-decomposable metrics (e.g. precision/recall/F1) to per_sample_aggregate.
        # These then get auto-statistics.
        self._add_derived_metrics(per_sample_aggregate, aggregate, task_results, all_score_dicts, all_answers)

        # Fuse per_sample_aggregate means into pass@1[avg-of-k] and statistics into pass@1[avg-of-*].
        # Everything in per_sample_aggregate gets: mean → aggregate value, std/err → fused stats.
        if k > 1:
            statistics = compute_statistics(per_sample_aggregate)
            for agg_key in aggregate:
                if not agg_key.startswith("pass@1[avg-of-"):
                    continue
                # Fuse means for any per_sample_aggregate keys not already in aggregate
                for name, values in per_sample_aggregate.items():
                    if name not in aggregate[agg_key]:
                        aggregate[agg_key][name] = sum(values) / len(values)
                # Fuse statistics
                for score_name, stat_dict in statistics.items():
                    if score_name in aggregate[agg_key]:
                        for stat_name, stat_val in stat_dict.items():
                            aggregate[agg_key][f"{score_name}_{stat_name}"] = stat_val

        # Per-task details
        per_task = self._compute_per_task(task_results, all_score_dicts, all_answers, score_names, k)

        # Usage
        usage = self._compute_usage(task_results)

        return MetricsOutput(
            aggregate=aggregate,
            per_sample_aggregate=per_sample_aggregate,
            per_task=per_task,
            usage=usage,
        )

    def _compute_pass_at_k(
        self,
        all_score_dicts: List[List[Dict[str, float]]],
        score_names: List[str],
        k: int,
    ) -> Dict[str, float]:
        """Compute pass@k for each score: max of first k scores per task, averaged across tasks."""
        result = {}
        for name in score_names:
            values = []
            for task_scores in all_score_dicts:
                task_vals = [s.get(name) for s in task_scores if name in s]
                if len(task_vals) == 0 or k > len(task_vals):
                    continue
                values.append(max(task_vals[:k]))

            if values:
                result[name] = 100.0 * sum(values) / len(values)

        return result

    def _compute_avg_of_k(
        self,
        all_score_dicts: List[List[Dict[str, float]]],
        score_names: List[str],
        k: int,
    ) -> Dict[str, float]:
        """Compute pass@1[avg-of-k]: mean of first k scores per task, then averaged across tasks."""
        result = {}
        for name in score_names:
            values = []
            for task_scores in all_score_dicts:
                task_vals = [s.get(name) for s in task_scores[:k] if name in s]
                if task_vals:
                    values.append(sum(task_vals) / len(task_vals))

            if values:
                result[name] = 100.0 * sum(values) / len(values)

        return result

    def _compute_majority_at_k(
        self,
        all_score_dicts: List[List[Dict[str, float]]],
        all_answers: List[List[Optional[str]]],
        score_names: List[str],
        k: int,
    ) -> Dict[str, float]:
        """Compute majority@k: pick the most common answer, use its score."""
        result = {}
        for name in score_names:
            values = []
            for task_idx, (task_scores, task_answers) in enumerate(zip(all_score_dicts, all_answers)):
                # Collect (answer, score) pairs for first k results
                answer_scores = []
                for i, (scores, answer) in enumerate(zip(task_scores[:k], task_answers[:k])):
                    if answer is not None and answer is not NOT_FOUND and name in scores:
                        answer_scores.append((answer, scores[name]))

                if not answer_scores:
                    continue

                # Count answers
                answer_counter = Counter(a for a, _ in answer_scores)
                most_common_answer = answer_counter.most_common(1)[0][0]

                # Use the score of the first occurrence of the most common answer
                for answer, score in answer_scores:
                    if answer == most_common_answer:
                        values.append(score)
                        break

            if values:
                result[name] = 100.0 * sum(values) / len(values)

        return result

    def _compute_per_sample_aggregate(
        self,
        all_score_dicts: List[List[Dict[str, float]]],
        score_names: List[str],
        k: int,
    ) -> Dict[str, List[float]]:
        """Per-sample pass@1 values: element i = mean score across all tasks using only rollout i.

        Returns {"accuracy": [82.0, 84.0, ...]} where each element is a pass@1 percentage.
        mean(list) = pass@1[avg-of-k] in aggregate.
        """
        result: Dict[str, List[float]] = {name: [] for name in score_names}
        for sample_idx in range(k):
            sample_scores: Dict[str, List[float]] = {name: [] for name in score_names}
            for task_scores in all_score_dicts:
                if sample_idx < len(task_scores):
                    for name in score_names:
                        if name in task_scores[sample_idx]:
                            sample_scores[name].append(task_scores[sample_idx][name])

            for name in score_names:
                if sample_scores[name]:
                    result[name].append(100.0 * sum(sample_scores[name]) / len(sample_scores[name]))

        # Remove score names that had no data
        return {name: values for name, values in result.items() if values}

    def _compute_per_task(
        self,
        task_results: List[List[dict]],
        all_score_dicts: List[List[Dict[str, float]]],
        all_answers: List[List[Optional[str]]],
        score_names: List[str],
        k: int,
    ) -> List[Dict[str, Any]]:
        """Compute per-task metrics."""
        per_task = []
        for task_idx, (results, task_scores, task_answers) in enumerate(
            zip(task_results, all_score_dicts, all_answers)
        ):
            n = len(task_scores)
            task_entry: Dict[str, Any] = {"task_index": task_idx, "num_rollouts": n}

            # Per-rollout scores
            rollout_scores = []
            for i, (scores, answer) in enumerate(zip(task_scores, task_answers)):
                entry = {"rollout_index": i, **scores}
                if answer is not None:
                    entry["answer"] = answer
                rollout_scores.append(entry)
            task_entry["rollouts"] = rollout_scores

            # Task-level aggregations
            task_agg: Dict[str, float] = {}
            for name in score_names:
                vals = [s.get(name) for s in task_scores if name in s]
                if not vals:
                    continue
                task_agg[f"mean/{name}"] = sum(vals) / len(vals)
                task_agg[f"max/{name}"] = max(vals)
                task_agg[f"min/{name}"] = min(vals)

                # pass@k for this task: max of first k_val scores
                for k_val in range(1, min(k, n) + 1):
                    task_agg[f"pass@{k_val}/{name}"] = max(vals[:k_val])

            task_entry["aggregations"] = task_agg
            per_task.append(task_entry)

        return per_task

    def _compute_usage(self, task_results: List[List[dict]]) -> Dict[str, Dict[str, float]]:
        """Extract usage metrics with recursive dict flattening and mean+stdev."""
        all_values: Dict[str, List[float]] = {}

        def _flatten(d: dict, prefix: str = "") -> None:
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (int, float)):
                    all_values.setdefault(full_key, []).append(float(value))
                elif isinstance(value, dict):
                    _flatten(value, full_key)

        for results in task_results:
            for result in results:
                usage = (result.get("response") or {}).get("usage", {})
                _flatten(usage)

        output: Dict[str, Dict[str, float]] = {}
        for key in sorted(all_values):
            vals = all_values[key]
            mean = sum(vals) / len(vals)
            if len(vals) >= 2:
                std_dev = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
            else:
                std_dev = 0.0
            output[key] = {"mean": mean, "std_dev": std_dev}
        return output
