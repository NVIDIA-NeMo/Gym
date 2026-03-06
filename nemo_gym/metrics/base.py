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


class MetricsOutput(BaseModel):
    """Structured output from metrics computation."""

    # Aggregate metrics keyed by aggregation mode
    # e.g. {"pass@1[avg-of-5]": {"accuracy": 85.0}, "pass@5": {"accuracy": 95.0}}
    aggregate: Dict[str, Dict[str, float]]

    # Per-sample aggregate: accuracy you'd get using only sample i
    # e.g. {"sample_0": {"accuracy": 82.0}, "sample_1": {"accuracy": 84.0}}
    per_sample_aggregate: Dict[str, Dict[str, float]]

    # Variance statistics per score name
    # e.g. {"accuracy": {"std_dev_across_runs": 2.24, "std_err_across_runs": 1.0, "avg_sample_std_dev": 0.35}}
    statistics: Dict[str, Dict[str, float]]

    # Per-task metrics (one entry per task with per-rollout scores and task-level aggregations)
    per_task: List[Dict[str, Any]]

    # Usage stats (token counts etc.)
    usage: Dict[str, float]


class BaseMetrics(ABC):
    @abstractmethod
    def get_score_dict(self, result: dict) -> Dict[str, Union[float, bool]]:
        """Extract named scores from a single verify result.

        Returns a dict mapping score names to values.
        Boolean values are treated as binary (0/1) for aggregation.
        """
        ...

    def get_answer(self, result: dict) -> Optional[str]:
        """Extract answer string for majority@k voting. Return None to skip majority@k."""
        return None

    def compute(self, task_results: List[List[dict]]) -> MetricsOutput:
        """Compute all metrics from grouped task results.

        Args:
            task_results: task_results[i] = list of k results for task i (sorted by rollout index).

        Returns:
            MetricsOutput with all computed metrics.
        """
        if not task_results:
            return MetricsOutput(aggregate={}, per_sample_aggregate={}, statistics={}, per_task=[], usage={})

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

        # Determine which scores are binary (all values are 0 or 1)
        is_binary = {name: True for name in score_names}
        for task_scores in all_score_dicts:
            for scores in task_scores:
                for name in score_names:
                    if name in scores and scores[name] not in (0, 0.0, 1, 1.0):
                        is_binary[name] = False

        # Compute aggregate metrics
        aggregate: Dict[str, Dict[str, float]] = {}

        # pass@k and avg-of-k for each k value
        for k_val in range(1, k + 1):
            pass_at_k = self._compute_pass_at_k(all_score_dicts, score_names, is_binary, k_val)
            if pass_at_k:
                aggregate[f"pass@{k_val}"] = pass_at_k

            avg_of_k = self._compute_avg_of_k(all_score_dicts, score_names, k_val)
            if avg_of_k:
                aggregate[f"pass@1[avg-of-{k_val}]"] = avg_of_k

        # majority@k for each k value (only if get_answer() returns non-None for some results)
        has_answers = any(any(a is not None for a in task_answers) for task_answers in all_answers)
        if has_answers:
            for k_val in range(1, k + 1):
                majority_at_k = self._compute_majority_at_k(all_score_dicts, all_answers, score_names, k_val)
                if majority_at_k:
                    aggregate[f"majority@{k_val}"] = majority_at_k

        # Per-sample aggregate
        per_sample_aggregate = self._compute_per_sample_aggregate(all_score_dicts, score_names, k)

        # Statistics
        statistics = self._compute_statistics(all_score_dicts, score_names, k)

        # Per-task details
        per_task = self._compute_per_task(task_results, all_score_dicts, all_answers, score_names, is_binary, k)

        # Usage
        usage = self._compute_usage(task_results)

        return MetricsOutput(
            aggregate=aggregate,
            per_sample_aggregate=per_sample_aggregate,
            statistics=statistics,
            per_task=per_task,
            usage=usage,
        )

    def _compute_pass_at_k(
        self,
        all_score_dicts: List[List[Dict[str, float]]],
        score_names: List[str],
        is_binary: Dict[str, bool],
        k: int,
    ) -> Dict[str, float]:
        """Compute pass@k for each score.

        For binary scores: uses combinatorial formula 1 - C(n-c, k) / C(n, k).
        For continuous scores: uses max of first k scores.
        """
        result = {}
        for name in score_names:
            values = []
            for task_scores in all_score_dicts:
                task_vals = [s.get(name) for s in task_scores if name in s]
                n = len(task_vals)
                if n == 0:
                    continue
                if k > n:
                    continue

                if is_binary[name]:
                    c = sum(1 for v in task_vals if v >= 1.0)
                    values.append(self._pass_at_k_combinatorial(n, c, k))
                else:
                    values.append(max(task_vals[:k]))

            if values:
                result[name] = 100.0 * sum(values) / len(values)

        return result

    @staticmethod
    def _pass_at_k_combinatorial(n: int, c: int, k: int) -> float:
        """Compute pass@k using 1 - C(n-c, k) / C(n, k).

        n: total number of samples
        c: number of correct samples
        k: k value
        """
        if n - c < k:
            return 1.0
        return 1.0 - math.comb(n - c, k) / math.comb(n, k)

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
                    if answer is not None and name in scores:
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
    ) -> Dict[str, Dict[str, float]]:
        """For each sample index i, compute the mean score across all tasks using only sample i."""
        result = {}
        for sample_idx in range(k):
            sample_scores: Dict[str, List[float]] = {name: [] for name in score_names}
            for task_scores in all_score_dicts:
                if sample_idx < len(task_scores):
                    for name in score_names:
                        if name in task_scores[sample_idx]:
                            sample_scores[name].append(task_scores[sample_idx][name])

            sample_agg = {}
            for name in score_names:
                if sample_scores[name]:
                    sample_agg[name] = 100.0 * sum(sample_scores[name]) / len(sample_scores[name])

            if sample_agg:
                result[f"sample_{sample_idx}"] = sample_agg

        return result

    def _compute_statistics(
        self,
        all_score_dicts: List[List[Dict[str, float]]],
        score_names: List[str],
        k: int,
    ) -> Dict[str, Dict[str, float]]:
        """Compute variance statistics per score name.

        - std_dev_across_runs: std dev of per-sample aggregate scores
        - std_err_across_runs: std_dev / sqrt(k)
        - avg_sample_std_dev: average per-task std dev across rollouts
        """
        if k <= 1:
            return {}

        result = {}

        # Per-sample aggregates (mean score per sample index)
        per_sample = self._compute_per_sample_aggregate(all_score_dicts, score_names, k)

        for name in score_names:
            sample_means = []
            for sample_key in sorted(per_sample.keys()):
                if name in per_sample[sample_key]:
                    sample_means.append(per_sample[sample_key][name])

            if len(sample_means) < 2:
                continue

            # std_dev_across_runs
            mean_of_means = sum(sample_means) / len(sample_means)
            variance = sum((x - mean_of_means) ** 2 for x in sample_means) / (len(sample_means) - 1)
            std_dev = math.sqrt(variance)

            # std_err_across_runs
            std_err = std_dev / math.sqrt(len(sample_means))

            # avg_sample_std_dev: average per-task std dev
            task_std_devs = []
            for task_scores in all_score_dicts:
                task_vals = [s.get(name) for s in task_scores if name in s]
                if len(task_vals) >= 2:
                    task_mean = sum(task_vals) / len(task_vals)
                    task_var = sum((v - task_mean) ** 2 for v in task_vals) / (len(task_vals) - 1)
                    task_std_devs.append(math.sqrt(task_var))

            avg_sample_std = 0.0
            if task_std_devs:
                avg_sample_std = sum(task_std_devs) / len(task_std_devs)

            result[name] = {
                "std_dev_across_runs": std_dev,
                "std_err_across_runs": std_err,
                "avg_sample_std_dev": avg_sample_std,
            }

        return result

    def _compute_per_task(
        self,
        task_results: List[List[dict]],
        all_score_dicts: List[List[Dict[str, float]]],
        all_answers: List[List[Optional[str]]],
        score_names: List[str],
        is_binary: Dict[str, bool],
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

                # pass@k for this task
                for k_val in range(1, min(k, n) + 1):
                    if is_binary[name]:
                        c = sum(1 for v in vals if v >= 1.0)
                        task_agg[f"pass@{k_val}/{name}"] = self._pass_at_k_combinatorial(n, c, k_val)
                    else:
                        task_agg[f"pass@{k_val}/{name}"] = max(vals[:k_val])

            task_entry["aggregations"] = task_agg
            per_task.append(task_entry)

        return per_task

    def _compute_usage(self, task_results: List[List[dict]]) -> Dict[str, float]:
        """Extract and average token counts from result["response"]["usage"]."""
        usage_sums: Dict[str, float] = {}
        usage_counts: Dict[str, int] = {}

        for results in task_results:
            for result in results:
                usage = (result.get("response") or {}).get("usage", {})
                for key, value in usage.items():
                    if isinstance(value, (int, float)):
                        usage_sums[key] = usage_sums.get(key, 0.0) + value
                        usage_counts[key] = usage_counts.get(key, 0) + 1

        return {key: usage_sums[key] / usage_counts[key] for key in sorted(usage_sums.keys())}
