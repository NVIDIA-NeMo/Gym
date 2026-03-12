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
"""Pluggable metrics system for computing pass@k, majority@k, and per-sample statistics.

Three tiers of customization:
  Tier 1: Override get_score_dict() for custom scores. Optionally get_answer() for majority@k.
  Tier 2: Also override _add_derived_metrics() for cross-sample metrics (precision/recall/F1).
  Tier 3: Override compute() entirely for fully custom aggregation (e.g. arena win-rate).
"""

import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


# Sentinel: extraction attempted but failed. Distinct from None (not applicable).
NOT_FOUND = "__NOT_FOUND__"


class MetricsOutput(BaseModel):
    """Structured output from BaseMetrics.compute()."""

    aggregate: Dict[str, Dict[str, Any]]
    """Keyed by mode: pass@k, pass@1[avg-of-k], majority@k. Values are {score_name: value}.
    Statistics (std_dev/std_err) fused into pass@1[avg-of-*] entries."""

    per_sample_aggregate: Dict[str, List[float]]
    """Element i = pass@1 using only rollout i from each task. Auto-statistics computed from this."""

    per_task: List[Dict[str, Any]]
    """Per-task rollout scores and aggregations."""

    usage: Dict[str, Dict[str, float]]
    """Token usage: {metric: {mean, std_dev}}. Nested dicts flattened with dot keys."""


def compute_statistics(per_sample_agg: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Compute std_dev/std_err across runs for each key in per_sample_aggregate."""
    result = {}
    for name, values in per_sample_agg.items():
        if len(values) < 2:
            continue
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        std_dev = math.sqrt(variance)
        result[name] = {
            "std_dev_across_runs": std_dev,
            "std_err_across_runs": std_dev / math.sqrt(len(values)),
        }
    return result


class BaseMetrics(ABC):
    """Base class for computing metrics from grouped rollout results."""

    @abstractmethod
    def get_score_dict(self, result: dict) -> Dict[str, Union[float, bool]]:
        """Extract named scores from a single verify result. Bools become 0/1."""
        ...

    def get_answer(self, result: dict) -> Optional[str]:
        """Extract answer for majority@k. Returns str, NOT_FOUND, or None (skip majority)."""
        return None

    @property
    def no_answer_label(self) -> str:
        """Label for the no-answer metric. Override to rename."""
        return "no_answer"

    def _add_derived_metrics(
        self,
        per_sample_aggregate: Dict[str, List[float]],
        aggregate: Dict[str, Dict[str, Any]],
        task_results: List[List[dict]],
        all_score_dicts: List[List[Dict[str, float]]],
        all_answers: List[List[Optional[str]]],
    ) -> None:
        """Hook: add derived metrics (precision/recall/F1) to per_sample_aggregate before statistics."""
        pass

    def compute(self, task_results: List[List[dict]]) -> MetricsOutput:
        """Compute all metrics. task_results[i] = list of k results for task i."""
        if not task_results:
            return MetricsOutput(aggregate={}, per_sample_aggregate={}, per_task=[], usage={})

        k = max(len(results) for results in task_results)

        # Extract scores and answers
        all_score_dicts: List[List[Dict[str, float]]] = []
        all_answers: List[List[Optional[str]]] = []
        for results in task_results:
            task_scores, task_answers = [], []
            for result in results:
                raw = self.get_score_dict(result)
                task_scores.append({n: int(v) if isinstance(v, bool) else v for n, v in raw.items()})
                task_answers.append(self.get_answer(result))
            all_score_dicts.append(task_scores)
            all_answers.append(task_answers)

        score_names = sorted({n for ts in all_score_dicts for s in ts for n in s})
        aggregate: Dict[str, Dict[str, float]] = {}

        # pass@k and pass@1[avg-of-k] for each k value
        for k_val in range(1, k + 1):
            pass_k, avg_k = self._compute_pass_and_avg(all_score_dicts, score_names, k_val)
            if pass_k:
                aggregate[f"pass@{k_val}"] = pass_k
            if avg_k:
                aggregate[f"pass@1[avg-of-{k_val}]"] = avg_k

        # majority@k (only if get_answer is overridden)
        has_answers = any(any(a is not None and a is not NOT_FOUND for a in ta) for ta in all_answers)
        if has_answers:
            for k_val in range(1, k + 1):
                maj = self._compute_majority_at_k(all_score_dicts, all_answers, score_names, k_val)
                if maj:
                    aggregate[f"majority@{k_val}"] = maj

        # Per-sample aggregate
        per_sample_aggregate = self._compute_per_sample_aggregate(all_score_dicts, score_names, k)

        # Per-sample no_answer
        if has_answers:
            no_answer_vals = []
            for sample_idx in range(k):
                not_found = sum(1 for ta in all_answers if sample_idx < len(ta) and ta[sample_idx] is NOT_FOUND)
                total = sum(1 for ta in all_answers if sample_idx < len(ta))
                no_answer_vals.append(100.0 * not_found / total if total else 0.0)
            if any(v > 0 for v in no_answer_vals):
                per_sample_aggregate[self.no_answer_label] = no_answer_vals

        # Derived metrics hook (runs before statistics)
        self._add_derived_metrics(per_sample_aggregate, aggregate, task_results, all_score_dicts, all_answers)

        # Fuse statistics into pass@1[avg-of-*] entries
        if k > 1:
            statistics = compute_statistics(per_sample_aggregate)
            for agg_key in aggregate:
                if not agg_key.startswith("pass@1[avg-of-"):
                    continue
                for name, values in per_sample_aggregate.items():
                    if name not in aggregate[agg_key]:
                        aggregate[agg_key][name] = sum(values) / len(values)
                for score_name, stat_dict in statistics.items():
                    if score_name in aggregate[agg_key]:
                        for stat_name, stat_val in stat_dict.items():
                            aggregate[agg_key][f"{score_name}_{stat_name}"] = stat_val

        per_task = self._compute_per_task(all_score_dicts, all_answers, score_names)
        usage = self._compute_usage(task_results)

        return MetricsOutput(
            aggregate=aggregate,
            per_sample_aggregate=per_sample_aggregate,
            per_task=per_task,
            usage=usage,
        )

    def _compute_pass_and_avg(
        self,
        all_score_dicts,
        score_names,
        k,
    ) -> tuple[Dict[str, float], Dict[str, float]]:
        """Compute pass@k (max of first k) and pass@1[avg-of-k] (mean of first k) in one pass."""
        pass_k: Dict[str, float] = {}
        avg_k: Dict[str, float] = {}
        for name in score_names:
            pass_vals, avg_vals = [], []
            for task_scores in all_score_dicts:
                task_vals = [s.get(name) for s in task_scores if name in s]
                if task_vals:
                    first_k = task_vals[:k]
                    avg_vals.append(sum(first_k) / len(first_k))
                    if len(task_vals) >= k:
                        pass_vals.append(max(first_k))
            if pass_vals:
                pass_k[name] = 100.0 * sum(pass_vals) / len(pass_vals)
            if avg_vals:
                avg_k[name] = 100.0 * sum(avg_vals) / len(avg_vals)
        return pass_k, avg_k

    def _compute_majority_at_k(self, all_score_dicts, all_answers, score_names, k) -> Dict[str, float]:
        """Majority@k: pick most common answer among first k rollouts, use its score."""
        result = {}
        for name in score_names:
            values = []
            for task_scores, task_answers in zip(all_score_dicts, all_answers):
                answer_scores = [
                    (a, s[name])
                    for s, a in zip(task_scores[:k], task_answers[:k])
                    if a is not None and a is not NOT_FOUND and name in s
                ]
                if not answer_scores:
                    continue
                most_common = Counter(a for a, _ in answer_scores).most_common(1)[0][0]
                values.append(next(score for a, score in answer_scores if a == most_common))
            if values:
                result[name] = 100.0 * sum(values) / len(values)
        return result

    def _compute_per_sample_aggregate(self, all_score_dicts, score_names, k) -> Dict[str, List[float]]:
        """Element i = pass@1 using only rollout i across all tasks."""
        result: Dict[str, List[float]] = {name: [] for name in score_names}
        for sample_idx in range(k):
            for name in score_names:
                vals = [
                    ts[sample_idx][name] for ts in all_score_dicts if sample_idx < len(ts) and name in ts[sample_idx]
                ]
                if vals:
                    result[name].append(100.0 * sum(vals) / len(vals))
        return {name: values for name, values in result.items() if values}

    def _compute_per_task(self, all_score_dicts, all_answers, score_names) -> List[Dict[str, Any]]:
        """Per-task rollout scores and mean/max/min aggregations."""
        per_task = []
        for task_idx, (task_scores, task_answers) in enumerate(zip(all_score_dicts, all_answers)):
            rollouts = []
            for i, (scores, answer) in enumerate(zip(task_scores, task_answers)):
                entry = {"rollout_index": i, **scores}
                if answer is not None:
                    entry["answer"] = answer
                rollouts.append(entry)

            agg: Dict[str, float] = {}
            for name in score_names:
                vals = [s.get(name) for s in task_scores if name in s]
                if vals:
                    agg[f"mean/{name}"] = sum(vals) / len(vals)
                    agg[f"max/{name}"] = max(vals)
                    agg[f"min/{name}"] = min(vals)

            per_task.append(
                {
                    "task_index": task_idx,
                    "num_rollouts": len(task_scores),
                    "rollouts": rollouts,
                    "aggregations": agg,
                }
            )
        return per_task

    def _compute_usage(self, task_results: List[List[dict]]) -> Dict[str, Dict[str, float]]:
        """Token usage stats with recursive dict flattening."""
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
                _flatten((result.get("response") or {}).get("usage", {}))

        output: Dict[str, Dict[str, float]] = {}
        for key in sorted(all_values):
            vals = all_values[key]
            mean = sum(vals) / len(vals)
            std_dev = math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) >= 2 else 0.0
            output[key] = {"mean": mean, "std_dev": std_dev}
        return output
