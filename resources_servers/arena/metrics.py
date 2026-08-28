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
"""Shared metric computation for online evaluation and saved rollouts."""

import logging
from collections import defaultdict
from typing import Any, Protocol

import numpy as np
import tiktoken

from resources_servers.arena.arena import (
    _VERDICT_LABEL_BOTH_BAD,
    _VERDICT_LABELS_TIE,
    _bootstrap_per_category,
    _compute_raw_style_feature,
    _weighted_scores_as_a,
    _weighted_scores_as_b,
)
from resources_servers.arena.taxonomy import MIN_SLICE_PROMPTS, PROMPT_CATEGORY_ORDER


logger = logging.getLogger(__name__)
DECISIVE_VERDICTS = {"[[A>>B]]", "[[A>B]]", "[[B>A]]", "[[B>>A]]"}


class MetricsConfig(Protocol):
    verdict_weight: int
    max_rollout_failure_rate: float
    generation_only: bool
    score_both_bad_as_tie: bool
    style_control_method: str
    style_length_ratio_range: tuple[float, float] | None
    style_short_reference_max_tokens: int | None
    style_short_response_max_tokens: int | None
    style_norm_mean: dict[str, list[float]] | None
    style_norm_std: dict[str, list[float]] | None
    style_coefs: dict[str, list[float]] | None
    tokenizer_model: str
    bootstrap_rounds: int
    bootstrap_seed: int


class ArenaMetrics:
    """Compute the same metrics for online and saved rollouts."""

    def __init__(self, config: MetricsConfig) -> None:
        self.config = config
        self._bradley_terry_constants = None
        if config.style_control_method == "bradley_terry":
            means, stds, coefs = config.style_norm_mean, config.style_norm_std, config.style_coefs
            if not means or not stds or not coefs:
                raise ValueError("Bradley-Terry style constants are required")
            self._bradley_terry_constants = {
                category: (np.array(mean), np.array(stds[category]), np.array(coefs[category]))
                for category, mean in means.items()
            }
        elif config.style_control_method == "reference_length":
            ratio_range = config.style_length_ratio_range
            if ratio_range is None or not 0 <= ratio_range[0] < ratio_range[1]:
                raise ValueError("style_length_ratio_range must satisfy 0 <= lower < upper")
            if (config.style_short_reference_max_tokens is None) != (config.style_short_response_max_tokens is None):
                raise ValueError("both short-reference limits must be set or both must be null")
        else:
            raise NotImplementedError(f"Unknown style_control_method: {config.style_control_method}")

    def compute(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        """Compute metrics; each task contains repeated rollouts for one prompt."""
        total_rollouts = failed_rollouts = self_comparison_rollouts = 0
        missing_judgment_rollouts = parse_failure_rollouts = 0
        reasoning_only_response_rollouts = max_token_reached_rollouts = context_window_exceeded_rollouts = 0
        any_both_bad_rollouts = any_tie_rollouts = 0
        verbosity_accepted_rollouts = verbosity_eligible_rollouts = 0

        # Each rollout contributes its two swapped-order judge scores.
        scores_by_category: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        offsets_by_category: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        controlled_scores_by_category: defaultdict[str, list[np.ndarray]] = defaultdict(list)
        # Optional prompt slices, populated when validation metadata provides labels.
        slice_prompt_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        slice_scores: defaultdict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        slice_offsets: defaultdict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        slice_controlled_scores: defaultdict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
        slice_verbosity_counts: defaultdict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        lmarena_v2_prompt_scores: list[np.ndarray] = []
        lmarena_v2_prompt_offsets: list[np.ndarray] = []
        lmarena_v2_prompt_controlled_scores: list[np.ndarray] = []
        response_tokens: list[int] = []
        reasoning_tokens: list[int] = []
        encoding = tiktoken.encoding_for_model(self.config.tokenizer_model)

        # Collect valid judge scores and the data needed for each aggregate.
        for rollouts in tasks:
            # Slice membership belongs to the prompt and is shared by repeated rollouts.
            prompt_slices = rollouts[0].get("prompt_slices") or {}
            slice_keys = {(namespace, label) for namespace, labels in prompt_slices.items() for label in labels}
            task_games = [game for rollout in rollouts for game in rollout.get("games") or []]
            if task_games and all(game.get("verdict") in DECISIVE_VERDICTS for game in task_games):
                slice_keys.add(("arena", "exclude-ties"))
            for key in slice_keys:
                slice_prompt_counts[key] += 1

            for rollout in rollouts:
                response = rollout.get("policy_answer")
                reasoning = rollout.get("policy_reasoning")
                if "policy_answer" in rollout:
                    response_tokens.append(len(encoding.encode(response, disallowed_special=())) if response else 0)
                if "policy_reasoning" in rollout:
                    reasoning_tokens.append(len(encoding.encode(reasoning, disallowed_special=())) if reasoning else 0)

                if self.config.generation_only:
                    continue
                if rollout.get("self_comparison"):
                    self_comparison_rollouts += 1
                    continue

                # Only complete, parsed judge pairs contribute to either win rate.
                total_rollouts += 1
                reasoning_only_response_rollouts += int(bool(reasoning) and not response)
                incomplete_reason = ((rollout.get("response") or {}).get("incomplete_details") or {}).get("reason")
                if self.config.style_control_method == "reference_length" and incomplete_reason == "max_output_tokens":
                    # Context-window rejections have no usage because generation never started.
                    if (rollout.get("response") or {}).get("usage") is None:
                        context_window_exceeded_rollouts += 1
                    else:
                        max_token_reached_rollouts += 1
                    category = rollout.get("category")
                    if not category:
                        raise ValueError("category is required for scoring")

                    # A response that reaches the generation limit is a zero, not an infrastructure failure.
                    scores = np.zeros(2)
                    scores_by_category[category].append(scores)
                    if rollout.get("is_lmarena_v2_prompt"):
                        lmarena_v2_prompt_scores.append(scores)
                    for key in slice_keys:
                        slice_scores[key].append(scores)

                    controlled_scores_by_category[category].append(scores)
                    verbosity_eligible_rollouts += 1
                    if rollout.get("is_lmarena_v2_prompt"):
                        lmarena_v2_prompt_controlled_scores.append(scores)
                    for key in slice_keys:
                        slice_controlled_scores[key].append(scores)
                        slice_verbosity_counts[key][1] += 1
                    continue

                games = rollout.get("games") or []
                missing_judgment = len(games) != 2
                if not missing_judgment:
                    missing_judgment = any(
                        "response" in (game or {}) and not ((game or {}).get("response") or {}).get("output")
                        for game in games
                    )
                parse_failure = not missing_judgment and any((game or {}).get("verdict") is None for game in games)
                if missing_judgment:
                    missing_judgment_rollouts += 1
                if parse_failure:
                    parse_failure_rollouts += 1
                if missing_judgment or parse_failure:
                    failed_rollouts += 1

                verdicts = [(game or {}).get("verdict") for game in games]
                if any(verdict == _VERDICT_LABEL_BOTH_BAD for verdict in verdicts):
                    any_both_bad_rollouts += 1
                if any(verdict in _VERDICT_LABELS_TIE or verdict == _VERDICT_LABEL_BOTH_BAD for verdict in verdicts):
                    any_tie_rollouts += 1
                if missing_judgment or parse_failure:
                    continue

                verdict_a, verdict_b = verdicts
                if not self.config.score_both_bad_as_tie and _VERDICT_LABEL_BOTH_BAD in verdicts:
                    continue

                scores = _weighted_scores_as_a(verdict_a, self.config.verdict_weight)
                scores += _weighted_scores_as_b(verdict_b, self.config.verdict_weight)
                scores = np.asarray(scores, dtype=np.float64)
                category = rollout.get("category")
                if not category:
                    raise ValueError("category is required for scoring")
                scores_by_category[category].append(scores)
                if rollout.get("is_lmarena_v2_prompt"):
                    lmarena_v2_prompt_scores.append(scores)
                for key in slice_keys:
                    slice_scores[key].append(scores)

                # lmarena_v3 uses reference lengths; lmarena_v2 uses Bradley-Terry.
                if self.config.style_control_method == "reference_length":
                    reference_tokens = rollout.get("style_reference_token_count")
                    if not reference_tokens:
                        raise ValueError("style_reference_token_count is required for reference_length")
                    policy_tokens = len(encoding.encode(response or "", disallowed_special=()))
                    accepted = self._is_comparable_to_reference_length(policy_tokens, reference_tokens)
                    verbosity_eligible_rollouts += 1
                    verbosity_accepted_rollouts += int(accepted)
                    controlled_scores = scores if accepted else np.zeros_like(scores)
                    controlled_scores_by_category[category].append(controlled_scores)
                    if rollout.get("is_lmarena_v2_prompt"):
                        lmarena_v2_prompt_controlled_scores.append(controlled_scores)
                    for key in slice_keys:
                        slice_controlled_scores[key].append(controlled_scores)
                        slice_verbosity_counts[key][0] += int(accepted)
                        slice_verbosity_counts[key][1] += 1
                elif self.config.style_control_method == "bradley_terry":
                    policy_text = rollout.get("policy_answer")
                    baseline_text = rollout.get("baseline_answer")
                    if not policy_text or not baseline_text:
                        raise ValueError("policy_answer and baseline_answer are required for bradley_terry")
                    mean, std, coefs = self._get_bradley_terry_constants(category)
                    feature = _compute_raw_style_feature(policy_text, baseline_text, encoding)
                    offset = float((feature - mean) / std @ coefs)
                    repeated_offsets = np.full(len(scores), offset)
                    offsets_by_category[category].append(repeated_offsets)
                    if rollout.get("is_lmarena_v2_prompt"):
                        lmarena_v2_prompt_offsets.append(repeated_offsets)
                    for key in slice_keys:
                        slice_offsets[key].append(repeated_offsets)
                else:
                    raise NotImplementedError(f"Unknown style_control_method: {self.config.style_control_method}")

        # Token statistics are also available for generation-only runs.
        metrics = self._token_metrics(response_tokens, reasoning_tokens)
        if self.config.generation_only:
            return metrics

        # Report diagnostics before computing scores.
        if self_comparison_rollouts:
            logger.warning(
                "%d self-comparison rollout(s) excluded from scoring (baseline and policy from the same model).",
                self_comparison_rollouts,
            )
        if total_rollouts == 0:
            return {}

        failure_rate = failed_rollouts / total_rollouts
        if failure_rate > self.config.max_rollout_failure_rate:
            raise ValueError(
                f"Too many failed rollouts: {failed_rollouts}/{total_rollouts} "
                f"({failure_rate * 100:.1f}%) exceeds max_rollout_failure_rate="
                f"{self.config.max_rollout_failure_rate * 100:.1f}%. "
                "Check gen_answer and gen_judgment API errors."
            )

        if self.config.style_control_method == "reference_length":
            metrics["max_token_reached_rate"] = max_token_reached_rollouts / total_rollouts
            metrics["context_window_exceeded_rate"] = context_window_exceeded_rollouts / total_rollouts
        metrics.update(
            {
                "rollout_failure_rate": failure_rate,
                "reasoning_only_response_rate": reasoning_only_response_rollouts / total_rollouts,
                "missing_judgment_rate": missing_judgment_rollouts / total_rollouts,
                "parse_failure_rate": parse_failure_rollouts / total_rollouts,
                "any_both_bad_rate": any_both_bad_rollouts / total_rollouts,
                "any_tie_rate": any_tie_rollouts / total_rollouts,
            }
        )
        if not scores_by_category:
            return metrics

        no_sc, no_sc_lower, no_sc_upper = self._bootstrap(scores_by_category)
        metrics.update(
            win_rate_no_SC=no_sc,
            win_rate_no_SC_ci95_lower=no_sc_lower,
            win_rate_no_SC_ci95_upper=no_sc_upper,
        )

        if self.config.style_control_method == "reference_length":
            metrics["verbosity_acceptance_rate"] = verbosity_accepted_rollouts / verbosity_eligible_rollouts
            controlled, controlled_lower, controlled_upper = self._bootstrap(controlled_scores_by_category)
        elif self.config.style_control_method == "bradley_terry":
            controlled, controlled_lower, controlled_upper = self._bootstrap(scores_by_category, offsets_by_category)
        else:
            raise NotImplementedError(f"Unknown style_control_method: {self.config.style_control_method}")
        metrics.update(
            win_rate=controlled,
            win_rate_ci95_lower=controlled_lower,
            win_rate_ci95_upper=controlled_upper,
        )

        # Also report lmarena_v3 on the prompts inherited from lmarena_v2.
        if lmarena_v2_prompt_scores:
            metrics["win_rate_no_SC_lmarena_v2_prompts"] = self._bootstrap_point(lmarena_v2_prompt_scores)
            if self.config.style_control_method == "reference_length":
                controlled_lmarena_v2_score = self._bootstrap_point(lmarena_v2_prompt_controlled_scores)
            elif self.config.style_control_method == "bradley_terry":
                controlled_lmarena_v2_score = self._bootstrap_point(
                    lmarena_v2_prompt_scores, lmarena_v2_prompt_offsets
                )
            else:
                raise NotImplementedError(f"Unknown style_control_method: {self.config.style_control_method}")
            metrics["win_rate_lmarena_v2_prompts"] = controlled_lmarena_v2_score

        # Add sufficiently large metadata-defined prompt slices.
        self._add_slice_metrics(
            metrics,
            len(tasks),
            slice_prompt_counts,
            slice_scores,
            slice_offsets,
            slice_controlled_scores,
            slice_verbosity_counts,
        )
        return metrics

    def get_key_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Select and round the metrics printed by NeMo Gym."""
        primary_metric_order = (
            "total_prompts",
            "win_rate",
            "win_rate_ci95_lower",
            "win_rate_ci95_upper",
            "win_rate_no_SC",
            "win_rate_no_SC_ci95_lower",
            "win_rate_no_SC_ci95_upper",
            "win_rate_lmarena_v2_prompts",
            "win_rate_no_SC_lmarena_v2_prompts",
        )
        token_metric_order = tuple(
            f"{kind}/{stat}"
            for kind in ("response_tokens", "reasoning_tokens")
            for stat in ("mean", "median", "min", "max", "p5", "p95")
        )
        diagnostic_metric_order = (
            "mean/reward",
            "max_token_reached_rate",
            "context_window_exceeded_rate",
            "rollout_failure_rate",
            "reasoning_only_response_rate",
            "missing_judgment_rate",
            "parse_failure_rate",
            "any_both_bad_rate",
            "any_tie_rate",
            "verbosity_acceptance_rate",
        )
        names = (
            token_metric_order
            if self.config.generation_only
            else primary_metric_order + token_metric_order + diagnostic_metric_order
        )
        if not self.config.generation_only:
            names += tuple(
                name
                for namespace in ("arena", "taxonomy-language", "taxonomy-task-type")
                for name in metrics
                if name.startswith(f"{namespace}/")
            )

        key_metrics = {}
        for name in names:
            if name not in metrics:
                continue
            value = metrics[name]
            if name.startswith(("response_tokens/", "reasoning_tokens/")):
                value = round(value)
            elif isinstance(value, (float, np.floating)):
                value = round(float(value), 4)
            key_metrics[name] = value
        return key_metrics

    def _get_bradley_terry_constants(self, category: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._bradley_terry_constants is None or category not in self._bradley_terry_constants:
            raise ValueError(f"No Bradley-Terry settings configured for category {category!r}")
        return self._bradley_terry_constants[category]

    def _is_comparable_to_reference_length(self, policy_tokens: int, reference_tokens: int) -> bool:
        short_reference_max = self.config.style_short_reference_max_tokens
        if short_reference_max is not None and reference_tokens <= short_reference_max:
            return policy_tokens <= self.config.style_short_response_max_tokens
        lower, upper = self.config.style_length_ratio_range
        return lower < policy_tokens / reference_tokens < upper

    def _bootstrap(
        self, scores: dict[str, list[np.ndarray]], offsets: dict[str, list[np.ndarray]] | None = None
    ) -> tuple[float, float, float]:
        return _bootstrap_per_category(
            scores,
            cat_offsets=offsets,
            n_rounds=self.config.bootstrap_rounds,
            seed=self.config.bootstrap_seed,
        )

    def _bootstrap_point(self, scores: list[np.ndarray], offsets: list[np.ndarray] | None = None) -> float:
        values = {"all": scores}
        style_offsets = None if offsets is None else {"all": offsets}
        return self._bootstrap(values, style_offsets)[0]

    @staticmethod
    def _token_metrics(response: list[int], reasoning: list[int]) -> dict[str, Any]:
        metrics = {}
        for name, counts in (("response_tokens", response), ("reasoning_tokens", reasoning)):
            if not counts:
                continue
            values = np.asarray(counts)
            metrics.update(
                {
                    f"{name}/mean": float(np.mean(values)),
                    f"{name}/median": float(np.median(values)),
                    f"{name}/min": int(np.min(values)),
                    f"{name}/max": int(np.max(values)),
                    f"{name}/p5": float(np.percentile(values, 5)),
                    f"{name}/p95": float(np.percentile(values, 95)),
                }
            )
        return metrics

    def _add_slice_metrics(
        self,
        metrics: dict[str, Any],
        total_prompts: int,
        prompt_counts: dict[tuple[str, str], int],
        scores: dict[tuple[str, str], list[np.ndarray]],
        offsets: dict[tuple[str, str], list[np.ndarray]],
        controlled_scores: dict[tuple[str, str], list[np.ndarray]],
        verbosity_counts: dict[tuple[str, str], list[int]],
    ) -> None:
        if not prompt_counts:
            return

        # Slice scores reuse the same scoring and bootstrap rules as the overall score.
        metrics["total_prompts"] = total_prompts
        metrics["arena/overall/win_rate"] = metrics["win_rate"]
        metrics["arena/overall/win_rate_no_SC"] = metrics["win_rate_no_SC"]
        if "verbosity_acceptance_rate" in metrics:
            metrics["arena/overall/verbosity_acceptance_rate"] = metrics["verbosity_acceptance_rate"]

        arena_order = {name: index for index, name in enumerate((*PROMPT_CATEGORY_ORDER, "exclude-ties"))}

        def sort_key(key: tuple[str, str]) -> tuple[int, int, str]:
            namespace, label = key
            namespace_order = {"arena": 0, "taxonomy-language": 1, "taxonomy-task-type": 2}
            if namespace == "arena":
                return namespace_order[namespace], arena_order.get(label, len(arena_order)), label
            return namespace_order.get(namespace, 3), -prompt_counts[key], label

        for key in sorted(prompt_counts, key=sort_key):
            prompts = prompt_counts[key]
            if prompts < MIN_SLICE_PROMPTS or key not in scores:
                continue
            namespace, label = key
            no_sc = self._bootstrap_point(scores[key])
            if self.config.style_control_method == "reference_length":
                controlled = self._bootstrap_point(controlled_scores[key])
            elif self.config.style_control_method == "bradley_terry":
                controlled = self._bootstrap_point(scores[key], offsets[key])
            else:
                raise NotImplementedError(f"Unknown style_control_method: {self.config.style_control_method}")
            prefix = f"{namespace}/{label}"
            metrics[f"{prefix}/prompts"] = prompts
            metrics[f"{prefix}/win_rate"] = controlled
            metrics[f"{prefix}/win_rate_no_SC"] = no_sc
            if key in verbosity_counts:
                accepted, eligible = verbosity_counts[key]
                metrics[f"{prefix}/verbosity_acceptance_rate"] = accepted / eligible
