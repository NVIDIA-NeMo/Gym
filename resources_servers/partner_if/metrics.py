# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark metrics owned by the scorer: all-IF pass and partial IF credit, overall and per origin.

Gym excludes masked (unscorable) rollouts from these rates and reports them under `coverage/*`.
"""

from typing import Any

from nemo_gym.reward_profile import (
    AggregateMetricsMixin,
    compute_pass_majority_metrics,
    compute_subset_metrics,
    highest_k_metrics,
)


ALL_IF_PASS = "all_if_pass"
IF_PARTIAL = "if_partial"


def score(rollout: dict[str, Any]) -> dict[str, float]:
    """The reward is 1 only when every IF rubric passes; partial credit is the passed fraction."""
    result = rollout.get("if_result") or {}
    # A rollout counted as zero without running has no IF result and earns no partial credit.
    partial = result["passed"] / result["total"] if result.get("total") else float(rollout["reward"])
    return {ALL_IF_PASS: float(rollout["reward"]), IF_PARTIAL: partial}


def origin_subset(rollout: dict[str, Any]) -> str | None:
    origin = ((rollout.get("verifier_metadata") or {}).get("partner_if") or {}).get("origin")
    return f"origin/{origin.casefold()}" if isinstance(origin, str) and origin else None


class PartnerIFMetricsMixin(AggregateMetricsMixin):
    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        metrics = compute_pass_majority_metrics(tasks, score_fn=score)[0]
        labeled = [[{**rollout, "origin": origin_subset(rollout)} for rollout in task] for task in tasks]
        metrics.update(compute_subset_metrics(labeled, subset_key="origin", score_fn=score))
        return metrics

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        return {
            **highest_k_metrics(agent_metrics, "pass@1[avg-of-{k}]", score_names=[ALL_IF_PASS, IF_PARTIAL]),
            **highest_k_metrics(agent_metrics, "pass@{k}", score_names=[ALL_IF_PASS]),
        }
