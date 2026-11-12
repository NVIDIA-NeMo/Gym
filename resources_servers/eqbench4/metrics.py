# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EQ-Bench 4 aggregate metrics: the policy's rating on the published leaderboard scale."""

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from resources_servers.eqbench4.rating import Outcome, fit_ratings, rate_with_intervals


# The evaluated model's name in the rating fit; published models are named by their provider IDs.
POLICY = "policy"
KEY_METRICS = (
    "eqbench4_elo",
    "eqbench4_elo_ci_low",
    "eqbench4_elo_ci_high",
    "eqbench4_rank",
    "eqbench4_win_rate",
)
# Why a policy gave no visible reply: the prompt overflowed its context window, its reply was cut at the output-token
# limit before any visible text (often all reasoning), a provider filter withheld it, or it simply ended empty.
POLICY_STOP_REASONS = ("context_limit", "output_budget", "content_filter", "empty_reply")
_STOP_ADVICE = {
    "context_limit": "serve the policy with a longer context window",
    "output_budget": "raise max_output_tokens or lower the policy's reasoning effort",
    "content_filter": "check the provider's content filter",
    "empty_reply": "inspect the policy's replies in the rollouts",
}


def dropped_conversation_metrics(verify_responses: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How many conversations were left out of the score because the policy or the persona stopped, over all rows."""
    total = len(verify_responses)
    stopped = [row for row in verify_responses if row.get("conversation_status") == "policy_failed"]
    metrics: dict[str, Any] = {
        "eqbench4_conversations": total,
        "eqbench4_policy_stopped_conversations": len(stopped),
        "eqbench4_policy_stopped_rate": len(stopped) / total if total else 0.0,
        "eqbench4_persona_failed_rate": (
            sum(row.get("conversation_status") == "persona_failed" for row in verify_responses) / total
            if total
            else 0.0
        ),
    }
    for reason in POLICY_STOP_REASONS:
        count = sum((row.get("policy_stop_reason") or "empty_reply") == reason for row in stopped)
        metrics[f"eqbench4_policy_stopped/{reason}"] = count
        metrics[f"eqbench4_policy_stopped_rate/{reason}"] = count / total if total else 0.0
    return metrics


def dropped_conversation_alerts(metrics: dict[str, Any]) -> list[str]:
    """Warnings for left-out conversations, naming each cause and what to check."""
    total = metrics["eqbench4_conversations"]
    alerts = []
    if metrics["eqbench4_policy_stopped_conversations"]:
        causes = "; ".join(
            f"{reason} {metrics[f'eqbench4_policy_stopped/{reason}']} ({_STOP_ADVICE[reason]})"
            for reason in POLICY_STOP_REASONS
            if metrics[f"eqbench4_policy_stopped/{reason}"]
        )
        alerts.append(
            f"EQ-Bench 4: the policy stopped replying in {metrics['eqbench4_policy_stopped_conversations']} of "
            f"{total} conversations ({metrics['eqbench4_policy_stopped_rate']:.1%}), which are left out of the "
            f"score: {causes}."
        )
    persona_failed = round(metrics["eqbench4_persona_failed_rate"] * total)
    if persona_failed:
        alerts.append(
            f"EQ-Bench 4: the persona simulator gave no usable reply in {persona_failed} of {total} conversations "
            f"({metrics['eqbench4_persona_failed_rate']:.1%}), which are left out of the score; check the persona "
            "model, which should be the leaderboard's for comparable ratings."
        )
    return alerts


def _rollouts(tasks: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [rollout for task in tasks for rollout in task]


def policy_outcomes(rollouts: Sequence[dict[str, Any]]) -> list[Outcome]:
    """One outcome per judged comparison, with the policy as model A."""
    return [
        Outcome(POLICY, comparison["opponent"], float(comparison["fraction"]))
        for rollout in rollouts
        for comparison in rollout.get("comparisons") or []
    ]


def _models(reference: Sequence[Outcome]) -> list[str]:
    return sorted({outcome.model_a for outcome in reference} | {outcome.model_b for outcome in reference} | {POLICY})


def eqbench4_metrics(
    tasks: Sequence[Sequence[dict[str, Any]]], reference: Sequence[Outcome], *, resamples: int
) -> dict[str, Any]:
    """The policy's Bradley-Terry rating among the published models, its 95% interval, and conversation statistics."""
    rollouts = _rollouts(tasks)
    outcomes = policy_outcomes(rollouts)
    metrics: dict[str, Any] = {
        "eqbench4_judged_conversations": len(rollouts),
        "eqbench4_comparisons": len(outcomes),
    }
    if rollouts:
        statuses = [rollout.get("conversation_status") for rollout in rollouts]
        metrics["eqbench4_persona_ended_rate"] = statuses.count("persona_ended") / len(rollouts)
        metrics["eqbench4_mean_policy_turns"] = sum(rollout.get("policy_turns", 0) for rollout in rollouts) / len(
            rollouts
        )
    traits: dict[str, list[float]] = defaultdict(list)
    for rollout in rollouts:
        for name, value in (rollout.get("trait_scores") or {}).items():
            traits[name].append(value)
    for name, values in sorted(traits.items()):
        metrics[f"eqbench4_trait/{name}"] = sum(values) / len(values)
    if not outcomes:
        return metrics

    models = _models(reference)
    ratings = rate_with_intervals(models, [*reference, *outcomes], resamples=resamples)
    policy = ratings[POLICY]
    metrics |= {
        "eqbench4_elo": policy.rating,
        "eqbench4_elo_ci_low": policy.ci_low,
        "eqbench4_elo_ci_high": policy.ci_high,
        "eqbench4_rank": 1 + sum(rating.rating > policy.rating for rating in ratings.values()),
        "eqbench4_models_ranked": len(ratings),
        "eqbench4_win_rate": sum(outcome.fraction_a for outcome in outcomes) / len(outcomes),
    }
    for model, rating in sorted(ratings.items(), key=lambda item: -item[1].rating):
        if model != POLICY:
            metrics[f"eqbench4_ladder/{model}"] = rating.rating
    return metrics


def eqbench4_repeat_metrics(tasks: Sequence[Sequence[dict[str, Any]]], reference: Sequence[Outcome]) -> dict[str, Any]:
    """The policy's rating and win rate from one repeat's comparisons alone (no interval)."""
    outcomes = policy_outcomes(_rollouts(tasks))
    if not outcomes:
        return {}
    ratings = fit_ratings(_models(reference), [*reference, *outcomes])
    return {
        "eqbench4_elo": ratings[POLICY],
        "eqbench4_win_rate": sum(outcome.fraction_a for outcome in outcomes) / len(outcomes),
    }


def key_metrics(agent_metrics: dict[str, Any]) -> dict[str, Any]:
    selected = {name: agent_metrics[name] for name in KEY_METRICS if name in agent_metrics}
    if "mean/reward" in agent_metrics:
        selected["mean/reward"] = agent_metrics["mean/reward"]
    return selected
