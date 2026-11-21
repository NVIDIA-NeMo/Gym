# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""EQ-Bench 4 aggregate metrics: the policy's rating on the published leaderboard scale."""

import logging
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from resources_servers.eqbench4.upstream_rating import Outcome, UpstreamSolver


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RatingInputs:
    """The published comparisons and the leaderboard's own solver, both written by the benchmark's prepare."""

    reference: Sequence[Outcome]
    solver: UpstreamSolver


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
    "empty_reply": (
        "inspect the policy's replies in the rollouts; a server reasoning parser hides the whole reply as reasoning "
        "when the model does not close </think>"
    ),
}


def policy_stop_metrics(verify_responses: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """How many conversations the policy stopped, each scored as a loss to every opponent, and why, over all rows."""
    total = len(verify_responses)
    stopped = [row for row in verify_responses if row.get("conversation_status") == "policy_failed"]
    metrics: dict[str, Any] = {
        "eqbench4_conversations": total,
        "eqbench4_policy_stopped_conversations": len(stopped),
        "eqbench4_policy_stopped_rate": len(stopped) / total if total else 0.0,
    }
    for reason in POLICY_STOP_REASONS:
        count = sum((row.get("policy_stop_reason") or "empty_reply") == reason for row in stopped)
        metrics[f"eqbench4_policy_stopped/{reason}"] = count
        metrics[f"eqbench4_policy_stopped_rate/{reason}"] = count / total if total else 0.0
    return metrics


def policy_stop_alerts(metrics: dict[str, Any]) -> list[str]:
    """A warning when the policy stopped any conversation, naming each cause and what to check."""
    if not metrics["eqbench4_policy_stopped_conversations"]:
        return []
    causes = "; ".join(
        f"{reason} {metrics[f'eqbench4_policy_stopped/{reason}']} ({_STOP_ADVICE[reason]})"
        for reason in POLICY_STOP_REASONS
        if metrics[f"eqbench4_policy_stopped/{reason}"]
    )
    return [
        f"EQ-Bench 4: the policy stopped replying in {metrics['eqbench4_policy_stopped_conversations']} of "
        f"{metrics['eqbench4_conversations']} conversations ({metrics['eqbench4_policy_stopped_rate']:.1%}), each "
        f"scored as a loss to every opponent: {causes}."
    ]


def _rollouts(tasks: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [rollout for task in tasks for rollout in task]


def _policy_stopped(rollout: dict[str, Any]) -> bool:
    """Whether the row is a policy stop the resources server scored as a loss without a judge (as in ``verify``)."""
    return rollout.get("conversation_status") == "policy_failed" or rollout.get("policy_turns") == 0


def policy_outcomes(rollouts: Sequence[dict[str, Any]]) -> list[Outcome]:
    """One outcome per judged comparison, with the policy as model A."""
    return [
        Outcome(POLICY, comparison["opponent"], float(comparison["fraction"]))
        for rollout in rollouts
        for comparison in rollout.get("comparisons") or []
    ]


def _models(reference: Sequence[Outcome]) -> list[str]:
    return sorted({outcome.model_a for outcome in reference} | {outcome.model_b for outcome in reference} | {POLICY})


def _rateable(
    outcomes: Sequence[Outcome], rating_inputs: Callable[[], RatingInputs], metrics: dict[str, Any]
) -> tuple[RatingInputs, list[str]] | None:
    """The rating inputs and the fit's models, or None, with a warning and the count in ``metrics``, when the policy
    cannot be rated: the prepared files are missing, or some comparisons are against a model the published graph does
    not hold (the synthetic example rows' opponent), which the fit would leave out."""
    try:
        inputs = rating_inputs()
    except (FileNotFoundError, ValueError) as error:
        logger.warning("EQ-Bench 4: no rating is reported: %s", error)
        return None
    models = _models(inputs.reference)
    unrated = [outcome for outcome in outcomes if outcome.model_b not in models]
    if unrated:
        metrics["eqbench4_unrated_comparisons"] = len(unrated)
        logger.warning(
            "EQ-Bench 4: %d of %d comparisons are against models missing from the published comparisons (%s), which "
            "the fit would leave out; no rating is reported",
            len(unrated),
            len(outcomes),
            ", ".join(sorted({outcome.model_b for outcome in unrated})[:5]),
        )
        return None
    return inputs, models


def eqbench4_metrics(
    tasks: Sequence[Sequence[dict[str, Any]]], rating_inputs: Callable[[], RatingInputs], *, resamples: int
) -> dict[str, Any]:
    """The policy's Bradley-Terry rating among the published models, its 95% interval, and conversation statistics.

    The rating and win rate count every scored comparison, a policy stop's losses included; the conversation
    statistics describe the judged conversations only, which leaves out the policy stops."""
    rollouts = _rollouts(tasks)
    outcomes = policy_outcomes(rollouts)
    judged = [rollout for rollout in rollouts if not _policy_stopped(rollout)]
    metrics: dict[str, Any] = {
        "eqbench4_judged_conversations": len(judged),
        "eqbench4_comparisons": len(outcomes),
    }
    if judged:
        statuses = [rollout.get("conversation_status") for rollout in judged]
        metrics["eqbench4_persona_ended_rate"] = statuses.count("persona_ended") / len(judged)
        metrics["eqbench4_mean_policy_turns"] = sum(rollout.get("policy_turns", 0) for rollout in judged) / len(judged)
    traits: dict[str, list[float]] = defaultdict(list)
    for rollout in judged:
        for name, value in (rollout.get("trait_scores") or {}).items():
            traits[name].append(value)
    for name, values in sorted(traits.items()):
        metrics[f"eqbench4_trait/{name}"] = sum(values) / len(values)
    if not outcomes:
        return metrics
    metrics["eqbench4_win_rate"] = sum(outcome.fraction_a for outcome in outcomes) / len(outcomes)
    rateable = _rateable(outcomes, rating_inputs, metrics)
    if rateable is None:
        return metrics
    inputs, models = rateable
    ratings = inputs.solver.rate_with_intervals(models, [*inputs.reference, *outcomes], resamples=resamples)
    policy = ratings[POLICY]
    metrics |= {
        "eqbench4_elo": policy.rating,
        "eqbench4_elo_ci_low": policy.ci_low,
        "eqbench4_elo_ci_high": policy.ci_high,
        "eqbench4_rank": 1 + sum(rating.rating > policy.rating for rating in ratings.values()),
        "eqbench4_models_ranked": len(ratings),
    }
    for model, rating in sorted(ratings.items(), key=lambda item: -item[1].rating):
        if model != POLICY:
            metrics[f"eqbench4_ladder/{model}"] = rating.rating
    return metrics


def eqbench4_repeat_metrics(
    tasks: Sequence[Sequence[dict[str, Any]]], rating_inputs: Callable[[], RatingInputs]
) -> dict[str, Any]:
    """The policy's rating and win rate from one repeat's comparisons alone (no interval)."""
    outcomes = policy_outcomes(_rollouts(tasks))
    if not outcomes:
        return {}
    metrics: dict[str, Any] = {"eqbench4_win_rate": sum(outcome.fraction_a for outcome in outcomes) / len(outcomes)}
    rateable = _rateable(outcomes, rating_inputs, metrics)
    if rateable is None:
        return metrics
    inputs, models = rateable
    return metrics | {"eqbench4_elo": inputs.solver.fit(models, [*inputs.reference, *outcomes])[POLICY]}


def key_metrics(agent_metrics: dict[str, Any]) -> dict[str, Any]:
    selected = {name: agent_metrics[name] for name in KEY_METRICS if name in agent_metrics}
    if "mean/reward" in agent_metrics:
        selected["mean/reward"] = agent_metrics["mean/reward"]
    return selected
