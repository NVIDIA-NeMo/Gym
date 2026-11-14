# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Identity-only verification using UserSim's existing session lifecycle."""

from typing import Any

from fastapi import Request
from usersim.taxonomy.eval_cell import scorer_state

from benchmarks.model_identity.report import CORRECTNESS_METRICS, validate_correctness_metrics
from benchmarks.model_identity.scoring import OTHER_MISTAKE_KINDS, score_model_identity
from resources_servers.nemo_user_sim.app import UserSimResourcesServer, _ResourcesModelFacade
from resources_servers.nemo_user_sim.episode_contracts import (
    UserSimSimulationResult,
    UserSimVerification,
    UserSimVerifyRequest,
)


# The identity rates shown in Gym's run summary; the other-mistake kinds go to the full metrics only.
KEY_METRICS = (
    "mean/correct_developer",
    "mean/correct_model",
    "mean/false_attribution",
    "mean/other_mistake",
)


def _outcome(rollout: dict[str, Any]) -> dict[str, Any] | None:
    data = rollout.get("verifier_data") or (rollout.get("verification") or {}).get("verifier_data") or {}
    return (data.get("scorer_result") or {}).get("identity_outcome")


def identity_metrics(tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
    """Identity rates over the unmasked rollouts, as report.py computes them.

    No identity claim means 0 correctness; indirect questions can still pass.
    Missing scoring results count as 0 correctness and are excluded from mistake rates.
    """
    outcomes = [_outcome(rollout) for task in tasks for rollout in task]
    graded = [o for o in outcomes if o]
    for result in graded:
        validate_correctness_metrics(result)
    if not outcomes:
        return {}

    def mean(metric: str, pool: list[dict[str, Any]], total: int | None = None) -> float | None:
        size = len(pool) if total is None else total
        return sum(o["metrics"].get(metric, 0.0) for o in pool) / size if size else None

    metrics = {
        **{f"mean/{metric}": mean(metric, graded, len(outcomes)) for metric in CORRECTNESS_METRICS},
        "mean/false_attribution": mean("false_attribution", graded),
        "mean/other_mistake": mean("other_mistake", graded),
        **{f"mean/{kind}": mean(kind, graded) for kind in OTHER_MISTAKE_KINDS},
        # Samples with a model name outside UserSim's vocabulary.
        "mean/unrecognized_name": mean("unrecognized_name", graded),
    }
    return {key: value for key, value in metrics.items() if value is not None}


class ModelIdentityResourcesServer(UserSimResourcesServer):
    async def _evaluate(self, resolved_row: dict, usersim_result: UserSimSimulationResult) -> tuple:
        model = _ResourcesModelFacade(self, self.config.probe_scorer_model)
        scorer = await score_model_identity(
            {**resolved_row, **usersim_result.model_dump(mode="python")}, {"judge_model": model}
        )
        if scorer_state(scorer) == "not_applicable":
            scorer = {"error": "Identity judgment is missing", "status_proposal": None}
        # The inherited verifier gates this value on the scorer passing and masks
        # extraction errors. No conversation-quality evaluator is invoked.
        evaluation = {"scorers": {"identity_disclosure": scorer}, "skipped": True}
        return evaluation, {}, float(scorer.get("status_proposal") is True)

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return {**super().compute_metrics(tasks), **identity_metrics(tasks)}

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        """Gym's default (mean reward and coverage), then the main identity rates."""
        key = {k: v for k, v in super().get_key_metrics(agent_metrics).items() if not k.startswith("mean/")}
        return {
            "mean/reward": agent_metrics.get("mean/reward"),
            **{k: agent_metrics[k] for k in KEY_METRICS if k in agent_metrics},
            **key,
        }

    async def verify(self, request: Request, body: UserSimVerifyRequest) -> UserSimVerification:
        verified = await super().verify(request, body)
        verified.reward_components.pop("assistant_quality", None)
        return verified


if __name__ == "__main__":
    ModelIdentityResourcesServer.run_webserver()
