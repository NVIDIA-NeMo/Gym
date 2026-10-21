# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-free HarnessOptimizer used by examples and contract tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.agent_harness_optimization import (
    HarnessEvaluator,
    SystemPromptCandidate,
)
from nemo_gym.prompt import PromptConfig


class CandidateSweepOptimizerConfig(BaseModel):
    """Configured system prompts evaluated with native mean reward."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["candidate_sweep"] = "candidate_sweep"
    candidate_systems: list[str] = Field(default_factory=list)
    output_dir: str | None = None


class CandidateSweepIteration(BaseModel):
    """One candidate observation produced by the reference optimizer."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    iteration: int
    candidate: SystemPromptCandidate
    score: float
    feedback: dict[str, Any]


class CandidateSweepHarnessOptimizer:
    """Evaluate configured candidates and select the highest mean reward."""

    def __init__(self, config: CandidateSweepOptimizerConfig) -> None:
        self.config = config
        self.iterations: tuple[CandidateSweepIteration, ...] = ()

    async def optimize(
        self,
        initial_candidate: SystemPromptCandidate,
        evaluator: HarnessEvaluator[SystemPromptCandidate, PromptConfig],
    ) -> SystemPromptCandidate:
        observations: list[CandidateSweepIteration] = []
        best_candidate = initial_candidate
        best_score = float("-inf")

        for iteration, candidate in enumerate(self._candidates(initial_candidate)):
            rollouts = await evaluator.evaluate(candidate)
            score = sum(float(rollout["reward"]) for rollout in rollouts) / len(rollouts)
            feedback = {
                "num_rollouts": len(rollouts),
                "num_with_trajectory": sum("ng_trajectory" in rollout for rollout in rollouts),
                "failed_task_indices": [
                    rollout["_ng_task_index"] for rollout in rollouts if float(rollout["reward"]) <= 0
                ],
            }
            observation = CandidateSweepIteration(
                iteration=iteration,
                candidate=candidate,
                score=score,
                feedback=feedback,
            )
            observations.append(observation)
            self._write_iteration(observation, rollouts)

            if score > best_score:
                best_candidate = candidate
                best_score = score

        self.iterations = tuple(observations)
        return best_candidate

    def _candidates(self, initial_candidate: SystemPromptCandidate) -> Sequence[SystemPromptCandidate]:
        return [
            initial_candidate,
            *[SystemPromptCandidate.from_system(system=system) for system in self.config.candidate_systems],
        ]

    def _write_iteration(
        self,
        observation: CandidateSweepIteration,
        rollouts: Sequence[Mapping[str, Any]],
    ) -> None:
        if self.config.output_dir is None:
            return
        directory = Path(self.config.output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        artifact = observation.model_dump(mode="json") | {"rollouts": list(rollouts)}
        (directory / f"iteration_{observation.iteration:03d}.json").write_text(json.dumps(artifact, indent=2))
