# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config-driven workflow for shared HarnessOptimizer implementations."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.agent_harness_optimization import (
    HarnessEvaluator,
    SystemPromptCandidate,
    SystemPromptConfigAdapter,
)
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.prompt import PromptConfig
from nemo_gym.reference_harness_optimizer import (
    CandidateSweepHarnessOptimizer,
    CandidateSweepOptimizerConfig,
)


class HarnessRolloutConfig(BaseModel):
    """Fixed rollout settings used for every candidate comparison."""

    model_config = ConfigDict(extra="forbid")

    agent_name: str
    head_server: BaseServerConfig
    num_repeats: int = Field(default=1, ge=1)
    max_concurrency: int = Field(default=1, ge=1)


class SystemPromptTargetConfig(BaseModel):
    """A system-prompt optimization target and its seed candidate."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["prompt_config.system"]
    module_name: str
    prompt_config: PromptConfig

    @model_validator(mode="after")
    def require_initial_system_prompt(self) -> SystemPromptTargetConfig:
        if self.prompt_config.system is None:
            raise ValueError("prompt_config.system is required as the initial candidate")
        return self


class HarnessOptimizationRecipe(BaseModel):
    """Typed configuration for a system-prompt optimization run."""

    model_config = ConfigDict(extra="forbid")

    train_jsonl_fpath: str
    validation_jsonl_fpath: str | None = None
    train_limit: int | None = Field(default=None, ge=1)
    validation_limit: int | None = Field(default=None, ge=1)
    rollout: HarnessRolloutConfig
    target: SystemPromptTargetConfig
    optimizer: CandidateSweepOptimizerConfig


def load_source_rows(path: str, limit: int | None) -> list[dict[str, Any]]:
    """Load prompt-agnostic copies of native Gym dataset rows."""

    rows: list[dict[str, Any]] = []
    with Path(path).open() as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} must contain a JSON object")
            row["responses_create_params"] = {
                key: value for key, value in row.get("responses_create_params", {}).items() if key != "input"
            }
            rows.append(row)
            if limit is not None and len(rows) == limit:
                break
    if not rows:
        raise ValueError(f"no rows found in {path}")
    return rows


def build_system_prompt_evaluator(
    recipe: HarnessOptimizationRecipe,
    train_rows: Sequence[Mapping[str, Any]],
    validation_rows: Sequence[Mapping[str, Any]] | None = None,
) -> HarnessEvaluator[SystemPromptCandidate, PromptConfig]:
    """Construct the evaluator for the recipe's declared target."""

    return HarnessEvaluator(
        train_rows=train_rows,
        validation_rows=validation_rows,
        config_adapter=SystemPromptConfigAdapter(
            module_name=recipe.target.module_name,
            base_prompt=recipe.target.prompt_config,
        ),
        num_repeats=recipe.rollout.num_repeats,
        agent_name=recipe.rollout.agent_name,
        head_server_config=recipe.rollout.head_server,
        max_concurrency=recipe.rollout.max_concurrency,
    )


def build_optimizer(recipe: HarnessOptimizationRecipe) -> CandidateSweepHarnessOptimizer:
    """Construct the typed reference HarnessOptimizer."""

    return CandidateSweepHarnessOptimizer(recipe.optimizer)


async def run_harness_optimization(
    recipe: HarnessOptimizationRecipe,
) -> Mapping[str, Any]:
    """Run the configured optimizer and freeze its selected candidate."""

    train_rows = load_source_rows(recipe.train_jsonl_fpath, recipe.train_limit)
    validation_rows = (
        load_source_rows(recipe.validation_jsonl_fpath, recipe.validation_limit)
        if recipe.validation_jsonl_fpath is not None
        else None
    )
    evaluator = build_system_prompt_evaluator(recipe, train_rows, validation_rows)
    optimizer = build_optimizer(recipe)
    initial_system = recipe.target.prompt_config.system
    assert initial_system is not None  # Validated by SystemPromptTargetConfig.
    initial_candidate = SystemPromptCandidate.from_system(system=initial_system)
    selected_candidate = await optimizer.optimize(initial_candidate, evaluator)
    frozen_artifact = evaluator.freeze(selected_candidate)
    return {
        "optimizer": recipe.optimizer.type,
        "selected_candidate": selected_candidate.model_dump(mode="json"),
        "frozen_artifact": frozen_artifact.model_dump(mode="json"),
        "iterations": [iteration.model_dump(mode="json") for iteration in optimizer.iterations],
    }
