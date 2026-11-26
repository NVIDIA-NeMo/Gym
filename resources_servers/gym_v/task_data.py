# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the Gym-V resources server.

Rows are flat and select one Gym-V environment plus the arguments needed to reproduce its
initial state.  The agent forwards those fields as ``GymVTaskRow`` to ``/seed_session``;
``env_id`` and ``seed`` are wire-required, while ``env_kwargs`` and the bookkeeping fields use
the same defaults as that request model.  ``horizon_cap`` is also consumed by the agent so
recovery-only turns cannot bypass the task's step budget.

``task_idx`` is preparation-time routing metadata emitted by NeMo Gym's collation and mixed-
manifest builders. It is optional because inline rows do not need an index, but declaring it
keeps dependency-light validation aligned with the Gym-V agent and seed-session wire models.

``responses_create_params`` is deliberately absent: it is a framework-owned row key removed by
``normalize_task_fields`` before validation, not task data owned by this resources server.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class InfoBoolSuccessCriterion(BaseModel):
    """Read a canonical boolean emitted by an audited environment."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["info_bool"]
    path: list[str] = Field(min_length=1)
    require_env_terminated: bool = True


class TerminalStepRewardSuccessCriterion(BaseModel):
    """Classify one audited environment from its terminal raw step reward."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["terminal_step_reward"]
    comparison: Literal["equal", "greater_than"]
    threshold: float = Field(allow_inf_nan=False)
    absolute_tolerance: float = Field(default=1e-9, ge=0, allow_inf_nan=False)


SuccessCriterion = Annotated[
    Union[InfoBoolSuccessCriterion, TerminalStepRewardSuccessCriterion],
    Field(discriminator="type"),
]


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    env_id: str = Field(
        description="Gym-V registry identifier, for example 'Games/FrozenLake-v0'.",
        json_schema_extra={"consumed_by": ["verify", "metrics", "provenance"]},
    )
    env_kwargs: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-task arguments passed to gym_v.make(), overriding server defaults.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    seed: int = Field(
        description="Seed passed to env.reset() to reproduce the initial state.",
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    task_id: str | None = Field(
        default=None,
        description="Optional stable identifier for the generated task row.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    horizon_cap: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum number of rollout turns, enforced by both agent and server.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Open task-generation metadata retained for analysis and provenance.",
        json_schema_extra={"consumed_by": ["metrics", "provenance"]},
    )
    success_criterion: SuccessCriterion | None = Field(
        default=None,
        description=(
            "Audited binary success predicate for this exact env/config. Omit only for legacy "
            "reward-only consumers; mixed-suite task accuracy requires this field."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    task_idx: int | None = Field(
        default=None,
        ge=0,
        description="Optional row index emitted by manifest builders and accepted by legacy routing.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
