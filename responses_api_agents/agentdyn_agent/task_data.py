# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the self-contained agentdyn_agent (no resources server).

A row selects one AgentDyn benchmark selector. The suite and user task say which task to run; adding
an injection task and an attack turns that same task into its attacked counterpart, which is why
the attacked and clean arms share a schema rather than being two row types.

Required-ness mirrors `AgentDynRunRequest` in `app.py`, not what the rollout happens to read.
Only `suite` and `user_task_id` are required there: everything else has a wire default, and
`defense` in particular is normally left unset so a run config's `default_defense` supplies the
treatment without duplicating the 620-row task matrix per defense.

The literals are the wire's literals. A row naming a suite or defense the adapter does not
implement is a dataset bug that should fail here, at validation, rather than twenty minutes
into an evaluation run.
"""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    # `extra="allow"` matches the run request: rows may carry fields the agent does not read.
    model_config = ConfigDict(extra="allow")

    suite: Literal["shopping", "github", "dailylife"] = Field(
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    user_task_id: str = Field(
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    # None marks the clean arm of the matrix; a string names the injection task to run alongside
    # the user task. The adapter rejects this without `attack`, and `attack` without this.
    injection_task_id: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    attack: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    defense: Optional[
        Literal[
            "prompt_guard_2_detector",
            "piguard_detector",
            "camel",
            "progent",
            "drift",
            "transformers_pi_detector",
            "spotlighting_with_delimiting",
            "repeat_user_prompt",
            "tool_filter",
        ]
    ] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    # Which arm the row belongs to. Written by `prepare.py` for readability and filtering; the
    # adapter does not read it -- the arm is determined by whether `injection_task_id` is set.
    condition: Optional[Literal["clean", "attacked"]] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    benchmark_version: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    # Defense-scoped provenance. These are not part of the task matrix -- they record which
    # detector weights and which policy endpoint a treatment actually used, so a row stays
    # attributable after the fact.
    external_model_base_url: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    detector_model_name: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    detector_model_revision: Optional[str] = Field(
        default=None,
        json_schema_extra={"consumed_by": ["provenance"]},
    )
