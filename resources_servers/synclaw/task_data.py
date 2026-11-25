# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the BioNeMo-SynClaw server.

Retrosynthesis rows carry their task fields top-level (there is no verifier_metadata).
Required-ness mirrors ``SynClawRunRequest`` (schema.py): ``task_id``, ``target_smiles``,
``system_prompt`` and ``question`` are required on the wire. The row's ``excluded_tools``
adds to the agent's exclusions; ``available_tools`` remains informational.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


TaskBudget = Annotated[int, Field(strict=True, gt=0)] | Literal["inf"]


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description="BioNeMo-SynClaw task identifier (e.g. 'enamine-0249'); forwarded to the rollout loop for tracing.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    target_smiles: str = Field(
        description=(
            "SMILES of the molecule to synthesize. Passed to tool calls as the target and scored against "
            "the submitted pathway's final product in verify()."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    system_prompt: str = Field(
        description="System prompt text; /seed_session builds the initial messages from it.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    question: str = Field(
        description="User prompt naming the target molecule; /seed_session builds the initial messages from it.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    excluded_tools: list[str] = Field(
        default_factory=list,
        description="Tools hidden from the model for this task, on top of the agent's excluded_tools config.",
        json_schema_extra={"consumed_by": ["prompt", "tools"]},
    )
    available_tools: list[str] | None = Field(
        default=None,
        description=(
            "Tool names derived from the task's excluded_tools by BioNeMo-SynClaw's dataset builder; "
            "informational only."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    max_turns: TaskBudget | None = Field(
        None,
        description="Assistant-turn budget; 'inf' removes the cap. null inherits the agent setting.",
        json_schema_extra={"consumed_by": ["prompt", "policy", "transport"]},
    )
    max_tool_calls: TaskBudget | None = Field(
        None,
        description="Tool-call budget; 'inf' removes the cap. null inherits the agent setting.",
        json_schema_extra={"consumed_by": ["prompt", "policy", "transport"]},
    )
    max_model_len: TaskBudget | None = Field(
        None,
        description="Task context limit; may lower the agent ceiling. 'inf' hides but enforces it. null inherits it.",
        json_schema_extra={"consumed_by": ["prompt", "policy", "transport"]},
    )
    max_tokens: TaskBudget | None = Field(
        None,
        description="Per-response output cap, including reasoning; 'inf' uses remaining context. null inherits the agent setting.",
        json_schema_extra={"consumed_by": ["prompt", "policy", "transport"]},
    )
    budget_feedback: bool | None = Field(
        None,
        strict=True,
        description="Include budget progress reminders; enforcement always applies. null inherits the agent setting.",
        json_schema_extra={"consumed_by": ["prompt", "policy", "transport"]},
    )
