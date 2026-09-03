# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the self-contained tau2 agent (no resources server).

``Tau2RunRequest`` (app.py) is the strictest agent wire model: it requires ``config`` (a tau2
``TextRunConfig``), ``task`` (a tau2 ``Task``), ``seed``, ``evaluation_type``, and pins the
remaining knobs to constants via ``Literal`` (the harness only supports text simulation with
full review and no audio). ``config``/``task`` stay ``Dict[str, Any]`` here because their full
shapes belong to the tau2 package and this module must stay dependency-light; the wire validates
them precisely.
"""

from typing import Any, Dict, Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    config: Dict[str, Any] = Field(
        description="tau2 TextRunConfig dict (domain, task_split_name, llm settings, ...).",
        json_schema_extra={"consumed_by": ["run"]},
    )
    task: Dict[str, Any] = Field(
        description="tau2 Task dict (id, description, user_scenario, evaluation_criteria, ...).",
        json_schema_extra={"consumed_by": ["run", "verify"]},
    )
    seed: int = Field(
        description="Simulation seed for the user model.",
        json_schema_extra={"consumed_by": ["run"]},
    )
    evaluation_type: str = Field(
        description="tau2 evaluation mode; committed rows use 'all'.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    save_dir: Literal[None] = Field(
        description="Wire-pinned to null; tau2 result saving is managed by the harness.",
        json_schema_extra={"consumed_by": ["run"]},
    )
    user_voice_settings: Literal[None] = Field(
        description="Wire-pinned to null; audio simulation is unsupported.",
        json_schema_extra={"consumed_by": ["run"]},
    )
    user_persona_config: Literal[None] = Field(
        description="Wire-pinned to null.",
        json_schema_extra={"consumed_by": ["run"]},
    )
    verbose_logs: Literal[False] = Field(
        description="Wire-pinned to false.",
        json_schema_extra={"consumed_by": ["run"]},
    )
    audio_debug: Literal[False] = Field(
        description="Wire-pinned to false; audio simulation is unsupported.",
        json_schema_extra={"consumed_by": ["run"]},
    )
    audio_taps: Literal[False] = Field(
        description="Wire-pinned to false; audio simulation is unsupported.",
        json_schema_extra={"consumed_by": ["run"]},
    )
    auto_review: Literal[False] = Field(
        description="Wire-pinned to false.",
        json_schema_extra={"consumed_by": ["run"]},
    )
    review_mode: Literal["full"] = Field(
        description="Wire-pinned to 'full'.",
        json_schema_extra={"consumed_by": ["run"]},
    )
    hallucination_feedback: Literal[None] = Field(
        description="Wire-pinned to null.",
        json_schema_extra={"consumed_by": ["run"]},
    )
