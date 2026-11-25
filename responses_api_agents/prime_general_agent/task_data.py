# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for general-agent rows, run by the verifiers agent (see verifiers_agent/task_data.py)."""

from pydantic import BaseModel, ConfigDict, Field


class GeneralAgentInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_dir: str = Field(
        description="Task directory relative to the general-agent package's tasks/ dir, resolved by the agent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    vf_env_id: str | None = None
    answer: str = ""
    example_id: int | str = 0
    # The local solver loads the task's tools, DB and gold solution from info.task_dir.
    info: GeneralAgentInfo
