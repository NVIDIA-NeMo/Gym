# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for datasets run by the generic verifiers agent."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_idx: int
    task_id: str | None = Field(
        default=None,
        description="Stable task identifier, when the benchmark provides one (AutomationBench: `<task_name>/<example_id>`).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    vf_env_id: str | None = None
    question: str = ""
    answer: str = ""
    task: str = "default"
    example_id: int | str = 0
    info: dict[str, Any] = Field(default_factory=dict)
