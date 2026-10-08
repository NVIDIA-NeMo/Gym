# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for LMArena v2 and v3 prompt rows."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Shared by lmarena_v2 and lmarena_v3.
    question_id: str
    question: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    baseline_answer: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    baseline_model: str | None = None
    category: str = Field(json_schema_extra={"consumed_by": ["verify", "metrics"]})
    metadata: dict[str, Any] | None = Field(default=None, json_schema_extra={"consumed_by": ["prompt"]})
    prompt_slices: dict[str, list[str]] = Field(default_factory=dict, json_schema_extra={"consumed_by": ["metrics"]})
    self_comparison: bool = Field(default=False, json_schema_extra={"consumed_by": ["verify", "metrics"]})

    # lmarena_v3 reference-length scoring and human-battle provenance.
    style_reference_token_count: int | None = Field(default=None, json_schema_extra={"consumed_by": ["metrics"]})
    is_lmarena_v2_prompt: bool = Field(default=False, json_schema_extra={"consumed_by": ["metrics"]})
    other_answer: str | None = None
    other_model: str | None = None
    winner: str | None = None
