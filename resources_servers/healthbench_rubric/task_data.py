# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task fields shared by HealthBench preparation, collation, and verification."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    uuid: str | None = Field(default=None, json_schema_extra={"consumed_by": ["provenance"]})
    source_id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    conversation: list[dict[str, str]] = Field(json_schema_extra={"consumed_by": ["verify", "prompt"]})
    rubrics: list[dict[str, Any]] = Field(min_length=1, json_schema_extra={"consumed_by": ["verify"]})
    metadata: dict[str, Any] = Field(default_factory=dict, json_schema_extra={"consumed_by": ["provenance"]})

    @model_validator(mode="after")
    def validate_task(self) -> TaskData:
        """Require the same rubric and conversation contract before collation or a judge call."""
        for rubric in self.rubrics:
            if not isinstance(rubric.get("criterion"), str) or not rubric["criterion"].strip():
                raise ValueError("Each rubric requires non-empty criterion text")
            points = rubric.get("points")
            if isinstance(points, bool) or not isinstance(points, (int, float)) or not math.isfinite(points):
                raise ValueError("Rubric points must be finite numbers")
        if not any(rubric["points"] > 0 for rubric in self.rubrics):
            raise ValueError("Rubrics have no positive points")
        for index, message in enumerate(self.conversation):
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError(f"conversation[{index}] has an unsupported role")
            if "content" not in message:
                raise ValueError(f"conversation[{index}] requires content")
        if not self.conversation or self.conversation[-1].get("role") != "user":
            raise ValueError("Conversation must end with a user message")
        return self
