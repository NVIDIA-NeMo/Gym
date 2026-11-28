# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent schema for the original AGA export; unknown provenance is retained."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")
    expected_answer: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    image_tools_base_agent_ref: dict[str, str] = Field(json_schema_extra={"consumed_by": ["verify"]})
    question: str | None = None
    source_task: str | None = None
    id: str | int | None = None
    dataset: str | None = None
    verifier: Any = None
    source_tool_names: list[str] | None = None
    pass_rate: float | None = None
    notools_pass_rate: float | None = None
    pass_rate_source_line: int | None = None
    notools_source_line: int | None = None
    metadata: dict[str, Any] | None = None
    options: list[dict[str, str | None]] | None = None
    grading_mode: str | None = None
    extraction_mode: str | None = None
    case_sensitive: bool | None = None
    aga_family: Literal["code_assisted_vqa", "agent_x"]
    source_images: dict[str, str] = Field(default_factory=dict, json_schema_extra={"consumed_by": ["seed"]})
