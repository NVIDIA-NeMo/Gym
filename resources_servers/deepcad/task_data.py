# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task fields shared by dataset validation and the runtime verifier."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")
    reference_operations: list[str] = Field(
        default_factory=list, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    task_id: str = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    reference_step: str = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    view_mode: Literal["axis", "diagonal"] = Field(
        default="axis", json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    foreground_color: Literal["blue", "green", "teal", "steel_blue", "violet", "amber"] | None = Field(
        default=None, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    input_augmentation_seed: int | None = Field(
        default=None, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    engineering_annotations: bool = Field(default=True, json_schema_extra={"legacy_location": "verifier_metadata"})
    difficulty: int | None = Field(default=None, ge=0, json_schema_extra={"legacy_location": "verifier_metadata"})
    construction_metrics: dict[str, int] | None = Field(
        default=None, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
