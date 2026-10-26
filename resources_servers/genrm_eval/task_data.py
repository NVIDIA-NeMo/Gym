# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dataset fields consumed by configurable GenRM preference scorers."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(json_schema_extra={"consumed_by": ["provenance"]})
    ground_truth_label: float | None = Field(
        default=None, json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"}
    )
    data_type: str | None = Field(
        default=None, json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"}
    )
    pair_id: str | None = Field(
        default=None, json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"}
    )
    source: str | None = Field(
        default=None, json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"}
    )
    category: str | None = Field(
        default=None, json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"}
    )
    subcategory: str | None = Field(
        default=None, json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"}
    )
    task: str | None = Field(
        default=None, json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"}
    )
    response_modality: str | None = Field(
        default=None, json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"}
    )
    label: str | None = Field(
        default=None, json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"}
    )
    correct: str | None = Field(
        default=None, json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"}
    )
    domain: str | None = Field(
        default=None, json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"}
    )
    response_1_style_value: int | None = Field(
        default=None, json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"}
    )
    response_2_style_value: int | None = Field(
        default=None, json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"}
    )
