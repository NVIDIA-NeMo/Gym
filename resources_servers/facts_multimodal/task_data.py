# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for FACTS Multimodal rows."""

from pydantic import BaseModel, ConfigDict, Field


class RubricItem(BaseModel):
    """One human-authored atomic fact from the public benchmark release."""

    fact: str
    tags: list[str] = Field(default_factory=list)


class TaskData(BaseModel):
    """Fields consumed by the FACTS Multimodal verifier and aggregate metrics."""

    model_config = ConfigDict(extra="allow")

    id: str = Field(json_schema_extra={"consumed_by": ["provenance"]})
    prompt: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    image_url: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    image_path: str = Field(json_schema_extra={"consumed_by": ["provenance"]})
    image_mime_type: str = Field(json_schema_extra={"consumed_by": ["provenance"]})
    image_data_url: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    rubric_items: list[RubricItem] = Field(json_schema_extra={"consumed_by": ["verify"]})
    prompt_category: str = Field(json_schema_extra={"consumed_by": ["metrics"]})
    image_category: str = Field(json_schema_extra={"consumed_by": ["metrics"]})
    user_intent_majority: str = Field(json_schema_extra={"consumed_by": ["metrics"]})
    external_information_majority: str = Field(json_schema_extra={"consumed_by": ["metrics"]})
    reasoning_requirement_majority: str = Field(json_schema_extra={"consumed_by": ["metrics"]})
