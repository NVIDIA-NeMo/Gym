# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LongFact task fields read by reset."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    # reset falls back to the session key without an ID, but every prepared row carries one.
    id: str | None = Field(default=None, json_schema_extra={"consumed_by": ["provenance"]})
    benchmark_variant: Literal["longfact-safe-v1"]
    prompt: str = Field(pattern=r"\S", json_schema_extra={"consumed_by": ["prompt", "verify"]})
    subset: Literal["objects", "concepts"] | None = Field(default=None, json_schema_extra={"consumed_by": ["metrics"]})
    topic: str | None = Field(default=None, json_schema_extra={"consumed_by": ["metrics"]})
    source_provenance: dict[str, Any] = Field(json_schema_extra={"consumed_by": ["provenance"]})
