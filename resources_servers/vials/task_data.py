# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VIALS task fields: the question, its ground-truth final answer and the optional numeric bounds."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


_VM = {"legacy_location": "verifier_metadata"}


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(json_schema_extra=_VM | {"consumed_by": ["provenance"]})
    question: str = Field(json_schema_extra=_VM | {"consumed_by": ["verify"]})
    gtfa: str = Field(json_schema_extra=_VM | {"consumed_by": ["verify"]})
    domain: str = Field(json_schema_extra=_VM | {"consumed_by": ["metrics"]})
    secondary_tags: list[str] = Field(default_factory=list, json_schema_extra=_VM | {"consumed_by": ["provenance"]})
    answer_lower_bound: Optional[float] = Field(default=None, json_schema_extra=_VM | {"consumed_by": ["verify"]})
    answer_upper_bound: Optional[float] = Field(default=None, json_schema_extra=_VM | {"consumed_by": ["verify"]})
