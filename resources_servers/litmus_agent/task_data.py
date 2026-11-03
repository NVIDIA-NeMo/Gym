# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the domain-agnostic Litmus answer verifier.

Ground truth is carried in top-level row fields. The scorer requires an explicit
answer type and accepts either a row-provided regex or a named answer format.
Domain metadata passes through unchanged; metrics group rows by method and property.
"""

from typing import Any, Dict, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: Union[str, float, int] = Field(
        description="Gold answer, compared after parsing per the answer type.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    answer_type: str = Field(
        description="How the captured answer is parsed: one of 'float', 'bool', 'string'.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    output_regex: Optional[str] = Field(
        default=None,
        description="Extraction regex with exactly one capture group; takes precedence over answer_format.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    answer_format: Optional[str] = Field(
        default=None,
        description="Extraction-format registry name (fmt_00..fmt_30); defaults to fmt_00 when output_regex is absent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    match: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Per-row reward-rule override: {'rule': <name>, **params}; defaults to the answer type's rule.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    method: Optional[str] = Field(
        default=None,
        description="Task-generation method, passed through and used for metric grouping.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    property: Optional[str] = Field(
        default=None,
        description="Property being asked about, passed through and used for metric grouping.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    dataset: Optional[str] = Field(
        default=None,
        description="Source dataset identifier, retained unchanged.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question_family: Optional[str] = Field(
        default=None,
        description="Source question category, retained for subset selection and provenance.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question_license: Optional[str] = Field(
        default=None,
        description="Source question license, retained unchanged.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Source identifiers and question UUID, retained unchanged.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
