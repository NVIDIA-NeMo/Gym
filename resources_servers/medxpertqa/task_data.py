# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the medxpertqa server.

A gymnasium-style environment: step() sends the answer trigger after the first reply and grades the
second. The fields are MedXpertQA's native row fields, top-level row extras.
"""

from typing import Dict

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(
        description="MedXpertQA question id, e.g. 'Text-0'; reported as info.question_id.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question: str = Field(
        description="Question text with its answer choices, as released.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    options: Dict[str, str] = Field(
        description="Option letter to option text; the letters set the trigger range and the allowed answers.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    label: str = Field(
        description="The correct option letter.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    medical_task: str = Field(
        description="Diagnosis, Treatment or Basic Science.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    body_system: str = Field(
        description="Body system label, e.g. 'Nervous'.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    question_type: str = Field(
        description="Reasoning or Understanding.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
