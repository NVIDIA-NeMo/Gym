# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the false_statement_judge server.

The false statement in ``question`` and its unperturbed source in
``original_problem`` fill the judge prompt. Both fields are optional to mirror
``FalseStatementRunRequest``, which handles missing values as empty strings.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: Optional[str] = Field(
        default=None,
        description="False statement shown to the policy and supplied to the judge.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    original_problem: Optional[str] = Field(
        default=None,
        description="Unperturbed source statement supplied to the judge for comparison.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
