# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the FACTS Parametric resources server.

The prepared public rows are flat and mirror ``FACTSParametricRunRequest``. All
fields are optional on the wire so verifier fixtures and dry-run requests can
exercise malformed or partial cases explicitly; production preparation writes
every field on all 1,052 rows.
"""

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[int, str]] = Field(
        default=None,
        description="Stable public-row identifier used for reconciliation and evidence links.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question: Optional[str] = Field(
        default=None,
        description="Closed-book factual query rendered verbatim into the policy and grader prompts.",
        json_schema_extra={"consumed_by": ["prompt", "verify"]},
    )
    expected_answer: Optional[str] = Field(
        default=None,
        description="Gold answer inserted into the official FACTS Parametric grader template.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    topic: Optional[str] = Field(
        default=None,
        description="Public dataset topic label used for diagnostic accuracy and hedging slices.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    source_url: Optional[str] = Field(
        default=None,
        description="Wikipedia source URL supplied by the public dataset for provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
