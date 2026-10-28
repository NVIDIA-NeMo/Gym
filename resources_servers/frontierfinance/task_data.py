# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the frontierfinance server.

``FrontierFinanceTaskFields`` is also the base of the server's run and verify requests, so the
schema and the wire contract share one definition.
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class FrontierFinanceRubric(BaseModel):
    rubric_id: int
    rubric_text: str
    must_have: bool = False
    rubric_type: Optional[str] = None
    rubric_subtype: Optional[str] = None
    data_source_type: Optional[str] = None
    data_source_subtype: Optional[str] = None
    data_source_detailed: List[str] = Field(default_factory=list)


class FrontierFinanceTaskFields(BaseModel):
    question: str = Field(
        default="",
        description="The investor research query, passed to the judges.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    query_id: str = Field(default="", json_schema_extra={"consumed_by": ["verify", "metrics"]})
    as_of_date: Optional[str] = Field(
        default=None,
        description="YYYY-MM-DD date the query is answered as of. Caps dated tool calls and is passed "
        "to the judges. Null leaves tool dates uncapped.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    use_cases: List[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["metrics"]})
    capabilities: List[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["provenance"]})
    rubrics: List[FrontierFinanceRubric] = Field(
        default_factory=list,
        description="Binary grading criteria. An empty list masks the sample at verify.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )


class TaskData(FrontierFinanceTaskFields):
    model_config = ConfigDict(extra="allow")
