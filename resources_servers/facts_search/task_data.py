# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the FACTS Search resources server.

Every prepared row is a public Search-On question with its gold answer and
source fingerprint. The agent receives ``responses_create_params``; the
resources server reads the fields below while grading the final answer.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = Field(
        default=None,
        description="Public FACTS Search example identifier.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    problem: Optional[str] = Field(
        default=None,
        description="Question the search agent must answer.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    gold_answer: Optional[str] = Field(
        default=None,
        description="Public reference answer inserted into the A/B/C grader prompt.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    row_sha256: Optional[str] = Field(
        default=None,
        description="SHA-256 fingerprint of the source CSV row.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
