# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public OfficeQA Pro task rows; reference answers live in the trusted bundle."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """Task fields shared by catalog validation, session seeding and verification."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        pattern=r"^officeqa_pro_[0-9a-f]{24}$", json_schema_extra={"consumed_by": ["seed_session", "verify"]}
    )
    workspace_id: str = Field(
        pattern=r"^workspace_[0-9a-f]{24}$", json_schema_extra={"consumed_by": ["seed_session", "verify"]}
    )
    prompt_version: str = Field(min_length=1, json_schema_extra={"consumed_by": ["seed_session", "verify"]})
    uid: str = Field(min_length=1, json_schema_extra={"consumed_by": ["provenance"]})
    question: str = Field(min_length=1, json_schema_extra={"consumed_by": ["provenance"]})
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$", json_schema_extra={"consumed_by": ["provenance"]})
