# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Official reference answers stored outside the agent-visible corpus."""

from pydantic import BaseModel, ConfigDict, Field


class OfficeQAProTask(BaseModel):
    """Trusted preparation artifact; never mounted in a document sandbox."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(pattern=r"^officeqa_pro_[0-9a-f]{24}$")
    workspace_id: str = Field(pattern=r"^workspace_[0-9a-f]{24}$")
    uid: str = Field(min_length=1)
    question: str = Field(min_length=1)
    expected_answer: str = Field(min_length=1)
    source_revision: str = Field(pattern=r"^[0-9a-f]{40}$")
