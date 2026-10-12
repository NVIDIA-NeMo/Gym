# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for FACTS Grounding v2.

Rows are prepared from the public Kaggle release. The resources server uses the request,
context, and ordered metadata fields while the remaining fields preserve source provenance
and support slicing and reproducibility.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str = Field(description="Stable id from the pinned public dataset.")
    system_instruction: str = Field(description="Instruction included in the official full prompt.")
    user_request: str = Field(
        description="The user's request; used by the eligibility and grounding judges.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    context_document: str = Field(
        description="The source document used to assess sentence-level grounding.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    full_prompt: str = Field(description="The official prompt sent as the single user message.")
    domain: str = Field(description="Source dataset domain label used for result slices.")
    type: str = Field(description="Source dataset task-type label.")
    high_level_type: str = Field(description="Source dataset high-level task-type label.")
    context_document_chars: int = Field(description="Character count of context_document.")
    full_prompt_chars: int = Field(description="Character count of full_prompt.")
    row_sha256: str = Field(description="Stable digest of the pinned source fields.")
    upstream: dict[str, Any] = Field(description="Pinned source dataset and license metadata.")
