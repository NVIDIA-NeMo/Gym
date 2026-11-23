# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the hle_web_tools server.

A tool server: ``verify`` grades nothing, so no field is required. ``seed_session`` reads the row's
question (``question_field``, ``question`` by default) to drive the verbatim-span refusal and falls back to
the last user message of ``responses_create_params``. The CI example rows also carry ``placeholder``, a
note that they are synthetic and not HLE items; the server never reads it.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: Optional[str] = Field(
        default=None,
        description=(
            "The rollout's question. seed_session reads it to refuse a web_search query that repeats a long "
            "verbatim span of it; without it the last user message of responses_create_params is used."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    placeholder: Optional[str] = Field(
        default=None,
        description="Marks a synthetic CI example row (data/example.jsonl) as not an HLE item; not read by the server.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
