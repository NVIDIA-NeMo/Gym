# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the nullable_tool_args server."""

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


_VM = {"legacy_location": "verifier_metadata"}


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    null_args: List[str] = Field(
        description="Tool arguments that must be JSON null in the first function call.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    position: Literal["first", "mid"] = Field(
        description="'first' if the conversation has no earlier tool calls, 'mid' if it has one.",
        json_schema_extra={"consumed_by": ["metrics"], **_VM},
    )
