# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional task provenance; music rewards depend only on the generated answer."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """Describe task-owned fields without requiring a reference composition."""

    model_config = ConfigDict(extra="allow")

    task_id: str | None = Field(
        default=None,
        description="Optional task identifier, retained for provenance and not scored.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source: str | None = Field(
        default=None,
        description="Optional dataset/source label, retained for provenance and not scored.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
