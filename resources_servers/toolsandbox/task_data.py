# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the toolsandbox server.

Pointer rows: the row's only task datum is ``task_idx``, an index into the deterministic sorted
registry of vendored ToolSandbox scenarios built in-process at runtime — the task definition lives
in code, OUT of the row (wire-required by ToolSandboxSeedSessionRequest). verify() is a
cached-reward lookup keyed by env_id (milestone similarity computed at /close, defaulting to 0.0).
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_idx: int = Field(
        description=(
            "Index into the sorted registry of vendored ToolSandbox scenarios. Wire-required by "
            "ToolSandboxSeedSessionRequest and consumed by /seed_session to instantiate the episode; "
            "not read by verify()."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
