# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the self-contained pinchbench agent (no resources server).

The committed rows nest ``task_id`` inside ``verifier_metadata``; the agent accepts BOTH
placements (``meta.get("task_id") or record.get("task_id")`` in app.py), so the field carries no
``legacy_location`` marker. ``task_id`` selects the PinchBench task and is exported into the
sandbox environment as ``TASK_ID``.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description="PinchBench task name; selects the task and is exported to the sandbox as TASK_ID.",
        json_schema_extra={"consumed_by": ["run"]},
    )
