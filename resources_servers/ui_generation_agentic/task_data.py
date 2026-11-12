# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the ui_generation_agentic server.

A row is one UI task, as ``environments/ui_generation_agentic/prepare.py`` writes it: the agent's prompt travels in
``responses_create_params``; the task-owned fields name the stack, the entry file the agent must create, and the user
prompt the judge grades against. ``UiGenTaskFields`` declares those fields once: the seed's ``UiGenTask`` and
``UiGenVerifyRequest`` (app.py) inherit them, so the row schema and the runtime models cannot drift apart. Unknown
fields are kept (``extra="allow"``).
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class UiGenTaskFields(BaseModel):
    """The task fields the server reads at seed and verify."""

    stack: str = Field(
        description="The task's stack, a key of the server's `stacks` config (ui-next, ui-vite or ui-static).",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    entry_path: str = Field(
        description="The deliverable, relative to the stack's workdir; it must match the stack's configured entry path.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    user_prompt: str = Field(
        description="The user's request alone (without the delivery instructions), which the judge grades against.",
        json_schema_extra={"consumed_by": ["verify"]},
    )


class TaskData(UiGenTaskFields):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description="Stable task id, `<stack>-<source id>`.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Row id in the source JSONL.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
