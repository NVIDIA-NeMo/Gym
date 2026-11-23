# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the deepswe server.

Pointer rows: the row names a task and pins its sandbox image, and everything else (verifier
files, collect hook, solution patch, resource requirements) lives OUT of the row in an on-disk
Pier-format task store resolved from ``config.tasks_dir`` (one ``task.toml`` directory per task,
typed by task_schema.py's ``TaskConfig``), keyed by ``task_id``. Required-ness mirrors
``DeepSWEInstanceRequest`` (app.py:82, ``extra="allow"``): ``image`` required, ``task_id``
wire-Optional even though ``_resolve_task_id`` (app.py:154) errors when it is absent from both the
top level and ``verifier_metadata``. Committed rows duplicate ``task_id`` in both places with
equal values; the flat schema keeps the single top-level field (``verify()`` errors when the two
copies conflict). ``sandbox_handle`` (DeepSWEVerifyRequest) is injected by seed_session at verify
time and is not a task field.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    image: str = Field(
        description=(
            "Sandbox image reference; wire-required. verify()/seed_session validate it against the pinned "
            "image recorded in the on-disk task store for this task_id (app.py:170) and reject mismatches."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task_id: Optional[str] = Field(
        default=None,
        description=(
            "Key into the on-disk Pier task store (e.g. 'random-tree-generator'). Wire-Optional because "
            "today's rows also duplicate it inside verifier_metadata (_resolve_task_id accepts either "
            "placement and errors on conflict or on both missing)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    split: Optional[str] = Field(
        default=None,
        description="Source split (e.g. 'test'); pass-through provenance via extra='allow', never read.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    subset: Optional[str] = Field(
        default=None,
        description="Source subset (e.g. 'deepswe-v1.1'); pass-through provenance via extra='allow', never read.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
