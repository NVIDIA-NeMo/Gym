# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the evalplus server.

Pointer rows: the only task-owned field is ``verifier_metadata.task_id``, an EvalPlus key such as
'HumanEval/0' or 'Mbpp/2'. Prompts, tests, and ground truth live OUT of the row in the
``evalplus`` package, selected by ``config.dataset`` (humaneval | mbpp) at server startup. Both
dataset flows share this row shape — only the task_id namespace differs.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description=(
            "Key into the out-of-row EvalPlus problem registry, namespaced 'HumanEval/<n>' or 'Mbpp/<n>'. "
            "Wire-optional: verify() reads it via .get() and scores 0.0 with an error when missing or unknown."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
