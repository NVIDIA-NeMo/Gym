# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the code_fim server.

Pointer rows: the only task-owned field is ``verifier_metadata.task_id``. The actual problem
payload (prefix/prompt, suffix, tests, entry point) lives OUT of the row, in the
``human_eval_infilling`` package: it is loaded at server startup via
``read_problems(config.split)`` (split: single_line | multi_line | random_span |
random_span_light) and looked up by task_id at verify time. evalplus's TaskData subclasses this
model — same pointer shape, different task-id namespace.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: Optional[str] = Field(
        default=None,
        description=(
            "Key into the out-of-row problem registry (a HumanEval-Infilling split task id here). "
            "Wire-optional: verify() reads it via .get() and scores 0.0 with an error when missing or unknown."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
