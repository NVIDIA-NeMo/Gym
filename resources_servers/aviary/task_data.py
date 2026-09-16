# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the aviary server family.

Stateful environment: the row is a pointer whose only task datum is ``task_idx``, an integer
index into an aviary ``TaskDataset`` supplied OUT of the row by the concrete subclass server
(the aviary_bbh / aviary_bixbench / aviary_gsm8k / aviary_hotpotqa environments). Reward
accumulates server-side across /step calls keyed by env_id; verify() reads that state, never the
row. toolsandbox's TaskData subclasses this model.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_idx: int = Field(
        description=(
            "Index into the server's task registry. Wire-required by AviarySeedSessionRequest and consumed "
            "by /seed_session to instantiate the episode (initial observations + tools); not read by verify()."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
