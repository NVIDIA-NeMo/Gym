# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the example_multi_turn_gymnasium server.

A stateful, gymnasium-style environment: there is no /verify endpoint. Scoring happens inside
step(), which replays ``follow_ups`` as user turns and substring-matches ``expected_answer``
against the final assistant message. Both fields reach the server as top-level row extras.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    follow_ups: List[str] = Field(
        description="User turns replayed by the environment after the first response.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_answer: str = Field(
        description="Substring the final assistant message must contain.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
