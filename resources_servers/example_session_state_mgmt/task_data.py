# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the example_session_state_mgmt (stateful counter) server.

Ground truth is split across the session lifecycle: ``initial_count`` seeds the per-session
counter (consumed by /seed_session), and ``expected_count`` is what verify() compares the
server-side counter against. Both are required top-level row fields on the wire.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    initial_count: int = Field(
        description="Counter value the session starts from (consumed at seed_session time).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_count: int = Field(
        description="Counter value the session must reach for reward 1.0.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
