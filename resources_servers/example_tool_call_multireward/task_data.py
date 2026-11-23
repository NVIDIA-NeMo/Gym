# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the example_tool_call_multireward server.

Ground truth is a single top-level ``expected_call`` dict ({name, arguments}) compared against
the rollout's function calls. Typed ``dict`` with an empty default to mirror the wire
(``ToolCallMultiRewardVerifyRequest.expected_call: Dict[str, Any] = {}``); tightening it to a
required, structured model is a server-owned follow-up, not a schema-layer decision.
"""

from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_call: Dict[str, Any] = Field(
        default_factory=dict,
        description="The one tool call the model is expected to make: {name: str, arguments: dict}.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
