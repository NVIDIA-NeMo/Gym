# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the mcp_atlas server.

The wire requires a ``verifier_metadata`` dict and reads these fields from it with fallbacks, so they are
optional here; ``benchmarks/mcp_atlas/prepare.py`` writes all of them. ``task_id`` is also a top-level key.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        default="", json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["provenance"]}
    )
    enabled_tools: list[str] = Field(
        default_factory=list,
        description="The task's tools, as <server>_<tool>; every other tool name is refused.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
    servers: list[str] = Field(
        default_factory=list,
        description="The MCP servers the task's sandbox serves.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
    claims: list[str] = Field(
        default_factory=list,
        description="The ground-truth claims the judge checks one by one against the final answer.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["verify"]},
    )
    dropped_tools: list[str] = Field(
        default_factory=list,
        description="Distractor tools of servers the image does not ship; kept for the audit only.",
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["provenance"]},
    )
