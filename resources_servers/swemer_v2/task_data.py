# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the Swemer Agentic-v2 resources server.

Mirrors ``app.SwemerV2InstanceRequest``: required-ness follows the wire contract, not what
``verify()`` happens to read. Unlike swemer_v1, ``FAIL_TO_PASS``/``PASS_TO_PASS`` here use a
dotted-path id convention rather than each framework's own real node ids -- see verification.py.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One Swemer Agentic-v2 task."""

    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    delivery: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    language: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    workdir: str = Field(default="/workspace/repo", json_schema_extra={"consumed_by": ["verify"]})
    image_ref: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    test_patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    problem_statement: str = Field(default="", json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    test_framework: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    test_command: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    FAIL_TO_PASS: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    PASS_TO_PASS: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
