# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the SWE-rebench-V2 resources server.

Mirrors ``app.SWERebenchInstanceRequest``: required-ness follows the wire contract, not what
``verify()`` happens to read.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One nebius/SWE-rebench-V2 task."""

    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    repo: str = Field(json_schema_extra={"consumed_by": ["provenance"]})
    base_commit: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    test_patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    problem_statement: str = Field(default="", json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    language: str = Field(default="", json_schema_extra={"consumed_by": ["verify", "provenance"]})
    image_name: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    # Carries the row's own install/test commands and its named upstream log parser (see
    # log_parsers.py) -- there is no per-language template to fall back to.
    install_config: dict[str, Any] = Field(default_factory=dict, json_schema_extra={"consumed_by": ["verify"]})
    # Upstream spells these in caps; kept as-received so a row round-trips unchanged.
    FAIL_TO_PASS: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    PASS_TO_PASS: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
