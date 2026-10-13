# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the Scale-SWE resources server.

Mirrors ``app.ScaleSWEInstanceRequest``: required-ness follows the wire contract, not what
``verify()`` happens to read.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One AweAI-Team/Scale-SWE task."""

    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    repo: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    language: str = Field(default="python", json_schema_extra={"consumed_by": ["verify", "provenance"]})
    workdir: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    image_url: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    pre_commands: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    problem_statement: str = Field(default="", json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    f2p_patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    f2p_script: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    # JSON-encoded strings on the wire in this dataset; kept typed str | list[str] so a row
    # round-trips unchanged either way (see verification.as_id_list).
    FAIL_TO_PASS: str | list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    PASS_TO_PASS: str | list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
