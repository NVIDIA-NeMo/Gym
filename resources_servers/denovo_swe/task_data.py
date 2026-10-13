# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the DeNovoSWE resources server.

Mirrors ``app.DeNovoSWEInstanceRequest``: required-ness follows the wire contract, not what
``verify()`` happens to read. ``patch`` is legitimately empty for every committed row -- DeNovoSWE
has no golden model-style patch (the pre-existing image source at ``base_commit`` IS the golden
answer); it is only ever non-empty for an agent-produced rollout.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One AweAI-Team/DeNovoSWE task."""

    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    repo: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    github_url: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    language: str = Field(default="python", json_schema_extra={"consumed_by": ["verify", "provenance"]})
    workdir: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    image_ref: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    base_commit: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    test_patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    document: str = Field(default="", json_schema_extra={"consumed_by": ["verify", "prompt", "provenance"]})
    problem_statement: str = Field(default="", json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    pypi_name: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    import_names: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["provenance"]})
    passed_ptp: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    failed_ptp: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["provenance"]})
    test_binary_archive_b64: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    expected_coverage_percent: float = Field(default=0.0, json_schema_extra={"consumed_by": ["provenance"]})
