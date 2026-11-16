# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the swe_internal_v1 resources server.

Mirrors ``app.SweInternalV1InstanceRequest``: required-ness follows the wire contract. The
``run_script`` / ``parsing_script`` pair is what the verifier runs; ``FAIL_TO_PASS`` /
``PASS_TO_PASS`` are the ids it grades.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One internal-v1 SWE task."""

    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    delivery: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    language: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    repo: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    workdir: str = Field(default="/app", json_schema_extra={"consumed_by": ["verify"]})
    image_ref: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    nydus_ref: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    base_commit: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    solution_commit: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    test_patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    problem_statement: str = Field(default="", json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    run_script: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    parsing_script: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    test_files: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    test_patch_checkout_cmd: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    env_exports: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    test_framework: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    FAIL_TO_PASS: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    PASS_TO_PASS: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["verify"]})
    # Vendor metadata carried through for provenance; none of it is read by verify().
    original_issue_url: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    issue_categories: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["provenance"]})
    issue_specificity: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    license: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    evaluation_time: float | None = Field(default=None, json_schema_extra={"consumed_by": ["provenance"]})
