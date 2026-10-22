# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the SWE-Next resources server.

Mirrors ``app.SWENextInstanceRequest``: required-ness follows the wire contract, not what
``verify()`` happens to read.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One TIGER-Lab/SWE-Next task."""

    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    repo: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    language: str = Field(default="python", json_schema_extra={"consumed_by": ["verify", "provenance"]})
    workdir: str = Field(default="/testbed", json_schema_extra={"consumed_by": ["verify"]})
    image_ref: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    base_commit: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    test_patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    problem_statement: str = Field(default="", json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    FAIL_TO_PASS: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["provenance"]})
    PASS_TO_PASS: list[str] = Field(default_factory=list, json_schema_extra={"consumed_by": ["provenance"]})
    # A {test_name: "PASSED"|"FAILED"|"ERROR"} snapshot of the whole suite on the golden-patched
    # instance -- the actual grading signal (see verification.py module docstring); a strict
    # superset of FAIL_TO_PASS/PASS_TO_PASS, which are carried above for provenance only now.
    expected_output_json: str = Field(default="{}", json_schema_extra={"consumed_by": ["verify"]})
