# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the SWE-bench Pro resources server."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One prepared ScaleAI SWE-bench Pro task and its pinned evaluator assets."""

    model_config = ConfigDict(extra="allow")

    repo: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    instance_id: str = Field(json_schema_extra={"consumed_by": ["verify", "provenance"]})
    base_commit: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    patch: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    test_patch: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    problem_statement: str = Field(json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    requirements: str = Field(default="", json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    interface: str = Field(default="", json_schema_extra={"consumed_by": ["prompt", "provenance"]})
    repo_language: str = Field(default="", json_schema_extra={"consumed_by": ["verify", "provenance"]})
    fail_to_pass: str | list[str] = Field(json_schema_extra={"consumed_by": ["verify"]})
    pass_to_pass: str | list[str] = Field(json_schema_extra={"consumed_by": ["verify"]})
    issue_specificity: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    issue_categories: str = Field(default="", json_schema_extra={"consumed_by": ["provenance"]})
    before_repo_set_cmd: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    selected_test_files_to_run: str | list[str] = Field(json_schema_extra={"consumed_by": ["verify"]})
    dockerhub_tag: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    image_digest: str = Field(default="", json_schema_extra={"consumed_by": ["verify", "provenance"]})
    run_script: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    parser_script: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    base_dockerfile: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    instance_dockerfile: str = Field(default="", json_schema_extra={"consumed_by": ["verify"]})
    subset: str = Field(default="pro", json_schema_extra={"consumed_by": ["provenance"]})
    split: str = Field(default="test", json_schema_extra={"consumed_by": ["provenance"]})
    evaluator_commit: str = Field(json_schema_extra={"consumed_by": ["provenance"]})
    dataset_revision: str = Field(json_schema_extra={"consumed_by": ["provenance"]})
