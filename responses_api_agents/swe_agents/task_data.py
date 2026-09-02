# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the self-contained swe_agents harness (no resources server).

Rows are SWE-bench instances flattened top-level in the HuggingFace export shape:
``FAIL_TO_PASS``/``PASS_TO_PASS`` are JSON-encoded strings (like the swebench resources server),
and ``environment_setup_commit``/``difficulty`` are present. ``SWEBenchRunRequest`` (app.py)
declares no task fields (``extra="allow"``); the instance dict rides the wire as extras and is
handed to the configured SWE harness and its evaluation.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(
        description="SWE-bench instance id, e.g. 'astropy__astropy-12907'; keys the evaluation.",
        json_schema_extra={"consumed_by": ["run", "verify"]},
    )
    repo: str = Field(
        description="GitHub repository the instance patches.",
        json_schema_extra={"consumed_by": ["run"]},
    )
    base_commit: str = Field(
        description="Commit the repo is checked out at before the agent works.",
        json_schema_extra={"consumed_by": ["run"]},
    )
    patch: str = Field(
        description="Gold patch diff; used by evaluation, not shown to the agent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    test_patch: str = Field(
        description="Test-file diff applied before running the evaluation tests.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    problem_statement: str = Field(
        description="Issue text describing the bug.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    hints_text: str = Field(
        description="Optional hints; rides along in the instance dump (may be empty).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    created_at: str = Field(
        description="Instance creation timestamp; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    version: str = Field(
        description="Repo version tag used by the SWE-bench environment spec.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    FAIL_TO_PASS: str = Field(
        description="JSON-encoded list of test ids that must flip to passing.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    PASS_TO_PASS: str = Field(
        description="JSON-encoded list of test ids that must keep passing.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    environment_setup_commit: str = Field(
        description="Commit used to build the evaluation environment.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    difficulty: str = Field(
        description="Human difficulty band, e.g. '15 min - 1 hour'; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
