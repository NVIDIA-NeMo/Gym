# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the self-contained mini_swe_agent (no resources server).

Rows are SWE-bench instances flattened top-level, the same family as the swebench resources
server's rows, except ``FAIL_TO_PASS``/``PASS_TO_PASS`` are real JSON lists here (swebench keeps
them JSON-encoded strings) and there is no environment_setup_commit/difficulty/subset/split.
``MiniSWEAgentRunRequest`` (app.py) declares no task fields (``extra="allow"``); the whole
instance dict rides the wire as extras and is handed to the SWE-bench evaluation harness.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    instance_id: str = Field(
        description="SWE-bench instance id, e.g. 'getmoto__moto-7365'; keys the evaluation.",
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
    FAIL_TO_PASS: List[str] = Field(
        description="Test ids that must flip to passing; a JSON list (not the encoded-string form).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    PASS_TO_PASS: List[str] = Field(
        description="Test ids that must keep passing; a JSON list (not the encoded-string form).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    subset: str = Field(
        description="SWE-bench subset the instance came from, e.g. 'verified'; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    split: str = Field(
        description="Dataset split the instance came from, e.g. 'test'; provenance only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
