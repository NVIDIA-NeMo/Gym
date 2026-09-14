# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the swebench server.

Rows are complete HuggingFace SWE-bench instances flattened top-level (no verifier_metadata, ever)
and this is already the repo's most strictly typed wire contract: ``SWEBenchInstanceRequest``
(app.py:70) requires all fifteen fields as ``str``, so this schema does too. verify() and
seed_session() pass the entire instance dump to SWE-bench's ``make_test_spec()`` to build the
evaluation TestSpec; several fields (hints_text, created_at, difficulty, subset, split) merely
ride along in that dump but remain wire-required. ``FAIL_TO_PASS``/``PASS_TO_PASS`` are
JSON-encoded list-of-test-id strings and stay typed ``str``.

This file is the parent of the SWE-bench instance family: ``swerl_gen`` nests the same instance
core (minus environment_setup_commit/difficulty/subset/split, plus setup/test/regression script
fields) inside its open ``instance`` dict.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    repo: str = Field(
        description="GitHub repository the instance patches, e.g. 'astropy/astropy'.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    instance_id: str = Field(
        description="SWE-bench instance id; also keys sandbox metadata and multilingual patch hooks.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    base_commit: str = Field(
        description="Commit the repo is checked out at before the model's patch is applied.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    patch: str = Field(
        description=(
            "Gold patch diff; read directly when config.is_verifying_golden_patch=true, otherwise "
            "part of the TestSpec build."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    test_patch: str = Field(
        description="Test-file diff applied before running the evaluation tests.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    problem_statement: str = Field(
        description="Issue text describing the bug; duplicated into responses_create_params for the agent.",
        json_schema_extra={"consumed_by": ["prompt", "provenance"]},
    )
    hints_text: str = Field(
        description="Issue discussion hints; wire-required but only passes through the TestSpec dump.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    created_at: str = Field(
        description="Upstream issue creation timestamp; wire-required but only passes through the dump.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    version: str = Field(
        description="Repo version tag used to select the SWE-bench environment spec.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    FAIL_TO_PASS: str = Field(
        description="JSON-encoded list-of-str: test ids that must flip from fail to pass.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    PASS_TO_PASS: str = Field(
        description="JSON-encoded list-of-str: test ids that must keep passing.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    environment_setup_commit: str = Field(
        description="Commit used to build the evaluation environment image.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    difficulty: str = Field(
        description="Annotated difficulty bucket, e.g. '15 min - 1 hour'; wire-required, unread.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    subset: str = Field(
        description="SWE-bench subset the row came from, e.g. 'verified'; wire-required, unread.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    split: str = Field(
        description="Dataset split the row came from, e.g. 'test'; wire-required, unread.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
