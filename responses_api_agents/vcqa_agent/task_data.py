# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the self-contained vcqa_agent (no resources server).

The wire (``VcqaAgentRunRequest``) requires a ``verifier_metadata`` object and reads task fields
ONLY from it, so every field here carries ``legacy_location: verifier_metadata``. Required-ness
mirrors ``VcqaVerifierMetadata`` (app.py): ``dataset_kind`` and ``artifact_key`` are wire-required,
the rest are optional. ``language``, ``task_type``, ``task_kind``, and ``verifier_kind`` are
open extras on the wire model that the committed rows carry; they are declared here for
documentation and typo protection.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


_VM = {"legacy_location": "verifier_metadata"}


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    dataset_kind: Literal["fileset", "githistory"] = Field(
        description="Which VCQA artifact family the row belongs to.",
        json_schema_extra={"consumed_by": ["run"], **_VM},
    )
    artifact_key: str = Field(
        description="Artifact path joined with the configured base URL to fetch the repo tarball.",
        json_schema_extra={"consumed_by": ["run"], **_VM},
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Task identifier; provenance label.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    repo_full_name: Optional[str] = Field(
        default=None,
        description="GitHub repo the artifact snapshots, e.g. 'huggingface/transformers'.",
        json_schema_extra={"consumed_by": ["run"], **_VM},
    )
    pre_merge_sha: Optional[str] = Field(
        default=None,
        description="Base commit for githistory tasks.",
        json_schema_extra={"consumed_by": ["run"], **_VM},
    )
    head_sha: Optional[str] = Field(
        default=None,
        description="Head commit for githistory tasks.",
        json_schema_extra={"consumed_by": ["run"], **_VM},
    )
    rubric: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Judge rubric ({'judge': [...]}) used to score the final answer.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    verifiers: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Alternative verifier spec used by bisect-style tasks.",
        json_schema_extra={"consumed_by": ["verify"], **_VM},
    )
    max_turns: Optional[int] = Field(
        default=None,
        description="Turn budget for the exploration loop.",
        json_schema_extra={"consumed_by": ["run"], **_VM},
    )
    language: Optional[str] = Field(
        default=None,
        description="Repo language tag; open extra on the wire, informational.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    task_type: Optional[str] = Field(
        default=None,
        description="Task flavor, e.g. 'fileset_explore'; open extra on the wire, informational.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    task_kind: Optional[str] = Field(
        default=None,
        description="Task flavor for githistory rows, e.g. 'bisect'; open extra on the wire.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
    verifier_kind: Optional[str] = Field(
        default=None,
        description="Which verifier family scores the row, e.g. 'judge'; open extra on the wire.",
        json_schema_extra={"consumed_by": ["provenance"], **_VM},
    )
