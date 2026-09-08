# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the gdpval server.

GDPVal economically-valuable-work rows, all fields top-level (no ``verifier_metadata``).
Required-ness mirrors ``GDPValVerifyRequest`` (app.py:236): ``task_id`` is the only required
field. ``deliverables_dir`` and ``reference_ids`` on that request are injected per verify call by
the agent/driver (deliverable collection and comparison-mode reference filtering) and are NOT
dataset columns. ``reference_files`` is the reverse case: present in the committed data but
undeclared on the wire model, so today it is silently dropped at the verify boundary (default
Pydantic ``extra="ignore"``); it is declared here so tooling sees the column.
"""

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_id: str = Field(
        description=(
            "GDPVal task UUID; wire-required. Seeds the per-task rng (make_rng), locates the reference "
            "deliverable directory (task_{task_id}) in comparison mode, and tags logs."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    sector: Optional[str] = Field(
        default=None,
        description="Economic sector label (e.g. 'Professional, Scientific, and Technical Services'); "
        "declared on the wire but pass-through only.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    occupation: Optional[str] = Field(
        default=None,
        description=(
            "Occupation label (e.g. 'Accountants and Auditors'); not read by verify() but used for "
            "stratified task sampling by multistage_orchestrator.py:164."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    prompt: Optional[str] = Field(
        default=None,
        description="Full task prompt; read as `body.prompt or ''` and interpolated into the judge "
        "prompt as the task description.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    rubric_json: Optional[Any] = Field(
        default=None,
        description=(
            "Rubric for rubric-mode scoring. Typed Any on the wire; in the committed data it is a "
            "JSON-encoded string (~12KB) of an array of {score: int, criterion: str, required, "
            "rubric_item_id, ...} objects — per-criterion 'score'/'weight' entries weight structured-mode "
            "scoring. At least one of rubric_json/rubric_pretty must be present for rubric mode "
            "(app.py:368)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    rubric_pretty: Optional[str] = Field(
        default=None,
        description="Human-readable rubric text ('[+2] ...' lines); read as `body.rubric_pretty or ''` "
        "and shown to the judge alongside/instead of rubric_json.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    reference_file_urls: Optional[List[str]] = Field(
        default=None,
        description=(
            "Public URLs of task input files (huggingface openai/gdpval reference_files); declared on the "
            "wire but unread by verify() — the agent downloads them when building the workspace."
        ),
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    reference_files: Optional[List[str]] = Field(
        default=None,
        description=(
            "Repo-relative local paths of the same task input files. Present in the committed data but "
            "UNDECLARED on GDPValVerifyRequest, so it is silently dropped at the verify boundary today."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
