# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the safe_child_llm server (Safe-Child-LLM developmental safety).

Fields are top-level row columns written by ``benchmarks/safe_child_llm/prepare.py`` from the two
released XLSX splits. Required-ness mirrors ``SafeChildLLMVerifyRequest``: only ``safe_child_id``
and ``age_group`` are required on the wire; the judge is shown ``category`` and ``prompt`` when
present and falls back to the last user message for the prompt.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    safe_child_id: str = Field(
        description=(
            "Identifies one prompt: `safe-child-<age_group>-<index>` where `index` is the row's `Index` "
            "column in the upstream workbook, e.g. `safe-child-6-12-001`. Unique across both splits."
        ),
        json_schema_extra={"consumed_by": ["verify", "provenance"]},
    )
    age_group: Literal["6-12", "13-17"] = Field(
        description=(
            "Which released split the prompt comes from. The judge is told to score for this age group, and "
            "compute_metrics reports the safe response rate per group; the paper treats the two as separate "
            "developmental stages, not one population."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    category: str = Field(
        default="",
        description=(
            "The upstream `category` column, one of eight harm categories such as `Assisting illegal "
            "activities` or `Mental Health or Overreliance Crisis`. Shown to the judge and used as a metrics "
            "slice."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    prompt: str = Field(
        default="",
        description=(
            "The upstream `query` column verbatim; also the sole user message in "
            "`responses_create_params.input`. The judge is shown this field."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    source: str = Field(
        default="",
        description=(
            "The upstream `source` column: the public red-teaming set the prompt was drawn from "
            "(DoNotAnswer, AdvBench, SG-Bench, ForbiddenQuestions, Strongreject, ...). Provenance only."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    upstream_revision: str = Field(
        default="",
        description="Git revision of the upstream repository the row was built from.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
