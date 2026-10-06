# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the chemreason_bench server.

Verification is local and deterministic: ``task_type`` selects one of six
scorers and ``ground_truth`` carries that task's gold record. Both are required
on the wire. ``task_id`` and ``benchmark_id`` are provenance only -- the latter
identifies which of the 500 source reactions an instance came from, which is
what upstream's ``range_1_400`` / ``range_401_500`` reporting slices on.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task_type: str = Field(
        description=(
            "One of ordering, contrastive_choice, step_validation, "
            "condition_validation, step_completion, rationalization. Selects the "
            "scorer and therefore the metric this row contributes to."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    ground_truth: Dict[str, Any] = Field(
        description=(
            "Upstream gold record for this instance; its shape depends on "
            "task_type (e.g. correct_order for ordering, label for the binary "
            "tasks, action+slots for step_completion)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    protocol: str = Field(
        default="gen",
        description=(
            "'gen' (JSON reply) or 'lm' (bare decision token). The three discriminative "
            "tasks carry both; the published metric averages them."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    question: Optional[str] = Field(
        default=None,
        description="Rendered user prompt; materialized into responses_create_params, never read by the server.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    expected_step_ids: Optional[List[str]] = Field(
        default=None,
        description="ORDERING only: the legal step ids in presentation order, for upstream's post_ordering.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    options: Optional[List[Any]] = Field(
        default=None,
        description="CONTRASTIVE_CHOICE only: the option list, for upstream's post_contrastive.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    legend: Optional[Dict[str, str]] = Field(
        default=None,
        description="STEP_COMPLETION only: placeholder->name map, for upstream's canonicalize_slots.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Upstream instance id, e.g. 'ordering_001_1'; carried for traceability.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    dataset_name: Optional[str] = Field(
        default=None,
        description="Always 'chemreason_bench'; written by prepare.py so rows stay identifiable once merged.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    split: Optional[str] = Field(
        default=None,
        description="Always 'test'. Upstream publishes a single split; declared so the field is not silently extra.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    benchmark_id: Optional[int] = Field(
        default=None,
        description="Source reaction id in 1..500; upstream slices its report on this.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
