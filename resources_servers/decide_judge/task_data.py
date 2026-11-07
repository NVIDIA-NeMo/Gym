# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the decide_judge server.

Rows are open (``extra="allow"``): any field named in the config's ``reference_fields`` is copied into the judge
state, looked up at top level and then in ``metadata`` / ``verifier_metadata``. The default reference field is
``expected_answer``. A row may bring its own checklist in ``decide_checklist``.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class ChecklistItemData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique item name ([A-Za-z0-9_-], up to 64 chars).")
    question: str = Field(description="Yes/no question about the rollout; yes must mean good.")
    weight: float = Field(default=1.0, ge=0, description="Reward weight; 0 logs the item without using it.")
    required: bool = Field(default=False, description="If p(yes) < pass_threshold the reward is required_fail_reward.")


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: Optional[str] = Field(
        default=None,
        description="Reference answer shown to the judge as state.reference.expected_answer (default reference field).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    question: Optional[str] = Field(
        default=None,
        description="Question text; copied into state.reference only when listed in reference_fields.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    decide_checklist: Optional[List[ChecklistItemData]] = Field(
        default=None,
        description="Per-row checklist; replaces the config checklist when allow_row_checklist is true.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Fallback bucket for reference fields missing at top level.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
