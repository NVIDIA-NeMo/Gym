# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the ns_tools server.

Delegating verifier: ``NSToolsRunRequest`` (app.py:97, extra="allow") types only an optional
``verifier_type``; verify() picks a downstream verifier by that field (falling back to
``config.default_verifier``) and forwards the entire row via ``body.model_dump()``. The effective
schema is therefore this model unioned with the delegated verifier's request model — for the one
verifier_type in committed data ('math_with_judge'), that downstream wire requires ``question``
and ``expected_answer``. They stay Optional here because ns_tools' own wire does not require
them; rows routed to a math judge must still carry both or the delegated verify 422s.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    verifier_type: Optional[str] = Field(
        default=None,
        description="Which downstream verifier scores this row (e.g. 'math_with_judge'); falls back to "
        "config.default_verifier when absent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    question: Optional[str] = Field(
        default=None,
        description="Problem statement forwarded to the delegated verifier; required by the math_with_judge "
        "downstream wire.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_answer: Optional[str] = Field(
        default=None,
        description="Ground-truth answer forwarded to the delegated verifier; required by the math_with_judge "
        "downstream wire.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    id: Optional[str] = Field(
        default=None,
        description="Task identifier, e.g. 'aime25-0'; unread by server code.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    reference_solution: Optional[str] = Field(
        default=None,
        description="Reference solution text; present for some subsets (aime24/25), absent for others (hmmt).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    subset_for_metrics: Optional[str] = Field(
        default=None,
        description="Cross-benchmark prepare.py convention naming the metrics subset (e.g. 'aime25').",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
