# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the ui_generation server.

Rows keep their verifier fields in the ``verifier_metadata`` bucket (schemas are flat: core splices its
contents up before validating; ``legacy_location`` records today's wire placement) and the task id at the
top level, where task materialization reads it. The verifier fields mirror ``UiGenerationVerifierMetadata``
in app.py (``reference_answer`` required, ``category`` optional, extra fields allowed);
``tests/test_app.py`` fails when the two declarations drift apart.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    reference_answer: str = Field(
        description=(
            "The task's reference answer. The judge compares the policy's code blocks with its code blocks as a "
            "guideline; it is never shown to the policy."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    category: Optional[str] = Field(
        default=None,
        description="Optional task label, such as the target stack; compute_metrics reports mean scores per label.",
        json_schema_extra={"consumed_by": ["metrics"], "legacy_location": "verifier_metadata"},
    )
    task_id: Optional[str] = Field(
        default=None,
        description=(
            "Stable task identifier from the input file (prepare.py's id). Gym's task identity for resume and "
            "run comparison; never read by server code."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
