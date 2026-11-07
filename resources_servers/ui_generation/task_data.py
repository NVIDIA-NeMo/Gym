# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the ui_generation server.

Rows keep their task fields in the ``verifier_metadata`` bucket (schemas are flat: core splices its
contents up before validating; ``legacy_location`` records today's wire placement). Required-ness
mirrors ``UiGenerationVerifierMetadata`` in app.py: ``reference_answer`` is required, the rest is
optional, and extra fields are allowed.
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
    id: Optional[str] = Field(
        default=None,
        description="Task identifier from the input file. Never read by server code.",
        json_schema_extra={"consumed_by": ["provenance"], "legacy_location": "verifier_metadata"},
    )
