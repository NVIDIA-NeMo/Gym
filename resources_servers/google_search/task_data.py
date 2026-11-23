# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the google_search server.

Ground truth rides as top-level row fields; there is no verifier_metadata. Mirrors
``GoogleSearchRunRequest`` (app.py): both fields are wire-required, so both stay required here —
including ``task_difficulty_qwen3_32b_avg_8``, which no verify/metrics code ever reads but whose
absence would 422 today's wire. Relaxing that dead-required field is a per-server follow-up with
data migration, not a schema decision.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: str = Field(
        description=(
            "Gold answer, exact-matched (case-sensitive) against the \\boxed{...} span parsed "
            "from the last assistant message. A single option letter (e.g. 'C') in committed data."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task_difficulty_qwen3_32b_avg_8: float = Field(
        description=(
            "Pre-computed difficulty score (Qwen3-32B pass rate over 8 attempts). Required by the "
            "wire request model but never read by verify() or metrics — pure passthrough."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
