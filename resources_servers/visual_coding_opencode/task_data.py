# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the visual_coding_opencode server.

A row is one image question: the image and prompt travel in ``responses_create_params``; the
task-owned fields are the gold answer the GPT judge grades the final ``\\boxed{}`` answer
against, the question text the judge sees, and provenance carried through to metrics.
Required-ness mirrors ``VisualCodingOpenCodeVerifyRequest`` (app.py), which inherits this model:
``expected_answer`` is required, ``question`` defaults to "", and unknown fields are kept
(``extra="allow"``).
"""

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: str = Field(
        description="Gold answer the judge compares the model's final \\boxed{} answer against.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    question: str = Field(
        default="",
        description="Question text shown to the judge (the prompt without the visual-reasoning instruction).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Stable task id for eval planning and task selection (BabyVision's taskId as a string).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source_id: Optional[Union[str, int]] = Field(
        default=None,
        description="Row id in the source dataset (BabyVision's taskId).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    category: Optional[str] = Field(
        default=None,
        description="Top-level task category (e.g. 'Visual Tracking').",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    l2_category: Optional[str] = Field(
        default=None,
        description="Second-level task category (e.g. 'Find the different').",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    ans_type: Optional[str] = Field(
        default=None,
        description="Answer format: 'blank' (free-form) or 'choice' (option letter).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
