# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the frontierscience_judge server.

An optional id/question/expected_answer, all read defensively by the LLM-judge ``verify()``, plus
a ``subject`` metrics column and an optional ``rubric`` used by the research judge_mode.
Required-ness mirrors ``FrontierScienceJudgeRunRequest`` (app.py): every task field is Optional
on the wire with ``extra="allow"``.
"""

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[Union[int, str]] = Field(
        default=None,
        description="Ride-along task identifier; never read by verify() or metrics.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question: Optional[str] = Field(
        default=None,
        description=(
            "Question text interpolated into the LLM-judge prompt ({question} placeholder); "
            "verify() falls back to '' when absent."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_answer: Optional[str] = Field(
        default=None,
        description=(
            "Ground-truth answer interpolated into the judge prompt; also the rubric fallback in the "
            "research judge_mode when ``rubric`` is absent. verify() falls back to '' when absent."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    subject: Optional[str] = Field(
        default=None,
        description=(
            "Science subject (e.g. 'chemistry'); compute_metrics reports per-subject subsets via "
            "compute_subset_metrics(subset_key='subject'). Not read by verify()."
        ),
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    rubric: Optional[str] = Field(
        default=None,
        description=(
            "Grading rubric consumed by the research judge_mode; verify() falls back to expected_answer "
            "when absent. Declared on the wire model but present in no committed row."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
