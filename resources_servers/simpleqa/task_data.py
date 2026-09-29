# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the simpleqa server.

Parent of the defensive judge-QA family (simpleqa, omniscience, frontierscience_judge,
abstention): an optional id + question + ground-truth answer, every field read defensively by an
LLM-judge ``verify()`` (a missing field becomes an empty string in the judge prompt), with
``extra="allow"`` provenance passthrough. Heirs import ``JudgeQATaskDataCore`` (id + question;
abstention names its answer field ``answer``) or ``TaskData`` (adds ``expected_answer``) instead
of redefining these fields. Required-ness mirrors ``SimpleQARunRequest`` (app.py): all task
fields are Optional on the wire. The judge prompt template and judge model come from server
config, not the row.
"""

from typing import Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class JudgeQATaskDataCore(BaseModel):
    """Shared id + question core of the judge-QA family."""

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


class TaskData(JudgeQATaskDataCore):
    expected_answer: Optional[str] = Field(
        default=None,
        description=(
            "Ground-truth answer interpolated into the judge prompt ({expected_answer} placeholder) and "
            "echoed on the verify response; verify() falls back to '' when absent."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
