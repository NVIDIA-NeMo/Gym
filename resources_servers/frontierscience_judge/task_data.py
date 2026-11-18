# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the frontierscience_judge server.

Judge-QA family heir: extends simpleqa's ``TaskData`` (optional id/question/expected_answer,
all read defensively by the LLM-judge ``verify()``) with a ``subject`` metrics column and an
optional ``rubric`` used by the research judge_mode. Required-ness mirrors
``FrontierScienceJudgeRunRequest`` (app.py): every task field is Optional on the wire with
``extra="allow"``.
"""

from typing import Optional

from pydantic import Field

from resources_servers.simpleqa.task_data import TaskData as SimpleQATaskData


class TaskData(SimpleQATaskData):
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
