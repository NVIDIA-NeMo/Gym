# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the abstention server.

Judge-QA family heir: extends simpleqa's ``JudgeQATaskDataCore`` (optional id + question) with
the ground truth under the name ``answer`` instead of the family's ``expected_answer``. The LLM
judge grades the extracted ``\\boxed{}`` answer against it; an abstention-token match
short-circuits the judge. Required-ness mirrors ``AbstentionRunRequest`` (app.py): every task
field is Optional on the wire (verify() reads ``body.answer or ""`` / ``body.question or ""``)
with ``extra="allow"``.
"""

from typing import Optional

from pydantic import Field

from resources_servers.simpleqa.task_data import JudgeQATaskDataCore


class TaskData(JudgeQATaskDataCore):
    answer: Optional[str] = Field(
        default=None,
        description=(
            "Ground-truth answer the LLM judge grades the extracted \\boxed{} answer against; verify() "
            "falls back to '' when absent. Family twin of simpleqa's expected_answer under another name."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
