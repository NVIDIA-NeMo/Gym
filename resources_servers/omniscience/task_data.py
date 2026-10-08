# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the omniscience server.

Judge-QA family heir: extends simpleqa's ``TaskData`` (optional id/question/expected_answer,
all read defensively by the LLM-judge ``verify()``) with ``domain``/``topic`` provenance
columns. Required-ness mirrors ``OmniscienceRunRequest`` (app.py): every task field is Optional
on the wire with ``extra="allow"``.
"""

from typing import Optional

from pydantic import Field

from resources_servers.simpleqa.task_data import TaskData as SimpleQATaskData


class TaskData(SimpleQATaskData):
    domain: Optional[str] = Field(
        default=None,
        description=(
            "Top-level knowledge domain (e.g. 'Science Engineering and Mathematics'); never read by "
            "verify() or compute_metrics, echoed through for downstream analysis."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    topic: Optional[str] = Field(
        default=None,
        description=(
            "Topic within the domain (e.g. 'Physics'); never read by verify() or compute_metrics, "
            "echoed through for downstream analysis."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
