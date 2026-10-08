# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the lc_niah server.

Rows are graphwalks-derived long-context needle-in-a-haystack tasks, so the shared core
(``expected_answer`` + ``n_tokens``) is imported from ``resources_servers.graphwalks.task_data``
instead of redefined. Only ``expected_answer`` is typed on the wire (LCNIAHRunRequest,
extra='allow'); every other task field arrives as an untyped extra and is therefore Optional
here even though committed rows always carry it.
"""

from typing import Optional

from pydantic import Field

from resources_servers.graphwalks.task_data import GraphWalksCore


class TaskData(GraphWalksCore):
    problem_type: Optional[str] = Field(
        default=None,
        description="Task family ('parents' | 'bfs'); untyped wire extra, unread by verify().",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    question: Optional[str] = Field(
        default=None,
        description="The bare question text (also embedded in the prompt); untyped wire extra, unread.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source: Optional[str] = Field(
        default=None,
        description="Origin dataset tag, e.g. 'graphwalks'; untyped wire extra, unread.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
