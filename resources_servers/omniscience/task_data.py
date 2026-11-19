# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the omniscience server.

Judge-QA family heir: extends simpleqa's ``TaskData`` (optional id/question/expected_answer,
all read defensively by the LLM-judge ``verify()``) with ``domain``/``subtopic`` columns and the
dataset version and revision the row was prepared from. Rows prepared by earlier versions of the
benchmark (AA-Omniscience v1) carry ``topic`` and no ``subtopic``. Required-ness mirrors
``OmniscienceRunRequest`` (app.py): every task field is Optional on the wire with ``extra="allow"``.
"""

from typing import Optional

from pydantic import Field

from resources_servers.simpleqa.task_data import TaskData as SimpleQATaskData


class TaskData(SimpleQATaskData):
    domain: Optional[str] = Field(
        default=None,
        description=(
            "Top-level knowledge domain (e.g. 'Science Engineering and Mathematics'); filled into the answer "
            "prompt, never read by verify() or compute_metrics, echoed through for downstream analysis."
        ),
        json_schema_extra={"consumed_by": ["prompt", "provenance"]},
    )
    subtopic: Optional[str] = Field(
        default=None,
        description=(
            "Subtopic within the domain (e.g. 'Biology'); filled into the answer prompt, never read by "
            "verify() or compute_metrics, echoed through for downstream analysis."
        ),
        json_schema_extra={"consumed_by": ["prompt", "provenance"]},
    )
    topic: Optional[str] = Field(
        default=None,
        description=(
            "Topic within the domain (e.g. 'Physics') in rows prepared from AA-Omniscience v1; never read by "
            "verify() or compute_metrics, echoed through for downstream analysis."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    aa_omniscience_version: Optional[str] = Field(
        default=None,
        description="AA-Omniscience version the row was prepared from (e.g. '1.1'); not read by verify() or metrics.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    aa_omniscience_dataset_revision: Optional[str] = Field(
        default=None,
        description="Hugging Face dataset revision the row was prepared from; never read by verify() or metrics.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
