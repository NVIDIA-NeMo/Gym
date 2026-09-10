# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the omniscience server.

An optional id/question/expected_answer, all read defensively by the LLM-judge ``verify()``, plus
``domain``/``topic`` provenance columns. Required-ness mirrors ``OmniscienceRunRequest`` (app.py):
every task field is Optional on the wire with ``extra="allow"``.
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
            "Ground-truth answer interpolated into the judge prompt ({expected_answer} placeholder) and "
            "echoed on the verify response; verify() falls back to '' when absent."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
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
