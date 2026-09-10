# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the abstention server.

An optional id + question + ground truth (named ``answer``). The LLM judge grades the extracted
``\\boxed{}`` answer against ``answer``; an abstention-token match short-circuits the judge.
Required-ness mirrors ``AbstentionRunRequest`` (app.py): every task field is Optional on the wire
(verify() reads ``body.answer or ""`` / ``body.question or ""``) with ``extra="allow"``.
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
    answer: Optional[str] = Field(
        default=None,
        description=(
            "Ground-truth answer the LLM judge grades the extracted \\boxed{} answer against; verify() "
            "falls back to '' when absent."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
