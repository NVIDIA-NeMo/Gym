# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the string_match server.

All task fields are top-level (no ``verifier_metadata``) and already fully typed on the wire:
mirrors ``StringMatchRunRequest`` (app.py:34). ``verify()`` extracts an answer from the last
assistant message according to ``extraction_mode`` and compares it to ``expected_answer``.
``metadata`` is declared on the wire but absent from committed data and never read.
"""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: str = Field(
        description="Reference answer the extracted model answer is compared against.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    extraction_mode: Literal["boxed", "final_answer", "last_line", "full_response"] = Field(
        default="final_answer",
        description="How the candidate answer is extracted from the last assistant message.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    case_sensitive: bool = Field(
        default=False,
        description="Whether the string comparison is case-sensitive.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Untyped passthrough; accepted by the wire but absent from committed data.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
