# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the math_with_code server (math QA with a Python-executor tool).

Task fields ride at the row top level; there is no verifier_metadata. Required-ness mirrors
``PythonMathRunRequest`` (app.py:150): ``expected_result`` required, ``expected_code_contains``
defaulted to ''. ``id`` is present in committed rows but undeclared on today's wire model
(default ``extra="ignore"``, so it is silently dropped in transit); declared here as provenance.
verify() extracts \\boxed{} from assistant messages, falling back to function_call_output stdout.
"""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_result: str = Field(
        description=(
            "Gold final answer, often LaTeX (e.g. '32' or '\\\\(\\\\frac{333}{1997}\\\\)'); compared "
            "against the \\boxed{} extraction (or executor stdout fallback) after normalization."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    expected_code_contains: str = Field(
        default="",
        description=(
            "Declared on the wire as 'optional validation' but NEVER read by verify() — dead field; "
            "empty string in all committed rows. Kept because the wire declares it with default ''."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    id: Optional[int] = Field(
        default=None,
        description=(
            "Row identifier. Undeclared on today's wire model and silently dropped in transit "
            "(extra='ignore'); provenance only."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
