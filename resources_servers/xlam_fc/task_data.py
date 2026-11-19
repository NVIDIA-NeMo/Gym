# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the xlam_fc server.

xLAM function-calling: expected calls live in top-level ``expected_answers`` (no
verifier_metadata). ``XlamFcVerifyRequest`` (app.py:32) declares only ``expected_answers`` with a
[] default — an EMPTY list is meaningful (reward 1.0 only when the model also made zero calls) —
so it stays optional-with-default here. Items are {'name': str, 'arguments': dict} in all
committed rows; the verifier reads both via ``.get`` and ``_normalize_arguments`` also accepts a
JSON-string ``arguments`` (falling back to {} on parse failure), matching is
subset-on-expected-keys with exact equality. The top-level row ``id`` is NOT declared on the wire
model, so today's pydantic default ``extra="ignore"`` silently drops it at verify time; it is
declared here as provenance so the row field is typed rather than invisible.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answers: List[Dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Expected function calls: each item {'name': str, 'arguments': dict of expected kwargs "
            "(a JSON-encoded string is also tolerated by the verifier)}. Empty list means the correct "
            "behavior is to make NO calls."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    id: Optional[Union[int, str]] = Field(
        default=None,
        description=(
            "Row identifier (int in committed data). Undeclared on today's wire model and therefore "
            "dropped by its extra='ignore' before verify() runs; provenance only."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
