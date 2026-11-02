# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the ruler server.

Task fields ride at the row top level; there is no verifier_metadata. Mirrors
``RulerVerifyRequest`` (app.py), the strictest wire model in the survey: all three fields are
required with no defaults, so a row missing any of them 422s today and fails validation here.
``length`` stays required despite never being scored against — required-ness mirrors the wire,
not verify()'s reads.
"""

from typing import List

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    outputs: List[str] = Field(
        description="Reference strings the response is substring-matched against (needle values, answers).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    length: int = Field(
        description=(
            "Context-length bucket of the row (e.g. 4096). Wire-required but only echoed through "
            "the verify response; never scored against."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    subset: str = Field(
        description=(
            "RULER subset name (one of the 13 enumerated by RulerVerifyResponse, e.g. 'vt', "
            "'cwe', 'qa_1'). Routes scoring — qa_1/qa_2 take string_match_part_single (max), all "
            "others string_match_all_single (avg) — and becomes a dynamic per-subset key in the "
            "verify response for metrics aggregation."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
