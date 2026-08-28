# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the math_advanced_calculations server (MultiVerse math-hard tool use).

Task fields ride at the row top level; there is no verifier_metadata. Required-ness mirrors
``MultiVerseMathHardVerifyRequest`` (app.py:59): all four fields are wire-required, including
``id``/``depth``/``breadth`` which scoring never reads (they are only echoed via model_dump into
the verify response). The wire nominally types ``ground_truth`` as ``list[float] | str``, but
verify() unconditionally ``json.loads(body.ground_truth)`` (app.py:108), so a bare list would
crash — only the JSON-encoded-string form works, and every committed row uses it; per protocol
the field stays typed ``str`` with the encoded shape documented.
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    ground_truth: str = Field(
        description=(
            "JSON-encoded list of expected floats, one per required tool-call solution, e.g. "
            "'[-3.0, 0.5, 9.9, 0.28366218546322625]'. verify() json.loads it and float-compares "
            "against the json.loads(output)['solution'] values scavenged from function_call_output "
            "items. The wire model also admits a bare list[float], but that branch crashes verify()."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    id: int = Field(
        description="Row identifier; wire-required but never read by scoring — echoed into the response.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    depth: int = Field(
        description=(
            "Nesting depth of the generated expression tree; wire-required but never read by "
            "scoring — echoed into the response."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    breadth: int = Field(
        description=(
            "Number of independent sub-expressions (equals the expected solution count); "
            "wire-required but never read by scoring — echoed into the response."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
