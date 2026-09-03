# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the mrcr server (OpenAI MRCR long-context retrieval).

All four task fields live at the row top level (no ``verifier_metadata``) and all four are
required, mirroring ``MRCRVerifyRequest`` (app.py) which declares them without defaults.
``verify()`` scores with ``expected_answer`` + ``random_string_to_prepend`` (prefix gate then
SequenceMatcher ratio); ``n_needles`` feeds the per-needle subset breakdown in
``compute_metrics``; ``n_tokens`` is wire-required but never read (carried through only).
"""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    expected_answer: str = Field(
        description="Reference answer; graded against the response via SequenceMatcher ratio after prefix strip.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    random_string_to_prepend: str = Field(
        description="Random prefix the response must start with (hard gate: no prefix means reward 0.0).",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    n_needles: int = Field(
        description="Number of needles in the haystack; subset key for per-needle metric breakdowns.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
    n_tokens: int = Field(
        description="Approximate context length of the task. Wire-required but never read by the server.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
