# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the code_gen server.

Rows use the legacy ``verifier_metadata`` bucket for ground truth (schemas are flat: core splices
its contents up before validating; ``legacy_location`` records today's wire placement). The wire
model ``CompCodingVerifyRequest`` (app.py:65) keeps ``verifier_metadata`` as
``Optional[Dict[str, Any]]``, so nothing here is wire-required — but note the sharp edge:
verify() hard-indexes ``verifier_metadata['unit_tests']`` after the empty-output early return, so
a row missing it CRASHES verify rather than scoring 0. ``UnitTests`` mirrors the app.py model of
the same name (LiveCodeBench format), which verify() applies via ``model_validate``.

Top-level ``hash_id``/``dataset``/``source`` are provenance columns undeclared on today's wire
request model (silently dropped in transit by pydantic's default extra='ignore').
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class UnitTests(BaseModel):
    """LiveCodeBench-format test bundle; mirrors ``UnitTests`` at app.py:55."""

    model_config = ConfigDict(extra="allow")

    inputs: List[str] = Field(description="Stdin (or argument) payloads, one per test case.")
    outputs: List[str] = Field(description="Expected outputs, parallel to inputs.")
    fn_name: Optional[str] = Field(
        default=None,
        description=(
            "Function name for call-based grading; absent/None means stdin/stdout grading. "
            "Absent from all committed example rows."
        ),
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    unit_tests: Optional[UnitTests] = Field(
        default=None,
        description=(
            "Test cases the extracted code must pass. Optional on the wire (verifier_metadata is "
            "Optional[Dict]), but verify() hard-indexes it once the response is non-empty — a row "
            "without it crashes verify instead of scoring 0."
        ),
        json_schema_extra={"consumed_by": ["verify"], "legacy_location": "verifier_metadata"},
    )
    difficulty: Optional[str] = Field(
        default=None,
        description=(
            "Difficulty label read defensively ((verifier_metadata or {}).get('difficulty')) and "
            "echoed into the verify response for per-difficulty subset metrics. Absent from the "
            "committed example rows (the difficulty-bearing datasets are gitignored)."
        ),
        json_schema_extra={"consumed_by": ["verify", "metrics"], "legacy_location": "verifier_metadata"},
    )
    hash_id: Optional[str] = Field(
        default=None,
        description="Stable row hash. Never read by server code; dropped in transit (extra='ignore' wire).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    dataset: Optional[str] = Field(
        default=None,
        description="Source dataset label (e.g. 'taco'). Never read; dropped in transit.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source: Optional[str] = Field(
        default=None,
        description="Upstream problem source (e.g. 'codeforces'). Never read; dropped in transit.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
