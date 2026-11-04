# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ACMG answer key and case provenance, carried inside verifier_metadata and never shown to the model."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Label = Literal["P", "LP", "VUS", "LB", "B"]
Arm = Literal["with_rules", "without_rules"]


class Reference(BaseModel):
    """The answer key in the shape the upstream scorer's `reward(response, reference, arm)` reads."""

    model_config = ConfigDict(extra="forbid")

    label: Label = Field(description="The expert panel's classification.")
    label_scoring: Literal["scored", "any_of", "excluded"] = Field(
        description="scored: one accepted label; any_of: any of accepted_labels counts; excluded: the label is not scored."
    )
    accepted_labels: list[Label] = Field(
        description="Labels that earn label credit; empty when the label is excluded."
    )
    required: dict[str, list[str]] = Field(
        description="Criterion code -> strengths at which it must be returned (recall side of the criteria score)."
    )
    acceptable: list[tuple[str, str]] = Field(
        description="(code, strength) pairs a response may return without penalty (precision side)."
    )
    combining: dict[str, Any] | None = Field(
        default=None,
        description="The rule document's combining rules; used for self-consistency in the with-rules arm only.",
    )


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    record_id: str = Field(
        description="`<case_id>:<arm>`, the upstream response identity.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
    case_id: str = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    arm: Arm = Field(
        description="with_rules: the system prompt carries the gene's rule document; without_rules: it does not.",
        json_schema_extra={"legacy_location": "verifier_metadata"},
    )
    reference: Reference = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    rule_document: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    gene: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    role: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    skill: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    consequence: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    approved_2026: bool | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    version_status: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    confirmation: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
