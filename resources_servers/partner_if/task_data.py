# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light task contract shared by Gym data tooling and the IF verifier."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool, StringConstraints, model_validator


Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Dimension = Literal["if", "completion"]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VerifierSpec(StrictModel):
    id: Text
    params: dict[str, JsonValue] = Field(default_factory=dict)


class GradingItem(StrictModel):
    id: Text
    constraint: Text
    verifier: VerifierSpec
    dimension: Dimension = "if"
    category: Literal["always", "task_context"] = "always"
    depends_on: list[Text] = Field(default_factory=list)
    guidance: str = ""
    # Source grounding and analogue data are preserved but are not extra scoring rules.
    provenance: dict[str, JsonValue] = Field(default_factory=dict)
    taxonomy: list[Text] = Field(default_factory=list)
    analogues: list[dict[str, JsonValue]] = Field(default_factory=list)


class GradingContract(StrictModel):
    schema_version: Literal["partner-if-v4"] = "partner-if-v4"
    task_id: Text
    origin: Text
    source: Text
    rubric_version: Text
    audit_signoff: StrictBool
    data_kind: Literal["reported", "reconstruction", "public_task", "synthetic_smoke"]
    evidence_mode: Literal["text_response", "frozen_tool_history", "tool_call_response", "live_agent"] = (
        "text_response"
    )
    rubrics: list[GradingItem] = Field(min_length=1)
    # Gates use the same verifier registry but never earn points. For example,
    # citation presence must not be confused with full citation coverage.
    prerequisites: list[GradingItem] = Field(default_factory=list)

    @property
    def checks(self) -> list[GradingItem]:
        """All registered checks, including explicitly unscored prerequisite gates."""
        return self.prerequisites + self.rubrics

    @model_validator(mode="after")
    def validate_graph(self):
        by_id = {r.id: r for r in self.checks}
        if len(by_id) != len(self.checks):
            raise ValueError("Duplicate rubric IDs")
        if not any(r.dimension == "if" for r in self.rubrics):
            raise ValueError("An IF benchmark task must have an IF dimension")
        visited, visiting = set(), set()

        def visit(identifier: str) -> None:
            if identifier not in by_id:
                raise ValueError(f"Unknown prerequisite rubric: {identifier}")
            if identifier in visiting:
                raise ValueError("Cyclic rubric dependencies")
            if identifier in visited:
                return
            visiting.add(identifier)
            for dependency in by_id[identifier].depends_on:
                visit(dependency)
                if by_id[identifier].dimension == "if" and by_id[dependency].dimension == "completion":
                    raise ValueError("An IF requirement cannot depend on a completion requirement")
            visiting.remove(identifier)
            visited.add(identifier)

        for rubric in self.checks:
            if len(set(rubric.depends_on)) != len(rubric.depends_on):
                raise ValueError("Duplicate prerequisites")
            visit(rubric.id)
        return self


SHA256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class CurationReceipt(StrictModel):
    """An immutable evidence artifact; never a curator's local path."""

    uri: Annotated[str, StringConstraints(pattern=r"^https://\S+$")]
    sha256: SHA256


class CurationEvaluation(StrictModel):
    """Versioned all-IF-pass counts, independent of task completion diagnostics."""

    model: Text
    passed: int | None = Field(default=None, ge=0, le=3, strict=True)
    attempts: Literal[3] = 3
    valid_attempts: int = Field(ge=0, le=3, strict=True)
    stage: Literal["discovery", "confirmation", "historical"]
    actor_sha256: SHA256
    rubric_hash: SHA256
    settings: dict[str, JsonValue]
    receipts: list[CurationReceipt] = Field(default_factory=list)

    @model_validator(mode="after")
    def complete_count(self):
        if (self.passed is not None) != (self.valid_attempts == self.attempts):
            raise ValueError("A passed/3 count requires exactly three valid attempts")
        if self.valid_attempts and not self.receipts:
            raise ValueError("Evaluated attempts require evidence receipts")
        return self


class CurationMetadata(StrictModel):
    """Non-scoring task annotations; never included in actor or judge input."""

    task_id: Text
    provenance: Literal["original", "modified", "reconstruction"]
    source: Text
    bug_ids: list[Text] | None = None
    related_bug_ids: list[Text] = Field(default_factory=list)
    source_defect_class: Text | None = None
    source_case_id: Text | None = None
    source_task_id: Text | None = None
    taxonomy_revision: Text
    rubric_families: dict[Text, list[Text]]
    rubric_if_types: dict[Text, list[Text]] = Field(default_factory=dict)
    evaluations: list[CurationEvaluation] = Field(default_factory=list)
    notes: list[Text] = Field(default_factory=list)


class TaskData(StrictModel):
    partner_if: GradingContract = Field(
        json_schema_extra={
            "legacy_location": "verifier_metadata",
            "consumed_by": ["verify", "metrics", "provenance"],
        }
    )
    curation: CurationMetadata | None = Field(default=None, exclude_if=lambda value: value is None)

    @model_validator(mode="after")
    def curation_identity(self):
        if self.curation is None:
            return self
        if self.curation.task_id != self.partner_if.task_id:
            raise ValueError("Curation task ID must match the scoring contract")
        if_ids = {r.id for r in self.partner_if.rubrics if r.dimension == "if"}
        if set(self.curation.rubric_families) != if_ids or any(
            not families for families in self.curation.rubric_families.values()
        ):
            raise ValueError("Curation families must cover every IF rubric exactly once")
        if self.curation.rubric_if_types and (
            set(self.curation.rubric_if_types) != if_ids
            or any(not types for types in self.curation.rubric_if_types.values())
        ):
            raise ValueError("Curated IF types must cover every IF rubric exactly once")
        return self
