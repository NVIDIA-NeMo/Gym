# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ChemEval question metadata, grading parameters, and English V2 judge context."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


RUBRIC_TO_TASK = {
    "fill_in_the_blank": "fill_blank",
    "short_answer": "short_answer",
    "calculation": "calculation",
    "abstract_generation": "paper_abstract",
    "outline_generation": "research_outline",
    "physicochemical": "molecular_description",
    "single_step_synthesis": "single_step_synthesis",
    "multi_step_synthesis": "multi_step_synthesis",
    "reaction_intermediate": "reaction_intermediate",
}


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    family: Literal[
        "mcq",
        "true_false",
        "classification",
        "classification_subset",
        "entity_extraction",
        "entity_recognition",
        "relation_extraction",
        "reagent_selection",
        "sider",
        "molecule_smiles",
        "molecule_formula",
        "molecule_iupac",
        "range_overlap",
        "regression",
        "judged",
    ] = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    expected_answer: str = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    task: str = Field(min_length=1, json_schema_extra={"legacy_location": "verifier_metadata"})
    level: Literal["L1", "L2", "L3", "L4"] = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    dimension: str = Field(min_length=1, json_schema_extra={"legacy_location": "verifier_metadata"})
    id: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    task_en: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    original_problem: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    subset_for_metrics: list[str] | None = Field(
        default=None, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    gold_span: float | None = Field(
        default=None, gt=0, allow_inf_nan=False, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    judge_protocol: Literal["english_v2"] | None = Field(
        default=None, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    judge_question: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    judge_scale: Literal["1-5", "0-1"] | None = Field(
        default=None, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    judge_rubric: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})

    @model_validator(mode="after")
    def validate_grading_fields(self) -> Self:
        if self.family != "judged" and not self.expected_answer.strip():
            raise ValueError("Deterministic tasks require a nonempty expected_answer")
        if self.family == "regression" and self.gold_span is None:
            raise ValueError("Regression requires a positive finite gold_span")
        if self.family == "judged":
            if self.judge_protocol != "english_v2" or not (self.judge_question or "").strip():
                raise ValueError(
                    "Judged tasks require English V2 metadata; rerun gym eval prepare --benchmark chemeval"
                )
            if self.judge_rubric not in RUBRIC_TO_TASK:
                raise ValueError("Judged tasks require a supported judge_rubric")
            expected_scale = "0-1" if self.judge_rubric == "fill_in_the_blank" else "1-5"
            if self.judge_scale != expected_scale:
                raise ValueError(f"judge_scale must be {expected_scale} for {self.judge_rubric}")
        return self
