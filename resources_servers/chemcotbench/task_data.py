# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public ChemCoTBench-V2 subtasks and reference-record contract."""

from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


SUBTASKS = {
    "mol_edit": ("add_v2", "delete_v2", "substitute_v2"),
    "rxn_pred": (
        "forward",
        "byproduct",
        "nepp",
        "retro",
        "rcr_catalyst",
        "rcr_reagent",
        "rcr_solvent",
        "condition_ranking",
        "yield_pred",
    ),
    "mol_und": ("fg_detect", "ring_count", "murcko_scaffold", "ring_sys_scaffold", "smiles_equivalent"),
    "mol_opt": (
        "logp",
        "qed",
        "solubility",
        "drd",
        "gsk",
        "jnk",
        "logp_qed",
        "logp_solubility",
        "qed_solubility",
        "drd_logp",
        "drd_solubility",
        "gsk_logp",
    ),
}


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")
    id: str = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    task_family: Literal["mol_edit", "rxn_pred", "mol_und", "mol_opt"] = Field(
        json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    subtask: str = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    subset_for_metrics: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    upstream_record: dict[str, Any] = Field(json_schema_extra={"legacy_location": "verifier_metadata"})

    @model_validator(mode="after")
    def validate_subtask(self) -> Self:
        if self.subtask not in SUBTASKS[self.task_family]:
            raise ValueError(f"Unsupported ChemCoTBench subtask: {self.task_family}/{self.subtask}")
        if not self.upstream_record:
            raise ValueError("upstream_record must contain the source reference fields")
        for key, expected in (
            ("task_family", self.task_family),
            ("subtask", self.subtask),
            ("anonymous_sample_id", self.id),
        ):
            if key in self.upstream_record and self.upstream_record[key] != expected:
                raise ValueError(f"upstream_record {key} conflicts with task metadata")
        return self
