# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light schema for captured next-turn and text-replay tasks."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StringConstraints, model_validator
from pydantic.fields import FieldInfo


Text = Annotated[str, StringConstraints(min_length=1, pattern=r"\S")]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


def metadata_field(**kwargs: Any) -> FieldInfo:
    return Field(
        json_schema_extra={"legacy_location": "verifier_metadata", "consumed_by": ["verify", "provenance"]}, **kwargs
    )


class SourceIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    benchmark: Text
    orig_trajectory_id: Text
    task_group_id: Text


class BaselineReference(BaseModel):
    """A private provenance pointer, never an answer or a source of new rewards."""

    model_config = ConfigDict(extra="allow")
    path: Text
    row_id: Text
    sha256: Digest


class CapturedMetadata(BaseModel):
    model_config = ConfigDict(extra="allow", allow_inf_nan=False)

    schema_version: Literal["captured-instruction-following/v1"] = metadata_field()
    evaluation_mode: Literal["native_next_turn", "single_turn_text_replay"] = metadata_field()
    source_identity: SourceIdentity = metadata_field()
    source_request_sha256: Digest = metadata_field()
    evaluation_request_sha256: Digest = metadata_field()
    private_baseline_ref: BaselineReference = metadata_field()
    criteria_status: Literal["native_reference_available", "pending_review"] = metadata_field()
    comparison_profile: Literal["case_insensitive_scalars", "normalized_scalars"] | None = metadata_field(default=None)
    expected_function: Text | None = metadata_field(default=None)
    expected_params: dict[str, JsonValue] | None = metadata_field(default=None)
    cohort_size: int | None = metadata_field(default=None, gt=0)

    @model_validator(mode="after")
    def mode_contract(self) -> "CapturedMetadata":
        if self.evaluation_mode == "native_next_turn":
            if (
                self.criteria_status != "native_reference_available"
                or self.comparison_profile is None
                or self.expected_function is None
                or self.expected_params is None
            ):
                raise ValueError(
                    "native_next_turn requires a function, parameter object, comparison profile and native reference status"
                )
            if any(isinstance(value, (dict, list)) for value in self.expected_params.values()):
                raise ValueError("native comparison profiles require scalar expected parameter values")
        elif self.criteria_status != "pending_review" or any(
            value is not None for value in (self.expected_function, self.expected_params, self.comparison_profile)
        ):
            raise ValueError("text replay is pending_review and cannot carry native scoring fields")
        return self


class TaskData(CapturedMetadata):
    """Flat task-owned fields; Gym splices verifier_metadata up for validation."""

    model_config = ConfigDict(extra="allow", allow_inf_nan=False)
    benchmark: Text
    orig_trajectory_id: Text
    task_group_id: Text
