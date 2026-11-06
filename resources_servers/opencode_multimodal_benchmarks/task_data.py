# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-owned fields shared by the six visual benchmark datasets."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


MetadataString = Annotated[str, Field(json_schema_extra={"legacy_location": "verifier_metadata"})]


class TaskData(BaseModel):
    """Flat schema; legacy rows carry these fields in verifier_metadata.

    Extra fields are retained for dataset provenance and caller annotations.
    References remain task data, never policy messages.
    """

    model_config = ConfigDict(extra="allow")

    benchmark: Literal["VStarBench", "HRBench8K", "VTCBench", "BabyVision", "AgentX_GoalAcc", "ZeroBench"] = Field(
        json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    index: MetadataString
    question: MetadataString
    answer: MetadataString
    options: dict[str, str] = Field(default_factory=dict, json_schema_extra={"legacy_location": "verifier_metadata"})
    category: MetadataString = ""
    l2_category: MetadataString = ""
    ans_type: MetadataString = ""
    cycle_category: MetadataString = ""
    split: MetadataString = ""
    expected_samples: int | None = Field(
        default=None, gt=0, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    dataset_revision: MetadataString = ""
    source_index: MetadataString = ""
    GT_json: MetadataString = ""
    query_type: MetadataString = ""
    query_type_raw: MetadataString = ""
    modality: MetadataString = ""
    source_media_sha256: dict[str, str] = Field(
        default_factory=dict, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
    video_sampling: dict[str, object] = Field(
        default_factory=dict, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
