# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the proof_jury_judge server: an olympiad problem whose proof the jury grades."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    problem: str = Field(
        description="Problem statement (LaTeX/markdown); fills the policy prompt and the judge prompt's {problem}.",
        json_schema_extra={"consumed_by": ["prompt", "verify"]},
    )
    problem_idx: Optional[str] = Field(
        default=None,
        description="Stable problem identifier (e.g. 'G50').",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source_dataset: Optional[str] = Field(
        default=None,
        description="URL of the dataset the problem was taken from.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    source_id: Optional[str] = Field(
        default=None,
        description="Problem identifier within the source dataset.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    dataset_license: Optional[str] = Field(
        default=None,
        description="License of the source dataset (e.g. 'CC-BY-4.0').",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    tier: Optional[str] = Field(
        default=None,
        description="Difficulty tier (EASY/MEDIUM/HARD/UNSOLVED); used by compute_metrics for subset breakdowns.",
        json_schema_extra={"consumed_by": ["metrics"]},
    )
