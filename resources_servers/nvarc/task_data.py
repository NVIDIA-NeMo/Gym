# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the nvarc server.

nvarc rows carry the ARC grid fields (train / test_input / expected_output / task_id) plus an
``agent_mode`` switch and augmentation/difficulty provenance. NVARCRunRequest does not set
extra='allow', so the untyped provenance fields below are silently dropped at today's verify
boundary (Pydantic default extra='ignore'); they are declared here as loose Optional passthrough
so the row data stays described without over-constraining it.
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    train: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Demonstration pairs [{'input': grid, 'output': grid}, ...] used to build the prompt.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    test_input: List[List[int]] = Field(
        default_factory=list,
        description="Test grid; prompt-side, and read by verify() in inductive agent_mode.",
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    expected_output: List[List[int]] = Field(
        default_factory=list,
        description="Ground-truth output grid; verify() checks exact equality against the extracted grid.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Upstream ARC task identifier.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    agent_mode: Optional[str] = Field(
        default=None,
        description="'transductive' | 'inductive'; verify() falls back to config.agent_mode when absent.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    problem_id: Optional[str] = Field(
        default=None,
        description="Upstream ARC problem id (typically equal to task_id).",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    variant: Optional[str] = Field(
        default=None,
        description="Row variant tag, e.g. 'transductive'.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    difficulty: Optional[float] = Field(
        default=None,
        description="Estimated task difficulty score.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    difficulty_bucket: Optional[str] = Field(
        default=None,
        description="Difficulty bucket label, e.g. 'hard'.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    augmentation: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Augmentation record: {augmentation_index: int, is_augmented: bool, d4_index, "
            "color_permutation, train_shuffle: nullable}."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    original_problem: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Pre-augmentation problem: {train, test_input, expected_output} in the arc_agi shape.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Generation provenance: {uuid: str, dataset_name: str|null, llm_uri: str, "
            "applied_augmentations: list of JSON-encoded strings}."
        ),
        json_schema_extra={"consumed_by": ["provenance"]},
    )
