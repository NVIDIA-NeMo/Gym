# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the nvarc server.

nvarc rows extend the arc_agi shape (imported, not redefined) with an ``agent_mode`` switch and
augmentation/difficulty provenance. NVARCRunRequest does not set extra='allow', so the untyped
provenance fields below are silently dropped at today's verify boundary (Pydantic default
extra='ignore'); they are declared here as loose Optional passthrough so the row data stays
described without over-constraining it.
"""

from typing import Any, Dict, List, Optional

from pydantic import Field

from resources_servers.arc_agi.task_data import TaskData as ARCAGITaskData


class TaskData(ARCAGITaskData):
    # Re-declared from the arc_agi parent because nvarc's verify() also consumes the test grid
    # (it feeds the extracted program in inductive mode).
    test_input: List[List[int]] = Field(
        default_factory=list,
        description="Test grid; prompt-side, and read by verify() in inductive agent_mode.",
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
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
