# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the arc_agi server.

Task fields ride at the row top level (no verifier_metadata). All four fields carry permissive
defaults on the wire (ARCAGIRunRequest), so none is required here. verify() reads only
``expected_output`` (exact grid equality against the parsed ``\\boxed{[[...]]}`` answer);
``train``/``test_input`` exist because the same fields seed the prompt. nvarc's TaskData
subclasses this model.
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
        description="Test grid the model must transform; prompt-side only for this server.",
        json_schema_extra={"consumed_by": ["prompt"]},
    )
    expected_output: List[List[int]] = Field(
        default_factory=list,
        description="Ground-truth output grid; verify() checks exact equality against the extracted grid.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    task_id: Optional[str] = Field(
        default=None,
        description="Upstream ARC-AGI task identifier.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
