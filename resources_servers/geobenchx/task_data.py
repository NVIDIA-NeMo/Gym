# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light schema for GeoBenchX dataset rows."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """Fields consumed by the GeoBenchX verifier and reporting."""

    model_config = ConfigDict(extra="allow")

    task_id: str
    activity: Literal["geobenchx"]
    group: str
    task_kind: str
    task_labels: list[str] = Field(default_factory=list)
    reference_solution_count: int
