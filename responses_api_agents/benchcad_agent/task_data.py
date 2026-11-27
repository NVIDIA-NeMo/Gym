# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task fields for BenchCAD's self-contained evaluation agent."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """Identify a prepared part and its native BenchCAD task variant."""

    model_config = ConfigDict(extra="allow")

    task: Literal["vision2code", "codeedit", "vision_qa", "code_qa"]
    record_id: str = Field(pattern=r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
    family: str = ""
