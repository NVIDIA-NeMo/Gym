# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the SA-V tracks resources server."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TrackTarget(BaseModel):
    model_config = ConfigDict(extra="allow")

    frame: int
    visible: bool = True
    bbox: Optional[list[float]] = Field(default=None, min_length=4, max_length=4)
    xy: Optional[list[float]] = Field(default=None, min_length=2, max_length=2)
    mask: Optional[dict[str, Any]] = None


class TrackObject(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: int
    reference_bbox: Optional[list[float]] = Field(default=None, min_length=4, max_length=4)
    reference_xy: Optional[list[float]] = Field(default=None, min_length=2, max_length=2)
    targets: list[TrackTarget]


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    task: Literal["pt", "box", "ref"] = "box"
    reference_frame: int = 0
    objects: list[TrackObject] = Field(default_factory=list)
    coordinate_space: Optional[str] = None
    verifier: Optional[str] = None
    source_id: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None
    require_answer_tags: bool = True
    penalize_extra_predictions: bool = True
    absence_score: float = Field(default=1.0, ge=0.0, le=1.0)
