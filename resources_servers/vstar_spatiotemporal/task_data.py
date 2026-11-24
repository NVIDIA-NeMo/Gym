# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the V-STaR resources server.

`VerifierMetadata` is the model `app.py` validates `verifier_metadata` with; `TaskData` is the whole row: the same
fields plus the row's top-level `task_id`.
"""

from typing import Literal, Optional, get_args

from pydantic import BaseModel, ConfigDict, Field


SubTask = Literal["vqa", "temporal", "spatial", "spatial_2", "temporal_2"]
SUB_TASKS: tuple[str, ...] = get_args(SubTask)

# Today's rows carry every `VerifierMetadata` field inside `verifier_metadata`.
_IN_METADATA = {"legacy_location": "verifier_metadata"}


class VstarBox(BaseModel):
    model_config = ConfigDict(extra="allow")

    # The whole second this box is annotated at.
    timestamp: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


class VerifierMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    # Position of the item in the annotation file; the five tasks of one item share it.
    item_index: int = Field(json_schema_extra=_IN_METADATA)
    sub_task: SubTask = Field(json_schema_extra=_IN_METADATA)
    vid: Optional[str] = Field(default=None, json_schema_extra=_IN_METADATA)
    domain: Optional[str] = Field(default=None, json_schema_extra=_IN_METADATA)
    fps: float = Field(gt=0, json_schema_extra=_IN_METADATA)
    frame_count: int = Field(ge=0, json_schema_extra=_IN_METADATA)
    width: float = Field(gt=0, json_schema_extra=_IN_METADATA)
    height: float = Field(gt=0, json_schema_extra=_IN_METADATA)
    question: str = Field(json_schema_extra=_IN_METADATA)
    answer: str = Field(json_schema_extra=_IN_METADATA)
    # Annotated [start, end] of the event, in seconds.
    timestamps: list[float] = Field(min_length=2, max_length=2, json_schema_extra=_IN_METADATA)
    bboxes: list[VstarBox] = Field(json_schema_extra=_IN_METADATA)
    # Times in seconds of the frames the prompt shows, in order.
    frame_times: Optional[list[float]] = Field(default=None, json_schema_extra=_IN_METADATA)


class TaskData(VerifierMetadata):
    # The row's own ID, beside `verifier_metadata`: `<vid>/<item_index>/<sub_task>`. The verifier does not read it.
    task_id: Optional[str] = None
