# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-object tracking task fields.

`MultiObjectTrackingVerifierMetadata` is the model `app.py` validates `verifier_metadata` with; `TaskData` is the
whole row: the same fields plus the row's top-level `task_id`.
"""

from pydantic import BaseModel, ConfigDict, Field

from resources_servers.got10k.task_data import Got10kTarget


class MultiObjectTrackingObject(BaseModel):
    """One tracked object: its id in the prompt and its ground truth per requested frame."""

    model_config = ConfigDict(extra="allow")

    # The id the prompt gives the object ("box 1", "box 2", ...) and the answer must repeat.
    id: int
    # Same shape as the single-object server: prompt-local frame, pixel xywh box, visibility.
    targets: list[Got10kTarget] = Field(min_length=1)


class MultiObjectTrackingVerifierMetadata(BaseModel):
    """One clip: the image size and, per tracked object, the ground truth of every requested image."""

    model_config = ConfigDict(extra="allow")

    # Bookkeeping; not read by the verifier.
    sequence: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    split: str | None = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    image_width: float = Field(gt=0, json_schema_extra={"legacy_location": "verifier_metadata"})
    image_height: float = Field(gt=0, json_schema_extra={"legacy_location": "verifier_metadata"})
    objects: list[MultiObjectTrackingObject] = Field(
        min_length=1, json_schema_extra={"legacy_location": "verifier_metadata"}
    )


class TaskData(MultiObjectTrackingVerifierMetadata):
    # The row's own ID, beside `verifier_metadata`. The verifier does not read it.
    task_id: str | None = None
