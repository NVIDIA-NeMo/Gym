# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Molmo2-VideoTrackEval task fields (see benchmarks/molmo2_videotrack/prepare.py).

`Molmo2VerifierMetadata` is the model the verify request validates `verifier_metadata` with
(app.py imports it from here), and `TaskData` extends it with the fields prepare.py writes for
bookkeeping. A row that passes dataset validation is therefore one the verifier accepts.
"""

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


VERIFIER_METADATA = {"legacy_location": "verifier_metadata"}


class Molmo2Object(BaseModel):
    """One matching object that is visible in a frame."""

    # Ground-truth point in pixels, used only to match predictions to objects.
    xy: list[float] = Field(min_length=2, max_length=2)
    # COCO RLE segmentation mask: {"size": [height, width], "counts": "..."}.
    mask: dict[str, Any]


class Molmo2Frame(BaseModel):
    # Prompt-local frame number (1..N), the number the model is asked to answer with.
    frame: int = Field(ge=1)
    # 0-based frame index inside the source clip (bookkeeping only).
    source_frame: Optional[int] = None
    # Matching objects visible in this frame; empty when none is.
    objects: list[Molmo2Object] = []


class Molmo2VerifierMetadata(BaseModel):
    """What the verifier reads from a task's `verifier_metadata`."""

    model_config = ConfigDict(extra="allow")

    category: Optional[str] = Field(
        default=None,
        description="Source category (misc, animal, dance, person, sports); metrics are also reported per category.",
        json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["metrics"]},
    )
    image_width: float = Field(
        gt=0,
        description="Frame width in pixels, used to scale 0-1000 grid answers.",
        json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["verify"]},
    )
    image_height: float = Field(
        gt=0,
        description="Frame height in pixels, used to scale 0-1000 grid answers.",
        json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["verify"]},
    )
    frames: list[Molmo2Frame] = Field(
        min_length=1,
        description="The frames shown in the prompt, each with the ground truth of its visible matching objects.",
        json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["verify"]},
    )


class TaskData(Molmo2VerifierMetadata):
    """A prepared row's task fields: the verifier's metadata, the row's task ID and the source annotation's fields."""

    task_id: Optional[str] = Field(
        default=None,
        description="Row-order-independent task ID, '<category>:<annotation id>'. A top-level row field.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    id: Optional[str] = Field(
        default=None,
        description="The source annotation's ID.",
        json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["provenance"]},
    )
    video: Optional[str] = Field(default=None, json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["provenance"]})
    clip: Optional[str] = Field(default=None, json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["provenance"]})
    expression: Optional[str] = Field(
        default=None,
        description="The referring expression, as it appears in the prompt.",
        json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["provenance"]},
    )
    fps: Optional[float] = Field(
        default=None,
        description="Frame rate of the source clip.",
        json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["provenance"]},
    )
