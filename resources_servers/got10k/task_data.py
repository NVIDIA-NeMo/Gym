# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GOT-10k task fields, carried inside verifier_metadata."""

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class Got10kTarget(BaseModel):
    """Ground truth for one requested image."""

    model_config = ConfigDict(extra="allow")

    # Prompt-local image id (1..N), the id the model is asked to answer with.
    frame: int = Field(ge=1)
    # 1-based frame number inside the source sequence (bookkeeping only).
    source_frame: Optional[int] = None
    # Ground-truth box in pixels, [x, y, width, height].
    bbox: list[float] = Field(min_length=4, max_length=4)
    # GOT-10k cover.label value; the frame is scored only when this is above 0.
    visible: float = 1.0


class Got10kVerifierMetadata(BaseModel):
    """One GOT-10k sequence: the image size and the ground truth of every requested image."""

    model_config = ConfigDict(extra="allow")

    sequence: Optional[str] = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    # Bookkeeping written by benchmarks/got10k/prepare.py; not read by the verifier.
    split: Optional[str] = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    anno_fps: Optional[float] = Field(default=None, json_schema_extra={"legacy_location": "verifier_metadata"})
    image_width: float = Field(gt=0, json_schema_extra={"legacy_location": "verifier_metadata"})
    image_height: float = Field(gt=0, json_schema_extra={"legacy_location": "verifier_metadata"})
    targets: list[Got10kTarget] = Field(min_length=1, json_schema_extra={"legacy_location": "verifier_metadata"})


TaskData = Got10kVerifierMetadata
