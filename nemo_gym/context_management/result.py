# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small semantic completion manifest; training tokens remain in ordinary capture."""

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.openai_utils import NeMoGymResponseOutputItem


# The framework may append its attempt UUID to the logical slot. Gym preserves
# that complete scope; the trainer maps it back to the stable owner on publish.
LOGICAL_ROLLOUT_ID_PATTERN = r"^[A-Za-z0-9_-]+_g[0-9]+(?:_a[0-9a-f]{32})?$"


def capture_rollout_id(logical_rollout_id: str, segment_index: int) -> str:
    if re.fullmatch(LOGICAL_ROLLOUT_ID_PATTERN, logical_rollout_id) is None:
        raise ValueError("Context management requires a framework logical owner ID (_gN, optionally _a<UUID hex>)")
    if type(segment_index) is not int or segment_index < 0:
        raise ValueError("Invalid segment ordinal")
    return f"{logical_rollout_id}_s{segment_index}"


class SelectedAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    response_id: str = Field(min_length=1)
    finish_reason: str | None
    last_output_item: NeMoGymResponseOutputItem | None
    # Per-action additions, not the full media prefix again. Equal assets may
    # appear repeatedly because occurrences, not unique images, are ordered.
    new_media_occurrence_refs: list[str] = Field(default_factory=list)


class LogicalCCSegment(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    capture_rollout_id: str = Field(min_length=1)
    segment_index: int = Field(ge=0)
    selected_actions: list[SelectedAction] = Field(min_length=1)
    media_occurrence_refs: list[str]


class LogicalCCResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    logical_rollout_id: str = Field(pattern=LOGICAL_ROLLOUT_ID_PATTERN)
    segments: list[LogicalCCSegment] = Field(min_length=1)
    media_assets: dict[str, dict[str, Any]]
    outcome: Literal["completed", "max_steps", "max_output_tokens", "execution_failure"]

    @model_validator(mode="after")
    def validate_selected_chain(self) -> "LogicalCCResult":
        selected: set[str] = set()
        media: set[str] = set()
        for index, segment in enumerate(self.segments):
            if segment.segment_index != index or segment.capture_rollout_id != capture_rollout_id(
                self.logical_rollout_id, index
            ):
                raise ValueError("Segments must have contiguous ordinals and owner-derived capture IDs")
            for action in segment.selected_actions:
                if action.response_id in selected:
                    raise ValueError("Selected response IDs must be unique")
                selected.add(action.response_id)
            if [ref for action in segment.selected_actions for ref in action.new_media_occurrence_refs] != (
                segment.media_occurrence_refs
            ):
                raise ValueError("Action media deltas must reproduce the segment's ordered occurrences")
            media.update(segment.media_occurrence_refs)
        if not media.issubset(self.media_assets):
            raise ValueError("Missing selected media asset")
        for media_id, asset in self.media_assets.items():
            if asset.get("media_id") != media_id or not isinstance(asset.get("source_part"), dict):
                raise ValueError("Malformed media asset")
        return self
