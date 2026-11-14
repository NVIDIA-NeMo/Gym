# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the video_caption server.

Both task fields ride inside ``verifier_metadata`` (``VideoCaptionVerifyRequest`` in app.py).
``atomic_events.visual_events`` is the reference event list every reward type judges against;
``reference_caption`` is only needed when ``visual_reward_type`` is ``precision`` or ``f1``, so it is
optional here, but a row that carries it must carry text. The verifier masks a row that violates
either rule instead of raising, so rejecting it at data-validation time is what keeps such a row
from spending a policy rollout.
"""

from pydantic import BaseModel, ConfigDict, Field, field_validator


VERIFIER_METADATA = {"legacy_location": "verifier_metadata"}


def _nonblank(value: str, what: str) -> str:
    if not value.strip():
        raise ValueError(f"{what} must not be blank")
    return value


class VisualAtomicEvents(BaseModel):
    model_config = ConfigDict(extra="allow")

    visual_events: list[str] = Field(
        min_length=1,
        description="Reference atomic visual events the judge checks the caption against, in video order.",
    )

    @field_validator("visual_events")
    @classmethod
    def events_have_text(cls, events: list[str]) -> list[str]:
        for event in events:
            _nonblank(event, "every atomic_events.visual_events entry")
        return events


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    atomic_events: VisualAtomicEvents = Field(
        description="Reference events; audio fields may ride along and are ignored by the verifier.",
        json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["verify"]},
    )
    reference_caption: str | None = Field(
        default=None,
        description=(
            "Human reference caption that precision and F1 judge the policy's extracted events against; "
            "unused by recall. Required by precision/F1 rows."
        ),
        json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["verify"]},
    )

    @field_validator("reference_caption")
    @classmethod
    def reference_has_text(cls, caption: str | None) -> str | None:
        if caption is not None:
            _nonblank(caption, "reference_caption")
        return caption
