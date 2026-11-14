# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schemas of the prepared video multiple-choice rows.

Mirrors the wire: ``VideoMCQAVerifyRequest`` and ``VideoMMEV2VerifyRequest`` read ``task_id`` and ``setting``
top-level and ``verifier_metadata`` as task.py's ``VerifierMetadata``, whose fields (``task_id`` aside, which both
places carry) only that container holds. Agentic rows also carry the ``video_binding`` that
``OpenCodeVideoRunRequest`` (responses_api_agents/opencode_video_agent/app.py) requires; there its ``source`` must
be an existing file on the agent host, so here it is only a path. ``annotation_warnings`` is preparation
provenance (``duplicate_option_text``), outside model inputs. ``VideoMCQATaskData`` is the shared shape that
VideoMME-v2 uses; ``TaskData`` adds the four-choice, ungrouped rule the VideoMME-v1/LVBench verifier enforces.
"""

from pathlib import PurePosixPath
from typing import Annotated, Literal, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_NESTED = {"legacy_location": "verifier_metadata"}


def _text(value: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError("must be a nonempty string without NUL bytes")
    return value


Text = Annotated[str, AfterValidator(_text)]
NestedText = Annotated[Text, Field(json_schema_extra=_NESTED)]


class VideoBinding(BaseModel):
    """The source video the OpenCode video agent checks and copies into its sandbox."""

    model_config = ConfigDict(extra="forbid")

    source: Text
    sandbox_path: Text
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def relative_path(self) -> Self:
        path = PurePosixPath(self.sandbox_path)
        if path.is_absolute() or ".." in path.parts or any(char in self.sandbox_path for char in "\r\n"):
            raise ValueError("sandbox video binding must be a safe workspace-relative path")
        return self


class VideoMCQATaskData(BaseModel):
    """One prepared question; extra keys are kept, as the verify request keeps them, and reported."""

    model_config = ConfigDict(extra="allow")

    task_id: Text
    setting: Literal["single_turn", "agentic"]
    expected_answer: str = Field(json_schema_extra=_NESTED)
    option_count: int = Field(ge=2, le=len(LABELS), json_schema_extra=_NESTED)
    dataset_source: NestedText
    dataset_revision: str = Field(pattern=r"^[0-9a-f]{40}$", json_schema_extra=_NESTED)
    video_id: NestedText
    video_source: NestedText
    video_sha256: str = Field(pattern=r"^[0-9a-f]{64}$", json_schema_extra=_NESTED)
    level: int | None = Field(default=None, ge=1, json_schema_extra=_NESTED)
    group_id: Text | None = Field(default=None, json_schema_extra=_NESTED)
    video_binding: VideoBinding | None = None
    annotation_warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def answer_and_binding(self) -> Self:
        if self.expected_answer not in tuple(LABELS[: self.option_count]):
            raise ValueError("expected_answer must be one available uppercase choice label")
        if self.setting == "agentic" and self.video_binding is None:
            raise ValueError("An agentic row needs the video_binding its agent reads")
        return self


class TaskData(VideoMCQATaskData):
    """VideoMME-v1 and LVBench: four choices and no VideoMME-v2 level or group."""

    @model_validator(mode="after")
    def four_ungrouped_choices(self) -> Self:
        if self.option_count != 4 or self.level is not None or self.group_id is not None:
            raise ValueError("VideoMME-v1/LVBench require four choices and no v2 group or level")
        return self
