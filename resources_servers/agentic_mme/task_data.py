# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agentic-MME task fields, carried inside verifier_metadata (see benchmarks/agentic_mme/prepare.py)."""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


VERIFIER_METADATA = {"legacy_location": "verifier_metadata"}


class GoldenAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: str | int | float
    match_type: Literal["exact", "contains", "numeric"] = "contains"
    tolerance: float = Field(default=0.0, ge=0, allow_inf_nan=False)

    @field_validator("value", mode="before")
    @classmethod
    def valid_target(cls, value: Any) -> str | int | float:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)) or not str(value).strip():
            raise ValueError("golden_answer.value must be a nonempty string or finite number")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("golden_answer.value must be finite")
        return value


class ImageFile(BaseModel):
    """One task image that the server loads at seed, for a lent sandbox or its image tools."""

    model_config = ConfigDict(extra="forbid")

    # Relative to image_root on the server's host; benchmarks/agentic_mme/prepare.py writes these files.
    source: str = Field(min_length=1)
    # Relative to the sandbox working directory, as the prompt names it.
    target: str = Field(min_length=1)


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    golden_answer: GoldenAnswer = Field(
        description="Final-answer reference; tolerance applies to numeric matches only.",
        json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["verify"]},
    )
    answer_tolerance: float | None = Field(
        default=None,
        ge=0,
        allow_inf_nan=False,
        description="The author's numeric tolerance from the final answer check; shown to the judge only.",
        json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["verify"]},
    )
    task_id: str | None = Field(default=None, json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["provenance"]})
    level: int | None = Field(
        default=None,
        description="Agentic-MME difficulty level (1-3).",
        json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["metrics"]},
    )
    domain: str | None = Field(default=None, json_schema_extra=VERIFIER_METADATA | {"consumed_by": ["metrics"]})
    process_evaluation: dict[str, Any] | None = Field(
        default=None,
        description="The release's process checkpoints, kept for process evaluation; never shown to the model.",
        json_schema_extra=VERIFIER_METADATA,
    )
    synthetic: bool = Field(default=False, json_schema_extra=VERIFIER_METADATA)
    retrieval_replay: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Recorded retrieval results for retrieval.mode=replay (read by the agent).",
    )
    image_files: list[ImageFile] | None = Field(
        default=None,
        description=(
            "The task images, in prompt order. The server reads them at seed when it lends a sandbox or serves "
            "the image tools; the Agentic-MME agent's own rows carry the images in the prompt instead."
        ),
    )
