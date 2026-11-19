# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verifier-owned metadata for visual coding training."""

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Benchmark = Literal["chart2code", "mmsvg_img2svg", "mmsvg_text2svg", "image2struct_latex", "design2code_full"]


class CodingTask(BaseModel):
    """A task identity and private reference; only the prompt is model-visible."""

    model_config = ConfigDict(extra="forbid")
    benchmark: Benchmark
    task_id: str = Field(min_length=1, max_length=256)
    prompt: str = Field(min_length=1)
    source_split: Literal["train", "validation", "example"]
    semantic_prompt: str = ""
    ground_truth: str = ""
    reference_image: str | None = None
    reference_sha256: str | None = Field(default=None, pattern="^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def validate_reference(self) -> "CodingTask":
        if not self.prompt.strip():
            raise ValueError("Task prompt must contain text")
        if self.benchmark == "mmsvg_text2svg":
            if self.reference_image is not None or self.reference_sha256 is not None:
                raise ValueError("Text2SVG has no reference screenshot")
        else:
            if self.reference_image is None or self.reference_sha256 is None:
                raise ValueError("Image tasks require a reference image and SHA256")
            path = PurePosixPath(self.reference_image)
            if path.is_absolute() or ".." in path.parts or not path.parts or "\\" in self.reference_image:
                raise ValueError("Reference image must be a relative path below reference_root")
        return self


class TaskData(BaseModel):
    """Task schema consumed by the resources server, outside model input."""

    model_config = ConfigDict(extra="forbid")

    task: CodingTask = Field(
        description="Training task identity, split, original prompt and private reference binding.",
        json_schema_extra={"consumed_by": ["seed", "verify", "provenance"]},
    )
    problem_id: str | None = Field(
        default=None,
        description="Flat-row task identity, scoped by benchmark for training without a collector task index.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )

    @model_validator(mode="after")
    def validate_problem_id(self) -> "TaskData":
        if self.problem_id is not None and self.problem_id != f"{self.task.benchmark}:{self.task.task_id}":
            raise ValueError("problem_id must match benchmark:task_id from the coding task")
        return self
