# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The typed task fields of every ClawEval task (``task_data.json``, written by the prepare script)."""

from pydantic import BaseModel, Field


class TaskData(BaseModel):
    # The package's category (video_qa, video_ocr, video_search, video_edit, video_chart, video_image, video_webpage,
    # video_workflow, ...): what a per-family breakdown of a rollouts file groups by.
    family: str = Field(min_length=1)
    # SHA-256 of the prompt the agent receives, and of the package's scoring inputs (task.yaml, grader.py and the
    # grader files; reference media by path and size), so a results row names the exact task it scored.
    prompt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_package_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
