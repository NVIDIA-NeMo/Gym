# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task identity for agent requests and reference data for server-side grading."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


type AnswerFormat = Literal[
    "bare_value",
    "value_with_unit",
    "tagged",
    "value_with_citation",
    "json_object",
    "structured_json",
    "bracket_list",
]


class PdfAnalystQaSeedRequest(BaseModel):
    """Task identity used to initialize an episode session."""

    model_config = ConfigDict(extra="allow")

    task_id: str = Field(pattern=r"^task_[0-9a-f]{24}$")
    workspace_id: str = Field(pattern=r"^workspace_[0-9a-f]{24}$")
    prompt_version: str = Field(min_length=1)
    development_trial: bool = False


TaskData = PdfAnalystQaSeedRequest


class AnswerFieldMetadata(BaseModel):
    """Answer field metadata used to distinguish labels from numbers."""

    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class AnswerMetadata(BaseModel):
    """Metadata from the data generation pipeline used for grading."""

    nodes: list[AnswerFieldMetadata] = Field(default_factory=list)


class PdfAnalystQaTask(BaseModel):
    """Task question, reference answer, and grading metadata kept on the server."""

    task_id: str = Field(pattern=r"^task_[0-9a-f]{24}$")
    workspace_id: str = Field(pattern=r"^workspace_[0-9a-f]{24}$")
    data_split: Literal["rl", "dev"]
    question: str
    final_answer: str = Field(description="Reference answer from the dataset; not the agent response.")
    answer_contract: AnswerFormat
    units: str | None = None
    answer_program: AnswerMetadata = Field(default_factory=AnswerMetadata)
