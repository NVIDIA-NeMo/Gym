# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task identity for agent requests and reference data for server-side grading."""

from typing import Literal

from pydantic import BaseModel, Field, JsonValue

from nemo_gym.server_runtime import BaseSeedSessionRequest
from resources_servers.pdf_analyst_qa.verifiers.answer_formats import AnswerFormat


class PdfAnalystQaSeedRequest(BaseSeedSessionRequest):
    """Task identity used to initialize an episode session."""

    task_id: str = Field(pattern=r"^task_[0-9a-f]{24}$")
    workspace_id: str = Field(pattern=r"^workspace_[0-9a-f]{24}$")
    prompt_version: str = Field(min_length=1)


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
