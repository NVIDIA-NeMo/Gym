# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task identity for agent requests and reference data for server-side grading."""

from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator, model_validator


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


class AnswerFieldMetadata(BaseModel):
    """Answer field metadata used to distinguish labels from numbers."""

    parameters: dict[str, JsonValue] = Field(default_factory=dict)


class AnswerMetadata(BaseModel):
    """Metadata from the data generation pipeline used for grading."""

    nodes: list[AnswerFieldMetadata] = Field(default_factory=list)


class PdfAnalystQaTask(BaseModel):
    """Task question, reference answer, and grading metadata kept on the server."""

    model_config = ConfigDict(extra="allow")
    task_id: str = Field(pattern=r"^task_[0-9a-f]{24}$")
    workspace_id: str = Field(pattern=r"^workspace_[0-9a-f]{24}$")
    data_split: Literal["rl", "dev"]
    question: str
    final_answer: str = Field(description="Reference answer from the dataset; not the agent response.")
    answer_contract: AnswerFormat
    units: str | None = None
    answer_program: AnswerMetadata = Field(default_factory=AnswerMetadata)


class DocumentRecord(BaseModel):
    """One selected PDF: its path under the bucket prefix, its path in the sandbox, and its checksum."""

    model_config = ConfigDict(extra="forbid")
    path: str = Field(min_length=1)
    target: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("path", "target")
    @classmethod
    def safe_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in value.split("/") or path.suffix.lower() != ".pdf":
            raise ValueError("document paths must be relative PDF paths")
        if "\\" in value or "\x00" in value:
            raise ValueError("invalid document path")
        return str(path)


class PdfAnalystQaTaskRow(PdfAnalystQaSeedRequest):
    """A self-contained task row: its identity, the reference kept on the server, and the PDFs the agent may read."""

    reference: PdfAnalystQaTask
    documents: list[DocumentRecord] = Field(min_length=1)

    @field_validator("documents")
    @classmethod
    def unique_targets(cls, value: list[DocumentRecord]) -> list[DocumentRecord]:
        targets = [document.target for document in value]
        if len(targets) != len(set(targets)):
            raise ValueError("document targets must be unique")
        return value

    @model_validator(mode="after")
    def reference_matches_identity(self) -> "PdfAnalystQaTaskRow":
        if (self.reference.task_id, self.reference.workspace_id) != (self.task_id, self.workspace_id):
            raise ValueError("reference task and workspace differ from the row's")
        return self


TaskData = PdfAnalystQaTaskRow
