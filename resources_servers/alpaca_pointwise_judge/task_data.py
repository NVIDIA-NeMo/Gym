# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task schema shared by preparation, Gym collate validation and verification."""

from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


LANGUAGES = {
    "bn": "Bengali",
    "gu": "Gujarati",
    "hi": "Hindi",
    "kn": "Kannada",
    "mr": "Marathi",
    "ml": "Malayalam",
    "ne": "Nepali",
    "or": "Odia",
    "pa": "Punjabi",
    "ta": "Tamil",
    "te": "Telugu",
    "ur": "Urdu",
}
LANGUAGE_ALIASES = {"ka": "kn", "mar": "mr", "mal": "ml"}


class AlpacaTask(BaseModel):
    model_config = ConfigDict(extra="allow")

    question: str = Field(min_length=1)
    row_id: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    language_code: str
    language_name: str
    expected_counts: dict[str, Annotated[int, Field(strict=True, gt=0)]] = Field(min_length=1)
    evaluation_set_id: str = Field(min_length=1)
    judge_pass_stage: str
    human_evaluation_pending: bool | None
    human_review_reason: str = ""

    @model_validator(mode="after")
    def validate_language(self) -> "AlpacaTask":
        if LANGUAGES.get(self.language_code) != self.language_name:
            raise ValueError("Language code/name must match a supported Indic language")
        if self.language_code not in self.expected_counts or not set(self.expected_counts) <= LANGUAGES.keys():
            raise ValueError("Expected coverage must include this row and supported languages only")
        if not self.question.strip() or not self.row_id.strip():
            raise ValueError("Question and row_id must be nonempty")
        return self


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: AlpacaTask
