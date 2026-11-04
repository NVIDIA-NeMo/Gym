# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light schema shared by IndicIFEval preparation and verification."""

from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


HARNESS_REVISION = "1bb5f1bc4936cb20b8544e185bfd7cbed8f31464"
IndicLanguage = Literal["bn", "gu", "hi", "kn", "mr", "ml", "ne", "or", "pa", "ta", "te", "ur"]
LANGUAGES: tuple[IndicLanguage, ...] = get_args(IndicLanguage)


class IndicIFEvalMetadata(BaseModel):
    """The aligned instruction lists and language required by the Trans harness."""

    model_config = ConfigDict(extra="allow")
    language: IndicLanguage
    prompt: str = Field(min_length=1)
    instruction_id_list: list[str] = Field(min_length=1)
    kwargs: list[dict[str, JsonValue]]
    grading_mode: Literal["binary", "fraction"] = "binary"

    @model_validator(mode="after")
    def aligned_instructions(self) -> "IndicIFEvalMetadata":
        if len(self.instruction_id_list) != len(self.kwargs):
            raise ValueError("instruction_id_list and kwargs must have the same length")
        return self
