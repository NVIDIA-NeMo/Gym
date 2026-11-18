# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Question-bank fields shared by dataset validation and the semantic verifier."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str | int | None = Field(default=None, exclude_if=lambda value: value is None)
    uuid: str | int | None = Field(default=None, exclude_if=lambda value: value is None)
    question: str | None = None
    expected_answer: str | None = None
    metadata: dict[str, Any] | None = None
    answer_format: Literal["free", "boxed"] | None = None
