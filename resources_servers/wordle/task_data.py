# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    custom_target: str = Field(
        description="Secret word for the game. reset() rejects rows without a valid one.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    word_length: int = Field(
        default=5,
        description="Word length. Only 5-letter words are in the word lists.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    max_turns: int = Field(
        default=6,
        description="Number of guesses allowed.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
