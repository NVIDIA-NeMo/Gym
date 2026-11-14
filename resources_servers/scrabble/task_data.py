# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the scrabble server.

Gymnasium-family environment: reward comes from /step, driven by gymnasium_agent. Each row pins one
position (board and rack) and its best achievable score, so every rollout of a row solves the same puzzle.
"""

from pydantic import BaseModel, ConfigDict, Field


class BestMove(BaseModel):
    """One highest-scoring move, kept for inspection; the reward uses only ``best_score``."""

    model_config = ConfigDict(extra="allow")

    word: str
    row: int = Field(ge=1, le=15)
    col: str = Field(pattern=r"^[A-O]$")
    direction: str = Field(pattern=r"^(across|down)$")
    score: int = Field(ge=1)


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    position_id: str = Field(
        description="Stable identifier of the position, for example 'train-000042'.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    board: list[str] = Field(
        min_length=15,
        max_length=15,
        description="15 rows of 15 characters: '.' empty, A-Z a tile, a-z a blank played as that letter.",
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    rack: str = Field(
        pattern=r"^[A-Z?]{1,7}$",
        description="Tiles the model may play; '?' is a blank.",
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    best_score: int = Field(
        ge=1,
        description="Highest score of any legal move, found by exhaustive search. Normalizes the reward.",
        json_schema_extra={"consumed_by": ["verify", "metrics"]},
    )
    best_move: BestMove | None = Field(
        default=None,
        description="One move that achieves best_score.",
        json_schema_extra={"consumed_by": ["provenance"]},
    )
    max_attempts: int = Field(
        default=3,
        ge=1,
        description="Moves the model may submit; an illegal move uses one and the episode ends after the last.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
