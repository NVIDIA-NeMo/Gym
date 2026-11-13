# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task-data schema for the chess server. reset() validates rows against it; the rules engine checks the FEN."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    """One game: where it starts, which side the model plays, and who it plays against."""

    model_config = ConfigDict(extra="allow")

    fen: Optional[str] = Field(
        default=None,
        description=(
            "Starting position as a full six-field FEN; omit it for the standard initial position. Must be a legal "
            "position whose game is not over."
        ),
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    agent_color: Literal["white", "black"] = Field(
        default="white",
        description="The side the model plays. When the other side is to move in the FEN, the opponent moves first.",
        json_schema_extra={"consumed_by": ["verify", "prompt"]},
    )
    opponent: Literal["random", "greedy", "lookahead"] = Field(
        default="random",
        description=(
            "Built-in opponent: random legal moves; greedy (mate in one, else the most valuable capture); or "
            "lookahead (a two-ply material search)."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
    seed: int = Field(
        default=0,
        description="Seed for the opponent's choices, so the same moves by the model always meet the same replies.",
        json_schema_extra={"consumed_by": ["verify"]},
    )
    max_consecutive_invalid_actions: int = Field(
        default=3,
        ge=0,
        description=(
            "Invalid actions (malformed or illegal moves, or turns without a move) tolerated in a row; one more "
            "forfeits the game. A legal move resets the count."
        ),
        json_schema_extra={"consumed_by": ["verify"]},
    )
