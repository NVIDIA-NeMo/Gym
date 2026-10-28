# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Explicit evaluation budget shared by the agent and its model gateway."""

from pydantic import BaseModel, Field, model_validator


FINAL_TURN_NOTICE = (
    "This is your final turn. You cannot use any more tools. "
    "Stop using tools and provide your final answer now using the evidence already available."
)


class EvalTurnBudget(BaseModel):
    regular_turns: int = Field(default=55, ge=1)
    grace_turns: int = Field(default=10, ge=1)
    grace_tool_turns: int = Field(default=2, ge=0)
    grace_tool_calls: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def reserve_text_only_turn(self) -> "EvalTurnBudget":
        if self.grace_tool_turns >= self.grace_turns:
            raise ValueError("The grace period must end with at least one text-only turn")
        return self

    @property
    def hard_limit(self) -> int:
        return self.regular_turns + self.grace_turns

    def tool_allowance(self, turn: int, used_grace_calls: int) -> int | None:
        """None preserves normal parallel tools; grace permits at most one per turn."""
        if turn <= self.regular_turns:
            return None
        if turn <= self.regular_turns + self.grace_tool_turns:
            return min(1, max(0, self.grace_tool_calls - used_grace_calls))
        return 0
