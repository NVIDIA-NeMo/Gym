# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small semantic completion manifest; training tokens remain in ordinary capture."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.openai_utils import NeMoGymResponseOutputItem


class SelectedAction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    response_id: str = Field(min_length=1)
    finish_reason: str | None
    last_output_item: NeMoGymResponseOutputItem | None


class LogicalCCResult(BaseModel):
    """Selected responses and outcome; physical training rows belong to RL."""

    model_config = ConfigDict(extra="forbid", strict=True)

    logical_rollout_id: str = Field(min_length=1)
    selected_actions: list[SelectedAction] = Field(min_length=1)
    outcome: Literal["completed", "max_steps", "max_output_tokens", "execution_failure", "incomplete_output"]

    @model_validator(mode="after")
    def validate_selection(self) -> "LogicalCCResult":
        ids = [action.response_id for action in self.selected_actions]
        if len(set(ids)) != len(ids):
            raise ValueError("Selected response IDs must be unique")
        return self
