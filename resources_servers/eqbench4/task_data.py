# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema of a prepared EQ-Bench 4 row: one scenario, its prompts, and the published transcripts it is judged against."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TranscriptMessage(BaseModel):
    """One message of a conversation; ``user`` is the simulated persona and ``assistant`` the evaluated model."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["assistant", "user"]
    content: str
    # The persona's full reply, including its private <internal> monologue (policy conversations).
    raw_response: str | None = None
    # The persona's private monologue, already extracted (published transcripts).
    internal: str | None = None


class SlottedPrompt(BaseModel):
    """A judge prompt split around named slots: literal, slot, literal, ..., literal."""

    model_config = ConfigDict(extra="forbid")

    literals: list[str] = Field(min_length=1)
    slots: list[str]


class Opponent(BaseModel):
    """A published model's conversation on the same scenario."""

    model_config = ConfigDict(extra="forbid")

    model: str
    # Position of this comparison in the whole benchmark; the judge panel rotates by it.
    comparison_index: int = Field(ge=0)
    transcript: list[TranscriptMessage] = Field(min_length=1)
    final_state: str


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")

    scenario_id: str
    scenario_index: int = Field(ge=0)
    source_type: str = Field(default="generated", json_schema_extra={"consumed_by": ["metrics"]})
    num_turns: int = Field(ge=1, json_schema_extra={"consumed_by": ["agent"]})
    evaluated_system_prompts: list[str] = Field(min_length=1, json_schema_extra={"consumed_by": ["agent", "prompt"]})
    opening_message: str = Field(json_schema_extra={"consumed_by": ["prompt"]})
    persona_system_prompt: str = Field(json_schema_extra={"consumed_by": ["agent"]})
    pairwise_prompt: SlottedPrompt = Field(json_schema_extra={"consumed_by": ["verify"]})
    pairwise_dimensions: dict[str, float] = Field(min_length=1, json_schema_extra={"consumed_by": ["verify"]})
    trait_prompt: SlottedPrompt = Field(json_schema_extra={"consumed_by": ["verify"]})
    trait_assistant_label: str = Field(json_schema_extra={"consumed_by": ["verify"]})
    opponents: list[Opponent] = Field(min_length=1, json_schema_extra={"consumed_by": ["verify", "metrics"]})
