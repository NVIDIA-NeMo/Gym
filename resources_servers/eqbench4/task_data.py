# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Schema of a prepared EQ-Bench 4 row: one scenario, its prompts, and the published transcripts it is judged against."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# The policy response that the eqbench4 environment server sends to /verify carries the conversation under this key of
# its metadata, as a JSON string, as Gym's Simple Agent carries its outcome there. Gym re-judges a saved answer with a
# verify request rebuilt from the task row and the saved response alone, so the conversation must be in the response
# for a re-judge to grade it; the grading binding then covers the whole conversation, too.
CONVERSATION_METADATA_KEY = "eqbench4_conversation"

# How a conversation ended and, for one the policy stopped, why: the eqbench4 environment server sends them to
# /verify, and the resources server reads them from there, so both declare them from here.
ConversationStatus = Literal["completed", "persona_ended", "policy_failed"]
PolicyStopReason = Literal["context_limit", "output_budget", "content_filter", "empty_reply"]


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

    # The row's identity for resume and comparison: the scenario id, so it holds across re-preparations that change
    # the opponents or the row order.
    task_id: str
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

    @model_validator(mode="after")
    def _a_system_prompt_per_policy_turn(self) -> "TaskData":
        check_system_prompts_cover_turns(self.evaluated_system_prompts, self.num_turns)
        return self


def check_system_prompts_cover_turns(evaluated_system_prompts: list[str], num_turns: int) -> None:
    """Every policy turn has its system prompt; checked before any model call, which a short list would waste."""
    if len(evaluated_system_prompts) < num_turns:
        raise ValueError(
            f"evaluated_system_prompts has {len(evaluated_system_prompts)} prompt(s) for {num_turns} policy turn(s); "
            "it must cover every policy turn"
        )
