# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Answer-dependent bash rewards using native OpenCode tool execution records."""

import json
import re
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from nemo_gym.utils import final_assistant_text


class OpenCodeToolRewardConfig(BaseModel, extra="forbid"):
    correct_answer_reward_threshold: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    correct_tool_call_bonuses: tuple[Annotated[float, Field(ge=0, allow_inf_nan=False)], ...] = Field(
        default=(0.0, 0.1, 0.1, 0.075, 0.05, 0.025, 0.01), min_length=1
    )
    tool_success_reward: float = Field(default=0.02, ge=0, allow_inf_nan=False)
    tool_success_reward_cap: float = Field(default=0.20, ge=0, allow_inf_nan=False)
    invalid_tool_call_penalty: float = Field(default=-0.05, le=0, allow_inf_nan=False)
    duplicate_tool_call_penalty: float = Field(default=-0.02, le=0, allow_inf_nan=False)

    correct_only: bool = Field(default=False, description="Apply auxiliary rewards only to correct answers.")
    correct_bash_call_penalties: tuple[Annotated[float, Field(le=0, allow_inf_nan=False)], ...] = Field(
        default=(0.0,),
        min_length=1,
        description="Penalty by total bash attempts on correct answers; the final entry caps the schedule.",
    )
    invalid_tool_call_names: tuple[str, ...] | None = Field(
        default=None,
        description="Tools eligible for failure penalties; None includes all tools.",
    )

    @field_validator("correct_tool_call_bonuses", "correct_bash_call_penalties")
    @classmethod
    def zero_calls_have_no_adjustment(cls, bonuses: tuple[float, ...]) -> tuple[float, ...]:
        if bonuses[0] != 0:
            raise ValueError("The zero-call adjustment must be zero")
        return bonuses


class OpenCodeToolReward(BaseModel):
    base_reward: float
    auxiliary_reward: float
    exploration_reward: float
    penalty_reward: float
    bash_success_count: int
    tool_call_count: int
    tool_error_count: int
    duplicate_tool_call_count: int
    bash_call_count: int = 0

    def for_answer(self, base_reward: float, config: OpenCodeToolRewardConfig) -> "OpenCodeToolReward":
        """Apply the verifier result to the recorded tool evidence."""
        auxiliary = 0.0 if config.correct_only else self.exploration_reward
        if base_reward >= config.correct_answer_reward_threshold:
            bonuses = config.correct_tool_call_bonuses
            penalties = config.correct_bash_call_penalties
            auxiliary = (
                bonuses[min(self.bash_success_count, len(bonuses) - 1)]
                + self.penalty_reward
                + penalties[min(self.bash_call_count, len(penalties) - 1)]
            )
        return self.model_copy(update={"base_reward": base_reward, "auxiliary_reward": auxiliary})


def _identity(item: dict) -> tuple[str, str]:
    try:
        arguments = json.loads(item["arguments"])
    except json.JSONDecodeError:
        # Native error records still let malformed calls receive the failure penalty.
        return item["name"], item["arguments"].strip()
    if not isinstance(arguments, dict):
        return item["name"], json.dumps(arguments, sort_keys=True)
    if item["name"] == "bash":
        # Descriptions do not change the command that is executed.
        arguments.pop("description", None)
        if isinstance(arguments.get("command"), str):
            arguments["command"] = arguments["command"].strip()
    return item["name"], json.dumps(arguments, sort_keys=True, separators=(",", ":"))


def score_opencode_tools(
    execution_records: dict[str, tuple[str, str, int | None]],
    output: list[dict],
    base_reward: float,
    config: OpenCodeToolRewardConfig,
) -> OpenCodeToolReward:
    """Score main-trajectory calls; child calls are not rewarded.

    A completed bash tool can still have a nonzero process exit code. Use the
    native state/exit metadata, never model-visible output text, to distinguish
    success from failure. Missing or unfinished execution records are errors so
    the caller can mask the rollout instead of inventing a reward.
    """
    calls = [item for item in output if item.get("type") == "function_call"]
    previous = None
    records = execution_records
    if len({item["call_id"] for item in calls}) != len(calls):
        raise ValueError("Repeated call ID in the live OpenCode trajectory")

    exploration = penalties = 0.0
    successes = errors = duplicates = 0
    for item in calls:
        identity = _identity(item)
        duplicate = identity == previous
        previous = identity
        record = records.get(item["call_id"])
        if record is None or record[0] != item["name"] or record[1] not in ("completed", "error"):
            raise ValueError(f"Missing or unfinished OpenCode tool record: {item['call_id']}")
        tool, status, exit_code = record
        if tool == "bash" and status == "completed" and not isinstance(exit_code, int):
            # Timeout/abort has a null exit code in OpenCode's shell tool.
            failed = True
        else:
            failed = status == "error" or (tool == "bash" and exit_code != 0)
        if failed:
            errors += 1
            if config.invalid_tool_call_names is None or tool in config.invalid_tool_call_names:
                penalties += config.invalid_tool_call_penalty
                exploration += config.invalid_tool_call_penalty
        elif duplicate:
            duplicates += 1
            penalties += config.duplicate_tool_call_penalty
            exploration += config.duplicate_tool_call_penalty
        elif tool == "bash":
            successes += 1
            exploration = min(exploration + config.tool_success_reward, config.tool_success_reward_cap)
    return OpenCodeToolReward(
        base_reward=base_reward,
        auxiliary_reward=0.0,
        exploration_reward=exploration,
        penalty_reward=penalties,
        bash_success_count=successes,
        tool_call_count=len(calls),
        tool_error_count=errors,
        duplicate_tool_call_count=duplicates,
        bash_call_count=sum(item["name"] == "bash" for item in calls),
    ).for_answer(base_reward, config)


class OpenCodeScoringConfig(BaseModel):
    """Scoring defaults shared by the agent and its environment without importing the agent runtime."""

    tool_rewards: OpenCodeToolRewardConfig | None = Field(default_factory=OpenCodeToolRewardConfig)
    strip_reasoning_for_grading: bool = True


class OpenCodeEvalScoringConfig(OpenCodeScoringConfig):
    tool_rewards: OpenCodeToolRewardConfig | None = None
    strip_reasoning_for_grading: bool = False


class EpisodeEvidence(BaseModel):
    """Validated native evidence passed explicitly across the responses self-call."""

    verification_error: str | None = None
    turns_used: int
    finished_naturally: bool
    tool_reward: OpenCodeToolReward | None


def final_answer(output: list[dict], *, strip_reasoning: bool = False) -> str:
    """Extract the scoring copy; optional training cleanup never mutates history."""
    text = final_assistant_text(output)
    if not strip_reasoning:
        return text
    text = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.DOTALL)
    text = re.split(r"<(?:think|thinking)>", text, maxsplit=1)[0]
    return re.split(r"</(?:think|thinking)>", text)[-1].strip()
