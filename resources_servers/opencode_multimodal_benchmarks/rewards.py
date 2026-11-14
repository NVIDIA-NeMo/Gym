# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Answer-dependent bash rewards using native OpenCode tool execution records."""

import json
import re
from typing import Annotated, Any, ClassVar

from pydantic import BaseModel, Field, field_validator, model_validator

from nemo_gym.server_runtime import BaseVerifyResponse
from nemo_gym.trajectory import AgentObservationBundle, TrajectoryRecord


def final_assistant_text(output: list[dict[str, Any]], *, block_separator: str = "") -> str:
    """Return the last assistant answer, stopping at an unfinished reasoning or tool turn."""
    for item in reversed(output):
        if item.get("type") in {"reasoning", "function_call"}:
            break
        if item.get("type") == "message" and item.get("role") == "assistant":
            return block_separator.join(part["text"] for part in item["content"] if part["type"] == "output_text")
    return ""


class OpenCodeToolRewardConfig(BaseModel, extra="forbid"):
    shell_tool_name: str = Field(
        default="bash",
        min_length=1,
        description="Shell tool used for call counts, duplicate normalization and bonuses.",
    )
    correct_answer_reward_threshold: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    correct_tool_call_bonuses: tuple[Annotated[float, Field(ge=0, allow_inf_nan=False)], ...] = Field(
        default=(0.0,), min_length=1
    )
    tool_success_reward: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    tool_success_reward_cap: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    invalid_tool_call_penalty: float = Field(default=-0.05, le=0, allow_inf_nan=False)
    duplicate_tool_call_penalty: float = Field(default=-0.02, le=0, allow_inf_nan=False)

    correct_only: bool = Field(default=True, description="Apply auxiliary rewards only to correct answers.")
    correct_bash_call_penalties: tuple[Annotated[float, Field(le=0, allow_inf_nan=False)], ...] = Field(
        default=(0.0, 0.0, 0.0, -0.025, -0.05, -0.075, -0.10),
        min_length=1,
        description="Penalty by total shell_tool_name attempts on correct answers; the final entry caps the schedule.",
    )
    invalid_tool_call_names: tuple[str, ...] | None = Field(
        default=("bash",),
        description="Tools eligible for failure penalties; defaults to shell_tool_name. None includes all tools.",
    )

    @model_validator(mode="after")
    def default_failure_filter_to_shell(self) -> "OpenCodeToolRewardConfig":
        if "invalid_tool_call_names" not in self.model_fields_set:
            self.invalid_tool_call_names = (self.shell_tool_name,)
        return self

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


def _identity(item: dict, config: OpenCodeToolRewardConfig) -> tuple[str, str]:
    try:
        arguments = json.loads(item["arguments"])
    except json.JSONDecodeError:
        # Native error records still let malformed calls receive the failure penalty.
        return item["name"], item["arguments"].strip()
    if not isinstance(arguments, dict):
        return item["name"], json.dumps(arguments, sort_keys=True)
    if item["name"] == config.shell_tool_name:
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
        raise ValueError("Repeated call ID in the live tool trajectory")

    exploration = penalties = 0.0
    successes = errors = duplicates = 0
    for item in calls:
        identity = _identity(item, config)
        duplicate = identity == previous
        previous = identity
        record = records.get(item["call_id"])
        if record is None or record[0] != item["name"] or record[1] not in ("completed", "error"):
            raise ValueError(f"Missing or unfinished tool record: {item['call_id']}")
        tool, status, exit_code = record
        if tool == config.shell_tool_name and status == "completed" and not isinstance(exit_code, int):
            # Timeout/abort has a null exit code in OpenCode's shell tool.
            failed = True
        else:
            failed = status == "error" or (tool == config.shell_tool_name and exit_code != 0)
        if failed:
            errors += 1
            if config.invalid_tool_call_names is None or tool in config.invalid_tool_call_names:
                penalties += config.invalid_tool_call_penalty
                exploration += config.invalid_tool_call_penalty
        elif duplicate:
            duplicates += 1
            penalties += config.duplicate_tool_call_penalty
            exploration += config.duplicate_tool_call_penalty
        elif tool == config.shell_tool_name:
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
        bash_call_count=sum(item["name"] == config.shell_tool_name for item in calls),
    ).for_answer(base_reward, config)


def final_answer(output: list[dict], *, strip_reasoning: bool = False, block_separator: str = "") -> str:
    """Extract the scoring copy; optional training cleanup never mutates history."""
    text = final_assistant_text(output, block_separator=block_separator)
    if not strip_reasoning:
        return text
    text = re.sub(r"<(think|thinking)>.*?</\1>", "", text, flags=re.DOTALL)
    text = re.split(r"<(?:think|thinking)>", text, maxsplit=1)[0]
    return re.split(r"</(?:think|thinking)>", text)[-1].strip()


def trajectory_tool_reward(
    response, trajectory, base_reward: float, config: OpenCodeToolRewardConfig, observations=None
):
    """Use the shared harness's canonical execution outcomes, including shell exit failures."""
    output = [item.model_dump(mode="json") for item in response.output]
    wanted = {item["call_id"] for item in output if item.get("type") == "function_call"}
    if not wanted:
        return score_opencode_tools({}, output, base_reward, config)
    from nemo_gym.trajectory import AgentInvocation, ToolCallObservation

    invocations = trajectory.invocations if trajectory is not None else []
    tools = trajectory.tool_calls if trajectory is not None else []
    if not tools and observations is not None:
        invocations = [item for item in observations.records if isinstance(item, AgentInvocation)]
        tools = [item for item in observations.records if isinstance(item, ToolCallObservation)]
    roots = {inv.invocation_id for inv in invocations if inv.parent_invocation_id is None}
    records = {}
    for tool in tools:
        if tool.invocation_id not in roots:
            continue
        call_id = tool.tool_call_id if tool.tool_call_id in wanted else tool.native_tool_call_id
        if call_id not in wanted:
            continue
        if call_id in records:
            raise ValueError(f"Ambiguous tool evidence for {call_id}")
        if tool.status in {"incomplete", "unknown"}:
            raise ValueError(f"Unfinished tool evidence for {call_id}")
        success = tool.status == "completed"
        records[call_id] = (tool.tool_name, "completed" if success else "error", 0 if success else None)
    return score_opencode_tools(records, output, base_reward, config)


class ToolRewardScoringConfig(BaseModel):
    """Opt-in verifier-side shaping for existing training tasks."""

    tool_rewards: OpenCodeToolRewardConfig | None = Field(default_factory=OpenCodeToolRewardConfig)
    strip_reasoning_for_grading: bool = True


class ToolRewardEvidence(BaseModel):
    """Execution evidence accepted only by the opt-in reward verifiers."""

    ng_trajectory: TrajectoryRecord | None = Field(default=None, exclude_if=lambda value: value is None)
    ng_agent_observations: AgentObservationBundle | None = Field(default=None, exclude_if=lambda value: value is None)


class ToolRewardVerifyResponse(BaseVerifyResponse, extra="allow"):
    pass


class ToolRewardVerifier:
    """Apply the existing training schedule at the verifier boundary, preserving capture."""

    _answer_block_separator: ClassVar[str] = ""

    async def verify(self, body):
        result = await super().verify(self._grading_body(body))
        return self._apply_tool_rewards(body, result)

    def _grading_body(self, body):
        original = body.response
        scoring = original.model_dump(mode="json")
        scoring["output"] = [
            {
                "id": "final",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "annotations": [],
                        "text": final_answer(
                            scoring["output"],
                            strip_reasoning=self.config.strip_reasoning_for_grading,
                            block_separator=self._answer_block_separator,
                        ),
                    }
                ],
            }
        ]
        return body.model_copy(update={"response": type(original).model_validate(scoring)})

    def _apply_tool_rewards(self, body, result):
        """Return the final Resources-owned grade before the shared observer records it."""
        original = body.response
        if result.mask_sample:
            return ToolRewardVerifyResponse.model_validate(result.model_dump(mode="json") | {"response": original})
        base = float(result.reward)
        tools = (
            trajectory_tool_reward(
                original, body.ng_trajectory, base, self.config.tool_rewards, body.ng_agent_observations
            )
            if self.config.tool_rewards is not None
            else None
        )
        bonus = tools.auxiliary_reward if tools is not None else 0.0
        source = result.model_dump(mode="json")
        source = {key: source[key] for key in ("response", "reward", "reward_components") if key in source}
        result = ToolRewardVerifyResponse.model_validate(
            result.model_dump(mode="json")
            | {
                "response": original,
                "reward": base + bonus,
                "opencode_tool_reward": tools,
                "reward_components": (source.get("reward_components") or {"verifier": base})
                | {"opencode_tool_reward": bonus},
            }
        )
        return result
