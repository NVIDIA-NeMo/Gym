# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The settings of a simulated turn (tool simulator, judges, scoring) and the bodies of the routes that serve one.

Both servers that run these turns use them: the ``agentic_turn_simulation`` resources server, and the ``agent_arena``
environment server, which reads its settings from that resources server's config.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyRequest, BaseVerifyResponse
from resources_servers.agentic_turn_simulation.harnesses.base import ToolCallProtocol
from resources_servers.agentic_turn_simulation.obstacles import ObstacleInjectionConfig


# How a tool call was answered: in code, by the simulator LLM, not at all (the simulator failed), or with an injected
# obstacle.
AnswerSource = Literal["exact", "llm", "failed", "injected"]


class JudgeSpec(BaseModel):
    model_server: ModelServerRef
    # Merged over judge_chat_completions_create_params for this judge only. A null value removes
    # the field, e.g. {temperature: null, reasoning_effort: null} for a model that rejects them.
    overrides: dict[str, Any] = Field(default_factory=dict)


class AgenticTurnSimulationSettings(BaseModel):
    """How a turn's tool calls are simulated and how the turn is scored."""

    # --- tool simulator (chat completions; reasoning on)
    tool_simulator_model_server: ModelServerRef
    tool_simulator_chat_completions_create_params: NeMoGymChatCompletionCreateParamsNonStreaming
    tool_simulator_max_attempts: int = 4
    tool_simulator_max_tokens_cap: int = 49_152
    tool_simulator_max_concurrency: int | None = 64

    # --- judge (chat completions with a strict JSON schema). One judge scores alone; two or more
    # are combined by majority vote.
    judges: list[JudgeSpec]
    judge_chat_completions_create_params: NeMoGymChatCompletionCreateParamsNonStreaming
    judge_max_attempts: int = 6
    judge_max_tokens_cap: int = 65_536
    judge_max_concurrency: int | None = 32

    # Failed attempts back off 1, 2, 4, ... seconds, up to this.
    retry_backoff_max_seconds: float = 20.0

    # policy: judge the policy's turn. reference_turn: judge the original model's recorded turn from
    # verifier_metadata (the baseline the policy is compared with).
    score_source: Literal["policy", "reference_turn"] = "policy"
    # Agent stop reasons that mask the rollout. The default masks the wall-clock cap, which tracks serving speed. A
    # context overflow before the first step (the row does not fit the policy's context) is masked whatever this lists.
    masked_stop_reasons: list[str] = Field(default_factory=lambda: ["time_cap"])
    # Agent stop reasons that leave the turn unfinished by the policy's own doing: scored 0 as code_gate
    # "unfinished_turn", without a judge, which passes many cut-off turns (it credits the progress made before the cut).
    # empty_output (a final reply with neither a message nor a call) counts only when the reply is the policy's: one
    # with no items at all, or one the model server marked unfinished (FAILED_RESPONSE_STATUSES, or an error), is a
    # failed request's and is judged.
    unfinished_stop_reasons: list[str] = Field(
        default_factory=lambda: ["max_steps", "max_output_tokens", "context_overflow", "empty_output"]
    )
    # Score 0 as code_gate "foreign_script", after the other gates, a policy turn whose final reply holds at least this
    # many letters of scripts its input never shows (`foreign_script_letters`: garbled text such as Chinese characters
    # in a Persian sentence). None: off; /verify reports the count either way.
    foreign_script_gate_min_letters: int | None = Field(default=None, ge=1)
    # Show the judge and the simulator the row's own system prompt (<agent_system_prompt>). None: each harness's
    # default (all but the sandbox do); true / false: every harness.
    show_agent_system_prompt: bool | None = None
    # The judge's paragraph about the <agent_system_prompt> block (a prompts/ file), when the block is shown.
    judge_system_prompt_note_file: str = "judge-system-prompt-note.md"
    # Recoverable failures injected into the policy's web calls (off by default; obstacles.py).
    obstacle_injection: ObstacleInjectionConfig = Field(default_factory=ObstacleInjectionConfig)
    # verifier_metadata keys (dotted: nested) that the metrics break turn_score down by, as turn_score_by_<key>.
    metric_breakdown_keys: list[str] = Field(default_factory=lambda: ["harness", "record_kind"])


class AgenticTurnSimulationRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: dict[str, Any] | None = None


class AgenticTurnSimulationVerifyRequest(AgenticTurnSimulationRunRequest, BaseVerifyRequest):
    pass


class HarnessResponse(BaseModel):
    """What the agent needs to know about the seeded row's harness."""

    # A call to one of these ends the turn unexecuted (the agent's config is the fallback).
    stop_tools: list[str]
    protocol: ToolCallProtocol


class SimulateToolCallRequest(BaseModel):
    """One tool call to answer, with the turn so far (the agent's running output, which includes
    this call and the outputs already produced in this step)."""

    call_id: str
    name: str
    arguments: str
    turn: list[dict[str, Any]] = Field(default_factory=list)


class SimulateToolCallResponse(BaseModel):
    output: str
    source: AnswerSource


class AgenticTurnSimulationVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    score_source: str = "policy"
    stop_reason: str | None = None
    steps: int | None = None
    # How the policy's turn ended, without harness knowledge: the stop tool whose call ended it (its stop_reason names
    # one of the row's stop tools), else None; and how many of its calls ran (got a result), stop tools excluded.
    ended_on_stop_tool: str | None = None
    executed_calls: int = 0
    # Letters in the policy turn's final reply of scripts its input never shows (foreign_script.py); None when the
    # recorded turn is scored.
    foreign_script_letters: int | None = None
    code_gate: str | None = None
    addresses_concern: str | None = None
    correct: str | None = None
    correct_ladder: str | None = None
    blocked_by: str | None = None
    request_questionable: str | None = None
    discard: str | None = None
    discard_reason: str | None = None
    judge_verdicts: list[dict[str, Any] | None] = Field(default_factory=list)
    # Failed calls of the harness's shell tool (whatever it is named); None: the row's harness has no shell tool
    # whose results this can be read from.
    shell_failures: int | None = None
    simulated_calls: dict[str, int] = Field(default_factory=dict)
    categories: list[str] = Field(default_factory=list)
    # The harness the row ran under (see harnesses).
    harness: str | None = None
