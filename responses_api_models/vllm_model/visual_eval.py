# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eval-only guard for additional tool images beyond the serving image capacity."""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import Body, HTTPException, Request
from pydantic import Field, PrivateAttr, model_validator

from nemo_gym.base_responses_api_model import start_model_execution
from nemo_gym.openai_utils import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.rollout_correlation import current_rollout_id
from responses_api_agents.opencode_multimodal_agent.eval_budget import FINAL_TURN_NOTICE, EvalTurnBudget
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
from responses_api_models.vllm_model.image_guard import IMAGE_LIMIT_NOTICE as IMAGE_LIMIT_NOTICE
from responses_api_models.vllm_model.image_guard import limit_request_images


LOG = logging.getLogger(__name__)


@dataclass
class _TurnState:
    turns: int = 0
    grace_tool_calls: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class VisualEvalModelConfig(VLLMModelConfig):
    max_images: int | None = Field(default=33, gt=0)
    turn_budget: EvalTurnBudget | None = None

    @model_validator(mode="after")
    def validate_turn_budget(self) -> "VisualEvalModelConfig":
        if self.turn_budget is not None and (self.num_workers or 1) != 1:
            raise ValueError("Enforced eval turn budgets require one model-gateway worker")
        if self.turn_budget is not None and self.use_completions_api:
            raise ValueError("Enforced eval turn budgets require the chat completions path")
        return self


class VisualEvalModel(VLLMModel):
    config: VisualEvalModelConfig
    _turn_states: dict[str, _TurnState] = PrivateAttr(default_factory=dict)

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        budget = self.config.turn_budget
        if budget is None:
            return await super().chat_completions(request, body)
        rollout_id = current_rollout_id()
        if not rollout_id or (body.n is not None and body.n != 1):
            raise HTTPException(400, "Enforced eval budget requires a rollout id and exactly one completion")
        state = self._turn_states.setdefault(rollout_id, _TurnState())
        async with state.lock:
            if state.turns >= budget.hard_limit:
                execution = start_model_execution(request, upstream_attempted=False)
                execution.update(response_source="local", local_response_reason="eval_turn_budget_exhausted")
                LOG.warning("visual_turn_budget rollout=%s hard_stop=%d", rollout_id, state.turns)
                # Non-retryable error: no further upstream generation is possible,
                # even if the client ignores a terminal response and asks again.
                raise HTTPException(400, "Evaluation hard turn limit reached; this episode is stopped")
            state.turns += 1
            turn = state.turns
            allowance = budget.tool_allowance(turn, state.grace_tool_calls)
            updates = {}
            if turn >= budget.regular_turns:
                updates["messages"] = [*body.messages, {"role": "user", "content": FINAL_TURN_NOTICE}]
            if allowance is not None:
                updates["parallel_tool_calls"] = False
                updates["tool_choice"] = "none" if allowance == 0 else "auto"
            guarded_body = body.model_copy(update=updates)
            request.state.visual_eval_budget = {"allowance": allowance}
            response = await super().chat_completions(request, guarded_body)
            if len(response.choices) != 1:
                raise ValueError("Enforced eval budget requires exactly one returned completion")
            removed = 0
            if allowance is not None:
                response = response.model_copy(deep=True)
                for choice in response.choices:
                    calls = choice.message.tool_calls or []
                    kept = calls[:allowance]
                    removed += len(calls) - len(kept)
                    choice.message.tool_calls = kept or None
                    # Do not let legacy function_call bypass the tools policy.
                    choice.message.function_call = None
                    state.grace_tool_calls += len(kept)
                    if not kept and choice.finish_reason in ("tool_calls", "function_call"):
                        choice.finish_reason = "stop"
                    if turn == budget.hard_limit:
                        choice.finish_reason = "stop"
            LOG.info(
                "visual_turn_budget rollout=%s turn=%d hard_limit=%d tool_choice=%s "
                "grace_tool_calls=%d removed_tool_calls=%d",
                rollout_id,
                turn,
                budget.hard_limit,
                guarded_body.tool_choice,
                state.grace_tool_calls,
                removed,
            )
            return response

    def _preprocess_chat_completion_create_params(self, request: Request, body_dict: dict[str, Any]) -> dict[str, Any]:
        body_dict = super()._preprocess_chat_completion_create_params(request, body_dict)
        enforcement = getattr(request.state, "visual_eval_budget", None)
        if isinstance(enforcement, dict):
            # Apply after provider/config overrides so extra_body cannot bypass
            # the gate. Tools remain in the template to preserve prefix reuse.
            body_dict["n"] = 1
            if enforcement["allowance"] is not None:
                body_dict["tool_choice"] = "none" if enforcement["allowance"] == 0 else "auto"
                body_dict["parallel_tool_calls"] = False
        return limit_request_images(body_dict, max_images=self.config.max_images)


if __name__ == "__main__":
    VisualEvalModel.run_webserver()
