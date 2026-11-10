# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Eval-only guard for additional tool images beyond the serving image capacity."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import Body, FastAPI, HTTPException, Request
from pydantic import Field, PrivateAttr, model_validator

from nemo_gym.agents import FINAL_TURN_NOTICE, EvalTurnBudget
from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming, current_rollout_id
from nemo_gym.models import start_model_execution
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
from responses_api_models.vllm_model.image_guard import IMAGE_LIMIT_NOTICE as IMAGE_LIMIT_NOTICE
from responses_api_models.vllm_model.image_guard import limit_request_images


LOG = logging.getLogger(__name__)


@dataclass
class _TurnState:
    turns: int = 0
    grace_tool_calls: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    users: int = 0
    expiry: asyncio.TimerHandle | None = None


class VisualEvalModelConfig(VLLMModelConfig):
    max_images: int | None = Field(default=None, gt=0)
    turn_budget: EvalTurnBudget | None = None
    turn_state_idle_timeout_seconds: float = Field(
        default=3600,
        gt=0,
        allow_inf_nan=False,
        description="Expire idle counters; must exceed the agent episode timeout.",
    )
    turn_state_max_sessions: int = Field(
        default=10000, gt=0, description="Reject new sessions at capacity without evicting existing budgets."
    )

    @model_validator(mode="after")
    def validate_turn_budget(self) -> "VisualEvalModelConfig":
        if self.turn_budget is not None and (self.num_workers or 1) != 1:
            raise ValueError("Enforced eval turn budgets require one model-gateway worker")
        if self.turn_budget is not None and self.use_completions_api:
            raise ValueError("Enforced eval turn budgets require the chat completions path")
        return self


class VisualEvalModel(VLLMModel):
    config: VisualEvalModelConfig
    _turn_states: dict[tuple[str, str], _TurnState] = PrivateAttr(default_factory=dict)

    def setup_webserver(self) -> FastAPI:
        """Cancel expiry callbacks when the gateway's serving lifecycle ends."""
        app = super().setup_webserver()
        parent_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
            try:
                async with parent_lifespan(app) as state:
                    yield state
            finally:
                for state in self._turn_states.values():
                    if state.expiry is not None:
                        state.expiry.cancel()
                self._turn_states.clear()

        app.router.lifespan_context = lifespan
        return app

    def _expire_turn_state(self, key: tuple[str, str], state: _TurnState) -> None:
        if self._turn_states.get(key) is state and state.users == 0:
            del self._turn_states[key]

    def _acquire_turn_state(self, key: tuple[str, str]) -> _TurnState:
        state = self._turn_states.get(key)
        if state is None:
            # Never evict live counters to admit another episode: that would let
            # an overloaded gateway silently reset an existing turn budget.
            if len(self._turn_states) >= self.config.turn_state_max_sessions:
                raise HTTPException(503, "Evaluation turn-state capacity reached; retry after idle sessions expire")
            state = self._turn_states[key] = _TurnState()
        if state.expiry is not None:
            state.expiry.cancel()
            state.expiry = None
        # Include requests waiting for the lock, so expiry cannot split a queue
        # across two counters after an upstream completion or cancellation.
        state.users += 1
        return state

    def _release_turn_state(self, key: tuple[str, str], state: _TurnState) -> None:
        state.users -= 1
        if state.users == 0:
            state.expiry = asyncio.get_running_loop().call_later(
                self.config.turn_state_idle_timeout_seconds, self._expire_turn_state, key, state
            )

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        budget = self.config.turn_budget
        if budget is None:
            return await super().chat_completions(request, body)
        rollout_id = current_rollout_id()
        if not rollout_id or (body.n is not None and body.n != 1):
            raise HTTPException(400, "Enforced eval budget requires a rollout id and exactly one completion")
        session_id = request.headers.get("x-opencode-session-id", "").strip()
        if not session_id:
            raise HTTPException(400, "Enforced eval budget requires an OpenCode session id")
        key = (rollout_id, session_id)
        state = self._acquire_turn_state(key)
        try:
            async with state.lock:
                if state.turns >= budget.hard_limit:
                    execution = start_model_execution(request, upstream_attempted=False)
                    execution.update(response_source="local", local_response_reason="eval_turn_budget_exhausted")
                    LOG.warning(
                        "visual_turn_budget rollout=%s session=%s hard_stop=%d", rollout_id, session_id, state.turns
                    )
                    # Non-retryable error: no further upstream generation is possible,
                    # even if the client ignores a terminal response and asks again.
                    raise HTTPException(422, "Evaluation hard turn limit reached; this episode is stopped")
                # Retry the same turn after an upstream failure or cancellation.
                # The lock serializes attempts; commit only a valid, processed completion.
                turn = state.turns + 1
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
                grace_tool_calls = state.grace_tool_calls
                if allowance is not None:
                    response = response.model_copy(deep=True)
                    for choice in response.choices:
                        calls = choice.message.tool_calls or []
                        kept = calls[:allowance]
                        removed += len(calls) - len(kept)
                        choice.message.tool_calls = kept or None
                        # Do not let legacy function_call bypass the tools policy.
                        choice.message.function_call = None
                        grace_tool_calls += len(kept)
                        if not kept and choice.finish_reason in ("tool_calls", "function_call"):
                            choice.finish_reason = "stop"
                        if turn == budget.hard_limit:
                            choice.finish_reason = "stop"
                state.turns = turn
                state.grace_tool_calls = grace_tool_calls
                LOG.info(
                    "visual_turn_budget rollout=%s session=%s turn=%d hard_limit=%d tool_choice=%s "
                    "grace_tool_calls=%d removed_tool_calls=%d",
                    rollout_id,
                    session_id,
                    turn,
                    budget.hard_limit,
                    guarded_body.tool_choice,
                    state.grace_tool_calls,
                    removed,
                )
                return response
        finally:
            self._release_turn_state(key, state)

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

elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VisualEvalModel.run_webserver()
