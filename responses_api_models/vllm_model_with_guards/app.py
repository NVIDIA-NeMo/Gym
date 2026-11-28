# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM model server with harness guards for models that keep calling tools after being told to stop."""

import json
import logging
from typing import Any, Dict, List, Optional

from fastapi import Body, Request
from pydantic import Field, model_validator

from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.models import start_model_execution, trailing_harness_note
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


LOG = logging.getLogger("nemo_gym.vllm_model_with_guards")


class VLLMModelWithGuardsConfig(VLLMModelConfig):
    """`VLLMModelConfig` plus the harness guards. All guards are off by default.

    A model that cannot answer in text keeps calling tools after the harness tells it to stop (the SFT checkpoints
    evaluated under opencode ignore its "maximum steps reached, answer with text only" notice and loop on no-op
    commands for hours), so the harness's own step limit never ends the run. These guards end it from the server
    instead: the request is answered with a final text message and no tool call, which a tool-calling harness treats
    as the end of the conversation, and the engine is not called.
    """

    # From this many assistant turns on, every request is sent with tool_choice "none", so the model can only answer
    # in text and the run ends. Asking such a model to answer in text does not work (replays of 18 late requests: 0 of
    # 18 ended in text when asked in the prompt, 18 of 18 with tool_choice "none").
    guard_text_only_after_assistant_turns: Optional[int] = Field(default=None, ge=1)
    # Stop when the conversation already holds this many assistant turns.
    guard_max_assistant_turns: Optional[int] = Field(default=None, ge=1)
    # Stop when the last this-many assistant turns made the identical tool call (same tool, same arguments).
    guard_identical_tool_calls: Optional[int] = Field(default=None, ge=2)

    @model_validator(mode="after")
    def _validate_text_only_guard(self) -> "VLLMModelWithGuardsConfig":
        if self.guard_text_only_after_assistant_turns is None:
            return self
        if self.use_completions_api:
            # /v1/completions renders the tools into the prompt and has no tool_choice, so "none" would be dropped.
            raise ValueError("guard_text_only_after_assistant_turns requires use_completions_api=false")
        pin = (self.sampling_overrides or {}).get("tool_choice", "none")
        if pin != "none":
            # A sampling pin is applied after the guard sets tool_choice, so it would undo the guard.
            raise ValueError(
                "guard_text_only_after_assistant_turns requires no tool_choice sampling pin, or a pin of 'none'"
            )
        return self


def _tool_call_signature(message: Dict[str, Any]) -> Optional[tuple]:
    """The tool calls of one assistant message as a comparable value, or None when it made none."""

    calls = message.get("tool_calls") or []
    if not calls:
        return None
    parts = []
    for call in calls:
        # A function call carries its name and JSON arguments; a custom tool call its name and free-form input.
        # Reading only "function" would give every custom call the same (None, null) signature, and a conversation
        # that is making progress with differing custom calls would be stopped as a loop.
        custom = None if call.get("function") else call.get("custom")
        if custom is not None:
            # Free-form input is not JSON: "1" and " 1 " are different inputs, so the raw string is the key.
            parts.append((call.get("type"), custom.get("name"), custom.get("input")))
            continue
        payload = call.get("function") or {}
        arguments = payload.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                pass
        parts.append((call.get("type"), payload.get("name"), json.dumps(arguments, sort_keys=True, default=str)))
    return tuple(parts)


class VLLMModelWithGuards(VLLMModel):
    """`VLLMModel` whose chat completions are cut short by the configured guards.

    Both `/v1/chat/completions` and the converted `/v1/responses` path call `chat_completions`, so the guards cover
    them. A model server with `is_responses_native: true` does not, because it never calls `chat_completions`.
    """

    config: VLLMModelWithGuardsConfig

    def model_post_init(self, context):
        if self.config.is_responses_native:
            raise ValueError("vllm_model_with_guards guards chat completions and does not support is_responses_native")
        super().model_post_init(context)

    def _guard_stop_reason(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """Why the guards end this conversation now, or None to let the request through."""

        turn_limit, run_length = self.config.guard_max_assistant_turns, self.config.guard_identical_tool_calls
        if turn_limit is None and run_length is None:
            return None
        assistant_turns = [m for m in messages if m.get("role") == "assistant"]
        if turn_limit is not None and len(assistant_turns) >= turn_limit:
            return f"the conversation reached {turn_limit} assistant turns"
        if run_length is not None and len(assistant_turns) >= run_length:
            signatures = [_tool_call_signature(m) for m in assistant_turns[-run_length:]]
            if signatures[0] is not None and all(signature == signatures[0] for signature in signatures):
                return f"the last {run_length} assistant turns made the same tool call"
        return None

    def _create_guard_completion(self, reason: str) -> NeMoGymChatCompletion:
        completion = self._create_empty_chat_completion()
        completion.choices[
            0
        ].message.content = (
            f"Stopping here: {reason}. The work above stands as the final result and nothing further will be run."
        )
        return completion

    async def chat_completions(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        guarded = any(
            limit is not None
            for limit in (
                self.config.guard_text_only_after_assistant_turns,
                self.config.guard_max_assistant_turns,
                self.config.guard_identical_tool_calls,
            )
        )
        # A compaction summary is not a turn of the task: it keeps the parent's bounded, checked summary.
        if not guarded or self._is_compaction_summary(request):
            return await super().chat_completions(request, body)

        messages = body.model_dump(exclude_unset=True).get("messages") or []
        # The harness's own step-limit notice is not a model turn: it neither counts as one nor breaks a streak.
        # Only the guards' view drops it; the body forwarded upstream is unchanged.
        if trailing_harness_note(messages) is not None:
            messages = messages[:-1]
        reason = self._guard_stop_reason(messages)
        if reason is not None:
            LOG.warning("`%s` ended a conversation: %s", self.config.name, reason)
            execution = start_model_execution(request, upstream_attempted=False)
            execution.update(response_source="local", local_response_reason="harness_guard")
            return self._create_guard_completion(reason)

        text_only_after = self.config.guard_text_only_after_assistant_turns
        if text_only_after is not None and sum(m.get("role") == "assistant" for m in messages) >= text_only_after:
            body = body.model_copy(update={"tool_choice": "none"})
        return await super().chat_completions(request, body)


if __name__ == "__main__":
    VLLMModelWithGuards.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = VLLMModelWithGuards.run_webserver()  # noqa: F401
