# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arena Proxy's request contract on Gym's vLLM model server."""

import json
from typing import Any

from fastapi import HTTPException, Request
from pydantic import Field, PositiveInt, model_validator

from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.models import ChatCompletionsConverter
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


_MAX_OUTPUT_TOKENS = 120000
_CONTEXT_RESERVE = 64
_BUDGET_FIELDS = ("max_tokens", "max_completion_tokens")


def _validate_budget(value: object, field: str) -> int:
    if type(value) is not int or not 0 < value <= _MAX_OUTPUT_TOKENS:
        raise ValueError(f"Arena Proxy requires {field} to be an integer from 1 to {_MAX_OUTPUT_TOKENS}")
    return value


class ArenaProxyGenerationConfig(VLLMModelConfig):
    """Reference chat generation with explicit rejection of incompatible profiles."""

    context_length: PositiveInt
    forward_request_chat_template_kwargs: bool = True
    chat_template_kwargs: dict[str, Any] = Field(default_factory=lambda: {"enable_thinking": True})

    @model_validator(mode="after")
    def validate_reference_protocol(self) -> "ArenaProxyGenerationConfig":
        required = {
            "is_responses_native": False,
            "use_completions_api": False,
            "dynamo_frontend": False,
            "uses_reasoning_parser": True,
            "preserve_reasoning_in_assistant_content": True,
            "preserve_thinking_blocks": False,
            "forward_request_chat_template_kwargs": True,
            "continue_final_assistant_message": False,
            "refit_output_budget_to_window": False,
            "refit_output_budget_to_prompt": False,
            "drop_unrepresentable_request_fields": False,
        }
        for name, value in required.items():
            if getattr(self, name) is not value:
                raise ValueError(f"Arena Proxy's reference protocol requires {name}={value}")
        if self.chat_template_kwargs.get("enable_thinking") is not True:
            raise ValueError("Arena Proxy's reference protocol requires chat_template_kwargs.enable_thinking=true")
        if "chat_template_kwargs" in (self.extra_body or {}):
            raise ValueError("Set Arena Proxy chat_template_kwargs directly, not inside extra_body")
        for source in ("extra_body", "sampling_overrides"):
            for name, value in (getattr(self, source) or {}).items():
                if name in (*_BUDGET_FIELDS, "max_output_tokens"):
                    _validate_budget(value, f"{source}.{name}")
        return self


class ArenaProxyConverter(ChatCompletionsConverter):
    """Retain reasoning items while exposing only the benchmark's final answer."""

    def _extract_reasoning_from_content(self, content: str) -> tuple[list[str], str]:
        reasoning, _ = super()._extract_reasoning_from_content(content)
        visible = content.rpartition("</think>")[2] if "</think>" in content else content
        return reasoning, "" if "<think>" in visible else visible


class ArenaProxyGenerationModel(VLLMModel):
    """Preserve Gym's inference lifecycle with the benchmark's measured token budget."""

    config: ArenaProxyGenerationConfig

    def get_converter(self) -> ArenaProxyConverter:
        """Apply the benchmark's answer extraction within Gym's response conversion."""
        return ArenaProxyConverter.model_validate(super().get_converter().model_dump())

    async def _prepare_chat_request(
        self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming
    ) -> dict[str, Any]:
        try:
            extra = json.loads((body.metadata or {}).get("extra_body") or "{}")
        except (TypeError, ValueError) as error:
            raise HTTPException(422, "Arena Proxy requires metadata.extra_body to encode a JSON object") from error
        if not isinstance(extra, dict):
            raise HTTPException(422, "Arena Proxy requires metadata.extra_body to encode a JSON object")
        if "chat_template_kwargs" in extra:
            raise HTTPException(422, "Use metadata.chat_template_kwargs, not metadata.extra_body.chat_template_kwargs")
        # Keep Gym's configured/request/metadata precedence and every supported
        # deployment field. Only text normalization and the reference budget differ.
        outbound = await super()._prepare_chat_request(request, body)
        if outbound.get("tools"):
            raise HTTPException(422, "Arena Proxy generation accepts text conversations without tools")
        if outbound.get("n", 1) != 1:
            raise HTTPException(422, "Arena Proxy requires one completion per task (n=1)")
        if (outbound.get("chat_template_kwargs") or {}).get("enable_thinking") is not True:
            raise HTTPException(422, "Arena Proxy requires chat_template_kwargs.enable_thinking=true")
        if outbound.get("continue_final_message") or outbound.get("add_generation_prompt", True) is not True:
            raise HTTPException(422, "Arena Proxy requires a new assistant turn, not assistant-prefix continuation")
        if outbound.get("truncate_prompt_tokens") is not None:
            raise HTTPException(
                422, "Arena Proxy requires the full conversation; truncate_prompt_tokens is unsupported"
            )
        budget_fields = [name for name in _BUDGET_FIELDS if name in outbound]
        if len(budget_fields) > 1:
            raise HTTPException(422, "Arena Proxy accepts one output budget: max_tokens or max_completion_tokens")
        budget_field = next(iter(budget_fields), "max_tokens")
        try:
            requested = _validate_budget(outbound.get(budget_field, _MAX_OUTPUT_TOKENS), budget_field)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

        messages = []
        for message in outbound["messages"]:
            content = message.get("content")
            if isinstance(content, list):
                if any(
                    part.get("type") != "text" or not isinstance(part.get("text"), str) or set(part) - {"type", "text"}
                    for part in content
                ):
                    raise HTTPException(422, "Arena Proxy requires plain text parts without additional content fields")
                content = "".join(part["text"] for part in content)
            if content is None and message["role"] == "assistant":
                content = ""
            if message["role"] not in ("system", "user", "assistant") or not isinstance(content, str):
                raise HTTPException(422, "Arena Proxy generation requires system/user/assistant text messages")
            messages.append({**message, "content": content})
        outbound["messages"] = messages
        for name, cast in (("temperature", float), ("top_p", float), ("top_k", int), ("seed", int)):
            if isinstance(outbound.get(name), str):
                try:
                    outbound[name] = cast(outbound[name])
                except ValueError as error:
                    raise HTTPException(422, f"Arena Proxy requires a numeric {name}") from error

        # Count with the SAME prompt-affecting fields used for generation, including
        # deployment template kwargs and reasoning effort. Gym owns this field list.
        tokenize_body = self._get_tokenize_chat_body(outbound)
        tokenize_body.setdefault("add_generation_prompt", True)
        client = self._resolve_client(request)
        tokenized = await self._call_endpoint(client, client.create_tokenize(**tokenize_body))
        count = tokenized.get("count")
        if type(count) is not int or count < 0:
            raise ValueError("The engine's /tokenize response must contain a nonnegative integer count")
        outbound[budget_field] = min(requested, self.config.context_length - count - _CONTEXT_RESERVE)
        if outbound[budget_field] <= 0:
            raise ValueError("The conversation leaves no output tokens after the 64-token context reserve")
        return outbound


if __name__ == "__main__":
    ArenaProxyGenerationModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ArenaProxyGenerationModel.run_webserver()
