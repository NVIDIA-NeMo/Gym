# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arena Proxy's request contract on Gym's vLLM model server."""

from typing import Any

from fastapi import Request
from pydantic import PositiveInt

from nemo_gym.core import NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.models import ChatCompletionsConverter
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class ArenaProxyGenerationConfig(VLLMModelConfig):
    """The served context window used to reserve space before each generation."""

    context_length: PositiveInt = 163840


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
        converted = await super()._prepare_chat_request(request, body)
        if converted.get("tools"):
            raise ValueError("Arena Proxy generation accepts text conversations without tools")
        messages = []
        for message in converted["messages"]:
            content = message.get("content")
            if isinstance(content, list):
                if any(part.get("type") != "text" or not isinstance(part.get("text"), str) for part in content):
                    raise ValueError("Arena Proxy generation requires text-only message content")
                content = "".join(part["text"] for part in content)
            if content is None and message["role"] == "assistant":
                content = ""
            if message["role"] not in ("user", "assistant") or not isinstance(content, str):
                raise ValueError("Arena Proxy generation requires user/assistant text messages")
            messages.append({"role": message["role"], "content": content})

        # Keep the source conversation and sampling payload; Gym-only metadata must
        # not alter the engine's chat template or its generation defaults.
        outbound = {
            "model": converted["model"],
            "messages": messages,
            "temperature": float(converted["temperature"]),
            "top_p": float(converted["top_p"]),
            "top_k": int(converted["top_k"]),
            "seed": int(converted["seed"]),
            "chat_template_kwargs": {"enable_thinking": True},
        }
        client = self._resolve_client(request)
        tokenized = await self._call_endpoint(
            client,
            client.create_tokenize(
                model=outbound["model"],
                messages=messages,
                add_generation_prompt=True,
                chat_template_kwargs=outbound["chat_template_kwargs"],
            ),
        )
        count = tokenized.get("count")
        if type(count) is not int or count < 0:
            raise ValueError("The engine's /tokenize response must contain a nonnegative integer count")
        requested = converted.get("max_completion_tokens", converted.get("max_tokens", 120000))
        if type(requested) is not int or requested <= 0:
            raise ValueError("The generation output budget must be a positive integer")
        outbound["max_tokens"] = min(requested, 120000, self.config.context_length - count - 64)
        if outbound["max_tokens"] <= 0:
            raise ValueError("The conversation leaves no output tokens after the 64-token context reserve")
        return outbound


if __name__ == "__main__":
    ArenaProxyGenerationModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = ArenaProxyGenerationModel.run_webserver()
