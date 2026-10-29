# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict one-message SSE model bridge; rejects executable actions before the agent sees them."""

import os
import re
from pathlib import Path
from typing import Literal

from fastapi import FastAPI

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from resources_servers.agentic_if_realworld.benchmark.core import PHASES, captured_context, judge, write
from resources_servers.agentic_if_realworld.benchmark.transport import delivered, wire_call


class TextModelConfig(BaseResponsesAPIModelConfig):
    endpoint: str
    model: str
    receipt_dir: str
    api_key_env: str = "AGENTIC_IF_POLICY_API_KEY"
    reasoning_effort: Literal["max"] | None = None


class TextModel(SimpleResponsesAPIModel):
    config: TextModelConfig

    def setup_webserver(self) -> FastAPI:
        # Fail at startup, before the driver journals any attempt; a dry run returns before this.
        if not os.environ.get(self.config.api_key_env):
            raise ValueError(f"{self.config.api_key_env} is not set")
        return super().setup_webserver()

    async def chat_completions(self, body: NeMoGymChatCompletionCreateParamsNonStreaming) -> NeMoGymChatCompletion:
        raise ValueError("Use the single-turn Responses interface")

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming) -> NeMoGymResponse:
        value = body.model_dump(exclude_none=True)
        if body.tools or body.instructions or body.previous_response_id:
            raise ValueError("Native tools and additional prompt instructions are forbidden")
        items = value["input"]
        if not isinstance(items, list) or len(items) != 1:
            raise ValueError("Expected exact single user wrapper")
        item = items[0]
        content = item["content"]
        if isinstance(content, list):
            if len(content) != 1 or content[0].get("type") != "input_text":
                raise ValueError("Unexpected prompt projection")
            content = content[0]["text"]
        request = {"messages": [{"role": item["role"], "content": content}]}
        captured_context(request)
        metadata = body.metadata or {}
        seed, cap = int(metadata["seed"]), body.max_output_tokens
        receipt_id = metadata["receipt_id"]
        if not re.fullmatch(r"[a-f0-9]{64}", receipt_id) or (seed, cap) not in PHASES or body.temperature != 0.7:
            raise ValueError("Generation outside frozen adaptive profile")
        upstream = {
            "model": self.config.model,
            **request,
            "temperature": 0.7,
            "seed": seed,
            "max_tokens": cap,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if self.config.reasoning_effort is not None:
            upstream["chat_template_kwargs"] = {"reasoning_effort": self.config.reasoning_effort}
        path = Path(self.config.receipt_dir) / "model" / (receipt_id + ".json")
        receipt = await wire_call(
            upstream,
            endpoint=self.config.endpoint,
            key_env=self.config.api_key_env,
            path=path,
        )
        complete = False
        try:
            canonical = delivered(receipt["response"])
            complete = True
        except (KeyError, ValueError, TypeError):
            canonical = None
        write(
            path.with_suffix(".binding.json"),
            {
                "input_sha256": judge.sha(request),
                "request_sha256": judge.sha(upstream),
                "canonical_response": canonical,
                "receipt_id": receipt_id,
            },
        )
        message = canonical["message"] if complete else {"content": ""}
        # Canonical raw fields remain in the receipt; this projection is for Gym's text agent.
        text = message.get("content") or message.get("refusal") or ""
        reported_usage = (receipt.get("response") or {}).get("usage") or {}
        usage = None
        if all(reported_usage.get(key) is not None for key in ["prompt_tokens", "completion_tokens", "total_tokens"]):
            usage = {
                "input_tokens": reported_usage["prompt_tokens"],
                "output_tokens": reported_usage["completion_tokens"],
                "total_tokens": reported_usage["total_tokens"],
                "input_tokens_details": {
                    "cached_tokens": (reported_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                },
                "output_tokens_details": {
                    "reasoning_tokens": (reported_usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                },
            }
        return NeMoGymResponse.model_validate(
            {
                "id": receipt_id,
                "created_at": 0,
                "object": "response",
                "model": self.config.model,
                "status": "completed" if complete else "incomplete",
                "metadata": {"receipt_id": receipt_id},
                "parallel_tool_calls": False,
                "tool_choice": "none",
                "tools": [],
                "usage": usage,
                "output": [
                    {
                        "id": "message_" + receipt_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed" if complete else "incomplete",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
            }
        )


if __name__ == "__main__":
    TextModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = TextModel.run_webserver()  # noqa: F401
