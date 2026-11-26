# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expose Gym-normalized reasoning in the fields OpenClaw retains and replays."""

from fastapi import Body, Request

from nemo_gym.core import NeMoGymChatCompletion, NeMoGymChatCompletionCreateParamsNonStreaming
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel


def restore_reasoning(payload: dict) -> dict:
    for choice in payload["choices"]:
        message = choice["message"]
        content = message.get("content")
        if isinstance(content, str) and content.startswith("<think>") and "</think>" in content:
            reasoning, answer = content[len("<think>") :].split("</think>", 1)
            message["reasoning"] = reasoning
            message["reasoning_content"] = message["reasoning"]
            message["content"] = answer or None
    return payload


class WildClawModel(VLLMModel):
    ray_enabled = False

    async def chat_completions(self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()):
        completion = await super().chat_completions(request, body)
        return NeMoGymChatCompletion.model_validate(restore_reasoning(completion.model_dump(exclude_unset=True)))


if __name__ == "__main__":
    WildClawModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = WildClawModel.run_webserver()
