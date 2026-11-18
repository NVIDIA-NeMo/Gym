# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Restore Gym's reasoning field convention for the unchanged OpenCode client."""

import re

from fastapi import Body, Request

from nemo_gym.core import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel


class AgentVistaOpenCodeModel(VLLMModel):
    ray_enabled = False

    async def chat_completions(self, request: Request, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()):
        response = await super().chat_completions(request, body)
        payload = response.model_dump(exclude_unset=True)
        for choice in payload["choices"]:
            message = choice["message"]
            content = message.get("content")
            match = (
                re.fullmatch(r"<think>\n?(.*?)</think>(.*)", content, re.DOTALL) if isinstance(content, str) else None
            )
            if match:
                reasoning, text = match.groups()
                message.update(reasoning_content=reasoning, reasoning=reasoning, content=text or None)
        return NeMoGymChatCompletion.model_validate(payload)


if __name__ == "__main__":
    AgentVistaOpenCodeModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = AgentVistaOpenCodeModel.run_webserver()
