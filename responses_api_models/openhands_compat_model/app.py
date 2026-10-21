# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""vLLM model server for the pinned OpenHands 0.62 agent.

Chat Completions backends only; tool messages are forwarded without `name`.
"""

from copy import deepcopy

from fastapi import Body, Request

from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint
from responses_api_models.vllm_model.app import VLLMModel


def normalize_openhands_tool_messages(body: dict) -> dict:
    """Drop `name` from tool messages; keep call IDs and output."""
    result = deepcopy(body)
    for message in result.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "tool":
            message.pop("name", None)
    return result


class OpenHandsCompatModel(VLLMModel):
    async def chat_completions_dispatch(self, request: Request, body: dict = Body()):
        # The pinned OpenHands0.62 history cannot replay opaque native Responses
        # reasoning state. Do not quietly discard it to make a GPT rollout run.
        if self.config.is_responses_native:
            raise ValueError("Legacy OpenHands requires a Chat Completions model; native Responses is not supported")
        return await super().chat_completions_dispatch(request, normalize_openhands_tool_messages(body))


if __name__ == "__main__":
    OpenHandsCompatModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenHandsCompatModel.run_webserver()
