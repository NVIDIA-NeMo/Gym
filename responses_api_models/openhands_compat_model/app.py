# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Legacy OpenHands tool-message compatibility, retained from the native-harness pilot."""

from copy import deepcopy

from fastapi import Body, Request

from responses_api_models.vllm_model.app import VLLMModel


def normalize_openhands_tool_messages(body: dict) -> dict:
    """Remove the redundant legacy tool name, not call IDs, output or validation."""
    result = deepcopy(body)
    for message in result.get("messages", []):
        if isinstance(message, dict) and message.get("role") == "tool":
            message.pop("name", None)
    return result


class OpenHandsCompatModel(VLLMModel):
    ray_enabled = False

    async def chat_completions_dispatch(self, request: Request, body: dict = Body()):
        # The pinned OpenHands0.62 history cannot replay opaque native Responses
        # reasoning state. Do not quietly discard it to make a GPT rollout run.
        if self.config.is_responses_native:
            raise ValueError("Legacy OpenHands requires a Chat Completions model; native Responses is not supported")
        return await super().chat_completions_dispatch(request, normalize_openhands_tool_messages(body))


if __name__ == "__main__":
    OpenHandsCompatModel.run_webserver()
