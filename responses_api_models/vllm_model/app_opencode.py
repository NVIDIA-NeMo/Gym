# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# OpenCode directive: MIT, Copyright (c) 2025 opencode.
# Source: https://github.com/anomalyco/opencode/blob/67aec2212010d67775c35e696d8b8b54902eb338/packages/core/src/session/runner/max-steps.ts
# Full license: responses_api_agents/opencode_multimodal_agent/third-party-notices.md
"""Opt-in OpenCode 1.17.11 final-turn adapter, before lineage and token capture."""

from fastapi import Request
from pydantic import Field

from nemo_gym.models import Body
from nemo_gym.models.token_capture import current_capture_context
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.opencode_agent.protocol import OPENCODE_1_17_11_MAX_STEPS as OPENCODE_1_17_11_MAX_STEPS
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig
from responses_api_models.vllm_model.image_guard import limit_request_images


# OpenCode's harness constant; keep exact matching version-scoped.
# prompt.ts appends this as an assistant message, although it is not model output.


def normalize_opencode_final_instruction(body: dict) -> dict:
    """Classify only the exact trailing harness directive as user input.

    Generated history remains untouched. Both lineage resolution and vLLM see
    the same normalized body, so the directive is captured as untrained prompt
    tokens. Unknown instructions still fail the ordinary lineage checks.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return body
    last = messages[-1]
    if (
        not isinstance(last, dict)
        or last.get("role") != "assistant"
        or last.get("content") != OPENCODE_1_17_11_MAX_STEPS
        or last.get("reasoning_content") not in (None, "")
        or set(last) - {"role", "content", "reasoning_content"}
    ):
        return body
    return {**body, "messages": [*messages[:-1], {"role": "user", "content": last["content"]}]}


class OpenCodeVLLMModelConfig(VLLMModelConfig):
    # Disabled by default. This is the accumulated
    # request limit, distinct from the agent's initial-attachment limit.
    max_images: int | None = Field(default=None, gt=0)


class OpenCodeVLLMModel(VLLMModel):
    config: OpenCodeVLLMModelConfig

    async def chat_completions_dispatch(self, request: Request, body: dict = Body()):
        guarded = limit_request_images(body, max_images=self.config.max_images)
        # Shared capture recognizes the assistant note, forwards it as user input, and excludes it from lineage.
        # Rewriting it here would hide that marker and break the next call when OpenCode drops the note.
        if current_capture_context() is None:
            guarded = normalize_opencode_final_instruction(guarded)
        return await super().chat_completions_dispatch(request, guarded)


if __name__ == "__main__":
    OpenCodeVLLMModel.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = OpenCodeVLLMModel.run_webserver()
