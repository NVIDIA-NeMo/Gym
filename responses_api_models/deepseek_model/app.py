# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek Chat Completions extensions for the DeepSeek Harness SDK."""

from typing import ClassVar, Literal

from openai.types.shared import ReasoningEffort
from typing_extensions import TypedDict

from nemo_gym.openai_utils import NeMoGymChatCompletionCreateParamsNonStreaming
from responses_api_models.openai_model.app import SimpleModelServer


class DeepSeekThinkingConfig(TypedDict):
    type: Literal["enabled", "disabled"]


class DeepSeekChatCompletionCreateParams(NeMoGymChatCompletionCreateParamsNonStreaming):
    reasoning_effort: ReasoningEffort | Literal["max"] | None = None
    thinking: DeepSeekThinkingConfig | None = None


class DeepSeekModel(SimpleModelServer):
    """Reuse the OpenAI backend, retries, and SSE bridge with DeepSeek-specific validation."""

    chat_completion_request_model: ClassVar[type[NeMoGymChatCompletionCreateParamsNonStreaming]] = (
        DeepSeekChatCompletionCreateParams
    )
    streaming_chat_extra_fields: ClassVar[frozenset[str]] = frozenset({"thinking"})


if __name__ == "__main__":
    DeepSeekModel.run_webserver()
