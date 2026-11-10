# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Model servers and what they share: the model server base, provider clients, converters, and model call capture."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.models._checkpoint import (
        PolicyModelParticipant,
        generation_cut_requester,
        note_generation_backend,
        note_generation_request,
    )
    from nemo_gym.models.anthropic_converter import AnthropicConverter
    from nemo_gym.models.base_responses_api_model import (
        PROPAGATED_REFUSAL_CODE_ATTRIBUTE,
        BaseResponsesAPIModelConfig,
        Body,
        SimpleResponsesAPIModel,
        context_overflow_error_fields,
        is_propagated_refusal,
    )
    from nemo_gym.models.chat_completions_converter import (
        ChatCompletionsConverter,
        VLLMConverter,
        VLLMConverterResponsesToChatCompletionsState,
        chat_completion_usage_to_response_usage,
        message_content_to_text,
        split_responses_input_output_items,
    )
    from nemo_gym.models.chat_streaming import synthesize_chat_completion_sse
    from nemo_gym.models.model_call_capture import (
        CaptureStore,
        ModelCallCaptureConfig,
        install_model_call_capture,
        start_model_execution,
    )
    from nemo_gym.models.responses_streaming import (
        flatten_namespace_tools,
        is_responses_lifecycle_chunk,
        restore_namespace_tool_calls,
        synthesize_responses_sse,
    )
    from nemo_gym.models.token_capture.external_capture import (
        ExternalCaptureHandler,
        VLLMWorkerCaptureHandler,
        make_external_capture_handler,
        validate_framework_context_options,
    )
    from nemo_gym.models.token_capture.harness_notes import HARNESS_NOTES
    from nemo_gym.models.token_capture.harness_notes.base import HarnessNote


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "AnthropicConverter": "anthropic_converter",
    "BaseResponsesAPIModelConfig": "base_responses_api_model",
    "Body": "base_responses_api_model",
    "CaptureStore": "model_call_capture",
    "ChatCompletionsConverter": "chat_completions_converter",
    "ExternalCaptureHandler": "token_capture.external_capture",
    "HARNESS_NOTES": "token_capture.harness_notes",
    "HarnessNote": "token_capture.harness_notes.base",
    "ModelCallCaptureConfig": "model_call_capture",
    "PROPAGATED_REFUSAL_CODE_ATTRIBUTE": "base_responses_api_model",
    "PolicyModelParticipant": "_checkpoint",
    "SimpleResponsesAPIModel": "base_responses_api_model",
    "VLLMConverter": "chat_completions_converter",
    "VLLMConverterResponsesToChatCompletionsState": "chat_completions_converter",
    "VLLMWorkerCaptureHandler": "token_capture.external_capture",
    "chat_completion_usage_to_response_usage": "chat_completions_converter",
    "context_overflow_error_fields": "base_responses_api_model",
    "flatten_namespace_tools": "responses_streaming",
    "generation_cut_requester": "_checkpoint",
    "install_model_call_capture": "model_call_capture",
    "is_propagated_refusal": "base_responses_api_model",
    "is_responses_lifecycle_chunk": "responses_streaming",
    "make_external_capture_handler": "token_capture.external_capture",
    "message_content_to_text": "chat_completions_converter",
    "note_generation_backend": "_checkpoint",
    "note_generation_request": "_checkpoint",
    "restore_namespace_tool_calls": "responses_streaming",
    "split_responses_input_output_items": "chat_completions_converter",
    "start_model_execution": "model_call_capture",
    "synthesize_chat_completion_sse": "chat_streaming",
    "synthesize_responses_sse": "responses_streaming",
    "validate_framework_context_options": "token_capture.external_capture",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
