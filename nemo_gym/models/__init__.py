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
    from nemo_gym.models._checkpoint import note_generation_backend, note_generation_request
    from nemo_gym.models.chat_completions_converter import (
        ResponsesConverter,
        VLLMConverter,
        VLLMConverterResponsesToChatCompletionsState,
        chat_completion_usage_to_response_usage,
        split_responses_input_output_items,
    )
    from nemo_gym.models.chat_streaming import synthesize_chat_completion_sse
    from nemo_gym.models.openai_utils import (
        MAX_NUM_TRIES,
        RATE_LIMIT_ERROR_CODES,
        RETRY_ERROR_CODES,
        NeMoGymAsyncOpenAI,
        PermanentEndpointError,
    )
    from nemo_gym.models.responses_streaming import (
        flatten_namespace_tools,
        restore_namespace_tool_calls,
        synthesize_responses_sse,
    )
    from nemo_gym.models.token_capture.config import token_id_capture_config
    from nemo_gym.models.token_capture.conformance import run_conformance
    from nemo_gym.models.token_capture.external_capture import (
        ExternalCaptureHandler,
        make_external_capture_handler,
        validate_framework_context_options,
    )
    from nemo_gym.models.token_capture.protocols import (
        CAPTURE_REFUSAL_HEADER,
        CaptureAdmissionError,
        CaptureInputError,
        TokenSource,
    )
    from nemo_gym.models.token_capture.sink import (
        CaptureContext,
        capture_engine_refusal,
        current_capture_context,
        reset_token_sink,
        resolve_parent,
        set_token_sink,
    )


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "CAPTURE_REFUSAL_HEADER": "token_capture.protocols",
    "CaptureAdmissionError": "token_capture.protocols",
    "CaptureContext": "token_capture.sink",
    "CaptureInputError": "token_capture.protocols",
    "ExternalCaptureHandler": "token_capture.external_capture",
    "MAX_NUM_TRIES": "openai_utils",
    "NeMoGymAsyncOpenAI": "openai_utils",
    "PermanentEndpointError": "openai_utils",
    "RATE_LIMIT_ERROR_CODES": "openai_utils",
    "RETRY_ERROR_CODES": "openai_utils",
    "ResponsesConverter": "chat_completions_converter",
    "TokenSource": "token_capture.protocols",
    "VLLMConverter": "chat_completions_converter",
    "VLLMConverterResponsesToChatCompletionsState": "chat_completions_converter",
    "capture_engine_refusal": "token_capture.sink",
    "chat_completion_usage_to_response_usage": "chat_completions_converter",
    "current_capture_context": "token_capture.sink",
    "flatten_namespace_tools": "responses_streaming",
    "make_external_capture_handler": "token_capture.external_capture",
    "note_generation_backend": "_checkpoint",
    "note_generation_request": "_checkpoint",
    "reset_token_sink": "token_capture.sink",
    "resolve_parent": "token_capture.sink",
    "restore_namespace_tool_calls": "responses_streaming",
    "run_conformance": "token_capture.conformance",
    "set_token_sink": "token_capture.sink",
    "split_responses_input_output_items": "chat_completions_converter",
    "synthesize_chat_completion_sse": "chat_streaming",
    "synthesize_responses_sse": "responses_streaming",
    "token_id_capture_config": "token_capture.config",
    "validate_framework_context_options": "token_capture.external_capture",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
