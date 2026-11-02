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
    from nemo_gym.models.chat_completions_converter import (
        ResponsesConverter,
        VLLMConverter,
        VLLMConverterResponsesToChatCompletionsState,
        chat_completion_usage_to_response_usage,
        split_responses_input_output_items,
    )
    from nemo_gym.models.chat_streaming import SSE_KEEPALIVE, synthesize_chat_completion_sse
    from nemo_gym.models.openai_utils import (
        MAX_NUM_TRIES,
        RATE_LIMIT_ERROR_CODES,
        RETRY_ERROR_CODES,
        NeMoGymAsyncOpenAI,
        PermanentEndpointError,
    )
    from nemo_gym.models.responses_streaming import (
        flatten_namespace_tools,
        is_responses_lifecycle_chunk,
        restore_namespace_tool_calls,
        synthesize_responses_sse,
    )
    from nemo_gym.models.token_capture.adapters.vllm import VLLMCaptureAdapter
    from nemo_gym.models.token_capture.config import token_capture_config
    from nemo_gym.models.token_capture.conformance import run_conformance
    from nemo_gym.models.token_capture.consumer import trajectories_from_source
    from nemo_gym.models.token_capture.external_capture import (
        ExternalCaptureHandler,
        VLLMWorkerCaptureHandler,
        make_external_capture_handler,
        validate_framework_context_options,
    )
    from nemo_gym.models.token_capture.lineage import FileLineageStore, InMemoryLineageStore, stamp_continuation
    from nemo_gym.models.token_capture.protocols import (
        CAPTURE_REFUSAL_HEADER,
        CaptureAdmissionError,
        CaptureInputError,
        LineageResolver,
        TokenCaptureSnapshot,
        TokenSink,
        TokenSource,
    )
    from nemo_gym.models.token_capture.records import (
        ENGINE_REFUSED_CALL_REASON,
        UNCOMMITTED_CALL_REASON,
        ParentResolutionStatus,
        TokenEntry,
    )
    from nemo_gym.models.token_capture.sink import (
        NG_CAPTURE_FIELD,
        NG_COMMIT_COORDS_FIELD,
        CaptureContext,
        capture_engine_refusal,
        current_capture_context,
        reset_token_sink,
        resolve_parent,
        set_token_sink,
    )
    from nemo_gym.models.token_capture.staging.capture import RolloutTokenCapture
    from nemo_gym.models.token_capture.staging.rebuild import verify_and_linearize
    from nemo_gym.models.token_capture.staging.records import (
        INVALID_COMMIT_COORDS_REASON,
        WORKER_MISSING_COMMIT_COORDS_REASON,
        CaptureAdmission,
        RolloutManifest,
        RolloutReceipt,
        StagedCallBaseSnapshot,
        StageResult,
    )
    from nemo_gym.models.token_capture.staging.terminal import select_terminal_call
    from nemo_gym.models.token_capture.store import TokenCaptureStore
    from nemo_gym.models.token_capture.terminal import resolve_terminal


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "AnthropicConverter": "anthropic_converter",
    "CAPTURE_REFUSAL_HEADER": "token_capture.protocols",
    "CaptureAdmission": "token_capture.staging.records",
    "CaptureAdmissionError": "token_capture.protocols",
    "CaptureContext": "token_capture.sink",
    "CaptureInputError": "token_capture.protocols",
    "ENGINE_REFUSED_CALL_REASON": "token_capture.records",
    "ExternalCaptureHandler": "token_capture.external_capture",
    "FileLineageStore": "token_capture.lineage",
    "INVALID_COMMIT_COORDS_REASON": "token_capture.staging.records",
    "InMemoryLineageStore": "token_capture.lineage",
    "LineageResolver": "token_capture.protocols",
    "MAX_NUM_TRIES": "openai_utils",
    "NG_CAPTURE_FIELD": "token_capture.sink",
    "NG_COMMIT_COORDS_FIELD": "token_capture.sink",
    "NeMoGymAsyncOpenAI": "openai_utils",
    "ParentResolutionStatus": "token_capture.records",
    "PermanentEndpointError": "openai_utils",
    "PolicyModelParticipant": "_checkpoint",
    "RATE_LIMIT_ERROR_CODES": "openai_utils",
    "RETRY_ERROR_CODES": "openai_utils",
    "ResponsesConverter": "chat_completions_converter",
    "RolloutManifest": "token_capture.staging.records",
    "RolloutReceipt": "token_capture.staging.records",
    "RolloutTokenCapture": "token_capture.staging.capture",
    "SSE_KEEPALIVE": "chat_streaming",
    "StageResult": "token_capture.staging.records",
    "StagedCallBaseSnapshot": "token_capture.staging.records",
    "TokenCaptureSnapshot": "token_capture.protocols",
    "TokenCaptureStore": "token_capture.store",
    "TokenEntry": "token_capture.records",
    "TokenSink": "token_capture.protocols",
    "TokenSource": "token_capture.protocols",
    "UNCOMMITTED_CALL_REASON": "token_capture.records",
    "VLLMCaptureAdapter": "token_capture.adapters.vllm",
    "VLLMConverter": "chat_completions_converter",
    "VLLMConverterResponsesToChatCompletionsState": "chat_completions_converter",
    "VLLMWorkerCaptureHandler": "token_capture.external_capture",
    "WORKER_MISSING_COMMIT_COORDS_REASON": "token_capture.staging.records",
    "capture_engine_refusal": "token_capture.sink",
    "chat_completion_usage_to_response_usage": "chat_completions_converter",
    "current_capture_context": "token_capture.sink",
    "flatten_namespace_tools": "responses_streaming",
    "generation_cut_requester": "_checkpoint",
    "is_responses_lifecycle_chunk": "responses_streaming",
    "make_external_capture_handler": "token_capture.external_capture",
    "note_generation_backend": "_checkpoint",
    "note_generation_request": "_checkpoint",
    "reset_token_sink": "token_capture.sink",
    "resolve_parent": "token_capture.sink",
    "resolve_terminal": "token_capture.terminal",
    "restore_namespace_tool_calls": "responses_streaming",
    "run_conformance": "token_capture.conformance",
    "select_terminal_call": "token_capture.staging.terminal",
    "set_token_sink": "token_capture.sink",
    "split_responses_input_output_items": "chat_completions_converter",
    "stamp_continuation": "token_capture.lineage",
    "synthesize_chat_completion_sse": "chat_streaming",
    "synthesize_responses_sse": "responses_streaming",
    "token_capture_config": "token_capture.config",
    "trajectories_from_source": "token_capture.consumer",
    "validate_framework_context_options": "token_capture.external_capture",
    "verify_and_linearize": "token_capture.staging.rebuild",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
