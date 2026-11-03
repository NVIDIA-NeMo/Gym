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
"""The definitions every package and every environment shares: Responses API types, episode identity, and failure kinds."""

# Public API
from typing import TYPE_CHECKING

from nemo_gym._public_api import lazy_exports


if TYPE_CHECKING:
    from nemo_gym.core.episode import (
        EpisodeId,
        TaskId,
        current_rollout_id,
        current_verifier_invocation_id,
        maybe_rollout_id_from_run_body,
        rollout_context,
        verifier_invocation_context,
    )
    from nemo_gym.core.failure_kinds import (
        AGENT_INVALID_OUTPUT,
        AGENT_LIMIT,
        AGENT_TIMEOUT,
        ENVIRONMENT_SERVER_FAILED,
        JUDGE_FAILED,
        JUDGE_UNPARSEABLE,
        MODEL_CONTEXT_LIMIT,
        MODEL_REQUEST_ERROR,
        MODEL_TRANSPORT_ERROR,
        PROVIDER_UNAVAILABLE,
        SESSION_EXPIRED,
        SESSION_LOST,
        VERIFIER_ERROR,
        FailureStage,
        is_namespaced,
        is_registered,
        validate_failure_kind,
    )
    from nemo_gym.core.operation_timing import OperationTiming, TimingBoundary
    from nemo_gym.core.responses_api_types import (
        CHAT_REQUEST_PROVIDER_EXTENSION_FIELDS,
        REQUIRED_TOKEN_METADATA_FIELDS,
        RESPONSES_TO_TRAIN,
        SSE_KEEPALIVE,
        TOKEN_METADATA_FIELDS,
        NeMoGymChatCompletion,
        NeMoGymChatCompletionAssistantMessageForTrainingParam,
        NeMoGymChatCompletionAssistantMessageParam,
        NeMoGymChatCompletionContentPartTextParam,
        NeMoGymChatCompletionCreateParamsNonStreaming,
        NeMoGymChatCompletionMessage,
        NeMoGymChatCompletionMessageForTraining,
        NeMoGymChatCompletionMessageParam,
        NeMoGymChatCompletionMessageToolCall,
        NeMoGymChatCompletionMessageToolCallFunctionParam,
        NeMoGymChatCompletionMessageToolCallParam,
        NeMoGymChatCompletionSystemMessageParam,
        NeMoGymChatCompletionToolMessageParam,
        NeMoGymChatCompletionUserMessageParam,
        NeMoGymChoice,
        NeMoGymEasyInputMessage,
        NeMoGymFunction,
        NeMoGymFunctionCallOutput,
        NeMoGymFunctionToolParam,
        NeMoGymMessage,
        NeMoGymResponse,
        NeMoGymResponseCreateParamsNonStreaming,
        NeMoGymResponseFunctionCallOutput,
        NeMoGymResponseFunctionToolCall,
        NeMoGymResponseFunctionToolCallForTraining,
        NeMoGymResponseInput,
        NeMoGymResponseInputItem,
        NeMoGymResponseInputText,
        NeMoGymResponseInputTokensDetails,
        NeMoGymResponseMcpCall,
        NeMoGymResponseOutputItem,
        NeMoGymResponseOutputMessage,
        NeMoGymResponseOutputMessageForTraining,
        NeMoGymResponseOutputRefusal,
        NeMoGymResponseOutputText,
        NeMoGymResponseOutputTokensDetails,
        NeMoGymResponseReasoningItem,
        NeMoGymResponseUsage,
        NeMoGymSummary,
        ReasoningEffort,
        accumulate_response_usage,
    )


# What code outside the library, such as servers, imports from this package. Each name's submodule loads
# on first use. Code inside nemo_gym imports the submodules directly.
_PUBLIC_API = {
    "OperationTiming": "operation_timing",
    "TimingBoundary": "operation_timing",
    "AGENT_INVALID_OUTPUT": "failure_kinds",
    "AGENT_LIMIT": "failure_kinds",
    "AGENT_TIMEOUT": "failure_kinds",
    "CHAT_REQUEST_PROVIDER_EXTENSION_FIELDS": "responses_api_types",
    "ENVIRONMENT_SERVER_FAILED": "failure_kinds",
    "EpisodeId": "episode",
    "FailureStage": "failure_kinds",
    "JUDGE_FAILED": "failure_kinds",
    "JUDGE_UNPARSEABLE": "failure_kinds",
    "MODEL_CONTEXT_LIMIT": "failure_kinds",
    "MODEL_REQUEST_ERROR": "failure_kinds",
    "MODEL_TRANSPORT_ERROR": "failure_kinds",
    "NeMoGymChatCompletion": "responses_api_types",
    "NeMoGymChatCompletionAssistantMessageForTrainingParam": "responses_api_types",
    "NeMoGymChatCompletionAssistantMessageParam": "responses_api_types",
    "NeMoGymChatCompletionContentPartTextParam": "responses_api_types",
    "NeMoGymChatCompletionCreateParamsNonStreaming": "responses_api_types",
    "NeMoGymChatCompletionMessage": "responses_api_types",
    "NeMoGymChatCompletionMessageForTraining": "responses_api_types",
    "NeMoGymChatCompletionMessageParam": "responses_api_types",
    "NeMoGymChatCompletionMessageToolCall": "responses_api_types",
    "NeMoGymChatCompletionMessageToolCallFunctionParam": "responses_api_types",
    "NeMoGymChatCompletionMessageToolCallParam": "responses_api_types",
    "NeMoGymChatCompletionSystemMessageParam": "responses_api_types",
    "NeMoGymChatCompletionToolMessageParam": "responses_api_types",
    "NeMoGymChatCompletionUserMessageParam": "responses_api_types",
    "NeMoGymChoice": "responses_api_types",
    "NeMoGymEasyInputMessage": "responses_api_types",
    "NeMoGymFunction": "responses_api_types",
    "NeMoGymFunctionCallOutput": "responses_api_types",
    "NeMoGymFunctionToolParam": "responses_api_types",
    "NeMoGymMessage": "responses_api_types",
    "NeMoGymResponse": "responses_api_types",
    "NeMoGymResponseCreateParamsNonStreaming": "responses_api_types",
    "NeMoGymResponseFunctionCallOutput": "responses_api_types",
    "NeMoGymResponseFunctionToolCall": "responses_api_types",
    "NeMoGymResponseFunctionToolCallForTraining": "responses_api_types",
    "NeMoGymResponseInput": "responses_api_types",
    "NeMoGymResponseInputItem": "responses_api_types",
    "NeMoGymResponseInputText": "responses_api_types",
    "NeMoGymResponseInputTokensDetails": "responses_api_types",
    "NeMoGymResponseMcpCall": "responses_api_types",
    "NeMoGymResponseOutputItem": "responses_api_types",
    "NeMoGymResponseOutputMessage": "responses_api_types",
    "NeMoGymResponseOutputMessageForTraining": "responses_api_types",
    "NeMoGymResponseOutputRefusal": "responses_api_types",
    "NeMoGymResponseOutputText": "responses_api_types",
    "NeMoGymResponseOutputTokensDetails": "responses_api_types",
    "NeMoGymResponseReasoningItem": "responses_api_types",
    "NeMoGymResponseUsage": "responses_api_types",
    "NeMoGymSummary": "responses_api_types",
    "PROVIDER_UNAVAILABLE": "failure_kinds",
    "REQUIRED_TOKEN_METADATA_FIELDS": "responses_api_types",
    "RESPONSES_TO_TRAIN": "responses_api_types",
    "ReasoningEffort": "responses_api_types",
    "SESSION_EXPIRED": "failure_kinds",
    "SESSION_LOST": "failure_kinds",
    "SSE_KEEPALIVE": "responses_api_types",
    "TOKEN_METADATA_FIELDS": "responses_api_types",
    "TaskId": "episode",
    "VERIFIER_ERROR": "failure_kinds",
    "accumulate_response_usage": "responses_api_types",
    "current_rollout_id": "episode",
    "current_verifier_invocation_id": "episode",
    "is_namespaced": "failure_kinds",
    "is_registered": "failure_kinds",
    "maybe_rollout_id_from_run_body": "episode",
    "rollout_context": "episode",
    "validate_failure_kind": "failure_kinds",
    "verifier_invocation_context": "episode",
}
__getattr__, __dir__ = lazy_exports(__name__, _PUBLIC_API)
__all__ = sorted(_PUBLIC_API)
