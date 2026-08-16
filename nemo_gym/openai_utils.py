# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
from asyncio import sleep
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Literal,
    NotRequired,
    Optional,
    Required,
    TypeAlias,
    Union,
)

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartInputAudioParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageToolCall,
    ChatCompletionMessageToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_assistant_message_param import (
    ContentArrayOfContentPart,
)
from openai.types.chat.chat_completion_content_part_param import File as ChatCompletionContentPartFileParam
from openai.types.chat.chat_completion_custom_tool_param import ChatCompletionCustomToolParam
from openai.types.chat.chat_completion_message_custom_tool_call import ChatCompletionMessageCustomToolCall
from openai.types.chat.chat_completion_message_custom_tool_call_param import ChatCompletionMessageCustomToolCallParam
from openai.types.chat.completion_create_params import (
    ChatCompletionAudioParam,
    ChatCompletionPredictionContentParam,
    ChatCompletionStreamOptionsParam,
    ChatCompletionToolChoiceOptionParam,
    ReasoningEffort,
    ResponseFormat,
    WebSearchOptions,
)
from openai.types.responses import (
    FunctionToolParam,
    Response,
    ResponseCodeInterpreterToolCall,
    ResponseComputerToolCall,
    ResponseCustomToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionWebSearch,
    ResponseInputTextParam,
)
from openai.types.responses.response_apply_patch_tool_call import ResponseApplyPatchToolCall
from openai.types.responses.response_apply_patch_tool_call_output import ResponseApplyPatchToolCallOutput
from openai.types.responses.response_compaction_item import ResponseCompactionItem
from openai.types.responses.response_create_params import (
    Metadata,
    Reasoning,
    ResponseIncludable,
    ResponsePromptParam,
    ResponsesModel,
    ResponseTextConfigParam,
    ToolChoice,
    ToolParam,
)
from openai.types.responses.response_function_call_output_item_list_param import (
    ResponseFunctionCallOutputItemListParam,
)
from openai.types.responses.response_function_shell_tool_call import ResponseFunctionShellToolCall
from openai.types.responses.response_function_shell_tool_call_output import ResponseFunctionShellToolCallOutput
from openai.types.responses.response_input_item import (
    ComputerCallOutput,
    LocalShellCallOutput,
    McpApprovalResponse,
    ResponseCustomToolCallOutput,
)
from openai.types.responses.response_input_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_output_item import (
    ImageGenerationCall,
    LocalShellCall,
    McpApprovalRequest,
    McpCall,
    McpListTools,
)
from openai.types.responses.response_output_text_param import Annotation, Logprob
from openai.types.responses.response_reasoning_item import (
    Content as ReasoningContent,
)
from openai.types.responses.response_reasoning_item import (
    Summary,
)
from openai.types.responses.response_usage import InputTokensDetails as ResponseInputTokensDetails
from openai.types.responses.response_usage import OutputTokensDetails as ResponseOutputTokensDetails
from openai.types.responses.response_usage import ResponseUsage
from openai.types.shared.chat_model import ChatModel
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field
from typing_extensions import TypedDict

from nemo_gym.server_utils import (
    _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG,
    MAX_NUM_TRIES,
    ClientResponse,
    get_response_json,
    raise_for_status,
    request,
)


########################################
# Training-specific
########################################

# Per-token routed expert indices with shape [tokens, num_moe_layers, topk], either as
# nested int lists or as an opaque string envelope produced by the training framework
# (e.g. NeMo-RL's "nrlre1:<dtype>:<SxLxK>:<base64>"). Gym never inspects the value; the
# string form keeps multi-MB payloads cheap to validate and re-serialize at every hop.
RoutedExperts: TypeAlias = Union[str, List[List[List[int]]]]


class TokenIDLogProbMixin(BaseModel):
    prompt_token_ids: List[int]
    generation_token_ids: List[int]
    generation_log_probs: List[float]
    routed_experts: Optional[RoutedExperts] = None


class TokenIDLogProbTypedDictMixin(TypedDict):
    prompt_token_ids: List[int]
    generation_token_ids: List[int]
    generation_log_probs: List[float]
    routed_experts: NotRequired[RoutedExperts]


_REQUIRED_TOKEN_METADATA_FIELDS = frozenset(
    {
        "prompt_token_ids",
        "generation_token_ids",
        "generation_log_probs",
    }
)
_TOKEN_METADATA_FIELDS = _REQUIRED_TOKEN_METADATA_FIELDS | {"routed_experts"}


def _validate_atomic_token_metadata(value: Any) -> Any:
    """Require complete token metadata when any token field is present."""
    if not isinstance(value, dict):
        return value

    present_fields = _TOKEN_METADATA_FIELDS.intersection(value)
    if not present_fields:
        return value

    missing_fields = _REQUIRED_TOKEN_METADATA_FIELDS.difference(present_fields)
    if missing_fields:
        missing = ", ".join(sorted(missing_fields))
        raise ValueError(f"Token metadata must include all required fields; missing: {missing}")

    TokenIDLogProbMixin.model_validate({field: value[field] for field in present_fields})
    return value


########################################
# Responses API inputs
########################################


class NeMoGymSummary(Summary):
    pass


class NeMoGymResponseReasoningItem(BaseModel):
    id: str
    # Override the Iterable to avoid lazy iterators in Pydantic validation.
    summary: List[NeMoGymSummary]
    type: Literal["reasoning"] = "reasoning"
    encrypted_content: Optional[str] = None
    content: Optional[List[ReasoningContent]] = None

    # As of Wed Sep 17, 2025, the OpenAI API with GPT-5 returns None for this status rather than a valid value here.
    # On subsequent calls to the OpenAI endpoints within a rollout, the status parameter is not accepted i.e. the OpenAI API returns a bad request when the status parameter is populated.
    # It's not clear whether or not this is intended. We comment out this status parameter here as a quick stop-gap to fix this issue in Gym re-queries.
    # status: Optional[Literal["in_progress", "completed", "incomplete"]] = None


class NeMoGymResponseOutputText(BaseModel):
    # Override the Iterable to avoid lazy iterators in Pydantic validation.
    annotations: List[Annotation]
    text: str
    type: Literal["output_text"] = "output_text"
    logprobs: Optional[List[Logprob]] = None


class NeMoGymResponseOutputRefusal(BaseModel):
    refusal: str
    type: Literal["refusal"] = "refusal"


NeMoGymContent: TypeAlias = Union[NeMoGymResponseOutputText, NeMoGymResponseOutputRefusal]


class NeMoGymResponseOutputMessage(BaseModel):
    id: str
    # Override the Iterable to avoid lazy iterators in Pydantic validation.
    content: List[NeMoGymContent]
    role: Literal["assistant"] = "assistant"
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    type: Literal["message"] = "message"


class NeMoGymEasyInputMessage(BaseModel):
    content: Union[str, ResponseInputMessageContentListParam]
    role: Literal["user", "assistant", "system", "developer"]
    type: Literal["message"] = "message"


class NeMoGymMessage(BaseModel):
    content: ResponseInputMessageContentListParam
    role: Literal["user", "system", "developer"]
    status: Literal["in_progress", "completed", "incomplete"] = "completed"
    type: Literal["message"] = "message"


class NeMoGymFunctionCallOutput(BaseModel):
    """
    We copy openai.types.responses.response_input_param.FunctionCallOutput, originally a TypedDict, as a BaseModel here
    so that we can use it in the NeMoGymResponseOutputItem below and be consistent with the other ResponseOutputItem types.
    """

    call_id: str
    output: Union[str, ResponseFunctionCallOutputItemListParam]
    type: Literal["function_call_output"] = "function_call_output"
    id: Optional[str] = None
    status: Optional[Literal["in_progress", "completed", "incomplete"]] = None


class NeMoGymResponseFunctionToolCall(BaseModel):
    arguments: str
    call_id: str
    name: str
    type: Literal["function_call"] = "function_call"
    id: Optional[str] = None
    status: Optional[Literal["in_progress", "completed", "incomplete"]] = None


class NeMoGymResponseMcpCall(McpCall):
    """A hosted-MCP tool call (OpenAI Responses ``mcp_call`` output item).

    Emitted when the upstream endpoint executes a tool *server-side* (e.g.
    NVIDIA-hosted gpt-oss surfacing its built-in python tool as MCP) instead of
    returning a client-executed ``function_call``. The ``output``/``error``
    fields are already populated by the server, so the agent parses and passes
    it through; there is no client-side execution and hence no training variant.

    Inherits the upstream ``McpCall`` typing and only relaxes the fields
    NVIDIA-hosted endpoints may omit or widen: ``id``/``server_label`` are made
    optional and ``status`` accepts any string (upstream pins it to a Literal).
    """

    type: Literal["mcp_call"] = "mcp_call"
    id: Optional[str] = None
    server_label: Optional[str] = None
    status: Optional[str] = None


class NeMoGymResponseMcpListTools(McpListTools):
    """A hosted-MCP tool listing (OpenAI Responses ``mcp_list_tools`` output item).

    Inherits the upstream ``McpListTools`` typing; only ``id``/``server_label``
    are relaxed to optional (NVIDIA-hosted endpoints may omit them) and ``tools``
    is widened to ``List[Any]`` so raw tool entries pass through without being
    coerced into the upstream ``McpListToolsTool`` schema.
    """

    type: Literal["mcp_list_tools"] = "mcp_list_tools"
    tools: List[Any] = Field(default_factory=list)
    id: Optional[str] = None
    server_label: Optional[str] = None


class NeMoGymResponseMcpApprovalRequest(McpApprovalRequest):
    """A hosted-MCP approval request (OpenAI Responses ``mcp_approval_request`` item).

    Inherits the upstream ``McpApprovalRequest`` typing; ``id``/``server_label``
    are relaxed to optional to tolerate endpoints that omit them.
    """

    type: Literal["mcp_approval_request"] = "mcp_approval_request"
    id: Optional[str] = None
    server_label: Optional[str] = None


class NeMoGymResponseFileSearchToolCall(ResponseFileSearchToolCall):
    """A hosted file-search call (OpenAI Responses ``file_search_call`` output item).

    The provider executes the search and returns the call in ``response.output``.
    Inherits the upstream typing unchanged.
    """


class NeMoGymResponseFunctionWebSearch(ResponseFunctionWebSearch):
    """A hosted web-search call (OpenAI Responses ``web_search_call`` output item)."""


class NeMoGymResponseComputerToolCall(ResponseComputerToolCall):
    """A computer-use action for the client to execute (``computer_call`` output item)."""


class NeMoGymImageGenerationCall(ImageGenerationCall):
    """A hosted image-generation call (OpenAI Responses ``image_generation_call`` output item)."""


class NeMoGymResponseCodeInterpreterToolCall(ResponseCodeInterpreterToolCall):
    """A hosted code-interpreter call (OpenAI Responses ``code_interpreter_call`` output item)."""


class NeMoGymResponseApplyPatchToolCall(ResponseApplyPatchToolCall):
    """A patch operation for the client to execute (``apply_patch_call`` output item)."""


class NeMoGymResponseShellToolCall(ResponseFunctionShellToolCall):
    """A shell command for the client to execute (``shell_call`` output item)."""


class NeMoGymLocalShellCall(LocalShellCall):
    """A local-shell command for the client to execute (``local_shell_call`` output item)."""


class NeMoGymResponseCustomToolCall(ResponseCustomToolCall):
    """A client-executed custom tool call (OpenAI Responses ``custom_tool_call`` output item)."""


# These models represent client-supplied results for the calls above.
# The installed SDK defines them in ``response_input_item``.
class NeMoGymComputerCallOutput(ComputerCallOutput):
    """The client's result of a computer-use action (``computer_call_output`` item)."""


class NeMoGymResponseCustomToolCallOutput(ResponseCustomToolCallOutput):
    """The client's result of a custom tool call (``custom_tool_call_output`` item)."""


class NeMoGymLocalShellCallOutput(LocalShellCallOutput):
    """The client's result of a local shell command (``local_shell_call_output`` item)."""


class NeMoGymMcpApprovalResponse(McpApprovalResponse):
    """The client's answer to a hosted-MCP approval request (``mcp_approval_response`` item)."""


class NeMoGymResponseApplyPatchToolCallOutput(ResponseApplyPatchToolCallOutput):
    """The client's result of a patch operation (``apply_patch_call_output`` item)."""


class NeMoGymResponseShellToolCallOutput(ResponseFunctionShellToolCallOutput):
    """The client's result of a shell command (``shell_call_output`` item)."""


class NeMoGymResponseCompactionItem(ResponseCompactionItem):
    """Opaque context-management data emitted by Responses (``compaction`` item)."""


class NeMoGymResponseInputText(ResponseInputTextParam):
    pass


class NeMoGymEasyInputMessageForTraining(NeMoGymEasyInputMessage, TokenIDLogProbMixin):
    pass


class NeMoGymMessageForTraining(NeMoGymMessage, TokenIDLogProbMixin):
    pass


class NeMoGymResponseOutputMessageForTraining(NeMoGymResponseOutputMessage, TokenIDLogProbMixin):
    pass


class NeMoGymResponseFunctionToolCallForTraining(NeMoGymResponseFunctionToolCall, TokenIDLogProbMixin):
    pass


class NeMoGymResponseReasoningItemForTraining(NeMoGymResponseReasoningItem, TokenIDLogProbMixin):
    pass


RESPONSES_TO_TRAIN = {
    NeMoGymEasyInputMessage: NeMoGymEasyInputMessageForTraining,
    NeMoGymMessage: NeMoGymMessageForTraining,
    NeMoGymResponseOutputMessage: NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseFunctionToolCall: NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseReasoningItem: NeMoGymResponseReasoningItemForTraining,
}

# The hosted-tool and client-executed call types have no variant here:
#   web_search_call, file_search_call, code_interpreter_call, image_generation_call,
#   mcp_call, computer_call, custom_tool_call, local_shell_call.
#
# training_variant_of() is reached only from ResponsesConverter.postprocess_assistant_message_dict,
# which passes response_output[-1].
# That list is local to the function.
# It holds only NeMoGymResponseReasoningItem, NeMoGymResponseOutputMessage
# or NeMoGymResponseFunctionToolCall, all registered above.
#
# Each variant is also another member of NeMoGymResponseInputItem.
# That union is validated in smart mode, so an unrecognised item reports the errors of every member.
# Variants that nothing can emit only make those errors harder to read.
#
# The upstream models permit extra fields.
# An item carrying token IDs without a declared variant still round-trips through its base class.
# Add a variant when a converter emits that type with sampled token IDs.


def training_variant_of(item_cls: type) -> type:
    """Return the ForTraining subclass that carries token IDs for ``item_cls``.

    Raises NotImplementedError rather than KeyError, so the message can name the class and the fix.
    Either register the pair in RESPONSES_TO_TRAIN, or stop attaching token IDs to that item.
    """
    try:
        return RESPONSES_TO_TRAIN[item_cls]
    except KeyError:
        raise NotImplementedError(
            f"{item_cls.__name__} has no ForTraining variant, so token IDs and logprobs cannot be "
            f"attached to it. Add it to RESPONSES_TO_TRAIN in nemo_gym/openai_utils.py if the policy "
            f"samples this item's tokens; provider-executed hosted calls should not reach this path."
        ) from None


NeMoGymResponseInputItem = Annotated[
    Union[
        NeMoGymEasyInputMessage,
        NeMoGymMessage,
        NeMoGymResponseOutputMessage,
        NeMoGymResponseFunctionToolCall,
        NeMoGymFunctionCallOutput,
        NeMoGymResponseReasoningItem,
        NeMoGymResponseMcpCall,
        NeMoGymResponseMcpListTools,
        NeMoGymResponseMcpApprovalRequest,
        # The SDK includes these items in both response output and request input.
        # Outputs are replayed as input on subsequent turns.
        NeMoGymResponseFileSearchToolCall,
        NeMoGymResponseFunctionWebSearch,
        NeMoGymResponseComputerToolCall,
        NeMoGymImageGenerationCall,
        NeMoGymResponseCodeInterpreterToolCall,
        NeMoGymResponseApplyPatchToolCall,
        NeMoGymResponseShellToolCall,
        NeMoGymLocalShellCall,
        NeMoGymResponseCustomToolCall,
        NeMoGymComputerCallOutput,
        NeMoGymResponseCustomToolCallOutput,
        NeMoGymResponseApplyPatchToolCallOutput,
        NeMoGymResponseShellToolCallOutput,
        NeMoGymResponseCompactionItem,
        NeMoGymLocalShellCallOutput,
        NeMoGymMcpApprovalResponse,
        # Training variants.
        NeMoGymEasyInputMessageForTraining,
        NeMoGymMessageForTraining,
        NeMoGymResponseOutputMessageForTraining,
        NeMoGymResponseFunctionToolCallForTraining,
        NeMoGymResponseReasoningItemForTraining,
    ],
    BeforeValidator(_validate_atomic_token_metadata),
]
NeMoGymResponseInput: TypeAlias = List[NeMoGymResponseInputItem]


class NeMoGymResponseCreateParamsNonStreaming(BaseModel):
    """
    This class is a copy of openai.types.responses.response_create_params.ResponseCreateParamsNonStreaming
    We make a copy of it here since ResponseCreateParamsNonStreaming is a TypedDict with no strict validation.
    We need to do server side validation here.
    """

    model_config = ConfigDict(extra="forbid")

    background: Optional[bool] = None
    include: Optional[List[ResponseIncludable]] = None
    input: Union[str, NeMoGymResponseInput]
    instructions: Optional[str] = None
    max_output_tokens: Optional[int] = None
    max_tool_calls: Optional[int] = None
    metadata: Optional[Metadata] = None
    model: Optional[ResponsesModel] = None
    parallel_tool_calls: bool = True  # OpenAI default
    previous_response_id: Optional[str] = None
    prompt: Optional[ResponsePromptParam] = None
    reasoning: Optional[Reasoning] = None
    service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] = None
    store: Optional[bool] = None
    temperature: Optional[float] = None
    text: Optional[ResponseTextConfigParam] = None
    tool_choice: ToolChoice = "auto"  # OpenAI default
    # Override the Iterable to avoid lazy iterators in Pydantic validation.
    tools: List[ToolParam] = Field(default_factory=list)
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = None
    truncation: Optional[Literal["auto", "disabled"]] = None
    user: Optional[str] = None
    stream: Optional[Literal[False]] = None


########################################
# Responses API outputs
########################################


def _require_response_output_item_type(value: Any) -> Any:
    """Prevent an untagged output item from being coerced into the wrong union member."""
    if isinstance(value, dict) and "type" not in value:
        raise ValueError("Responses API output items must include a type discriminator")
    return value


NeMoGymResponseOutputItem = Annotated[
    NeMoGymResponseInputItem,
    BeforeValidator(_require_response_output_item_type),
]


class NeMoGymResponseInputTokensDetails(ResponseInputTokensDetails):
    pass


class NeMoGymResponseOutputTokensDetails(ResponseOutputTokensDetails):
    pass


class NeMoGymResponseUsage(ResponseUsage):
    input_tokens_details: NeMoGymResponseInputTokensDetails
    output_tokens_details: NeMoGymResponseOutputTokensDetails

    @classmethod
    def sum_from_list(cls, usages: "NeMoGymResponseUsage") -> "NeMoGymResponseUsage":
        final_usage = NeMoGymResponseUsage(
            input_tokens=0,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        )
        for usage in usages:
            final_usage.input_tokens += usage.input_tokens
            final_usage.input_tokens_details.cached_tokens += usage.input_tokens_details.cached_tokens
            final_usage.output_tokens += usage.output_tokens
            final_usage.output_tokens_details.reasoning_tokens += usage.output_tokens_details.reasoning_tokens
            final_usage.total_tokens += usage.total_tokens

        return final_usage


def accumulate_response_usage(
    total: Optional[NeMoGymResponseUsage], additional: Optional[NeMoGymResponseUsage]
) -> Optional[NeMoGymResponseUsage]:
    """Accumulate top-level and detailed response token counts."""
    if additional is None:
        return total
    if total is None:
        return additional.model_copy(deep=True)

    result = total.model_copy(deep=True)
    result.input_tokens += additional.input_tokens
    result.output_tokens += additional.output_tokens
    result.total_tokens += additional.total_tokens
    if result.input_tokens_details is not None and additional.input_tokens_details is not None:
        result.input_tokens_details.cached_tokens += additional.input_tokens_details.cached_tokens
    if result.output_tokens_details is not None and additional.output_tokens_details is not None:
        result.output_tokens_details.reasoning_tokens += additional.output_tokens_details.reasoning_tokens
    return result


class NeMoGymResponse(Response):
    output: List[NeMoGymResponseOutputItem]
    usage: Optional[NeMoGymResponseUsage] = None


########################################
# Chat Completion API outputs
########################################


class NeMoGymFunction(BaseModel):
    arguments: str
    name: str


class NeMoGymChatCompletionMessageToolCall(ChatCompletionMessageToolCall):
    function: NeMoGymFunction


class NeMoGymChatCompletionMessageCustomToolCall(ChatCompletionMessageCustomToolCall):
    pass


NeMoGymChatCompletionMessageToolCallUnion = Annotated[
    Union[
        NeMoGymChatCompletionMessageToolCall,
        NeMoGymChatCompletionMessageCustomToolCall,
    ],
    Field(discriminator="type"),
]


class NeMoGymChatCompletionMessage(ChatCompletionMessage):
    tool_calls: Optional[List[NeMoGymChatCompletionMessageToolCallUnion]] = None


class NeMoGymChatCompletionMessageForTraining(NeMoGymChatCompletionMessage, TokenIDLogProbMixin):
    pass


NeMoGymChatCompletionOutputMessage: TypeAlias = Annotated[
    Union[NeMoGymChatCompletionMessage, NeMoGymChatCompletionMessageForTraining],
    BeforeValidator(_validate_atomic_token_metadata),
]


class NeMoGymChoice(Choice):
    message: NeMoGymChatCompletionOutputMessage


class NeMoGymChatCompletion(ChatCompletion):
    choices: List[NeMoGymChoice]


########################################
# Chat Completion API inputs
########################################


class NeMoGymFunctionDefinition(FunctionDefinition):
    pass


class NeMoGymChatCompletionToolParam(ChatCompletionToolParam):
    function: Required[NeMoGymFunctionDefinition]


class NeMoGymChatCompletionCustomToolParam(ChatCompletionCustomToolParam):
    pass


NeMoGymChatCompletionToolUnionParam = Annotated[
    Union[
        NeMoGymChatCompletionToolParam,
        NeMoGymChatCompletionCustomToolParam,
    ],
    Field(discriminator="type"),
]


class NeMoGymChatCompletionContentPartTextParam(ChatCompletionContentPartTextParam):
    pass


class NeMoGymChatCompletionContentPartImageParam(ChatCompletionContentPartImageParam):
    pass


class NeMoGymChatCompletionContentPartInputAudioParam(ChatCompletionContentPartInputAudioParam):
    pass


class NeMoGymChatCompletionContentPartFileParam(ChatCompletionContentPartFileParam):
    pass


NeMoGymChatCompletionContentPartParam = Union[
    NeMoGymChatCompletionContentPartTextParam,
    NeMoGymChatCompletionContentPartImageParam,
    NeMoGymChatCompletionContentPartInputAudioParam,
    NeMoGymChatCompletionContentPartFileParam,
]


class NeMoGymChatCompletionUserMessageParam(ChatCompletionUserMessageParam):
    # Override the iterable which is annoying to work with.
    content: Required[Union[str, List[NeMoGymChatCompletionContentPartParam]]]


class NeMoGymChatCompletionSystemMessageParam(ChatCompletionSystemMessageParam):
    # Override the iterable which is annoying to work with.
    content: Required[Union[str, List[NeMoGymChatCompletionContentPartTextParam]]]


class NeMoGymChatCompletionDeveloperMessageParam(ChatCompletionDeveloperMessageParam):
    # Override the iterable which is annoying to work with.
    content: Required[Union[str, List[NeMoGymChatCompletionContentPartTextParam]]]


class NeMoGymChatCompletionMessageToolCallFunctionParam(TypedDict, total=False):
    arguments: Required[str]
    name: Required[str]


class NeMoGymChatCompletionMessageToolCallParam(ChatCompletionMessageToolCallParam):
    function: NeMoGymChatCompletionMessageToolCallFunctionParam


class NeMoGymChatCompletionMessageCustomToolCallParam(ChatCompletionMessageCustomToolCallParam):
    pass


NeMoGymChatCompletionMessageToolCallUnionParam = Annotated[
    Union[
        NeMoGymChatCompletionMessageToolCallParam,
        NeMoGymChatCompletionMessageCustomToolCallParam,
    ],
    Field(discriminator="type"),
]


class NeMoGymChatCompletionAssistantMessageParam(ChatCompletionAssistantMessageParam, total=False):
    # Override the iterable which is annoying to work with.
    content: Union[str, List[ContentArrayOfContentPart], None]
    tool_calls: Optional[List[NeMoGymChatCompletionMessageToolCallUnionParam]] = None


class NeMoGymChatCompletionAssistantMessageForTrainingParam(
    NeMoGymChatCompletionAssistantMessageParam, TokenIDLogProbTypedDictMixin
):
    pass


class NeMoGymChatCompletionToolMessageParam(ChatCompletionToolMessageParam):
    # Override the iterable which is annoying to work with.
    content: Required[Union[str, List[NeMoGymChatCompletionContentPartTextParam]]]


class NeMoGymFunctionToolParam(FunctionToolParam):
    pass


NeMoGymChatCompletionMessageParam: TypeAlias = Annotated[
    Union[
        NeMoGymChatCompletionDeveloperMessageParam,
        NeMoGymChatCompletionSystemMessageParam,
        NeMoGymChatCompletionUserMessageParam,
        NeMoGymChatCompletionAssistantMessageParam,
        NeMoGymChatCompletionToolMessageParam,
        # Keep deprecated function messages out of this union.
        # NeMoGymChatCompletionFunctionMessageParam,
        # Training variants.
        NeMoGymChatCompletionAssistantMessageForTrainingParam,
    ],
    BeforeValidator(_validate_atomic_token_metadata),
]


class NeMoGymChatCompletionCreateParamsNonStreaming(BaseModel):
    messages: List[NeMoGymChatCompletionMessageParam]
    model: Optional[Union[str, ChatModel]] = None
    audio: Optional[ChatCompletionAudioParam] = None
    frequency_penalty: Optional[float] = None
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = None
    max_completion_tokens: Optional[int] = None
    max_tokens: Optional[int] = None
    metadata: Optional[Metadata] = None
    modalities: Optional[List[Literal["text", "audio"]]] = None
    n: Optional[int] = None
    parallel_tool_calls: bool = True  # OpenAI default
    prediction: Optional[ChatCompletionPredictionContentParam] = None
    presence_penalty: Optional[float] = None
    reasoning_effort: Optional[ReasoningEffort] = None
    response_format: Optional[ResponseFormat] = None
    seed: Optional[int] = None
    service_tier: Optional[Literal["auto", "default", "flex", "scale", "priority"]] = None
    stop: Union[Optional[str], List[str], None] = None
    store: Optional[bool] = None
    stream_options: Optional[ChatCompletionStreamOptionsParam] = None
    temperature: Optional[float] = None
    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None
    tools: Optional[List[NeMoGymChatCompletionToolUnionParam]] = None
    top_logprobs: Optional[int] = None
    top_p: Optional[float] = None
    user: Optional[str] = None
    web_search_options: Optional[WebSearchOptions] = None
    stream: Optional[Literal[False]] = None

    # Disallow deprecated args
    # function_call: FunctionCall
    # functions: Iterable[Function]


########################################
# Clients
########################################

# See https://platform.openai.com/docs/guides/error-codes/api-errors
# 500 is internal server error, which may sporadically occur
# 502 is Bad gateway (when the endpoint is overloaded)
# 504 is Gateway timeout (when the endpoint config has too low of a gateway timeout setting for the model to finish generating)
RATE_LIMIT_ERROR_CODES = [429, 502, 503, 504, 520]
RETRY_ERROR_CODES = RATE_LIMIT_ERROR_CODES + [500]


class NeMoGymAsyncOpenAI(BaseModel):  # pragma: no cover
    """This is just a stub class that wraps around aiohttp"""

    base_url: str
    api_key: str

    internal: bool = Field(
        default=False,
        description="Set this to true if this particular client is only used to call internal NeMo Gym servers.",
    )

    default_headers: Dict[str, str] = Field(
        default_factory=dict,
        description="Extra headers to include in every request.",
    )

    async def _request(self, **request_kwargs: Dict) -> ClientResponse:
        request_kwargs = request_kwargs | {
            "headers": self.default_headers
            | {
                "Authorization": f"Bearer {self.api_key}",
            },
            "_internal": self.internal,
        }
        return await self._request_with_retry(**request_kwargs)

    async def _request_with_retry(self, **request_kwargs: Dict) -> ClientResponse:
        max_num_tries = MAX_NUM_TRIES
        tries = 0
        while tries < max_num_tries:
            tries += 1
            response = await request(**request_kwargs)

            if response.status in RETRY_ERROR_CODES:
                # If we hit a rate limit, we don't want to hit max num tries, so we increment both.
                if response.status in RATE_LIMIT_ERROR_CODES:
                    max_num_tries += 1

                content = (await response.content.read()).decode()
                kind = "rate_limit" if response.status in RATE_LIMIT_ERROR_CODES else "server_error"
                print(
                    f"[model_retry url={request_kwargs.get('url')} status={response.status} kind={kind} try={tries} max_tries={max_num_tries} error_msg={content[:200]}]",
                    flush=True,
                )
                await sleep(0.5)
                continue
            else:
                return response

        # We've exited the loop
        await raise_for_status(response)

    async def _raise_for_status(self, response: ClientResponse, request_kwargs: Dict[str, Any]) -> None:
        if not response.ok and _GLOBAL_AIOHTTP_CLIENT_REQUEST_DEBUG:
            print(f"Request kwargs: {json.dumps(request_kwargs)}")

        await raise_for_status(response)

    async def create_models(self):
        request_kwargs = dict(url=f"{self.base_url}/models")
        response = await self._request(method="GET", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await get_response_json(response)

    async def create_chat_completion(self, **kwargs):
        request_kwargs = dict(
            url=f"{self.base_url}/chat/completions",
            json=kwargs,
        )
        response = await self._request(method="POST", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await get_response_json(response)

    async def create_completion(self, **kwargs):
        request_kwargs = dict(
            url=f"{self.base_url}/completions",
            json=kwargs,
        )
        response = await self._request(method="POST", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await get_response_json(response)

    async def create_response(self, **kwargs):
        request_kwargs = dict(
            url=f"{self.base_url}/responses",
            json=kwargs,
        )
        response = await self._request(method="POST", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await get_response_json(response)

    async def create_tokenize(self, **kwargs):
        base_url = self.base_url.removesuffix("/v1")
        request_kwargs = dict(
            url=f"{base_url}/tokenize",
            json=kwargs,
        )
        response = await self._request(method="POST", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await get_response_json(response)
