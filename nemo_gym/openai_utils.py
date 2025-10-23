# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from asyncio import sleep
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Required,
    TypeAlias,
    Union,
)

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
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
    ResponseInputTextParam,
)
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
from openai.types.responses.response_input_param import (
    ResponseInputMessageContentListParam,
)
from openai.types.responses.response_output_text_param import Annotation, Logprob
from openai.types.responses.response_reasoning_item import (
    Summary,
)
from openai.types.shared.chat_model import ChatModel
from openai.types.shared_params import FunctionDefinition
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from nemo_gym.server_utils import MAX_NUM_TRIES, ClientResponse, raise_for_status, request


########################################
# Training-specific
########################################


class TokenIDLogProbMixin(BaseModel):
    prompt_token_ids: List[int]
    generation_token_ids: List[int]
    generation_log_probs: List[float]


class TokenIDLogProbTypedDictMixin(TypedDict):
    prompt_token_ids: List[int]
    generation_token_ids: List[int]
    generation_log_probs: List[float]


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
    output: str
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


NeMoGymResponseInputItem = Union[
    NeMoGymEasyInputMessage,
    NeMoGymMessage,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseFunctionToolCall,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseReasoningItem,
    # For training:
    NeMoGymEasyInputMessageForTraining,
    NeMoGymMessageForTraining,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseFunctionToolCallForTraining,
    NeMoGymResponseReasoningItemForTraining,
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


NeMoGymResponseOutputItem = NeMoGymResponseInputItem


class NeMoGymResponse(Response):
    output: List[NeMoGymResponseOutputItem]


########################################
# Chat Completion API outputs
########################################


class NeMoGymFunction(BaseModel):
    arguments: str
    name: str


class NeMoGymChatCompletionMessageToolCall(ChatCompletionMessageToolCall):
    function: NeMoGymFunction


class NeMoGymChatCompletionMessage(ChatCompletionMessage):
    tool_calls: Optional[List[NeMoGymChatCompletionMessageToolCall]] = None


class NeMoGymChatCompletionMessageForTraining(NeMoGymChatCompletionMessage, TokenIDLogProbMixin):
    pass


class NeMoGymChoice(Choice):
    message: Union[NeMoGymChatCompletionMessage, NeMoGymChatCompletionMessageForTraining]


class NeMoGymChatCompletion(ChatCompletion):
    choices: List[NeMoGymChoice]


########################################
# Chat Completion API inputs
########################################


class NeMoGymFunctionDefinition(FunctionDefinition):
    pass


class NeMoGymChatCompletionToolParam(ChatCompletionToolParam):
    function: Required[NeMoGymFunctionDefinition]


class NeMoGymChatCompletionContentPartTextParam(ChatCompletionContentPartTextParam):
    pass


class NeMoGymChatCompletionUserMessageParam(ChatCompletionUserMessageParam):
    # Override the iterable which is annoying to work with.
    content: Required[Union[str, List[NeMoGymChatCompletionContentPartTextParam]]]


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


class NeMoGymChatCompletionAssistantMessageParam(ChatCompletionAssistantMessageParam):
    # Override the iterable which is annoying to work with.
    content: Union[str, List[ContentArrayOfContentPart], None]
    tool_calls: List[NeMoGymChatCompletionMessageToolCallParam]


class NeMoGymChatCompletionAssistantMessageForTrainingParam(
    NeMoGymChatCompletionAssistantMessageParam, TokenIDLogProbTypedDictMixin
):
    pass


class NeMoGymChatCompletionToolMessageParam(ChatCompletionToolMessageParam):
    # Override the iterable which is annoying to work with.
    content: Required[Union[str, List[NeMoGymChatCompletionContentPartTextParam]]]


class NeMoGymFunctionToolParam(FunctionToolParam):
    pass


NeMoGymChatCompletionMessageParam: TypeAlias = Union[
    NeMoGymChatCompletionDeveloperMessageParam,
    NeMoGymChatCompletionSystemMessageParam,
    NeMoGymChatCompletionUserMessageParam,
    NeMoGymChatCompletionAssistantMessageParam,
    NeMoGymChatCompletionToolMessageParam,
    # Don't add deprecated.
    # NeMoGymChatCompletionFunctionMessageParam,
    # Training:
    NeMoGymChatCompletionAssistantMessageForTrainingParam,
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
    tools: Optional[List[NeMoGymChatCompletionToolParam]] = None
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


class NeMoGymAsyncOpenAI(BaseModel):  # pragma: no cover
    """This is just a stub class that wraps around aiohttp"""

    base_url: str
    api_key: str

    async def _request(self, **request_kwargs: Dict) -> ClientResponse:
        tries = 0
        while tries < MAX_NUM_TRIES:
            tries += 1
            response = await request(**request_kwargs)
            # See https://platform.openai.com/docs/guides/error-codes/api-errors
            if response.status in (429, 500, 503):
                content = (await response.content.read()).decode()
                print(
                    f"Hit a {response.status} trying to query an OpenAI endpoint (try {tries}). Sleeping 0.5s. Error message: {content}"
                )
                await sleep(0.5)
                continue
            else:
                return response

        # We've exited the loop
        response.raise_for_status()

    async def _raise_for_status(self, response: ClientResponse, request_kwargs: Dict[str, Any]) -> None:
        if not response.ok:
            print(f"Request kwargs: {json.dumps(request_kwargs)}")

        await raise_for_status(response)

    async def create_chat_completion(self, **kwargs):
        request_kwargs = dict(
            url=f"{self.base_url}/chat/completions",
            json=kwargs,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response = await self._request(method="POST", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await response.json()

    async def create_response(self, **kwargs):
        request_kwargs = dict(
            url=f"{self.base_url}/responses",
            json=kwargs,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response = await self._request(method="POST", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await response.json()

    async def create_tokenize(self, **kwargs):
        base_url = self.base_url.removesuffix("/v1")
        request_kwargs = dict(
            url=f"{base_url}/tokenize",
            json=kwargs,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response = await self._request(method="POST", **request_kwargs)

        await self._raise_for_status(response, request_kwargs)
        return await response.json()
