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
"""NeMo Gym wrappers for the Anthropic Messages API (``POST /v1/messages``).

This mirrors ``nemo_gym/openai_utils.py``: we copy the Anthropic SDK schemas into
NeMo Gym ``BaseModel`` / ``TypedDict`` subclasses so that we can do strict server-side
validation, override the SDK's lazy ``Iterable`` annotations with ``List`` (which
Pydantic validates eagerly), and attach the training mixins (token IDs + log probs).

Types are prefixed ``NeMoGymAnthropic*`` to avoid colliding with the OpenAI Responses
API wrappers in ``openai_utils.py`` (e.g. ``NeMoGymMessage`` already exists there).
"""

from typing import (
    List,
    Literal,
    Optional,
    TypeAlias,
    Union,
)

from anthropic.types import (
    CacheControlEphemeralParam,
    DocumentBlockParam,
    ImageBlockParam,
    Message,
    MetadataParam,
    ModelParam,
    RedactedThinkingBlock,
    RedactedThinkingBlockParam,
    TextBlock,
    TextBlockParam,
    ThinkingBlock,
    ThinkingBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolResultBlockParam,
    ToolUnionParam,
    ToolUseBlock,
    ToolUseBlockParam,
    Usage,
)
from anthropic.types.message_create_params import OutputConfigParam
from anthropic.types.tool_result_block_param import Content as ToolResultContentBlockParam
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired, Required, TypedDict

from nemo_gym.openai_utils import (
    TokenIDLogProbMixin,
    TokenIDLogProbTypedDictMixin,
)


########################################
# Messages API inputs (request params)
########################################

# We deliberately narrow the input content-block surface to the blocks that show up
# in text + tool-calling rollouts (mirroring how openai_utils only wraps text/image
# content parts). Add more blocks here if an environment needs them.


class NeMoGymAnthropicTextBlockParam(TextBlockParam):
    pass


class NeMoGymAnthropicImageBlockParam(ImageBlockParam):
    pass


class NeMoGymAnthropicDocumentBlockParam(DocumentBlockParam):
    pass


class NeMoGymAnthropicToolUseBlockParam(ToolUseBlockParam):
    pass


class NeMoGymAnthropicToolResultBlockParam(ToolResultBlockParam):
    # Override the SDK's Iterable content annotation with List so Pydantic
    # materializes it eagerly instead of leaving a lazy ValidatorIterator
    # (same reason openai_utils overrides Iterable -> List throughout).
    content: NotRequired[Union[str, List[ToolResultContentBlockParam]]]


class NeMoGymAnthropicThinkingBlockParam(ThinkingBlockParam):
    pass


class NeMoGymAnthropicRedactedThinkingBlockParam(RedactedThinkingBlockParam):
    pass


NeMoGymAnthropicContentBlockParam: TypeAlias = Union[
    NeMoGymAnthropicTextBlockParam,
    NeMoGymAnthropicImageBlockParam,
    NeMoGymAnthropicDocumentBlockParam,
    NeMoGymAnthropicToolUseBlockParam,
    NeMoGymAnthropicToolResultBlockParam,
    NeMoGymAnthropicThinkingBlockParam,
    NeMoGymAnthropicRedactedThinkingBlockParam,
]


class NeMoGymAnthropicMessageParam(TypedDict):
    # Override the SDK's Iterable content annotation with List to avoid lazy
    # iterators in Pydantic validation.
    content: Required[Union[str, List[NeMoGymAnthropicContentBlockParam]]]
    role: Required[Literal["user", "assistant", "system"]]


class NeMoGymAnthropicMessageForTrainingParam(NeMoGymAnthropicMessageParam, TokenIDLogProbTypedDictMixin):
    pass


NeMoGymAnthropicMessageParamUnion: TypeAlias = Union[
    NeMoGymAnthropicMessageParam,
    # For training (assistant turns carry token IDs + log probs):
    NeMoGymAnthropicMessageForTrainingParam,
]


class NeMoGymAnthropicMessageCreateParamsNonStreaming(BaseModel):
    """A copy of ``anthropic.types.message_create_params.MessageCreateParamsNonStreaming``.

    The SDK type is a ``TypedDict`` with no strict validation; we copy it here as a
    Pydantic ``BaseModel`` (``extra="forbid"``) so the model server validates request
    bodies server-side, the same way ``NeMoGymResponseCreateParamsNonStreaming`` does.
    """

    model_config = ConfigDict(extra="forbid")

    max_tokens: int
    # Override the Iterable to avoid lazy iterators in Pydantic validation.
    messages: List[NeMoGymAnthropicMessageParamUnion]
    # model is optional here so the model server can fill it in from its own config,
    # matching NeMoGymResponseCreateParamsNonStreaming.
    model: Optional[ModelParam] = None
    cache_control: Optional[CacheControlEphemeralParam] = None
    container: Optional[str] = None
    inference_geo: Optional[str] = None
    metadata: Optional[MetadataParam] = None
    output_config: Optional[OutputConfigParam] = None
    service_tier: Optional[Literal["auto", "standard_only"]] = None
    stop_sequences: Optional[List[str]] = None
    system: Optional[Union[str, List[NeMoGymAnthropicTextBlockParam]]] = None
    temperature: Optional[float] = None
    thinking: Optional[ThinkingConfigParam] = None
    tool_choice: Optional[ToolChoiceParam] = None
    # Override the Iterable to avoid lazy iterators in Pydantic validation.
    tools: List[ToolUnionParam] = Field(default_factory=list)
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    stream: Optional[Literal[False]] = None


########################################
# Messages API outputs (response)
########################################


class NeMoGymAnthropicTextBlock(TextBlock):
    pass


class NeMoGymAnthropicToolUseBlock(ToolUseBlock):
    pass


class NeMoGymAnthropicThinkingBlock(ThinkingBlock):
    pass


class NeMoGymAnthropicRedactedThinkingBlock(RedactedThinkingBlock):
    pass


NeMoGymAnthropicContentBlock: TypeAlias = Union[
    NeMoGymAnthropicTextBlock,
    NeMoGymAnthropicToolUseBlock,
    NeMoGymAnthropicThinkingBlock,
    NeMoGymAnthropicRedactedThinkingBlock,
]


class NeMoGymAnthropicUsage(Usage):
    pass


class NeMoGymAnthropicMessage(Message):
    # Override the Iterable to avoid lazy iterators in Pydantic validation.
    content: List[NeMoGymAnthropicContentBlock]
    usage: NeMoGymAnthropicUsage


class NeMoGymAnthropicMessageForTraining(NeMoGymAnthropicMessage, TokenIDLogProbMixin):
    pass
