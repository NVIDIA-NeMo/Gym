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
"""NeMo Gym Pydantic types for the Anthropic Messages API boundary.

This mirrors the wrapping strategy in ``nemo_gym/openai_utils.py``, but deliberately stays
small. Anthropic is only a *wire format* at the proxy boundary — the internal canonical
representation is the Responses API (the ``NeMoGym*`` types in ``openai_utils``). So unlike
``openai_utils``, this module does not replicate the ``*ForTraining`` hierarchies or an async
client (the ``anthropic`` SDK is never used as a client — it uses httpx, whose O(n^2)
connection pooling hangs at high concurrency). Types only.

Two boundary types, one per direction:

* :class:`NeMoGymAnthropicMessageCreateParamsNonStreaming` — the **request** validator. The
  SDK's ``MessageCreateParams`` is a ``TypedDict`` with no runtime validation, so we copy it as
  a ``BaseModel`` for strict server-side validation at the ingress proxy endpoint, exactly as
  ``NeMoGymResponseCreateParamsNonStreaming`` does for the Responses API.
* :class:`NeMoGymAnthropicMessage` — the **response** validator. A thin subclass of the SDK's
  ``Message`` (already a ``BaseModel``), used to validate what we emit, mirroring
  ``NeMoGymResponse(Response)``.
"""

from typing import List, Literal, Optional, Union

from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    MetadataParam,
    ModelParam,
    OutputConfigParam,
    TextBlockParam,
    ThinkingConfigParam,
    ToolChoiceParam,
    ToolUnionParam,
)
from anthropic.types import Message as AnthropicMessage
from pydantic import BaseModel, ConfigDict, Field


########################################
# Messages API request
########################################


class NeMoGymAnthropicMessageCreateParamsNonStreaming(BaseModel):
    """Copy of ``anthropic.types.message_create_params.MessageCreateParamsBase`` as a BaseModel.

    The SDK ships this as a ``TypedDict`` (no runtime validation). We need server-side
    validation at the ingress proxy, so we re-declare it here, mirroring
    ``NeMoGymResponseCreateParamsNonStreaming``.

    The ``Iterable`` fields (``messages``, ``tools``, and the list arm of ``system``) are
    overridden to ``List`` so Pydantic eagerly validates them into real, re-iterable,
    indexable, JSON-serializable lists rather than single-use lazy ``ValidatorIterator``s.

    Note on ``extra="forbid"``: this matches the strict Responses-API policy and rejects
    unknown fields. Anthropic occasionally introduces beta body fields ahead of an SDK bump;
    if the ingress client (e.g. the Claude Code CLI) sends one, relax this to ``ignore``.
    """

    model_config = ConfigDict(extra="forbid")

    # Required by the Anthropic API.
    max_tokens: int
    messages: List[MessageParam]
    model: ModelParam

    cache_control: Optional[CacheControlEphemeralParam] = None
    container: Optional[str] = None
    inference_geo: Optional[str] = None
    metadata: Optional[MetadataParam] = None
    output_config: Optional[OutputConfigParam] = None
    service_tier: Optional[Literal["auto", "standard_only"]] = None
    stop_sequences: Optional[List[str]] = None
    system: Optional[Union[str, List[TextBlockParam]]] = None
    temperature: Optional[float] = None
    thinking: Optional[ThinkingConfigParam] = None
    tool_choice: Optional[ToolChoiceParam] = None
    tools: Optional[List[ToolUnionParam]] = Field(default=None)
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    # We synthesize SSE from a complete response; we never proxy a true upstream stream.
    stream: Optional[Literal[False]] = None


########################################
# Messages API response
########################################


class NeMoGymAnthropicMessage(AnthropicMessage):
    """Thin subclass of the SDK's ``Message`` response model, used to validate what we emit.

    ``Message.content`` is already typed as ``List[...]`` in the pinned ``anthropic`` version,
    so no iterable override is needed here; the subclass exists for symmetry with
    ``NeMoGymResponse`` and to give the egress/ingress response path a single ``NeMoGym*``
    validation point.
    """

    pass
