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
"""Token-ID tracking for OpenCode rollouts used in RL training.

OpenCode is a black-box Node.js client: it drops the token IDs and logprobs that the Gym
model server attaches to each chat completion, and it re-renders the conversation history as
text on every call. RL needs both the sampled tokens and a token sequence where each model
call's prompt extends the previous call's prompt + generation.

With ``track_token_ids`` the agent points OpenCode at its own chat-completions route and
forwards every call to the policy model server. For each call it:

- finds the assistant message in the history that an earlier call produced and re-attaches
  that call's ``prompt_token_ids`` / ``generation_token_ids`` / ``generation_log_probs``, so
  the policy server pins the prompt to the exact tokens sampled before (NeMo RL's vLLM server
  uses them as ``required_prefix_token_ids``);
- records the returned token bundle and strips it from the reply OpenCode sees.

After the run, the longest parent chain of calls becomes the rollout output: per call, a user
message with the images that call newly saw (tool-produced crops/plots), then the assistant
message carrying the token bundle, then its function calls. Side calls outside that chain (e.g.
a session-title request) are dropped.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from nemo_gym.openai_utils import (
    NeMoGymMessage,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessageForTraining,
    NeMoGymResponseOutputText,
)


TOKEN_KEYS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")
_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class TrackedCall:
    parent: Optional[int]
    depth: int
    tokens: dict[str, list]
    message: dict[str, Any]
    # Image URLs in the request that the prompt adds on top of the parent's prompt + generation
    # (for a root call: every image in the request).
    new_images: list[str] = field(default_factory=list)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def _visible_text(content: Any) -> str:
    return _THINK_RE.sub("", _text(content)).strip()


def _tool_call_ids(message: dict[str, Any]) -> list[str]:
    return [call.get("id") for call in message.get("tool_calls") or [] if isinstance(call, dict)]


def _same_assistant(history_message: dict[str, Any], produced: dict[str, Any]) -> bool:
    """Whether an assistant message in a request is the message an earlier call returned."""
    produced_ids = _tool_call_ids(produced)
    if produced_ids:
        return _tool_call_ids(history_message) == produced_ids
    return not _tool_call_ids(history_message) and _visible_text(history_message.get("content")) == _visible_text(
        produced.get("content")
    )


def _image_urls(messages: list[dict[str, Any]]) -> list[str]:
    urls = []
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        for part in content if isinstance(content, list) else []:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            url = image_url.get("url") if isinstance(image_url, dict) else image_url
            if isinstance(url, str):
                urls.append(url)
    return urls


def attach_parent_tokens(messages: list[dict[str, Any]], calls: list[TrackedCall]) -> tuple[Optional[int], list[str]]:
    """Re-attach the producing call's token bundle to the last assistant message.

    Returns (parent call index or None, image URLs the prompt adds beyond the parent's tokens).
    """
    last = next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "assistant"), None)
    if last is None:
        return None, _image_urls(messages)
    for index in range(len(calls) - 1, -1, -1):
        if _same_assistant(messages[last], calls[index].message):
            messages[last].update(calls[index].tokens)
            return index, _image_urls(messages[last + 1 :])
    return None, _image_urls(messages)


def record_call(
    calls: list[TrackedCall], parent: Optional[int], new_images: list[str], message: dict[str, Any]
) -> bool:
    """Pop the token bundle off a returned assistant message and record the call.

    Returns False (nothing recorded) when the reply carries no sampled tokens, e.g. the empty
    completion the model server returns for an over-length prompt.
    """
    tokens = {key: message.pop(key) for key in TOKEN_KEYS if key in message}
    message.pop("routed_experts", None)
    if len(tokens) != len(TOKEN_KEYS) or not tokens["generation_token_ids"]:
        return False
    depth = calls[parent].depth + 1 if parent is not None else 0
    calls.append(TrackedCall(parent=parent, depth=depth, tokens=tokens, message=message, new_images=new_images))
    return True


def main_chain(calls: list[TrackedCall]) -> list[TrackedCall]:
    """The deepest parent chain (latest on ties), root first."""
    if not calls:
        return []
    index: Optional[int] = max(range(len(calls)), key=lambda i: (calls[i].depth, i))
    chain = []
    while index is not None:
        chain.append(calls[index])
        index = calls[index].parent
    return chain[::-1]


def chain_output_items(calls: list[TrackedCall], num_input_images: int) -> list[Any]:
    """Rollout output items, with token IDs on one assistant message per model call.

    The root call's first ``num_input_images`` images are the task's input images, which the
    trainer already takes from ``responses_create_params``; only later images are emitted.
    """
    items: list[Any] = []
    for position, call in enumerate(main_chain(calls)):
        images = call.new_images[num_input_images:] if position == 0 else call.new_images
        if images:
            items.append(
                NeMoGymMessage(
                    type="message",
                    role="user",
                    content=[{"type": "input_image", "image_url": url, "detail": "auto"} for url in images],
                )
            )
        items.append(
            NeMoGymResponseOutputMessageForTraining(
                id=f"msg_{uuid4().hex}",
                content=[
                    NeMoGymResponseOutputText(
                        type="output_text", text=_visible_text(call.message.get("content")), annotations=[]
                    )
                ],
                role="assistant",
                status="completed",
                type="message",
                **call.tokens,
            )
        )
        for tool_call in call.message.get("tool_calls") or []:
            function = tool_call.get("function") or {}
            call_id = tool_call.get("id") or f"call_{uuid4().hex[:8]}"
            items.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=function.get("arguments") or "",
                    call_id=call_id,
                    name=function.get("name") or "",
                    type="function_call",
                    id=call_id,
                    status="completed",
                )
            )
    return items
