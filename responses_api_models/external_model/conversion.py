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
# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# Adapted from https://github.com/coreweave/inference-cookbook/blob/41f991fef0d45430d20cfd8bc142c4bfc5070369/libs/cwtrain/cwtrain/integrations/nemo_rl/gym_model.py
# Original license: Apache-2.0.

"""Render exact token prompts and validate external completion training metadata."""

import json
import math
import secrets
from typing import Protocol

from responses_api_models.external_model.tool_calls import parse_tool_calls


__all__ = ["chat_response", "completion_request", "parse_tool_calls"]

_TRAINING_KEYS = ("prompt_token_ids", "generation_token_ids")
# Gym's vllm_model turns a 400 whose body names the "context length" into an empty completion that
# ends the trajectory with finish_reason "length", instead of failing the whole rollout batch.
CONTEXT_FULL = b"context length: the prompt leaves no room in the deployment's max_model_len"


class ContextFull(ValueError):
    """The prompt already fills the deployment's context, so no token can be generated."""


class Tokenizer(Protocol):
    """Local chat-template tokenizer used to append a turn without rewriting its prefix."""

    eos_token: str | None

    def apply_chat_template(self, messages: list[dict], **kwargs: object) -> list[int] | str:
        """Render messages as IDs or text according to the tokenize keyword."""
        ...

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Encode only the new suffix, without inserting special tokens."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode a token to check whether generation included EOS."""
        ...


def is_context_overflow(status: int, body: str) -> bool:
    """Whether a CoreWeave error response says that prompt plus max_tokens exceeds the context."""
    text = body.lower()
    return status == 400 and "tokens" in text and ("context length" in text or "at most" in text)


def _text(content: object) -> str:
    if content is None:
        return ""
    if isinstance(content, list):
        if any(part.get("type") != "text" for part in content):
            raise ValueError("External token training requires text-only messages")
        content = "".join(part["text"] for part in content)
    if not isinstance(content, str):
        raise ValueError("External token training requires text-only messages")
    return content


def _template_message(item: dict) -> dict:
    """The chat-template form of one Gym message: text content, tool calls with dict arguments."""
    message = {"role": item["role"], "content": _text(item.get("content"))}
    if item.get("tool_calls"):
        if item["role"] != "assistant":
            raise ValueError("Only assistant messages can carry tool calls")
        calls = []
        for call in item["tool_calls"]:
            function = call["function"]
            arguments = function.get("arguments") or {}
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments.strip() else {}
            calls.append({"type": "function", "function": {"name": function["name"], "arguments": arguments}})
        message["tool_calls"] = calls
    if item["role"] == "tool" and item.get("tool_call_id"):
        message["tool_call_id"] = item["tool_call_id"]
    return message


def _render(tokenizer: Tokenizer, messages: list, tools: object, kwargs: dict, **options: object) -> list[int] | str:
    if tools:
        options["tools"] = tools
    return tokenizer.apply_chat_template(messages, return_dict=False, **options, **kwargs)


def _previous_turn(items: list) -> int | None:
    """The index of the last assistant message that carries its exact training token IDs."""
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if item.get("role") == "assistant" and any(key in item for key in _TRAINING_KEYS):
            if not all(isinstance(item.get(key), list) for key in _TRAINING_KEYS):
                raise ValueError("Previous assistant turn requires both prompt and generated token IDs")
            return index
    return None


def _prompt_ids(body: dict, tokenizer: Tokenizer) -> list[int]:
    items = body["messages"]
    messages = [_template_message(item) for item in items]
    tools = body.get("tools") or None
    kwargs = dict(body.get("chat_template_kwargs") or {})
    index = _previous_turn(items)
    if index is None:
        prompt = _render(tokenizer, messages, tools, kwargs, tokenize=True, add_generation_prompt=True)
        if not isinstance(prompt, list):
            raise ValueError("The trainer tokenizer must return a list of integer prompt IDs")
        return prompt
    # Keep earlier turns' text as generated; the Nemotron templates otherwise rewrite them.
    kwargs.setdefault("truncate_history_thinking", False)
    head = _render(tokenizer, messages[: index + 1], tools, kwargs, tokenize=False, add_generation_prompt=False)
    full = _render(tokenizer, messages, tools, kwargs, tokenize=False, add_generation_prompt=True)
    end = getattr(tokenizer, "eos_token", None)
    if (
        not isinstance(head, str)
        or not isinstance(full, str)
        or not full.startswith(head)
        or not end
        or end not in head
    ):
        raise ValueError(
            "The chat template renders the earlier turns differently in a later turn, so the "
            "trajectory's token prefix cannot be kept"
        )
    previous = items[index]
    generated = list(previous["generation_token_ids"])
    # A turn cut off by max_tokens has no end-of-turn token; close it in the next prompt.
    closed = bool(generated) and tokenizer.decode(generated[-1:]) == end
    suffix = ("" if closed else end) + head[head.rindex(end) + len(end) :] + full[len(head) :]
    added = tokenizer.encode(suffix, add_special_tokens=False)
    return list(previous["prompt_token_ids"]) + generated + list(added)


def completion_request(
    body: dict,
    tokenizer: Tokenizer,
    *,
    max_model_len: int | None = None,
    cache_salt: str | None = None,
    seed: bool = True,
) -> dict:
    """Convert a text chat request, with optional tools, into a token-ID completions request.

    With ``max_model_len``, ``max_tokens`` is capped to the room the prompt leaves, so a long
    trajectory generates until its context is full instead of failing the request. A
    ``cache_salt`` goes into ``nvext``. With ``seed``, a request without its own seed gets a
    random one.

    Raises:
        ContextFull: The prompt leaves no room for a generated token.
        ValueError: The request has a response format, ``n`` other than 1 or non-text content, or
            a later turn cannot keep the earlier turns' exact token IDs.
    """
    if body.get("response_format") or body.get("n", 1) != 1:
        raise ValueError("External token training supports one text completion per request")
    for field in ("top_k", "min_p", "logit_bias", "continue_final_message", "stream"):
        if body.get(field):
            raise ValueError(f"External token completions do not support {field}")
    if body.get("tool_choice") not in (None, "auto"):
        raise ValueError("External token completions support only automatic tool choice")
    prompt = _prompt_ids(body, tokenizer)
    if not prompt or any(type(token) is not int or token < 0 for token in prompt):
        raise ValueError("The trainer tokenizer must return nonempty integer prompt IDs")
    result = {
        key: body[key]
        for key in (
            "model",
            "temperature",
            "top_p",
            "seed",
            "stop",
            "presence_penalty",
            "frequency_penalty",
        )
        if body.get(key) is not None
    }
    max_tokens = body.get("max_completion_tokens") or body.get("max_tokens")
    if max_model_len:
        room = int(max_model_len) - len(prompt)
        if room < 1:
            raise ContextFull(f"prompt of {len(prompt)} tokens fills max_model_len {max_model_len}")
        max_tokens = min(max_tokens or room, room)
    result.update(
        prompt=prompt,
        max_tokens=max_tokens,
        logprobs=0,
        return_tokens_as_token_ids=True,
    )
    if cache_salt:
        result["nvext"] = {"cache_salt": cache_salt}
    # As cwtrain's sampler does: a request without its own seed takes the replica's random state,
    # which every replica starts from the same default (cwtrain.integrations.rollout, "Seeds").
    if seed and "seed" not in result:
        result["seed"] = secrets.randbelow(2**62)
    return result


def chat_response(response: dict, prompt: list[int], tools: list | None = None) -> dict:
    """Convert a completions response into a chat response with token IDs and log-probs.

    Tool calls in the generated text become the message's ``tool_calls``.

    Raises:
        ValueError: The response does not hold exactly one completion, lacks exact token IDs or
            finite, aligned log-probs, or has a prompt token count that differs from ``prompt``.
    """
    choices = response.get("choices") or []
    if len(choices) != 1 or (response.get("usage") or {}).get("prompt_tokens") != len(prompt):
        raise ValueError("External endpoint must return one completion with the exact prompt token count")
    choice = choices[0]
    lp = choice.get("logprobs") or {}
    pieces, probs = lp.get("tokens"), lp.get("token_logprobs")
    if not isinstance(pieces, list) or not isinstance(probs, list) or not pieces or len(pieces) != len(probs):
        raise ValueError("External token training requires generated token IDs and aligned log-probs")
    ids = []
    for piece, prob in zip(pieces, probs, strict=True):
        if not isinstance(piece, str) or not piece.startswith("token_id:") or not piece[9:].isdigit():
            raise ValueError("External endpoint omitted exact generated token IDs")
        if type(prob) not in (int, float) or not math.isfinite(prob):
            raise ValueError("External endpoint returned a non-finite generated-token log-prob")
        ids.append(int(piece[9:]))
    if choice.get("finish_reason") not in ("stop", "length"):
        raise ValueError("External completion must finish with stop or length")
    # An incomplete XML call is task output, not an executable tool request.
    content, tool_calls = (
        parse_tool_calls(choice["text"], tools)
        if tools and choice["finish_reason"] != "length"
        else (choice["text"], [])
    )
    message = {
        "role": "assistant",
        "content": content,
        "prompt_token_ids": prompt,
        "generation_token_ids": ids,
        "generation_log_probs": probs,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return {
        **{k: response[k] for k in ("id", "created", "model", "usage")},
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls" if tool_calls else choice["finish_reason"],
                "message": message,
            }
        ],
    }
