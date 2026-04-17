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
"""ChatCompletionsClient subclass with dynamic max_completion_tokens + graceful length-finish handling.

Stirrup's ``ChatCompletionsClient.generate()`` has two behaviours
that break on long-context models served by vLLM:

1. It sends ``max_completion_tokens = self._max_tokens`` with every call —
   a static value that does not account for the input size.  When the
   prompt consumes a non-trivial fraction of the model's context window,
   the server can return ``finish_reason=length`` with zero output tokens.

2. On any ``finish_reason in ("max_tokens", "length")`` it raises
   ``ContextOverflowError`` unconditionally, even when the response has
   valid partial content.  For reasoning models whose traces can be
   genuinely long, this turns a normal "ran out of output budget" event
   into a fatal error.

This subclass addresses both.  Before each call we tokenize the messages
and size ``max_completion_tokens`` as::

    context_window − tokenized(messages) − completion_token_buffer

clamped to a minimum of ``_MIN_COMPLETION_TOKENS``.  On the response
side, we replicate Stirrup parsing but do *not* raise on
``finish_reason=length`` — the agent loop will either terminate when the
model invokes the ``finish`` tool or exhaust ``max_turns``, yielding a
clean timeout instead of a crash.

``model_id`` selects the HuggingFace tokenizer (or local checkpoint path).
When unset, a conservative character-count fallback is used.
"""

from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Any, Optional

from openai import AsyncOpenAI
from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.clients.utils import to_openai_messages, to_openai_tools
from stirrup.core.models import (
    AssistantMessage,
    ChatMessage,
    Reasoning,
    SystemMessage,
    TokenUsage,
    Tool,
    ToolCall,
    ToolMessage,
    UserMessage,
)

LOGGER = logging.getLogger(__name__)

# Floor for per-call max_completion_tokens.  Below this the model basically
# cannot produce a useful answer — treat as a hard minimum.
_MIN_COMPLETION_TOKENS = 1024


def _load_tokenizer(model_id: Optional[str]):
    """Load a HuggingFace tokenizer, tolerating version differences in transformers."""
    if not model_id:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        LOGGER.warning(
            "transformers is not installed; dynamic max_tokens sizing will use "
            "a character-count fallback.  `pip install transformers` to enable."
        )
        return None
    # Some tokenizers (Mistral family) expose a ``fix_mistral_regex`` kwarg.
    # Try the richer call first, fall back to the common signature.
    for kwargs in (
        {"use_fast": True, "trust_remote_code": True, "fix_mistral_regex": True},
        {"use_fast": True, "trust_remote_code": True},
    ):
        try:
            return AutoTokenizer.from_pretrained(model_id, **kwargs)
        except TypeError:
            continue
        except Exception as exc:
            LOGGER.warning(f"Failed to load tokenizer for {model_id!r}: {exc}")
            return None
    return None


def _format_for_chat_template(m: ChatMessage) -> dict[str, str]:
    """Render a Stirrup ChatMessage into the OpenAI chat-template dict shape."""
    if isinstance(m, UserMessage):
        return {"role": "user", "content": m.content or ""}
    if isinstance(m, SystemMessage):
        return {"role": "system", "content": m.content or ""}
    if isinstance(m, ToolMessage):
        return {"role": "tool", "content": m.content or ""}
    # AssistantMessage and any other — best-effort to text.
    content = getattr(m, "content", "") or ""
    return {"role": "assistant", "content": content if isinstance(content, str) else str(content)}


class DynamicMaxTokensChatCompletionsClient(ChatCompletionsClient):
    """ChatCompletionsClient that sizes max_completion_tokens per call and
    does not raise on a length-finish response."""

    def __init__(
        self,
        *args: Any,
        model_id: Optional[str] = None,
        completion_token_buffer: int = 1000,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._completion_token_buffer = completion_token_buffer
        self._tokenizer = _load_tokenizer(model_id)
        if model_id and self._tokenizer is None:
            LOGGER.warning(
                f"model_id={model_id!r} provided but tokenizer could not be loaded. "
                "Dynamic max_tokens will use a character-count fallback."
            )

    # ------------------------------------------------------------------
    # Token counting
    # ------------------------------------------------------------------

    def _count_input_tokens(
        self,
        messages: list[ChatMessage],
        tools: Optional[dict[str, Tool]] = None,
    ) -> int:
        """Estimate the full prompt token count the server will see.

        We use Stirrup's own ``to_openai_messages``/``to_openai_tools``
        converters to produce the exact payload that gets serialised to the
        wire, then run the chat template over it.  This catches assistant
        ``tool_calls``, multimodal content blocks, and tool-schema injection.

        Counting strategy (in order, best -> worst):

        1. ``tokenizer.apply_chat_template(messages, tools=…)`` — ideal,
           but some chat templates don't support the ``tools`` kwarg.
        2. ``tokenizer.apply_chat_template(messages)`` + tokenise the tool
           JSON blob separately — still captures assistant ``tool_calls``
           via the chat template.
        3. Tokenise the JSON of ``to_openai_messages(..)`` and the tools
           blob — rough but serialises everything.
        4. Character-count fallback when no tokenizer is present.

        Any residual gap is absorbed by ``completion_token_buffer``.
        """
        import json as _json

        if self._tokenizer is None:
            # Pure character-count fallback.
            formatted = [_format_for_chat_template(m) for m in messages]
            total = sum(len(f.get("content") or "") for f in formatted) // 3
            if tools:
                try:
                    total += len(_json.dumps(to_openai_tools(tools))) // 3
                except Exception:
                    pass
            return total

        try:
            oai_messages = to_openai_messages(messages)
        except Exception as exc:
            LOGGER.warning(f"to_openai_messages failed ({exc}); falling back to legacy formatting.")
            oai_messages = [_format_for_chat_template(m) for m in messages]

        oai_tools = None
        if tools:
            try:
                oai_tools = to_openai_tools(tools)
            except Exception as exc:
                LOGGER.warning(f"to_openai_tools failed ({exc}).")

        # Strategy 1: apply_chat_template with tools=
        if oai_tools is not None:
            try:
                text = self._tokenizer.apply_chat_template(
                    oai_messages, tools=oai_tools, tokenize=False, add_generation_prompt=True
                )
                return len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
            except Exception as exc:
                LOGGER.debug(f"apply_chat_template(tools=) unsupported ({exc}); trying separate tool count.")

        # Strategy 2: apply_chat_template on messages only + separate tool JSON count
        try:
            text = self._tokenizer.apply_chat_template(
                oai_messages, tokenize=False, add_generation_prompt=True
            )
            total = len(self._tokenizer(text, add_special_tokens=False)["input_ids"])
            if oai_tools is not None:
                total += len(
                    self._tokenizer(_json.dumps(oai_tools), add_special_tokens=False)["input_ids"]
                )
            return total
        except Exception as exc:
            LOGGER.warning(f"apply_chat_template(messages) failed ({exc}); falling back to JSON count.")

        # Strategy 3: tokenise the full JSON payload
        try:
            blob = _json.dumps(oai_messages)
            total = len(self._tokenizer(blob, add_special_tokens=False)["input_ids"])
            if oai_tools is not None:
                total += len(
                    self._tokenizer(_json.dumps(oai_tools), add_special_tokens=False)["input_ids"]
                )
            return total
        except Exception as exc:
            LOGGER.warning(f"JSON tokenisation failed ({exc}); falling back to character count.")

        # Strategy 4: character count
        formatted = [_format_for_chat_template(m) for m in messages]
        total = sum(len(f.get("content") or "") for f in formatted) // 3
        return total

    async def generate(
        self,
        messages: list[ChatMessage],
        tools: dict[str, Tool],
    ) -> AssistantMessage:
        input_tokens = self._count_input_tokens(messages, tools)
        context_window = self._max_tokens
        dynamic_max = max(
            context_window - input_tokens - self._completion_token_buffer,
            _MIN_COMPLETION_TOKENS,
        )

        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": to_openai_messages(messages),
            "max_completion_tokens": dynamic_max,
            **self._kwargs,
        }
        if tools:
            request_kwargs["tools"] = to_openai_tools(tools)
            request_kwargs["tool_choice"] = "auto"
        if self._reasoning_effort:
            request_kwargs["reasoning_effort"] = self._reasoning_effort

        if LOGGER.isEnabledFor(logging.DEBUG):
            _msgs = request_kwargs["messages"]
            _tools = request_kwargs.get("tools") or []
            LOGGER.debug(
                "request: n_messages=%d first_role=%s last_role=%s "
                "msg_content_chars=%d n_tools=%d model=%r max_completion_tokens=%d",
                len(_msgs),
                _msgs[0].get("role") if _msgs else "?",
                _msgs[-1].get("role") if _msgs else "?",
                sum(len(str(m.get("content") or "")) for m in _msgs),
                len(_tools),
                request_kwargs.get("model"),
                request_kwargs.get("max_completion_tokens"),
            )

        request_start_time = perf_counter()
        try:
            response = await self._client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            LOGGER.error("API call raised %s: %s", type(exc).__name__, exc)
            raise
        request_end_time = perf_counter()

        choice = response.choices[0]
        msg = choice.message
        usage = response.usage
        prompt_tokens = usage.prompt_tokens if usage else 0
        completion_tokens = usage.completion_tokens if usage else 0
        reasoning_tokens = 0
        if usage and hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
            reasoning_tokens = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
        answer_tokens = completion_tokens - reasoning_tokens

        LOGGER.debug(
            "response: input_est=%d ctx=%d buf=%d -> max_completion=%d | "
            "actual prompt=%d completion=%d (reasoning=%d) finish=%s "
            "content_len=%d tool_calls=%d",
            input_tokens,
            context_window,
            self._completion_token_buffer,
            dynamic_max,
            prompt_tokens,
            completion_tokens,
            reasoning_tokens,
            choice.finish_reason,
            len(msg.content or ""),
            len(msg.tool_calls or []),
        )

        # Upstream raises ContextOverflowError on length/max_tokens; we don't.
        # The agent loop handles termination either via the finish tool or max_turns.

        reasoning: Optional[Reasoning] = None
        if hasattr(msg, "reasoning_content") and msg.reasoning_content:
            reasoning = Reasoning(content=msg.reasoning_content)

        tool_calls = [
            ToolCall(
                tool_call_id=tc.id,
                name=tc.function.name,
                arguments=tc.function.arguments or "",
            )
            for tc in (msg.tool_calls or [])
        ]

        return AssistantMessage(
            reasoning=reasoning,
            content=msg.content or "",
            tool_calls=tool_calls,
            token_usage=TokenUsage(
                input=prompt_tokens,
                answer=answer_tokens,
                reasoning=reasoning_tokens,
            ),
            request_start_time=request_start_time,
            request_end_time=request_end_time,
        )
