# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stirrup chat-completions client for the RL policy behind Gym's model server.

Differences from ``stirrup_agent.nemo_client.DynamicMaxTokensChatCompletionsClient``:

* **Per-call budget without a tokenizer.** The Gym model server reports exact
  ``usage`` on every reply, so the next call's prompt size is estimated as
  *last prompt + last completion + chars-added-since / 3* instead of running a
  HF tokenizer over the whole history on the agent server (thousands of
  concurrent rollouts share a few CPU cores there). The first call uses the
  character-count fallback.
* **Context exhaustion is a typed error.** A 400 from the server about the
  model's maximum context length, or an empty ``finish_reason=length`` reply
  when the budget already sat at the floor, raises
  :class:`PolicyContextLimitError` so the agent server can report
  ``agent_context_limit`` (masked in training when configured) instead of
  looping on an empty assistant turn.
* **Request log.** Every call's provider-serialized messages and the reply
  message are kept (``self.calls``) so the agent server can export the
  trajectory even when the Stirrup loop ends in an exception, and the served
  response id of the last call (``self.last_response_id``) is reused as the
  Gym response id for token-capture terminal attribution.

Read timeout and retries ride on the OpenAI SDK (``timeout`` / ``max_retries``):
a stalled policy call is re-issued statelessly, like the OpenHands client did.
"""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any, Optional

from stirrup.clients.chat_completions_client import ChatCompletionsClient
from stirrup.clients.utils import to_openai_tools
from stirrup.core.exceptions import ContextOverflowError
from stirrup.core.models import AssistantMessage, ChatMessage, Reasoning, TokenUsage, Tool, ToolCall

# Reuse the Gym-side serialization (restores tool-role messages for the provider)
# and the module-import-time stirrup patches (coercing finish tool, tool-arg error
# surfacing) that the eval harness also runs with.
from responses_api_agents.stirrup_agent import nemo_client as _nemo_client  # noqa: F401  (patches on import)
from responses_api_agents.stirrup_agent.stirrup_utils import to_provider_openai_messages

LOGGER = logging.getLogger(__name__)

_CONTEXT_ERROR_MARKERS = (
    "maximum context length",
    "context length",
    "context_length_exceeded",
    "exceeds the model's max",
    "prompt is too long",
    "too many tokens",
)


class PolicyContextLimitError(ContextOverflowError):
    """The policy's context window is exhausted for this rollout."""


def sanitize_provider_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strict OpenAI chat-completions shapes for Gym's policy model server.

    Stirrup's provider serialization dumps its internal ``ToolCall`` model into each
    assistant ``tool_calls`` entry (``name``/``arguments``/``tool_call_id``/``signature``
    beside ``id``/``type``/``function``), adds ``metadata`` / ``thinking_blocks`` on
    assistant turns and ``name`` on tool results. Gym validates request bodies with
    ``extra=forbid`` and rejected every second policy call of the first profiling
    smoke, so keep exactly the OpenAI fields.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "assistant":
            clean: dict[str, Any] = {"role": "assistant", "content": message.get("content")}
            if message.get("reasoning_content"):
                clean["reasoning_content"] = message["reasoning_content"]
            if message.get("tool_calls"):
                clean["tool_calls"] = [
                    {
                        "id": tc.get("id") or tc.get("tool_call_id"),
                        "type": "function",
                        "function": {
                            "name": (tc.get("function") or {}).get("name", tc.get("name")),
                            "arguments": (tc.get("function") or {}).get("arguments", tc.get("arguments")) or "",
                        },
                    }
                    for tc in message["tool_calls"]
                ]
            out.append(clean)
        elif role == "tool":
            out.append({"role": "tool", "content": message.get("content"), "tool_call_id": message.get("tool_call_id")})
        else:
            out.append({"role": role, "content": message.get("content")})
    return out


def _message_chars(messages: list[dict[str, Any]]) -> int:
    total = 0
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    total += len(str(part.get("text") or ""))
                else:
                    total += len(str(part))
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function") or {}
            total += len(str(fn.get("name") or "")) + len(str(fn.get("arguments") or ""))
        total += len(str(m.get("reasoning_content") or ""))
    return total


def _flatten_content(message: dict[str, Any]) -> dict[str, Any]:
    """Stirrup serializes text content as ``[{"type": "text", "text": ...}]`` parts.

    The exported Gym response (judge, logs, capture receipt) wants plain strings;
    the wire format sent to the policy is left untouched.
    """
    content = message.get("content")
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in ("text", "input_text", "output_text") or "text" in part:
                    parts.append(str(part.get("text") or ""))
                else:
                    parts.append(f"[{part.get('type', 'content')}]")
            else:
                parts.append(str(part))
        message["content"] = "".join(parts)
    return message


def _is_context_error(exc: BaseException) -> bool:
    status = getattr(exc, "status_code", None)
    text = str(exc).lower()
    if status == 400 and any(marker in text for marker in _CONTEXT_ERROR_MARKERS):
        return True
    return any(marker in text for marker in _CONTEXT_ERROR_MARKERS[:3])


class GDPValPolicyClient(ChatCompletionsClient):
    """ChatCompletionsClient with usage-based max_completion_tokens sizing and a request log."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        context_window: int,
        max_completion_tokens_cap: int,
        min_completion_tokens: int = 1024,
        completion_token_buffer: int = 1000,
        temperature: float = 1.0,
        top_p: float = 1.0,
        enable_thinking: bool = True,
        chat_template_kwargs_transport: str = "metadata",
        read_timeout_s: float = 600.0,
        read_retries: int = 1,
        api_key: str = "dummy",
        extra_body: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model=model,
            max_tokens=context_window,
            base_url=base_url,
            api_key=api_key,
            timeout=read_timeout_s,
            max_retries=read_retries,
        )
        self._context_window = int(context_window)
        self._cap = int(max_completion_tokens_cap)
        self._floor = int(min_completion_tokens)
        self._buffer = int(completion_token_buffer)
        self._temperature = temperature
        self._top_p = top_p
        self._enable_thinking = enable_thinking
        self._extra_body = dict(extra_body or {})
        if chat_template_kwargs_transport not in ("metadata", "extra_body"):
            raise ValueError("chat_template_kwargs_transport must be 'metadata' (Gym model server) or 'extra_body' (raw vLLM)")
        self._chat_template_kwargs_transport = chat_template_kwargs_transport
        # Running estimate state.
        self._last_prompt_tokens: Optional[int] = None
        self._last_completion_tokens: int = 0
        self._last_chars: int = 0
        # Request/response log.
        self.calls: list[dict[str, Any]] = []
        self.last_response_id: Optional[str] = None
        self.length_finishes = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    # ------------------------------------------------------------------
    # Budget
    # ------------------------------------------------------------------

    def _estimate_prompt_tokens(self, provider_messages: list[dict[str, Any]], tools: dict[str, Tool]) -> int:
        chars = _message_chars(provider_messages)
        if self._last_prompt_tokens is None:
            est = chars // 3
            if tools:
                try:
                    est += len(json.dumps(to_openai_tools(tools))) // 3
                except Exception:
                    pass
            return est
        delta = max(0, chars - self._last_chars)
        return self._last_prompt_tokens + self._last_completion_tokens + delta // 3

    def _budget(self, prompt_estimate: int) -> tuple[int, bool]:
        dynamic = self._context_window - prompt_estimate - self._buffer
        at_floor = dynamic <= self._floor
        return min(max(dynamic, self._floor), self._cap), at_floor

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _template_kwargs_request_fields(self) -> dict[str, Any]:
        """How enable_thinking (and any extra body) reaches the server.

        Gym's policy model server forbids unknown top-level body fields and instead
        merges ``metadata.chat_template_kwargs`` / ``metadata.extra_body`` (JSON strings)
        into the vLLM request; a raw vLLM endpoint takes them as ``extra_body``.
        """
        template_kwargs = {"enable_thinking": self._enable_thinking}
        if self._chat_template_kwargs_transport == "extra_body":
            return {"extra_body": {"chat_template_kwargs": template_kwargs} | self._extra_body}
        metadata = {"chat_template_kwargs": json.dumps(template_kwargs)}
        if self._extra_body:
            metadata["extra_body"] = json.dumps(self._extra_body)
        return {"metadata": metadata}

    async def generate(self, messages: list[ChatMessage], tools: dict[str, Tool]) -> AssistantMessage:
        provider_messages = sanitize_provider_messages(to_provider_openai_messages(messages))
        prompt_estimate = self._estimate_prompt_tokens(provider_messages, tools)
        max_completion, at_floor = self._budget(prompt_estimate)

        request_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": provider_messages,
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_completion_tokens": max_completion,
            **self._template_kwargs_request_fields(),
            **self._kwargs,
        }
        oai_tools = to_openai_tools(tools) if tools else None
        if oai_tools:
            request_kwargs["tools"] = oai_tools
            request_kwargs["tool_choice"] = "auto"

        start = perf_counter()
        try:
            response = await self._client.chat.completions.create(**request_kwargs)
        except Exception as exc:
            if _is_context_error(exc):
                self.calls.append(
                    {
                        "messages": provider_messages,
                        "tools": oai_tools,
                        "response": None,
                        "error": f"{type(exc).__name__}: {exc}"[:2000],
                        "prompt_estimate": prompt_estimate,
                        "max_completion_tokens": max_completion,
                        "seconds": perf_counter() - start,
                    }
                )
                raise PolicyContextLimitError(str(exc)[:500]) from exc
            raise
        end = perf_counter()

        choice = response.choices[0]
        msg = choice.message
        usage = response.usage
        prompt_tokens = (usage.prompt_tokens if usage else None) or 0
        completion_tokens = (usage.completion_tokens if usage else None) or 0
        reasoning_tokens = 0
        details = getattr(usage, "completion_tokens_details", None) if usage else None
        if details is not None:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0

        # Update the running estimate from the server's own accounting.
        if usage and prompt_tokens:
            self._last_prompt_tokens = prompt_tokens
            self._last_completion_tokens = completion_tokens
            self._last_chars = _message_chars(provider_messages)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.last_response_id = getattr(response, "id", None) or self.last_response_id

        message_dict = msg.model_dump(exclude_none=True) if hasattr(msg, "model_dump") else dict(msg)
        self.calls.append(
            {
                "response_id": getattr(response, "id", None),
                "messages": provider_messages,
                "tools": oai_tools,
                "response": {
                    "message": message_dict,
                    "finish_reason": choice.finish_reason,
                    "usage": usage.model_dump(exclude_none=True) if usage is not None else None,
                },
                "prompt_estimate": prompt_estimate,
                "max_completion_tokens": max_completion,
                "seconds": end - start,
            }
        )

        empty = not (msg.content or msg.tool_calls)
        if choice.finish_reason in ("length", "max_tokens"):
            self.length_finishes += 1
            if empty and (at_floor or prompt_tokens + completion_tokens >= self._context_window - self._buffer):
                raise PolicyContextLimitError(
                    f"empty length stop with prompt={prompt_tokens} completion={completion_tokens} "
                    f"budget={max_completion} (context window {self._context_window})"
                )

        reasoning: Optional[Reasoning] = None
        reasoning_text = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        if reasoning_text:
            reasoning = Reasoning(content=reasoning_text)

        tool_calls = [
            ToolCall(tool_call_id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "")
            for tc in (msg.tool_calls or [])
        ]
        return AssistantMessage(
            reasoning=reasoning,
            content=msg.content or "",
            tool_calls=tool_calls,
            token_usage=TokenUsage(
                input=prompt_tokens,
                answer=max(0, completion_tokens - reasoning_tokens),
                reasoning=reasoning_tokens,
            ),
            request_start_time=start,
            request_end_time=end,
        )

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def final_messages(self) -> list[dict[str, Any]]:
        """Chat-completions message list of the whole conversation.

        The last call's request messages plus its reply as an assistant message
        (the OpenHands exporter's convention). Calls that ended in a context
        error contribute their request messages only.
        """
        if not self.calls:
            return []
        last = self.calls[-1]
        messages = [_flatten_content(dict(m)) for m in last["messages"]]
        reply = (last.get("response") or {}).get("message")
        if reply:
            messages.append(_assistant_message(reply))
        return messages

    # ------------------------------------------------------------------
    # Context-compaction segments (NeMo RL multi-trace training)
    # ------------------------------------------------------------------

    def segments(self) -> list[dict[str, Any]]:
        """Group completed calls into context segments.

        Stirrup summarizes by sending the whole history plus its summarizer
        instruction as one policy call, then continues from ``task prefix +
        summary bridge``. NeMo RL trains each context as its own trace, so the
        calls are split into: the pre-summary segment, the single summary call
        (``segment_boundary_reason="compaction"``), and the continuation
        (``"post_compaction"``). Calls without a reply (context errors) are
        dropped; they produced no tokens.
        """
        from stirrup.prompts import MESSAGE_SUMMARIZER

        summarizer = MESSAGE_SUMMARIZER.strip()
        completed = [(i, c) for i, c in enumerate(self.calls) if (c.get("response") or {}).get("message")]
        segments: list[dict[str, Any]] = []
        for position, call in completed:
            last = call["messages"][-1] if call["messages"] else {}
            is_summary = last.get("role") == "user" and _flatten_content(dict(last)).get("content", "").strip() == summarizer
            entry = (position, call)
            if is_summary:
                segments.append({"reason": "compaction", "calls": [entry]})
            elif not segments:
                segments.append({"reason": "", "calls": [entry]})
            elif segments[-1]["reason"] == "compaction":
                segments.append({"reason": "post_compaction", "calls": [entry]})
            else:
                segments[-1]["calls"].append(entry)
        for index, segment in enumerate(segments):
            segment["index"] = index
        return segments

    @staticmethod
    def segment_messages(segment: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
        """Chat-completions messages of one segment with token metadata on each assistant turn.

        Returns ``(messages, prefix_message_count)``: the last call's request
        messages plus its reply, where ``prefix_message_count`` is the length of
        the segment's first request (context only, never trained). The k-th
        assistant message after the prefix is the k-th call's reply, so it
        carries that reply's ``prompt_token_ids`` / ``generation_token_ids`` /
        ``generation_log_probs`` as echoed by the model server.
        """
        calls = [call for _, call in segment["calls"]]
        prefix = len(calls[0]["messages"])
        last = calls[-1]
        messages = [_flatten_content(dict(m)) for m in last["messages"]]
        messages.append(_assistant_message(last["response"]["message"]))
        bundles = [c["response"]["message"] for c in calls]
        assistant_positions = [i for i, m in enumerate(messages) if i >= prefix and m.get("role") == "assistant"]
        if len(assistant_positions) != len(bundles):
            raise ValueError(
                f"segment {segment.get('index')}: {len(assistant_positions)} assistant messages after the "
                f"prefix but {len(bundles)} policy calls"
            )
        # Token metadata is attached when the model server echoed it (training). A server
        # that does not echo (profiling, eval) yields text-only segments; a partial echo is
        # an error because the trainer would otherwise train on a guess.
        echoed = [all(key in bundle for key in _TOKEN_KEYS) for bundle in bundles]
        if any(echoed) and not all(echoed):
            raise ValueError("policy replies carry token ids on some calls only; the model server echo is inconsistent")
        if all(echoed):
            for position, bundle in zip(assistant_positions, bundles):
                for key in _TOKEN_KEYS + _OPTIONAL_TOKEN_KEYS:
                    if key in bundle:
                        messages[position][key] = bundle[key]
        return messages, prefix


_TOKEN_KEYS = ("prompt_token_ids", "generation_token_ids", "generation_log_probs")
_OPTIONAL_TOKEN_KEYS = ("generation_topk_ids", "generation_topk_logprobs", "routed_experts")


def _assistant_message(reply: dict[str, Any]) -> dict[str, Any]:
    """Plain chat-completions assistant message from a stored reply (no token fields)."""
    assistant: dict[str, Any] = {"role": "assistant", "content": reply.get("content") or ""}
    if reply.get("reasoning_content"):
        assistant["reasoning_content"] = reply["reasoning_content"]
    if reply.get("tool_calls"):
        assistant["tool_calls"] = [
            {
                "id": tc.get("id"),
                "type": "function",
                "function": {
                    "name": (tc.get("function") or {}).get("name"),
                    "arguments": (tc.get("function") or {}).get("arguments") or "",
                },
            }
            for tc in reply["tool_calls"]
        ]
    return assistant
