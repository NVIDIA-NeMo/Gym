# Copyright 2026 Zapier, Inc.
# SPDX-License-Identifier: MIT

"""Custom API clients for the verifiers Client abstraction."""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
from collections.abc import Mapping
from datetime import datetime
from typing import Any, cast

import anthropic
import httpx
import openai
from openai import AsyncOpenAI

from verifiers.clients.client import Client
from verifiers.clients import AnthropicMessagesClient, OpenAIChatCompletionsClient
from verifiers.clients.openai_chat_completions_client import parse_reasoning_content
from verifiers.errors import ModelError, OverlongPromptError
from verifiers.types import (
    AssistantMessage,
    ClientConfig,
    FinishReason,
    Messages,
    Response,
    ResponseMessage,
    SamplingArgs,
    SystemMessage,
    TextMessage,
    Tool,
    ToolCall,
    ToolMessage,
    Usage,
    UserMessage,
)
from verifiers.utils.client_utils import setup_openai_client


# A single transient blip (rate limit, overloaded, gateway/server 5xx, dropped
# connection, timeout) must never kill a rollout. When an exception escapes the
# per-turn API call, the agentic loop ends mid-work and the partial transcript
# is recorded with a dangling assistant tool-call tail scored 0 — what we call
# an "abort". Retrying the full transient family with capped exponential backoff
# + jitter is the root-cause fix; only a genuinely non-transient error (e.g. a
# 400 bad request) or exhausting every attempt is allowed to propagate.
_RETRY_MAX_ATTEMPTS = 40

# Preserve every Responses API output item on the assistant message so a later
# tool turn can replay reasoning items together with function calls. OpenAI
# requires manual histories to include the complete prior response output, not
# only the user-visible text and tool-call projection.
_OPENAI_RESPONSES_OUTPUT_FIELD = "openai_responses_output"

# 4xx client errors that will never succeed on retry — fail fast on these so a
# misconfiguration surfaces immediately instead of burning the full backoff.
# Everything NOT in this set is treated as a transient blip and retried (see
# RetryingOpenAIChatCompletionsClient), which is what makes the proxy/alpha path
# resilient to proxy/gateway-wrapped exception types outside the openai.* hierarchy.
_NON_RETRYABLE_CHAT = (
    openai.BadRequestError,
    openai.AuthenticationError,
    openai.PermissionDeniedError,
    openai.NotFoundError,
    openai.UnprocessableEntityError,
    OverlongPromptError,
)

# Some OpenAI-compatible gateways wrap a provider safety rejection in their own
# exception type (or even in a generic 4xx/5xx wrapper).  Class hierarchy alone
# is therefore not enough to distinguish it from transient infrastructure.  Keep
# this deliberately narrow: malformed requests and ordinary provider errors must
# not be mislabeled as refusals.
_SAFETY_CLASSIFIER_MARKERS = (
    "contentpolicyviolationerror",
    "contentfilterfinishreasonerror",
    "prompt was flagged as potentially violating our usage policy",
)


def _retry_delay(attempt: int, retry_after: float | None = None) -> float:
    """Capped exponential backoff with jitter, honoring a server Retry-After."""
    base = retry_after if retry_after is not None else min(60.0, 2.0**attempt)
    return base + random.uniform(0, 1)


# _RETRY_MAX_ATTEMPTS attempts at min(60, 2**attempt) sum to ~35 minutes, all of it
# inside one API call with no output. That is correct behaviour (better a slow rollout
# than an aborted one) but it is indistinguishable from a hung run, and a whole
# debugging session has been lost to exactly that ambiguity. Announce a retry storm
# once it passes the point where a human would start suspecting a hang.
_RETRY_NOISE_THRESHOLD = 4


def _warn_retry_storm(attempt: int, err: Any, label: str) -> None:
    """Print once past _RETRY_NOISE_THRESHOLD so long backoffs are visible, not silent."""
    if attempt < _RETRY_NOISE_THRESHOLD:
        return
    kind = type(err).__name__ if err is not None else "empty-response"
    print(
        f"[retry] {label}: attempt {attempt + 1}/{_RETRY_MAX_ATTEMPTS} after {kind} — "
        f"sleeping ~{min(60.0, 2.0**attempt):.0f}s (a run can legitimately stall for "
        f"minutes here; it is retrying, not hung)",
        flush=True,
    )


def _parse_retry_after(err: Any) -> float | None:
    headers = getattr(getattr(err, "response", None), "headers", None)
    if not headers:
        return None
    ra = headers.get("retry-after")
    if not ra:
        return None
    try:
        return float(ra)
    except ValueError:
        return None


def _plain_response_value(value: Any) -> Any:
    """Convert an SDK response value (or batch namespace fallback) to JSON data."""
    if isinstance(value, Mapping):
        return {key: _plain_response_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_response_value(item) for item in value]
    if hasattr(value, "model_dump"):
        return _plain_response_value(value.model_dump(exclude_none=True, mode="json"))
    if hasattr(value, "__dict__"):
        return {
            key: _plain_response_value(item)
            for key, item in vars(value).items()
            if not key.startswith("_")
        }
    return value


def _response_output_item_dict(value: Any) -> dict[str, Any]:
    dumped = _plain_response_value(value)
    if not isinstance(dumped, dict):
        raise ValueError(f"Invalid Responses API output item: {type(value)}")
    return dumped


def _exception_chain_text(err: BaseException) -> str:
    """Return exception class names and messages across cause/context wrappers."""
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = err
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__
    return " -> ".join(parts)


def _is_safety_classifier_error(err: BaseException) -> bool:
    """Recognize only explicit provider-side content-policy rejections."""
    text = _exception_chain_text(err).lower()
    if any(marker in text for marker in _SAFETY_CLASSIFIER_MARKERS):
        return True
    return "invalid_prompt" in text and ("usage policy" in text or "content policy" in text)


def _record_safety_classifier_error(state: Any, err: BaseException) -> None:
    """Persist a non-retryable classifier stop in the task's exported debug data."""
    if not isinstance(state, dict):
        return
    debug = state.setdefault("_debug", {"finish_reasons": [], "empty_responses": [], "errors": []})
    message = _exception_chain_text(err)
    debug.setdefault("errors", []).append(
        {
            "type": "safety_classifier",
            "provider_exception": type(err).__name__,
            "message": message,
        }
    )
    # Export treats a final raw stop reason of "refusal" as a classifier-stopped
    # task.  Exception responses have no native response object, so record the
    # equivalent terminal marker here.
    debug.setdefault("stop_reasons", []).append("refusal")


def _perf(state: Any) -> dict | None:
    """Return the per-task performance accumulator on `state`, creating it if the
    task carries a mutable state dict. Returns None when there is no state to record
    into (e.g. ad-hoc calls outside a rollout). Each rollout has its own state, so
    accumulation here is concurrency-safe."""
    if not isinstance(state, dict):
        return None
    p = state.get("_perf")
    if p is None:
        p = {
            "model_time_s": 0.0,
            "model_calls": 0,
            "cached_input_tokens": 0,
            "reasoning_tokens": 0,
            "tool_time_s": 0.0,
            "tool_calls": 0,
        }
        state["_perf"] = p
    return p


def _record_model_call(state: Any, elapsed_s: float, native_response: Any) -> None:
    """Accumulate model-call wall time + cached-input / reasoning token counts from a
    native provider response into state['_perf']. Tolerates missing usage fields and
    different provider shapes (OpenAI Responses, Chat, Anthropic)."""
    p = _perf(state)
    if p is None:
        return
    p["model_time_s"] += elapsed_s
    p["model_calls"] += 1
    # Record the raw provider stop reason for every call. The vf finish_reason
    # mapping collapses reasons it doesn't know (e.g. Anthropic's "refusal") to
    # None, and the runner's env_response hook never sees an episode's final
    # call — so a completion killed by the refusal classifier was previously
    # indistinguishable from an ordinary failure in exports.
    if "_debug" not in state:
        state["_debug"] = {"finish_reasons": [], "empty_responses": [], "errors": []}
    if isinstance(native_response, dict):
        # Gemini Interactions (raw JSON dict): terminal status is the stop reason,
        # usage carries total_cached_tokens / total_thought_tokens.
        state["_debug"].setdefault("stop_reasons", []).append(native_response.get("status"))
        u = native_response.get("usage") or {}
        p["cached_input_tokens"] += int(u.get("total_cached_tokens") or 0)
        p["reasoning_tokens"] += int(u.get("total_thought_tokens") or 0)
        return
    stop = getattr(native_response, "stop_reason", None)  # Anthropic Messages
    if stop is None:
        choices = getattr(native_response, "choices", None)  # OpenAI Chat
        if choices:
            stop = getattr(choices[0], "finish_reason", None)
        else:
            stop = getattr(native_response, "status", None)  # OpenAI Responses
    state["_debug"].setdefault("stop_reasons", []).append(stop)
    u = getattr(native_response, "usage", None)
    if u is None:
        return
    # OpenAI Responses: usage.input_tokens_details.cached_tokens / output_tokens_details.reasoning_tokens
    itd = getattr(u, "input_tokens_details", None)
    if itd is not None and getattr(itd, "cached_tokens", None) is not None:
        p["cached_input_tokens"] += int(itd.cached_tokens)
    otd = getattr(u, "output_tokens_details", None)
    if otd is not None and getattr(otd, "reasoning_tokens", None) is not None:
        p["reasoning_tokens"] += int(otd.reasoning_tokens)
    # OpenAI Chat: usage.prompt_tokens_details.cached_tokens
    ptd = getattr(u, "prompt_tokens_details", None)
    if ptd is not None and getattr(ptd, "cached_tokens", None) is not None:
        p["cached_input_tokens"] += int(ptd.cached_tokens)
    # Anthropic: usage.cache_read_input_tokens. Unlike OpenAI, Anthropic's
    # usage.input_tokens EXCLUDES cache reads/writes, so track those separately
    # for export to add back into the input total ("extra_input_tokens").
    cr = getattr(u, "cache_read_input_tokens", None)
    if cr is not None:
        p["cached_input_tokens"] += int(cr)
        cw = getattr(u, "cache_creation_input_tokens", None) or 0
        p["extra_input_tokens"] = p.get("extra_input_tokens", 0) + int(cr) + int(cw)


def _apply_cache_control(messages, tools):
    """Add ephemeral prompt-caching breakpoints without mutating the inputs.

    Two breakpoints (Anthropic allows 4): the last tool caches the static
    tools+system prefix, and the last message's trailing block caches the
    rolling conversation prefix — so each agentic turn re-reads all prior
    turns from cache instead of paying full input price. Returns new
    (messages, tools); the caller's history objects are left untouched, and
    thinking blocks are never stamped (they must replay verbatim on fable).
    """
    new_tools = tools
    if tools:
        new_tools = list(tools)
        new_tools[-1] = {**new_tools[-1], "cache_control": {"type": "ephemeral"}}
    new_messages = messages
    if messages:
        new_messages = list(messages)
        last = dict(new_messages[-1])
        content = last.get("content")
        if isinstance(content, str):
            last["content"] = [
                {"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}
            ]
            new_messages[-1] = last
        elif isinstance(content, list) and content:
            tail = content[-1]
            if isinstance(tail, dict) and tail.get("type") not in (
                "thinking",
                "redacted_thinking",
            ):
                new_content = list(content)
                new_content[-1] = {**tail, "cache_control": {"type": "ephemeral"}}
                last["content"] = new_content
                new_messages[-1] = last
    return new_messages, new_tools


class StreamingAnthropicClient(AnthropicMessagesClient):
    """AnthropicMessagesClient that uses streaming to avoid 10-minute timeout."""

    @staticmethod
    def build_create_kwargs(prompt, model, sampling_args, tools=None, **kwargs) -> dict[str, Any]:
        """Build the exact Messages API request body for one turn.

        Shared by the interactive streaming path and the batch path
        (automationbench.batch) so both transports send byte-identical
        request bodies — the batch runner's score-equivalence guarantee
        rests on this being the single source of truth.

        Ephemeral cache_control breakpoints are injected here (see
        _apply_cache_control) so prompt caching applies on both transports.
        """

        def normalize_sampling_args(sa):
            sa = dict(sa)
            max_tokens = sa.pop("max_tokens", None)
            sa.pop("n", None)
            sa.pop("stop", None)
            reasoning_effort = sa.pop("reasoning_effort", None)
            if max_tokens is None:
                max_tokens = 4096
            if reasoning_effort is not None:
                sa["thinking"] = {"type": "adaptive"}
                sa["output_config"] = {"effort": reasoning_effort}
                sa["temperature"] = 1.0
                sa.pop("top_p", None)
            sa["max_tokens"] = max_tokens
            return {k: v for k, v in sa.items() if v is not None}

        normalized = normalize_sampling_args(sampling_args)
        messages, tools = _apply_cache_control(prompt, tools)
        create_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            **normalized,
            **kwargs,
        }
        if tools:
            create_kwargs["tools"] = tools

        # Prompt caching: the system prompt and tool definitions are identical
        # on every turn of a task, so marking the end of each as a cache
        # breakpoint lets Anthropic reuse them turn-to-turn instead of
        # re-processing the full prefix every call.
        system = create_kwargs.get("system")
        if isinstance(system, str) and system:
            create_kwargs["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if create_kwargs.get("tools"):
            cached_tools = [dict(t) for t in create_kwargs["tools"]]
            cached_tools[-1]["cache_control"] = {"type": "ephemeral"}
            create_kwargs["tools"] = cached_tools

        return create_kwargs

    async def get_native_response(self, prompt, model, sampling_args, tools=None, **kwargs):
        from anthropic.types import Message as AnthropicMessage

        state = kwargs.pop("state", None)

        create_kwargs = self.build_create_kwargs(prompt, model, sampling_args, tools, **kwargs)

        # Enable interleaved thinking so the model thinks between tool calls,
        # not just on turn 0 of the agentic loop.
        if "thinking" in create_kwargs:
            existing_headers = create_kwargs.pop("extra_headers", {}) or {}
            create_kwargs["extra_headers"] = {
                **existing_headers,
                "anthropic-beta": "interleaved-thinking-2025-05-14",
            }

        # Retry the full transient-error family (rate limit, overloaded, any 5xx
        # gateway/server error, dropped connection, timeout) with capped
        # exponential backoff + jitter, honoring Retry-After. See _RETRY_MAX_ATTEMPTS.
        for attempt in range(_RETRY_MAX_ATTEMPTS):
            try:
                t0 = time.monotonic()
                async with self.client.messages.stream(**create_kwargs) as stream:
                    response: AnthropicMessage = await stream.get_final_message()
                _record_model_call(state, time.monotonic() - t0, response)
                return response
            except (anthropic.APIConnectionError, anthropic.APITimeoutError):
                if attempt == _RETRY_MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_retry_delay(attempt))
            except anthropic.APIStatusError as e:
                status = getattr(e, "status_code", None)
                err_str = str(e)
                is_retryable = (
                    (status is not None and (status == 429 or status >= 500))
                    or "overloaded_error" in err_str
                    or "rate_limit_error" in err_str
                )
                if not is_retryable or attempt == _RETRY_MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_retry_delay(attempt, _parse_retry_after(e)))
        raise RuntimeError("unreachable")


class OpenAIResponsesClient(Client[AsyncOpenAI, list[dict], Any, dict]):
    """Client for the OpenAI Responses API (supports reasoning_effort + tools)."""

    def setup_client(self, config: ClientConfig) -> AsyncOpenAI:
        return setup_openai_client(config)

    async def close(self) -> None:
        await self.client.close()

    async def to_native_tool(self, tool: Tool) -> dict:
        """Convert vf.Tool to Responses API flat tool format."""
        result: dict[str, Any] = {"type": "function", "name": tool.name}
        if tool.description:
            result["description"] = tool.description
        result["parameters"] = tool.parameters
        return result

    async def to_native_prompt(self, messages: Messages) -> tuple[list[dict], dict]:
        """Convert vf.Messages to Responses API input items + instructions kwarg."""
        instructions_parts: list[str] = []
        items: list[dict] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                content = msg.content
                if isinstance(content, str):
                    instructions_parts.append(content)
                elif isinstance(content, list):
                    for part in content:
                        if hasattr(part, "text"):
                            instructions_parts.append(cast(str, part.text))
                        elif isinstance(part, dict) and part.get("type") == "text":
                            instructions_parts.append(part.get("text", ""))
            elif isinstance(msg, (UserMessage, TextMessage)):
                items.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AssistantMessage):
                raw_output = getattr(msg, _OPENAI_RESPONSES_OUTPUT_FIELD, None)
                if raw_output:
                    if not isinstance(raw_output, list):
                        raise ValueError(
                            f"{_OPENAI_RESPONSES_OUTPUT_FIELD} must be a list when present"
                        )
                    items.extend(_response_output_item_dict(item) for item in raw_output)
                    continue
                if msg.content:
                    items.append({"role": "assistant", "content": msg.content})
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        items.append(
                            {
                                "type": "function_call",
                                "call_id": tc.id,
                                "name": tc.name,
                                "arguments": tc.arguments,
                            }
                        )
            elif isinstance(msg, ToolMessage):
                output = msg.content if isinstance(msg.content, str) else str(msg.content)
                items.append(
                    {
                        "type": "function_call_output",
                        "call_id": msg.tool_call_id,
                        "output": output,
                    }
                )

        extra_kwargs: dict[str, Any] = {}
        if instructions_parts:
            extra_kwargs["instructions"] = "\n\n".join(instructions_parts)

        return items, extra_kwargs

    @staticmethod
    def build_call_kwargs(
        prompt: list[dict],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Build the exact Responses API request body for one turn.

        Shared by the interactive path and the batch path (automationbench.batch)
        so both transports send byte-identical request bodies.
        """
        call_kwargs: dict[str, Any] = {"model": model, "input": prompt}
        if tools:
            call_kwargs["tools"] = tools

        # Pass instructions from to_native_prompt
        if "instructions" in kwargs:
            call_kwargs["instructions"] = kwargs.pop("instructions")

        # Forward sampling args, skipping chat-completions-only keys
        _skip = {"extra_body", "max_completion_tokens", "max_tokens", "n", "stop"}
        for key, val in sampling_args.items():
            if key not in _skip and val is not None:
                call_kwargs[key] = val

        # Map reasoning_effort to Responses API reasoning param
        if sampling_args.get("reasoning_effort"):
            call_kwargs["reasoning"] = {"effort": sampling_args["reasoning_effort"]}
            call_kwargs.pop("reasoning_effort", None)

        return call_kwargs

    async def get_native_response(
        self,
        prompt: list[dict],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> Any:
        """Call the Responses API, retrying on rate limits."""
        # Keep state (not an API arg) so we can record per-task perf into it.
        state = kwargs.pop("state", None)

        call_kwargs = self.build_call_kwargs(prompt, model, sampling_args, tools, **kwargs)

        for attempt in range(_RETRY_MAX_ATTEMPTS):
            try:
                t0 = time.monotonic()
                resp = await self.client.responses.create(**call_kwargs)
                _record_model_call(state, time.monotonic() - t0, resp)
                return resp
            except openai.BadRequestError as e:
                error_text = getattr(e, "message", str(e)).lower()
                if "context length" in error_text or "too long" in error_text:
                    raise OverlongPromptError from e
                raise
            except (
                openai.RateLimitError,
                openai.APIConnectionError,
                openai.APITimeoutError,
                openai.InternalServerError,
            ) as e:
                if attempt == _RETRY_MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_retry_delay(attempt, _parse_retry_after(e)))
            except openai.APIStatusError as e:
                # Any other 5xx is transient; 4xx (except handled 400/429) is not.
                status = getattr(e, "status_code", None)
                if status is None or status < 500 or attempt == _RETRY_MAX_ATTEMPTS - 1:
                    raise
                await asyncio.sleep(_retry_delay(attempt, _parse_retry_after(e)))
        raise RuntimeError("unreachable")

    async def raise_from_native_response(self, response: Any) -> None:
        pass

    async def from_native_response(self, response: Any) -> Response:
        """Convert a Responses API response to vf.Response."""
        text_content = ""
        reasoning_chunks: list[str] = []
        tool_calls: list[ToolCall] = []
        raw_output_items: list[dict[str, Any]] = []

        for item in getattr(response, "output", []):
            raw_output_items.append(_response_output_item_dict(item))
            item_type = getattr(item, "type", None)
            if item_type == "message":
                for part in getattr(item, "content", []):
                    if getattr(part, "type", None) == "output_text":
                        text_content += getattr(part, "text", "")
                    elif getattr(part, "type", None) == "refusal":
                        text_content += getattr(part, "refusal", "")
            elif item_type == "reasoning":
                for part in getattr(item, "summary", []) or []:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        reasoning_chunks.append(text)
                for part in getattr(item, "content", []) or []:
                    text = getattr(part, "text", None)
                    if isinstance(text, str):
                        reasoning_chunks.append(text)
            elif item_type == "function_call":
                tool_calls.append(
                    ToolCall(
                        id=getattr(item, "call_id", getattr(item, "id", "")),
                        name=getattr(item, "name", ""),
                        arguments=getattr(item, "arguments", ""),
                    )
                )

        finish_reason: FinishReason = "tool_calls" if tool_calls else "stop"

        raw_usage = getattr(response, "usage", None)
        usage: Usage | None = None
        if raw_usage is not None:
            input_toks = getattr(raw_usage, "input_tokens", 0)
            output_toks = getattr(raw_usage, "output_tokens", 0)
            output_details = getattr(raw_usage, "output_tokens_details", None)
            reasoning_toks = getattr(output_details, "reasoning_tokens", 0) or 0
            usage = Usage(
                prompt_tokens=input_toks,
                completion_tokens=output_toks,
                reasoning_tokens=reasoning_toks,
                total_tokens=input_toks + output_toks,
            )

        message_data: dict[str, Any] = {
            "content": text_content or None,
            "reasoning_content": "\n".join(reasoning_chunks) or None,
            "tool_calls": tool_calls or None,
            "finish_reason": finish_reason,
            "is_truncated": False,
            _OPENAI_RESPONSES_OUTPUT_FIELD: raw_output_items,
        }

        return Response(
            id=getattr(response, "id", ""),
            created=getattr(response, "created_at", 0),
            model=getattr(response, "model", ""),
            usage=usage,
            message=ResponseMessage.model_validate(message_data),
        )


# Interactions API generation_config.thinking_level values. The CLI accepts a
# wider effort enum (none/xhigh/max are other providers'); anything outside this
# set has no Gemini mapping and must fail loud, not degrade silently.
_GEMINI_THINKING_LEVELS = ("minimal", "low", "medium", "high")

# sampling_args keys copied straight into generation_config.
_GEMINI_PASSTHROUGH_KEYS = ("temperature", "seed", "stop_sequences", "tool_choice")

# sampling_args keys build_call_kwargs consumes itself, plus the defaults
# verifiers seeds every run with (Environment sets {"n": 1, "extra_body": {}}).
# Anything outside these two sets would previously be dropped in silence, which
# could change eval semantics without a trace — so it now fails loud.
_GEMINI_CONSUMED_KEYS = (
    "extra_body",
    "reasoning_effort",
    "max_tokens",
    "max_completion_tokens",
    "n",
)

# A 400 is only an overlong prompt if it reports a size overflow. Matching a bare
# "token"/"context" substring misfires on ordinary bad-request text — e.g.
# "Invalid value at 'generation_config.max_output_tokens'" — which would mask a
# config bug as an overlong prompt (and OverlongPromptError is non-retryable).
_OVERLONG_SIZE_MARKERS = ("exceed", "too long", "too large", "too many")


def _is_overlong_prompt_error(text: str) -> bool:
    """True only for 400 bodies that actually report a prompt/context overflow."""
    lowered = text.lower()
    return any(marker in lowered for marker in _OVERLONG_SIZE_MARKERS) and (
        "token" in lowered or "context" in lowered
    )


def _parse_created(created: Any) -> int:
    """Best-effort unix timestamp from the API's `created` field.

    Cosmetic metadata: an epoch int, an ISO string, or something unparseable must
    never take down a rollout, so this degrades to 0 rather than raising.
    """
    if created is None:
        return 0
    if isinstance(created, (int, float)):
        return int(created)
    try:
        return int(datetime.fromisoformat(created).timestamp())
    except (TypeError, ValueError):
        return 0


class GeminiInteractionsClient(Client[httpx.AsyncClient, list[dict], dict, dict]):
    """Client for the Gemini Interactions API (POST /v1beta/interactions).

    Runs the API in stateless mode: every turn resends the full transcript as
    typed input steps with ``store: false``, mirroring the OpenAIResponsesClient
    transport pattern. The harness owns the transcript (exports, healing, and
    resume all re-run tasks from recorded messages), so server-side state via
    ``previous_interaction_id`` is deliberately not used.
    """

    DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def setup_client(self, config: ClientConfig) -> httpx.AsyncClient:
        api_key = os.environ.get(config.api_key_var)
        if not api_key:
            raise ValueError(f"No API key found in ${config.api_key_var}")
        return httpx.AsyncClient(
            base_url=config.api_base_url,
            headers={"x-goog-api-key": api_key, **config.extra_headers},
            timeout=httpx.Timeout(config.timeout, connect=config.connect_timeout),
            limits=httpx.Limits(
                max_connections=config.max_connections,
                max_keepalive_connections=config.max_keepalive_connections,
            ),
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def to_native_tool(self, tool: Tool) -> dict:
        """Convert vf.Tool to the Interactions API flat function-tool format."""
        result: dict[str, Any] = {"type": "function", "name": tool.name}
        if tool.description:
            result["description"] = tool.description
        result["parameters"] = tool.parameters
        return result

    @staticmethod
    def _text_blocks(content: Any) -> list[dict]:
        """Normalize vf MessageContent (str or content-part list) to Interactions
        text content blocks."""
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        blocks: list[dict] = []
        for part in content:
            if hasattr(part, "text"):
                blocks.append({"type": "text", "text": cast(str, part.text)})
            elif isinstance(part, dict) and part.get("type") == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
        return blocks

    async def to_native_prompt(self, messages: Messages) -> tuple[list[dict], dict]:
        """Convert vf.Messages to Interactions input steps + system_instruction kwarg."""
        instructions_parts: list[str] = []
        steps: list[dict] = []
        # function_result steps must carry the called function's name, which
        # vf.ToolMessage doesn't store — recover it from the function_call step
        # earlier in the same history (tool results always follow their call).
        call_names: dict[str, str] = {}

        for msg in messages:
            if isinstance(msg, SystemMessage):
                instructions_parts.extend(block["text"] for block in self._text_blocks(msg.content))
            elif isinstance(msg, (UserMessage, TextMessage)):
                steps.append({"type": "user_input", "content": self._text_blocks(msg.content)})
            elif isinstance(msg, AssistantMessage):
                if msg.content:
                    steps.append(
                        {"type": "model_output", "content": self._text_blocks(msg.content)}
                    )
                for tc in msg.tool_calls or []:
                    call_names[tc.id] = tc.name
                    steps.append(
                        {
                            "type": "function_call",
                            "id": tc.id,
                            "name": tc.name,
                            # vf keeps arguments as a JSON string; Interactions
                            # wants the decoded object.
                            "arguments": json.loads(tc.arguments) if tc.arguments else {},
                        }
                    )
            elif isinstance(msg, ToolMessage):
                output = msg.content if isinstance(msg.content, str) else str(msg.content)
                name = call_names.get(msg.tool_call_id)
                if name is None:
                    # A bare subscript here raised KeyError, which is not in the
                    # retry family — it ended the rollout mid-work with a dangling
                    # tool-call tail (an abort scored 0). Fail with something
                    # diagnosable instead of a one-word traceback.
                    raise ValueError(
                        f"tool result {msg.tool_call_id!r} has no matching function_call "
                        f"earlier in the transcript; known call ids: {sorted(call_names)}"
                    )
                steps.append(
                    {
                        "type": "function_result",
                        "name": name,
                        "call_id": msg.tool_call_id,
                        "result": [{"type": "text", "text": output}],
                    }
                )

        extra_kwargs: dict[str, Any] = {}
        if instructions_parts:
            extra_kwargs["system_instruction"] = "\n\n".join(instructions_parts)

        return steps, extra_kwargs

    @staticmethod
    def build_call_kwargs(
        prompt: list[dict],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Build the exact Interactions API request body for one turn."""
        payload: dict[str, Any] = {"model": model, "input": prompt, "store": False}
        if tools:
            payload["tools"] = tools
        if "system_instruction" in kwargs:
            payload["system_instruction"] = kwargs.pop("system_instruction")

        sa = dict(sampling_args or {})
        extra_body = sa.pop("extra_body", None)

        generation_config: dict[str, Any] = {}
        effort = sa.pop("reasoning_effort", None)
        if effort is not None:
            if effort not in _GEMINI_THINKING_LEVELS:
                raise ValueError(
                    f"reasoning_effort {effort!r} has no Interactions API thinking_level; "
                    f"valid values: {', '.join(_GEMINI_THINKING_LEVELS)}"
                )
            generation_config["thinking_level"] = effort
        max_tokens = sa.pop("max_tokens", None) or sa.pop("max_completion_tokens", None)
        if max_tokens is not None:
            generation_config["max_output_tokens"] = max_tokens
        for key in _GEMINI_PASSTHROUGH_KEYS:
            value = sa.pop(key, None)
            if value is not None:
                generation_config[key] = value
        if generation_config:
            payload["generation_config"] = generation_config

        # verifiers always seeds n=1; anything else asks for multiple completions,
        # which this transport does not implement.
        n = sa.pop("n", 1)
        if n not in (None, 1):
            raise ValueError(f"Gemini Interactions client supports n=1 only, got {n!r}")
        unsupported = sorted(key for key, value in sa.items() if value is not None)
        if unsupported:
            raise ValueError(
                f"sampling args have no Gemini Interactions mapping: {unsupported}. "
                f"Supported: {sorted((*_GEMINI_PASSTHROUGH_KEYS, *_GEMINI_CONSUMED_KEYS))}"
            )

        if extra_body:
            payload.update(extra_body)
        return payload

    async def get_native_response(
        self,
        prompt: list[dict],
        model: str,
        sampling_args: SamplingArgs,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> dict:
        """POST /interactions, retrying the transient family (429/5xx/connection/
        timeout/failed-status/empty-output) with capped backoff — see _RETRY_MAX_ATTEMPTS."""
        state = kwargs.pop("state", None)
        per_request_headers = kwargs.pop("extra_headers", None)

        payload = self.build_call_kwargs(prompt, model, sampling_args, tools, **kwargs)

        for attempt in range(_RETRY_MAX_ATTEMPTS):
            last_attempt = attempt == _RETRY_MAX_ATTEMPTS - 1
            try:
                t0 = time.monotonic()
                resp = await self.client.post(
                    "/interactions", json=payload, headers=per_request_headers
                )
            except httpx.TransportError:
                # Dropped connection / timeout — transient.
                if last_attempt:
                    raise
                await asyncio.sleep(_retry_delay(attempt))
                continue

            if resp.status_code == 429 or resp.status_code >= 500:
                if last_attempt:
                    resp.raise_for_status()
                retry_after = resp.headers.get("retry-after")
                await asyncio.sleep(
                    _retry_delay(attempt, float(retry_after) if retry_after else None)
                )
                continue
            if resp.status_code == 400 and _is_overlong_prompt_error(resp.text):
                raise OverlongPromptError(resp.text)
            if resp.status_code != 200:
                # Genuine client errors (bad request, auth, not-found, …) won't
                # change on retry — fail fast and loud.
                raise RuntimeError(f"Gemini Interactions API {resp.status_code}: {resp.text}")

            data = resp.json()
            _record_model_call(state, time.monotonic() - t0, data)
            if data.get("status") == "failed" and not last_attempt:
                # Server-side execution failure — transient (model unavailability
                # etc.); retrying is what keeps it from becoming an abort.
                await asyncio.sleep(_retry_delay(attempt))
                continue
            steps = data.get("steps") or []
            has_output = any(s.get("type") in ("model_output", "function_call") for s in steps)
            if not has_output and not last_attempt:
                # Fully empty interaction (a known Gemini flakiness on the chat
                # path) — treat as one more transient blip and retry.
                await asyncio.sleep(_retry_delay(attempt))
                continue
            return data
        raise RuntimeError("unreachable")

    async def raise_from_native_response(self, response: dict) -> None:
        if response.get("status") == "failed":
            raise RuntimeError(f"Gemini interaction failed: {response.get('error')}")

    async def from_native_response(self, response: dict) -> Response:
        """Convert an Interactions API response to vf.Response."""
        text_content = ""
        tool_calls: list[ToolCall] = []

        for index, step in enumerate(response.get("steps") or []):
            step_type = step.get("type")
            if step_type == "model_output":
                for block in step.get("content") or []:
                    if block.get("type") == "text":
                        text_content += block.get("text", "")
            elif step_type == "function_call":
                # to_native_prompt rebuilds a call_id -> name map to label
                # function_result steps, so ids must be unique. Defaulting a
                # missing id to "" made parallel calls collide and replay the
                # wrong tool name silently; synthesize a unique fallback instead.
                call_id = step.get("id") or f"{response.get('id') or 'interaction'}_call_{index}"
                tool_calls.append(
                    ToolCall(
                        id=call_id,
                        name=step.get("name", ""),
                        # vf keeps arguments as a JSON string.
                        arguments=json.dumps(step.get("arguments") or {}),
                    )
                )

        status = response.get("status")
        truncated = status in ("incomplete", "budget_exceeded")
        finish_reason: FinishReason = (
            "tool_calls" if tool_calls else ("length" if truncated else "stop")
        )

        raw_usage = response.get("usage")
        usage: Usage | None = None
        if raw_usage is not None:
            input_toks = int(raw_usage.get("total_input_tokens") or 0)
            output_toks = int(raw_usage.get("total_output_tokens") or 0)
            thought_toks = int(raw_usage.get("total_thought_tokens") or 0)
            # Unlike OpenAI, Gemini reports thought tokens OUTSIDE output tokens;
            # they bill as output, so fold them in for cross-provider parity.
            usage = Usage(
                prompt_tokens=input_toks,
                completion_tokens=output_toks + thought_toks,
                reasoning_tokens=thought_toks,
                total_tokens=input_toks + output_toks + thought_toks,
            )

        created_ts = _parse_created(response.get("created"))

        return Response(
            id=response.get("id", ""),
            created=created_ts,
            model=response.get("model", ""),
            usage=usage,
            message=ResponseMessage(
                content=text_content or None,
                tool_calls=tool_calls or None,
                finish_reason=finish_reason,
                is_truncated=truncated,
            ),
        )


class RetryingOpenAIChatCompletionsClient(OpenAIChatCompletionsClient):
    """Chat Completions client that retries the full transient-error family.

    The stock verifiers client lets connection/timeout/5xx errors propagate,
    which ends the rollout mid-work (a dangling tool-call tail scored 0). This is
    the path used for gateway-routed and alpha models, where transient proxy 5xx
    and dropped connections are the dominant abort cause — so it needs the same
    backoff guard as the Anthropic and Responses clients.
    """

    async def to_native_prompt(self, messages):
        # The stock serializer emits "reasoning_content": null on assistant
        # messages whose turn produced no reasoning text; strict validators
        # (e.g. Vercel AI Gateway) 400 on the null and abort the rollout.
        native, extra = await super().to_native_prompt(messages)
        for m in native:
            d = cast(dict, m)
            if d.get("reasoning_content") is None:
                d.pop("reasoning_content", None)
        return native, extra

    async def get_native_response(self, *args, **kwargs):
        state = kwargs.get("state")
        for attempt in range(_RETRY_MAX_ATTEMPTS):
            try:
                t0 = time.monotonic()
                resp = await super().get_native_response(*args, **kwargs)
                _record_model_call(state, time.monotonic() - t0, resp)
                # A fully empty message (no content, tool calls, or reasoning —
                # a Gemini flakiness) would raise EmptyModelResponseError in
                # from_native_response, outside this guard, killing the rollout.
                # Treat it as one more transient blip and retry here instead.
                choices = getattr(resp, "choices", None)
                if choices:
                    msg = choices[0].message
                    if not (msg.content or msg.tool_calls or parse_reasoning_content(msg)):
                        if attempt == _RETRY_MAX_ATTEMPTS - 1:
                            return resp
                        _warn_retry_storm(attempt, None, "chat/empty-response")
                        await asyncio.sleep(_retry_delay(attempt))
                        continue
                return resp
            except _NON_RETRYABLE_CHAT as e:
                # Genuine client errors (bad request, auth, not-found, …) won't
                # change on retry — fail fast and loud.
                if _is_safety_classifier_error(e):
                    _record_safety_classifier_error(state, e)
                    raise ModelError("provider safety classifier rejected the request") from e
                raise
            except Exception as e:  # noqa: BLE001
                # Gateway-wrapped safety rejections are deterministic model-side
                # failures, not infrastructure flakes.  Preserve them as refusal
                # stops instead of silently retrying into a different outcome.
                if _is_safety_classifier_error(e):
                    _record_safety_classifier_error(state, e)
                    raise ModelError("provider safety classifier rejected the request") from e
                # Everything else is treated as a transient blip and retried.
                # This deliberately broad net catches proxy/gateway-wrapped
                # exception types that don't subclass the openai.* classes
                # (the dominant abort cause on the proxy/alpha path), as well as
                # connection/timeout/5xx/rate-limit. After the last attempt it
                # propagates, and the --ensure-complete gate re-runs the task.
                if attempt == _RETRY_MAX_ATTEMPTS - 1:
                    raise
                _warn_retry_storm(attempt, e, "chat")
                await asyncio.sleep(_retry_delay(attempt, _parse_retry_after(e)))
        raise RuntimeError("unreachable")
