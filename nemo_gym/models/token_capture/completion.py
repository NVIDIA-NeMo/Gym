# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-free evidence from the response actually served by the model gateway."""

import re
from typing import Any

from nemo_gym._moved_paths import moved_names_getattr
from nemo_gym.models.anthropic_converter import AnthropicConverter
from nemo_gym.models.chat_completions_converter import ChatCompletionsConverter
from nemo_gym.models.token_capture.fingerprint import assistant_fingerprint
from nemo_gym.models.token_capture.records import response_to_output_items, strip_token_fields
from nemo_gym.models.token_capture.replay import replay_context
from nemo_gym.models.token_capture.staging.records import CompletionStatus, OutputValidationItem


def completion_replay_items(response: dict[str, Any]) -> list[dict]:
    """Return the served completion as the request items its echo converts to at ingress.

    Messages ingress uses the shared Anthropic converter, so the served assistant
    turn goes through the same converter. Messages clients drop blank text blocks
    when they echo history (the Messages API rejects them), so they are left out;
    an all-blank turn is kept as served. Other dialects use their token-free served
    items. Replay digests match the next request's prefix when the client echoes
    the served turn, apart from fields ignored by the converter.
    """
    if response.get("type") == "message" and response.get("role") == "assistant":
        content = response.get("content") or []
        echoed = [block for block in content if block.get("type") != "text" or block.get("text", "").strip()]
        params = AnthropicConverter().anthropic_request_to_responses(
            {
                "model": response.get("model", ""),
                "max_tokens": 1,
                "messages": [{"role": "assistant", "content": echoed or content}],
            }
        )
        assert isinstance(params.input, list)
        return [item.model_dump(mode="json") for item in params.input]
    return strip_token_fields(response_to_output_items(response))[0]


def completion_items(response: dict[str, Any]) -> list[dict]:
    """Describe served output without reinterpreting Chat text as reasoning."""
    if isinstance(response.get("choices"), list):
        # The gateway has already applied its configured reasoning conversion.
        # A Chat client sees content (including literal <think> tags) as text;
        # splitting it again invents items absent from the harness's output.
        converter = ChatCompletionsConverter(return_token_id_information=False, uses_reasoning_parser=False)
        items = []
        for choice in response["choices"]:
            for output in converter.postprocess_assistant_message_dict(choice["message"]):
                item = output.model_dump(mode="json")
                if item.get("type") != "function_call":
                    item.pop("id", None)
                items.append(item)
        return strip_token_fields(items)[0]
    if response.get("type") == "message" and response.get("role") == "assistant":
        # Describe a Messages turn by the items its echo converts to, so served tool-call ids and
        # per-item fingerprints match the Responses items a harness returns. Detectors read text parts.
        return [
            {**item, "content": [{"type": "output_text", "text": item["content"]}]}
            if item.get("type") == "message" and isinstance(item.get("content"), str)
            else item
            for item in completion_replay_items(response)
        ]
    return strip_token_fields(response_to_output_items(response))[0]


def completion_metadata(response: dict[str, Any]) -> tuple[CompletionStatus, str | None]:
    """Read existing dialect completion fields; never infer success from absence."""
    choices = response.get("choices")
    first = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
    incomplete = response.get("incomplete_details")
    incomplete = incomplete if isinstance(incomplete, dict) else {}
    reason = next(
        (
            value
            for value in (response.get("stop_reason"), first.get("finish_reason"), incomplete.get("reason"))
            if isinstance(value, str)
        ),
        None,
    )
    status = response.get("status")
    if response.get("error") is not None or status == "cancelled" or reason == "abort":
        status = "failed"
    if not isinstance(status, str):
        status = "unknown"
        if reason in ("stop", "tool_calls", "function_call", "end_turn", "tool_use", "stop_sequence"):
            status = "completed"
        elif reason in ("length", "max_tokens", "max_output_tokens", "content_filter"):
            status = "incomplete"
    if status not in ("completed", "incomplete", "failed"):
        return "unknown", reason
    return status, reason


def output_validation_item(item: dict[str, Any]) -> OutputValidationItem:
    """Keep original text used by output detectors, excluding transport arrays."""
    result: OutputValidationItem = {"type": item.get("type")}
    for key in ("content", "summary"):
        parts = item.get(key)
        if isinstance(parts, list):
            result[key] = [
                {"text": part["text"]} if isinstance(part, dict) and isinstance(part.get("text"), str) else {}
                for part in parts
            ]
    return result


# A model server's Chat Completions reply carries the model's reasoning as one <think>...</think> block at the start
# of its text (vllm_model wraps the parsed reasoning that way). A Chat client may return that text verbatim, or, like
# opencode_agent, return the block as a reasoning item and the rest as the answer.
_LEADING_REASONING_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _assistant_text_parts(item: dict[str, Any]) -> list[dict] | None:
    """Return an assistant message's content parts when every one is a text part, else None."""
    if item.get("type", "message") != "message" or item.get("role") != "assistant":
        return None
    content = item.get("content")
    if not isinstance(content, list) or not all(
        isinstance(part, dict) and part.get("type") == "output_text" and isinstance(part.get("text"), str)
        for part in content
    ):
        return None
    return content


def output_evidence_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Put generated output items in the one form that compares served output with a harness's returned output.

    Served evidence and returned history both pass through this function before their items are fingerprinted, so an
    output compares equal in either presentation of a Chat reply's leading reasoning block:

    - the block at the start of an assistant message's first text part becomes a reasoning item ahead of the message;
    - assistant message text is compared without surrounding whitespace;
    - an assistant message left with only empty text right after a reasoning item is dropped.

    Any other think tag stays the model's literal text. The function is idempotent and copies the items it changes.
    """
    result: list[dict[str, Any]] = []
    for item in items:
        parts = _assistant_text_parts(item)
        if parts is None:
            result.append(item)
            continue
        parts = [dict(part) for part in parts]
        match = _LEADING_REASONING_BLOCK.match(parts[0]["text"]) if parts else None
        if match is not None:
            result.append({"type": "reasoning", "summary": [{"type": "summary_text", "text": match.group(1)}]})
            parts[0]["text"] = parts[0]["text"][match.end() :]
        for part in parts:
            part["text"] = part["text"].strip()
        if result and result[-1].get("type") == "reasoning" and not any(part["text"] for part in parts):
            continue
        result.append({**item, "content": parts})
    return result


def output_item_fingerprint(item: dict[str, Any]) -> str:
    """Reuse dialect-neutral visible fingerprints and exact evidence for reasoning and custom tool calls.

    A custom tool call's ``input`` is free-form text in which whitespace and JSON formatting carry meaning,
    so its evidence is the replay digest. That digest hashes every field of the item except ``id``,
    ``status``, the token fields and null fields. A ``namespace`` or a ``caller`` field therefore counts:
    ``{namespace: "functions", name: "exec"}`` and ``{name: "functions__exec"}`` differ for a custom tool
    call, where the fingerprint of a function call joins them. Each such difference refuses a match and
    none accepts a false one.
    """
    if item.get("type") in ("reasoning", "custom_tool_call"):
        return replay_context([item]).items[0].digest
    result = assistant_fingerprint([item])
    if not result:
        raise ValueError("Expected a model-authored output item")
    return result


# Serves the converter this module imported before it was renamed ChatCompletionsConverter, with a warning.
__getattr__ = moved_names_getattr(__name__)
