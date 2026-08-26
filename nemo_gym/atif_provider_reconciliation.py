# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconcile captured provider responses with Gym's strict ATIF projection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from nemo_gym.atif_json import json_values_equal, strict_json_loads
from nemo_gym.relay_atif import AtifStep
from nemo_gym.rollout_observability import TrajectoryModelCall


class ProviderReconciliationError(ValueError):
    """Provider evidence cannot be reconciled with the exported ATIF step."""


@dataclass(frozen=True)
class _ProviderToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class _ProviderOutputEvidence:
    message_parts: tuple[str, ...] = ()
    reasoning: str | None = None
    tool_calls: tuple[_ProviderToolCall, ...] = ()


@dataclass(frozen=True)
class ReconciledTokenStats:
    """Token counts after matching normalized and provider-native evidence."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None


@dataclass(frozen=True)
class ProviderReconciliationResult:
    """Immutable values that the ATIF projection may safely consume."""

    finish_reason: str | None
    model_name: str | None
    response_id: str | None
    token_stats: ReconciledTokenStats


def _path_error(path: str, detail: str) -> ProviderReconciliationError:
    return ProviderReconciliationError(f"{path}: {detail}")


def _provider_termination_evidence(
    call: TrajectoryModelCall,
    *,
    path: str,
) -> tuple[str | None, str | None, bool]:
    """Read termination fields from provider responses whose dialect is known."""

    response = call.response
    dialect = call.response_metadata.dialect
    if dialect not in {"responses", "chat", "messages"}:
        return None, None, False
    if response is None:
        return None, None, False
    if not isinstance(response, dict):
        raise _path_error(path, f"{dialect} provider response is not an object")
    if response.get("error") is not None:
        raise _path_error(path, f"{dialect} provider response contains error evidence")

    if dialect == "responses":
        status = response.get("status")
        if status is not None and not isinstance(status, str):
            raise _path_error(path, "Responses status is not a string")
        incomplete_details = response.get("incomplete_details")
        if incomplete_details is None:
            return status, None, False
        if not isinstance(incomplete_details, dict):
            raise _path_error(path, "Responses incomplete_details is not an object")
        reason = incomplete_details.get("reason")
        if reason is not None and not isinstance(reason, str):
            raise _path_error(path, "Responses incomplete_details.reason is not a string")
        return status, reason, True
    if dialect == "chat":
        choices = response.get("choices")
        if choices is None:
            return None, None, False
        if not isinstance(choices, list):
            raise _path_error(path, "Chat choices is not an array")
        if not choices:
            return None, None, False
        if not isinstance(choices[0], dict):
            raise _path_error(path, "Chat first choice is not an object")
        finish_reason = choices[0].get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise _path_error(path, "Chat finish_reason is not a string")
        return None, finish_reason, False

    response_type = response.get("type")
    if response_type is not None and not isinstance(response_type, str):
        raise _path_error(path, "Messages type is not a string")
    if response_type == "error":
        raise _path_error(path, "messages provider response contains error evidence")
    stop_reason = response.get("stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        raise _path_error(path, "Messages stop_reason is not a string")
    return None, stop_reason, False


def _reconciled_finish_reason(call: TrajectoryModelCall, *, path: str) -> str | None:
    """Reject provider-native failure evidence and reconcile redundant metadata."""

    metadata = call.response_metadata
    raw_status, raw_finish_reason, raw_incomplete = _provider_termination_evidence(call, path=path)
    if (
        metadata.response_status is not None
        and raw_status is not None
        and metadata.response_status.strip().lower() != raw_status.strip().lower()
    ):
        raise _path_error(path, "normalized response_status conflicts with the provider response")
    if (
        metadata.finish_reason is not None
        and raw_finish_reason is not None
        and metadata.finish_reason.strip().lower() != raw_finish_reason.strip().lower()
    ):
        raise _path_error(path, "normalized finish_reason conflicts with the provider response")
    if raw_status is not None and raw_status.strip().lower() != "completed":
        raise _path_error(path, f"provider response is {raw_status!r}, not completed")
    if raw_incomplete:
        raise _path_error(path, "Responses provider response contains incomplete_details")
    return metadata.finish_reason if metadata.finish_reason is not None else raw_finish_reason


def _known_provider_response(call: TrajectoryModelCall, *, path: str) -> dict[str, Any] | None:
    dialect = call.response_metadata.dialect
    if dialect not in {"responses", "chat", "messages"} or call.response is None:
        return None
    if not isinstance(call.response, dict):
        raise _path_error(path, f"{dialect} provider response is not an object")
    return call.response


def _reconcile_response_identity(
    call: TrajectoryModelCall,
    *,
    path: str,
) -> tuple[str | None, str | None]:
    """Match raw response identity to normalized metadata without consulting the request."""

    response = _known_provider_response(call, path=path)
    metadata = call.response_metadata
    if response is None:
        return metadata.model, metadata.response_id

    def reconcile(field_name: str, normalized: str | None) -> str | None:
        if field_name not in response:
            return normalized
        raw = response[field_name]
        # Gym's Chat SSE reconstruction includes ``model: None`` when no chunk
        # reports a model. Treat that producer-owned shape as absent evidence;
        # the normalized record may still carry the requested model identity.
        if field_name == "model" and metadata.dialect == "chat" and raw is None:
            return normalized
        if not isinstance(raw, str) or not raw.strip():
            raise _path_error(f"{path}.response.{field_name}", "expected a non-empty string")
        if normalized is not None and raw != normalized:
            raise _path_error(
                path,
                f"normalized {field_name!r} conflicts with the provider response",
            )
        return raw

    return reconcile("model", metadata.model), reconcile("id", metadata.response_id)


def _usage_token(
    usage: dict[str, Any],
    aliases: tuple[tuple[str, ...], ...],
    *,
    label: str,
    path: str,
) -> int | None:
    """Read equivalent usage fields and reject malformed or conflicting aliases."""

    values: list[tuple[str, int]] = []
    for alias in aliases:
        current: Any = usage
        alias_path = path
        for index, field_name in enumerate(alias):
            alias_path = f"{alias_path}.{field_name}"
            if not isinstance(current, dict):
                raise _path_error(alias_path.rsplit(".", 1)[0], "expected an object")
            if field_name not in current:
                break
            current = current[field_name]
            if index < len(alias) - 1 and not isinstance(current, dict):
                raise _path_error(alias_path, "expected an object")
        else:
            if type(current) is not int or current < 0:
                raise _path_error(alias_path, "expected a non-negative integer")
            values.append((".".join(alias), current))

    if not values:
        return None
    first_name, first_value = values[0]
    for other_name, other_value in values[1:]:
        if other_value != first_value:
            raise _path_error(
                path,
                f"{label} aliases conflict ({first_name}={first_value}, {other_name}={other_value})",
            )
    return first_value


def _openai_usage_stats(usage: dict[str, Any], *, path: str) -> ReconciledTokenStats:
    prompt_tokens = _usage_token(
        usage,
        (("input_tokens",), ("prompt_tokens",)),
        label="prompt token",
        path=path,
    )
    completion_tokens = _usage_token(
        usage,
        (("output_tokens",), ("completion_tokens",)),
        label="completion token",
        path=path,
    )
    explicit_total = _usage_token(usage, (("total_tokens",),), label="total token", path=path)
    derived_total = (
        prompt_tokens + completion_tokens if prompt_tokens is not None and completion_tokens is not None else None
    )
    if explicit_total is not None and derived_total is not None and explicit_total != derived_total:
        raise _path_error(path, "total_tokens does not match provider prompt + completion tokens")
    cached_tokens = _usage_token(
        usage,
        (
            ("input_tokens_details", "cached_tokens"),
            ("prompt_tokens_details", "cached_tokens"),
            ("cached_input_tokens",),
        ),
        label="cached token",
        path=path,
    )
    reasoning_tokens = _usage_token(
        usage,
        (
            ("output_tokens_details", "reasoning_tokens"),
            ("completion_tokens_details", "reasoning_tokens"),
            ("reasoning_output_tokens",),
        ),
        label="reasoning token",
        path=path,
    )
    return ReconciledTokenStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=explicit_total if explicit_total is not None else derived_total,
        cached_tokens=cached_tokens,
    )


def _anthropic_usage_stats(usage: dict[str, Any], *, path: str) -> ReconciledTokenStats:
    input_tokens = _usage_token(usage, (("input_tokens",),), label="input token", path=path)
    completion_tokens = _usage_token(usage, (("output_tokens",),), label="output token", path=path)
    cached_tokens = _usage_token(
        usage,
        (("cache_read_input_tokens",),),
        label="cache-read token",
        path=path,
    )
    cache_creation_tokens = _usage_token(
        usage,
        (("cache_creation_input_tokens",),),
        label="cache-creation token",
        path=path,
    )
    cache_total = (cached_tokens or 0) + (cache_creation_tokens or 0)
    prompt_tokens = (input_tokens or 0) + cache_total if input_tokens is not None or cache_total > 0 else None
    total_tokens = (
        prompt_tokens + completion_tokens if prompt_tokens is not None and completion_tokens is not None else None
    )
    return ReconciledTokenStats(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
    )


def _provider_usage_stats(call: TrajectoryModelCall, *, path: str) -> ReconciledTokenStats:
    response = _known_provider_response(call, path=path)
    dialect = call.response_metadata.dialect
    if response is None or "usage" not in response or response["usage"] is None:
        return ReconciledTokenStats()
    usage = response["usage"]
    if not isinstance(usage, dict):
        raise _path_error(f"{path}.response.usage", "expected an object")
    if dialect in {"responses", "chat"}:
        return _openai_usage_stats(usage, path=f"{path}.response.usage")
    assert dialect == "messages"
    return _anthropic_usage_stats(usage, path=f"{path}.response.usage")


def _reconciled_token_stats(call: TrajectoryModelCall, *, path: str) -> ReconciledTokenStats:
    raw = _provider_usage_stats(call, path=path)
    normalized = call.token_stats

    def reconcile(field_name: str) -> int | None:
        raw_value = getattr(raw, field_name)
        normalized_value = getattr(normalized, field_name)
        if raw_value is not None and normalized_value is not None and raw_value != normalized_value:
            raise _path_error(
                path,
                f"normalized {field_name} conflicts with the provider response usage",
            )
        return normalized_value if normalized_value is not None else raw_value

    return ReconciledTokenStats(
        prompt_tokens=reconcile("prompt_tokens"),
        completion_tokens=reconcile("completion_tokens"),
        reasoning_tokens=reconcile("reasoning_tokens"),
        total_tokens=reconcile("total_tokens"),
        cached_tokens=reconcile("cached_tokens"),
    )


def _canonical_message_parts(step: AtifStep) -> tuple[str, ...]:
    if isinstance(step.message, str):
        return (step.message,) if step.message else ()
    return tuple(part.text for part in step.message if part.text is not None)


def _provider_tool_call(
    *,
    call_id: Any,
    name: Any,
    arguments: Any,
    arguments_are_json: bool,
    path: str,
) -> _ProviderToolCall:
    if not isinstance(call_id, str) or not call_id.strip():
        raise _path_error(path, "expected a non-empty tool call ID")
    if not isinstance(name, str) or not name.strip():
        raise _path_error(path, "expected a non-empty tool name")
    if arguments_are_json:
        if not isinstance(arguments, str):
            raise _path_error(f"{path}.arguments", "expected a JSON object string")
        try:
            arguments = strict_json_loads(arguments)
        except (TypeError, json.JSONDecodeError, ValueError) as exc:
            raise _path_error(f"{path}.arguments", "expected a JSON object string") from exc
    if not isinstance(arguments, dict):
        raise _path_error(f"{path}.arguments", "expected a JSON object")
    return _ProviderToolCall(call_id=call_id, name=name, arguments=arguments)


def _provider_reasoning_parts(
    value: Any,
    *,
    part_type: str,
    text_field: str,
    path: str,
) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise _path_error(path, "expected an array")
    parts: list[str] = []
    for index, part in enumerate(value):
        if not isinstance(part, dict) or part.get("type") != part_type:
            raise _path_error(f"{path}[{index}]", f"expected a {part_type!r} object")
        text = part.get(text_field)
        if not isinstance(text, str):
            raise _path_error(f"{path}[{index}].{text_field}", "expected a string")
        parts.append(text)
    return parts


def _responses_reasoning(item: dict[str, Any], *, path: str) -> str | None:
    if item.get("encrypted_content") is not None:
        raise _path_error(path, "encrypted Responses reasoning cannot be reconciled with ATIF text")
    summary = _provider_reasoning_parts(
        item.get("summary"),
        part_type="summary_text",
        text_field="text",
        path=f"{path}.summary",
    )
    content = _provider_reasoning_parts(
        item.get("content"),
        part_type="reasoning_text",
        text_field="text",
        path=f"{path}.content",
    )
    if summary and content:
        raise _path_error(path, "Responses reasoning contains both summary and content")
    parts = content or summary
    if len(parts) > 1:
        raise _path_error(path, "multiple Responses reasoning segments cannot be reconciled with one ATIF field")
    return parts[0] if parts else None


def _validate_provider_semantics(step: AtifStep, evidence: _ProviderOutputEvidence, *, path: str) -> None:
    if evidence.message_parts != _canonical_message_parts(step):
        raise _path_error(path, "raw provider message does not match the exported ATIF message")
    if evidence.reasoning != step.reasoning_content:
        raise _path_error(path, "raw provider reasoning does not match the exported ATIF reasoning")
    canonical_calls = tuple(
        _ProviderToolCall(call_id=call.tool_call_id, name=call.function_name, arguments=call.arguments)
        for call in step.tool_calls or []
    )
    if len(evidence.tool_calls) != len(canonical_calls) or any(
        raw.call_id != canonical.call_id
        or raw.name != canonical.name
        or not json_values_equal(raw.arguments, canonical.arguments)
        for raw, canonical in zip(evidence.tool_calls, canonical_calls, strict=True)
    ):
        raise _path_error(path, "raw provider tool calls do not match the exported ATIF tool calls")


def _validate_responses_output(response: dict[str, Any], step: AtifStep, *, path: str) -> None:
    if "output" not in response and "output_text" not in response:
        return
    output = response.get("output")
    output_text = response.get("output_text")
    if "output" in response and not isinstance(output, list):
        raise _path_error(f"{path}.response.output", "expected a Responses output array")
    if "output_text" in response and not isinstance(output_text, str):
        raise _path_error(f"{path}.response.output_text", "expected a string")

    message_parts: list[str] = []
    tool_calls: list[_ProviderToolCall] = []
    reasoning: str | None = None
    message_items = 0
    reasoning_items = 0
    saw_non_reasoning_output = False
    for index, item in enumerate(output or []):
        item_path = f"{path}.response.output[{index}]"
        if not isinstance(item, dict):
            raise _path_error(item_path, "expected a Responses output object")
        item_type = item.get("type")
        if item_type == "reasoning":
            if saw_non_reasoning_output:
                raise _path_error(item_path, "Responses reasoning cannot follow message or tool output")
        else:
            saw_non_reasoning_output = True
        if item_type in {"message", "reasoning", "function_call"}:
            item_status = item.get("status")
            if item_status is not None and (
                not isinstance(item_status, str) or item_status.strip().lower() != "completed"
            ):
                raise _path_error(item_path, f"Responses output status is {item_status!r}, not completed")
        if item_type == "message":
            message_items += 1
            if item.get("role") != "assistant":
                raise _path_error(f"{item_path}.role", "expected 'assistant'")
            content = item.get("content")
            if not isinstance(content, list):
                raise _path_error(f"{item_path}.content", "expected an output content array")
            for part_index, part in enumerate(content):
                part_path = f"{item_path}.content[{part_index}]"
                if not isinstance(part, dict) or part.get("type") != "output_text":
                    raise _path_error(part_path, "expected an 'output_text' object")
                text = part.get("text")
                if not isinstance(text, str):
                    raise _path_error(f"{part_path}.text", "expected a string")
                message_parts.append(text)
        elif item_type == "output_text":
            message_items += 1
            text = item.get("text")
            if not isinstance(text, str):
                raise _path_error(f"{item_path}.text", "expected a string")
            message_parts.append(text)
        elif item_type == "reasoning":
            reasoning_items += 1
            reasoning = _responses_reasoning(item, path=item_path)
        elif item_type == "function_call":
            tool_calls.append(
                _provider_tool_call(
                    call_id=item.get("call_id"),
                    name=item.get("name"),
                    arguments=item.get("arguments"),
                    arguments_are_json=True,
                    path=item_path,
                )
            )
        else:
            raise _path_error(item_path, f"unsupported Responses output type {item_type!r}")
    if message_items > 1:
        raise _path_error(f"{path}.response.output", "multiple Responses messages cannot be reconciled")
    if reasoning_items > 1:
        raise _path_error(f"{path}.response.output", "multiple Responses reasoning items cannot be reconciled")

    if isinstance(output_text, str):
        if message_parts:
            if output_text != "".join(message_parts):
                raise _path_error(
                    f"{path}.response.output_text",
                    "does not match the structured Responses message",
                )
        else:
            canonical_parts = _canonical_message_parts(step)
            if len(canonical_parts) > 1:
                raise _path_error(
                    f"{path}.response.output_text",
                    "flattened text cannot prove multiple ATIF message-part boundaries",
                )
            message_parts = [output_text] if output_text else []

    _validate_provider_semantics(
        step,
        _ProviderOutputEvidence(
            message_parts=tuple(message_parts),
            reasoning=reasoning,
            tool_calls=tuple(tool_calls),
        ),
        path=path,
    )


def _validate_chat_output(response: dict[str, Any], step: AtifStep, *, path: str) -> None:
    if "choices" not in response:
        return
    choices = response.get("choices")
    if not isinstance(choices, list):
        return  # The termination validator reports the more specific shape error.
    if not choices:
        raise _path_error(f"{path}.response.choices", "an empty Chat choices array cannot be reconciled")
    if len(choices) != 1:
        raise _path_error(f"{path}.response.choices", "expected exactly one Chat choice")
    choice = choices[0]
    if not isinstance(choice, dict):
        return  # The termination validator reports the more specific shape error.
    choice_index = choice.get("index")
    if choice_index is not None and (type(choice_index) is not int or choice_index != 0):
        raise _path_error(f"{path}.response.choices[0].index", "expected 0 when present")
    logprobs = choice.get("logprobs")
    if logprobs is not None and not isinstance(logprobs, dict):
        raise _path_error(f"{path}.response.choices[0].logprobs", "expected an object when present")
    message = choice.get("message")
    if message is None:
        finish_reason = choice.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason.strip():
            raise _path_error(
                f"{path}.response.choices[0]",
                "a sparse Chat choice requires an explicit finish_reason",
            )
        if logprobs:
            raise _path_error(
                f"{path}.response.choices[0].logprobs",
                "a sparse Chat choice cannot carry log probabilities",
            )
        if choice.get("delta") not in (None, {}):
            raise _path_error(
                f"{path}.response.choices[0].delta",
                "a sparse Chat choice contains output outside message in delta",
            )
        unsupported_fields = sorted(set(choice) - {"index", "finish_reason", "logprobs", "message", "delta"})
        if unsupported_fields:
            raise _path_error(
                f"{path}.response.choices[0]",
                "a sparse Chat choice contains unchecked fields: " + ", ".join(unsupported_fields),
            )
        return
    unchecked_output_fields = sorted(field for field in ("delta", "text") if field in choice)
    if unchecked_output_fields:
        raise _path_error(
            f"{path}.response.choices[0]",
            "a Chat choice contains output outside message: " + ", ".join(unchecked_output_fields),
        )
    if not isinstance(message, dict):
        raise _path_error(f"{path}.response.choices[0].message", "expected an object")
    if message.get("role") != "assistant":
        raise _path_error(f"{path}.response.choices[0].message.role", "expected 'assistant'")
    if message.get("refusal") not in (None, ""):
        raise _path_error(
            f"{path}.response.choices[0].message.refusal",
            "Chat refusal output is not represented in standard ATIF output",
        )
    if message.get("audio") is not None:
        raise _path_error(
            f"{path}.response.choices[0].message.audio",
            "Chat audio output is not represented in standard ATIF output",
        )
    if isinstance(logprobs, dict) and logprobs.get("refusal") not in (None, []):
        raise _path_error(
            f"{path}.response.choices[0].logprobs.refusal",
            "Chat refusal log probabilities are not represented in standard ATIF output",
        )

    content = message.get("content")
    if content is None:
        message_parts: list[str] = []
    elif isinstance(content, str):
        message_parts = [content] if content else []
    elif isinstance(content, list):
        message_parts = []
        for index, part in enumerate(content):
            part_path = f"{path}.response.choices[0].message.content[{index}]"
            if not isinstance(part, dict) or part.get("type") != "text":
                raise _path_error(part_path, "expected a text object")
            text = part.get("text")
            if not isinstance(text, str):
                raise _path_error(f"{part_path}.text", "expected a string")
            message_parts.append(text)
    else:
        raise _path_error(f"{path}.response.choices[0].message.content", "unsupported Chat content")

    raw_reasoning = [
        (field_name, message[field_name])
        for field_name in ("reasoning_content", "reasoning")
        if message.get(field_name) is not None
    ]
    if any(not isinstance(value, str) for _, value in raw_reasoning):
        raise _path_error(f"{path}.response.choices[0].message", "Chat reasoning must be scalar text")
    if len(raw_reasoning) == 2 and raw_reasoning[0][1] != raw_reasoning[1][1]:
        raise _path_error(f"{path}.response.choices[0].message", "Chat reasoning aliases conflict")
    reasoning = raw_reasoning[0][1] if raw_reasoning else None

    if message.get("function_call") is not None:
        raise _path_error(
            f"{path}.response.choices[0].message.function_call",
            "legacy Chat function_call output is not supported",
        )
    raw_tool_calls = message.get("tool_calls")
    if raw_tool_calls is None:
        raw_tool_calls = []
    if not isinstance(raw_tool_calls, list):
        raise _path_error(f"{path}.response.choices[0].message.tool_calls", "expected an array")
    tool_calls: list[_ProviderToolCall] = []
    for index, raw_call in enumerate(raw_tool_calls):
        call_path = f"{path}.response.choices[0].message.tool_calls[{index}]"
        function = raw_call.get("function") if isinstance(raw_call, dict) else None
        if not isinstance(function, dict):
            raise _path_error(call_path, "expected a function tool call object")
        if raw_call.get("type") not in (None, "function"):
            raise _path_error(f"{call_path}.type", "expected 'function'")
        tool_calls.append(
            _provider_tool_call(
                call_id=raw_call.get("id"),
                name=function.get("name"),
                arguments=function.get("arguments"),
                arguments_are_json=True,
                path=call_path,
            )
        )
    _validate_provider_semantics(
        step,
        _ProviderOutputEvidence(
            message_parts=tuple(message_parts),
            reasoning=reasoning,
            tool_calls=tuple(tool_calls),
        ),
        path=path,
    )


def _validate_anthropic_output(response: dict[str, Any], step: AtifStep, *, path: str) -> None:
    if "content" not in response:
        return
    content = response.get("content")
    if not isinstance(content, list):
        raise _path_error(f"{path}.response.content", "expected an Anthropic content array")
    if response.get("type") not in (None, "message"):
        raise _path_error(f"{path}.response.type", "expected 'message'")
    if response.get("role") not in (None, "assistant"):
        raise _path_error(f"{path}.response.role", "expected 'assistant'")

    message_parts: list[str] = []
    tool_calls: list[_ProviderToolCall] = []
    reasoning: str | None = None
    reasoning_items = 0
    saw_non_reasoning_output = False
    for index, block in enumerate(content):
        block_path = f"{path}.response.content[{index}]"
        if not isinstance(block, dict):
            raise _path_error(block_path, "expected an Anthropic content block")
        block_type = block.get("type")
        if block_type == "text":
            saw_non_reasoning_output = True
            text = block.get("text")
            if not isinstance(text, str):
                raise _path_error(f"{block_path}.text", "expected a string")
            message_parts.append(text)
        elif block_type == "thinking":
            if saw_non_reasoning_output:
                raise _path_error(block_path, "Anthropic thinking cannot follow text or tool output")
            reasoning_items += 1
            thinking = block.get("thinking")
            if not isinstance(thinking, str):
                raise _path_error(f"{block_path}.thinking", "expected a string")
            reasoning = thinking
        elif block_type == "redacted_thinking":
            raise _path_error(block_path, "redacted Anthropic reasoning cannot be reconciled with ATIF text")
        elif block_type == "tool_use":
            saw_non_reasoning_output = True
            caller = block.get("caller")
            if caller is not None and (
                not isinstance(caller, dict) or caller.get("type") != "direct" or set(caller) != {"type"}
            ):
                raise _path_error(block_path, "Anthropic server-tool callers are not supported")
            tool_calls.append(
                _provider_tool_call(
                    call_id=block.get("id"),
                    name=block.get("name"),
                    arguments=block.get("input"),
                    arguments_are_json=False,
                    path=block_path,
                )
            )
        else:
            raise _path_error(block_path, f"unsupported Anthropic content type {block_type!r}")
    if reasoning_items > 1:
        raise _path_error(f"{path}.response.content", "multiple Anthropic thinking blocks cannot be reconciled")
    _validate_provider_semantics(
        step,
        _ProviderOutputEvidence(
            message_parts=tuple(message_parts),
            reasoning=reasoning,
            tool_calls=tuple(tool_calls),
        ),
        path=path,
    )


def _validate_provider_output_coverage(call: TrajectoryModelCall, step: AtifStep, *, path: str) -> None:
    response = call.response
    dialect = call.response_metadata.dialect
    if response is None or dialect not in {"responses", "chat", "messages"}:
        return
    if not isinstance(response, dict):
        return  # The termination validator reports the more specific shape error.
    semantic_fields = {
        "responses": {"output", "output_text"},
        "chat": {"choices"},
        "messages": {"content"},
    }
    foreign_fields = sorted(
        field
        for family, fields in semantic_fields.items()
        if family != dialect
        for field in fields
        if field in response
    )
    if foreign_fields:
        raise _path_error(
            f"{path}.response",
            f"{dialect} provider response contains foreign semantic fields: {', '.join(foreign_fields)}",
        )
    if dialect == "responses":
        _validate_responses_output(response, step, path=path)
    elif dialect == "chat":
        _validate_chat_output(response, step, path=path)
    else:
        _validate_anthropic_output(response, step, path=path)


def reconcile_provider_model_call(
    call: TrajectoryModelCall,
    step: AtifStep,
    *,
    path: str,
) -> ProviderReconciliationResult:
    """Validate known provider evidence against one projected ATIF step."""

    finish_reason = _reconciled_finish_reason(call, path=path)
    model_name, response_id = _reconcile_response_identity(call, path=path)
    token_stats = _reconciled_token_stats(call, path=path)
    _validate_provider_output_coverage(call, step, path=path)
    return ProviderReconciliationResult(
        finish_reason=finish_reason,
        model_name=model_name,
        response_id=response_id,
        token_stats=token_stats,
    )
