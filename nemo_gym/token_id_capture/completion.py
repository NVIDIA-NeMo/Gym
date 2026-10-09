# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Token-free evidence from the response actually served by the model gateway."""

from typing import Any

from nemo_gym.responses_converter import ResponsesConverter
from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint
from nemo_gym.token_id_capture.records import response_to_output_items, strip_token_fields
from nemo_gym.token_id_capture.replay import replay_context


def completion_items(response: dict[str, Any]) -> list[dict]:
    """Describe served output without reinterpreting Chat text as reasoning."""
    if isinstance(response.get("choices"), list):
        # The gateway has already applied its configured reasoning conversion.
        # A Chat client sees content (including literal <think> tags) as text;
        # splitting it again invents items absent from the harness's output.
        converter = ResponsesConverter(return_token_id_information=False, uses_reasoning_parser=False)
        items = []
        for choice in response["choices"]:
            for output in converter.postprocess_assistant_message_dict(choice["message"]):
                item = output.model_dump(mode="json")
                if item.get("type") != "function_call":
                    item.pop("id", None)
                items.append(item)
        return strip_token_fields(items)[0]
    return strip_token_fields(response_to_output_items(response))[0]


def completion_metadata(response: dict[str, Any]) -> tuple[str | None, str | None]:
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
    if response.get("error") is not None:
        status = "failed"
    if not isinstance(status, str):
        status = None
        if reason in ("stop", "tool_calls", "function_call", "end_turn", "tool_use", "stop_sequence"):
            status = "completed"
        elif reason in ("length", "max_tokens", "max_output_tokens", "content_filter"):
            status = "incomplete"
    return status, reason


def output_validation_item(item: dict[str, Any]) -> dict[str, Any]:
    """Keep original text used by output detectors, excluding transport arrays."""
    result: dict[str, Any] = {"type": item.get("type")}
    for key in ("content", "summary"):
        parts = item.get(key)
        if isinstance(parts, list):
            result[key] = [
                {"text": part["text"]} if isinstance(part, dict) and isinstance(part.get("text"), str) else {}
                for part in parts
            ]
    return result


def output_item_fingerprint(item: dict[str, Any]) -> str:
    """Reuse dialect-neutral visible fingerprints and exact reasoning evidence."""
    if item.get("type") == "reasoning":
        return replay_context([item]).items[0].digest
    result = assistant_fingerprint([item])
    if not result:
        raise ValueError("Expected a model-authored output item")
    return result
