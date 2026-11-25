# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical looping assessment shared by eval workers and training consumers."""

import json
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from nemo_gym.rollouts.health.checks import _bind_policy_call_views, _normalized_trajectory_calls
from nemo_gym.trajectory import _looping_detector as detector
from nemo_gym.trajectory.looping import (
    MAX_OUTPUT_ITEMS,
    MAX_TOOL_CALLS,
    LoopingAssessment,
    LoopingInput,
    LoopingInputLimit,
    check_looping_input_size,
    detect_looping,
    unknown_looping_assessment,
)
from nemo_gym.trajectory.trajectory_record import TrajectoryRecord


class _Unassessable(ValueError):
    """A fixed diagnostic reason for incomplete or unsupported evidence."""


def _parts(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or len(value) > MAX_OUTPUT_ITEMS:
        raise _Unassessable("unsupported_text_parts")
    text_parts = [part.get("text", part.get("refusal")) for part in value]
    if any(not isinstance(text, str) for text in text_parts):
        raise _Unassessable("unavailable_text_part")
    return text_parts


def _response_events(response: dict[str, Any]) -> tuple[list[str], list[dict[str, Any]]]:
    """Read generated blocks in order; never flatten a producer's echoed conversation."""
    text_segments: list[str] = []
    tool_requests: list[dict[str, Any]] = []
    output = response.get("output")
    if isinstance(output, list):
        if len(output) > MAX_OUTPUT_ITEMS:
            raise LoopingInputLimit("output_item_limit")
        for item in output:
            item_type = item.get("type")
            if item_type == "reasoning":
                text_segments.extend(_parts(item.get("content") or item.get("summary")))
            elif item_type == "message" and item.get("role") == "assistant":
                text_segments.extend(_parts(item.get("content")))
            elif item_type == "function_call":
                tool_requests.append({key: item[key] for key in ("call_id", "name", "arguments") if key in item})
            else:
                raise _Unassessable("unsupported_output_item")
            if len(text_segments) + len(tool_requests) > MAX_OUTPUT_ITEMS:
                raise LoopingInputLimit("output_item_limit")
        return text_segments, tool_requests

    choices = response.get("choices")
    if isinstance(choices, list) and len(choices) == 1:
        message = choices[0].get("message") or {}
        text_segments.extend(_parts(message.get("reasoning_content")))
        text_segments.extend(_parts(message.get("content")))
        if len(message.get("tool_calls") or []) > MAX_TOOL_CALLS:
            raise LoopingInputLimit("tool_call_limit")
        for tool in message.get("tool_calls") or []:
            tool_requests.append({"call_id": tool.get("id"), **(tool.get("function") or {})})
        return text_segments, tool_requests

    content = response.get("content")
    if isinstance(content, list):
        if len(content) > MAX_OUTPUT_ITEMS:
            raise LoopingInputLimit("output_item_limit")
        for block in content:
            block_type = block.get("type")
            if block_type == "thinking":
                text_segments.extend(_parts(block.get("thinking")))
            elif block_type == "text":
                text_segments.extend(_parts(block.get("text")))
            elif block_type == "tool_use":
                tool_requests.append(
                    {"call_id": block.get("id"), "name": block.get("name"), "arguments": block.get("input")}
                )
            else:
                raise _Unassessable("unsupported_output_item")
        return text_segments, tool_requests
    raise _Unassessable("unsupported_policy_response")


def _policy_calls(
    trajectory: dict[str, Any],
    calls: list[dict[str, Any]],
    *,
    response_id: str | None,
    policy_model_server: str | None,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    invocations = trajectory.get("invocations") or []
    if len(invocations) > 1:
        raise _Unassessable("multiple_policy_invocations_unsupported")
    turns = trajectory.get("turns") or []
    turn_numbers = [turn["turn_no"] for turn in turns]
    if len(set(turn_numbers)) != len(turn_numbers):
        raise _Unassessable("policy_order_unavailable")
    turns = sorted(turns, key=lambda turn: turn["turn_no"])
    turn_bindings, _ = _bind_policy_call_views({"turns": turns}, calls)
    _, invocation_bindings = _bind_policy_call_views({"invocations": invocations}, calls)
    _, verifier_bindings = _bind_policy_call_views(
        {"invocations": trajectory.get("verifier_invocations") or []}, calls
    )
    if verifier_bindings.missing_references or verifier_bindings.duplicated_references:
        raise _Unassessable("verifier_ownership_unavailable")
    verifier_call_ids = {call["model_call_id"] for call in verifier_bindings.matched_calls}
    if not invocation_bindings.references and not turn_bindings.references:
        # Without ownership references, require an exact returned-response match.
        direct_calls = [
            call
            for call in calls
            if (call.get("model_ref") or {}).get("name") == policy_model_server
            and call["model_call_id"] not in verifier_call_ids
        ]
        if (
            policy_model_server
            and response_id
            and len(direct_calls) == 1
            and direct_calls[0]["response_id"] == response_id
        ):
            return direct_calls, ()
    policy_call_ids = {call["model_call_id"] for call in invocation_bindings.matched_calls} - verifier_call_ids
    if turn_bindings.references:
        policy_call_ids &= {call["model_call_id"] for call in turn_bindings.matched_calls}
    if not policy_call_ids:
        raise _Unassessable("policy_ownership_unavailable")
    gaps: tuple[str, ...] = ()
    if any(
        binding.missing_references or binding.duplicated_references for binding in (invocation_bindings, turn_bindings)
    ):
        gaps = ("policy_call_missing",)
    if not turn_bindings.references:
        if len(policy_call_ids) != 1:
            raise _Unassessable("policy_order_unavailable")
        return [call for call in calls if call["model_call_id"] in policy_call_ids], gaps

    # Resolution order follows the sorted turns rather than the physical capture list.
    ordered_calls = {}
    for resolved_calls in turn_bindings.resolutions.values():
        if len(resolved_calls) != 1:
            continue
        resolved_call = resolved_calls[0]
        if resolved_call["model_call_id"] in policy_call_ids:
            ordered_calls.setdefault(resolved_call["model_call_id"], resolved_call)
    return list(ordered_calls.values()), gaps


def extract_looping_events(
    trajectory: Mapping[str, Any],
    *,
    response_id: str | None = None,
    policy_model_server: str | None = None,
) -> LoopingInput:
    """Extract ordered fresh policy responses from canonical ownership and tool joins."""
    # Native validation also inspects unselected captures and echoed request bodies.
    check_looping_input_size(trajectory)
    if trajectory.get("schema_version", "1.0") not in {"1.0", "2.0"}:
        raise _Unassessable("unsupported_trajectory")
    for collection_name in ("model_calls", "invocations", "verifier_invocations", "turns", "tool_calls", "gaps"):
        if len(trajectory.get(collection_name) or []) > MAX_OUTPUT_ITEMS:
            raise LoopingInputLimit("output_item_limit")
    tool_observations = trajectory.get("tool_calls") or []
    if len(tool_observations) > MAX_TOOL_CALLS:
        raise LoopingInputLimit("tool_call_limit")
    reference_groups = [
        item.get("model_calls") or []
        for collection_name in ("invocations", "turns", "verifier_invocations")
        for item in trajectory.get(collection_name) or []
    ]
    if sum(len(references) for references in reference_groups) > MAX_OUTPUT_ITEMS:
        raise LoopingInputLimit("output_item_limit")
    normalized_calls = _normalized_trajectory_calls(dict(trajectory))
    if any(not isinstance(call["model_call_id"], str) or not call["model_call_id"] for call in normalized_calls):
        raise _Unassessable("model_call_id_unavailable")
    policy_calls, ownership_gaps = _policy_calls(
        dict(trajectory), normalized_calls, response_id=response_id, policy_model_server=policy_model_server
    )
    gaps = set(ownership_gaps)
    for gap in trajectory.get("gaps") or []:
        code = gap.get("code", "")
        if code in {
            "model_call_reference_ambiguous",
            "model_call_reference_conflict",
            "model_call_reference_unmatched",
        }:
            raise _Unassessable(code)
        if code.startswith("model_call_capture_"):
            gaps.add("model_call_capture_incomplete")
        elif code.startswith("model_call_ownership_"):
            gaps.add("policy_ownership_incomplete")
        elif code in {"model_calls_unavailable", "turn_model_call_scope_incomplete", "trajectory_projection_failed"}:
            gaps.add(code)
    text_known = not gaps
    tools_known = not gaps
    policy_call_ids = {call["model_call_id"] for call in policy_calls}
    tool_outputs_by_call = defaultdict(list)
    for tool in tool_observations:
        if tool.get("model_call_id") not in policy_call_ids:
            gaps.add("tool_ownership_unavailable")
            tools_known = False
        for tool_call_id in {tool.get("native_tool_call_id"), tool.get("tool_call_id")} - {None}:
            tool_outputs_by_call[(tool.get("model_call_id"), tool_call_id)].append(tool)

    text_segments = []
    tool_signatures = []
    paired_tools = []
    tool_request_count = 0
    output_item_count = 0
    for call in policy_calls:
        if call.get("status_code") is not None and call["status_code"] >= 400:
            continue
        try:
            response = call["response"]
            if isinstance(response, dict):
                response_items = response.get("output") or response.get("content") or response.get("choices") or []
                output_item_count += len(response_items)
                if output_item_count > MAX_OUTPUT_ITEMS:
                    raise LoopingInputLimit("output_item_limit")
            fresh_segments, tool_requests = _response_events(response)
        except (ValueError, TypeError, KeyError, AttributeError) as error:
            if isinstance(error, LoopingInputLimit):
                raise
            gaps.add("policy_output_unavailable")
            text_known = False
            tools_known = False
            continue
        text_segments.extend(fresh_segments)
        if len(text_segments) > MAX_OUTPUT_ITEMS:
            raise LoopingInputLimit("output_item_limit")
        if not any(fresh_segments) and not tool_requests and call.get("tokens_out") != 0:
            gaps.add("output_text_unavailable")
            text_known = False
        tool_request_count += len(tool_requests)
        if tool_request_count > MAX_TOOL_CALLS:
            raise LoopingInputLimit("tool_call_limit")
        for request in tool_requests:
            if not isinstance(request.get("name"), str) or not request["name"]:
                gaps.add("tool_request_incomplete")
                tools_known = False
                continue
            matched_tools = tool_outputs_by_call[(call["model_call_id"], request.get("call_id"))]
            if (
                len(matched_tools) != 1
                or not isinstance(matched_tools[0].get("output"), str)
                or matched_tools[0].get("status") not in {"completed", "failed", "timeout", "cancelled"}
            ):
                gaps.add("tool_pairing_incomplete")
                tools_known = False
                continue
            paired_tools.append((request, matched_tools[0]))

    for request, tool in paired_tools:
        arguments = request.get("arguments", "")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {"literal_arguments": arguments}
        tool_input_identity = [request.get("name"), arguments]
        input_signature = detector.sha(detector.canonical(tool_input_identity))
        normalized_output = tool["output"].replace("\r\n", "\n").strip()
        tool_result_identity = [*tool_input_identity, normalized_output, tool["status"]]
        tool_signatures.append(
            {
                "input_signature_sha256": input_signature,
                "signature_sha256": detector.sha(detector.canonical(tool_result_identity)),
            }
        )
    if any("<unk>" in segment for segment in text_segments):
        gaps.add("literal_unk_in_assistant_text")
        text_segments = []
        text_known = False
    return LoopingInput(
        segments=tuple(text_segments),
        tools=tuple(tool_signatures),
        text_known=text_known,
        tools_known=tools_known,
        gaps=tuple(sorted(gaps)),
    )


def assess_looping_record(record: Mapping[str, Any]) -> LoopingAssessment:
    """Assess canonical saved or prepared live evidence without altering the rollout."""
    try:
        trajectory = record.get("ng_trajectory")
        if not isinstance(trajectory, Mapping):
            return unknown_looping_assessment("trajectory_unavailable")
        response = record.get("response") or {}
        looping_input = extract_looping_events(
            trajectory,
            response_id=response.get("id"),
            policy_model_server=record.get("ng_policy_model_server"),
        )
        TrajectoryRecord.model_validate(trajectory)
        return detect_looping(looping_input)
    except (LoopingInputLimit, _Unassessable) as error:
        return unknown_looping_assessment(str(error))
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError, AssertionError):
        return unknown_looping_assessment("unsupported_trajectory")
