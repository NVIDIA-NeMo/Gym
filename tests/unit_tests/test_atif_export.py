# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from nemo_gym import __version__
from nemo_gym.atif_export import (
    AtifExportError,
    ExportAtifConfig,
    export_rollouts_to_atif,
    gym_rollout_to_atif,
)
from nemo_gym.relay_atif import AtifStepMetrics, AtifTrajectoryV1_7
from nemo_gym.rollout_observability import TrajectoryRecord


def _reasoning(item_id: str, text: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "reasoning",
        "summary": [{"type": "summary_text", "text": text}],
    }


def _function_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"item-{call_id}",
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(arguments, sort_keys=True),
        "status": "completed",
    }


def _function_output(call_id: str, output: str) -> dict[str, Any]:
    return {
        "type": "function_call_output",
        "call_id": call_id,
        "output": output,
        "status": "completed",
    }


def _assistant_message(item_id: str, text: str) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _canonical_rollout(
    *,
    task_index: int = 3,
    rollout_index: int = 7,
    agent_name: str = "simple_agent",
) -> dict[str, Any]:
    system = {"type": "message", "role": "system", "content": "Use tools when needed."}
    user = {"type": "message", "role": "user", "content": "Compare the two records."}
    first_reasoning = _reasoning("reason-1", "Read both records before comparing them.")
    search_call = _function_call("call-search", "search", {"query": "relay"})
    read_call = _function_call("call-read", "read", {"path": "/tmp/record.txt"})
    # Results deliberately arrive in the opposite order from the calls. Correct
    # conversion must correlate them by call_id, not list position or timing.
    read_output = _function_output("call-read", "record contents")
    search_output = _function_output("call-search", "search results")
    second_reasoning = _reasoning("reason-2", "The records agree.")
    final_message = _assistant_message("message-1", "Both records contain the same value.")
    first_question = [system, user]
    second_question = [
        system,
        user,
        first_reasoning,
        search_call,
        read_call,
        read_output,
        search_output,
    ]
    task_id = f"task-{task_index}"
    source_rollout_id = f"rollout-{task_index}-{rollout_index}"
    model_ref = {"type": "responses_api_models", "name": "policy"}

    return {
        "_ng_task_index": task_index,
        "_ng_rollout_index": rollout_index,
        "agent_ref": {"type": "responses_api_agents", "name": agent_name},
        "ng_trajectory": {
            "schema_version": "1.0",
            "task_id": task_id,
            "rollout_id": source_rollout_id,
            "invocations": [
                {
                    "invocation_id": "root",
                    "status": "completed",
                    "model_calls": [
                        {"model_call_id": "model-call-1"},
                        {"model_call_id": "model-call-2"},
                    ],
                    "conversation": [*second_question, second_reasoning, final_message],
                }
            ],
            "turns": [
                {
                    "invocation_id": "root",
                    "task_id": task_id,
                    "rollout_id": source_rollout_id,
                    "turn_no": 1,
                    "timestamp": 1_700_000_000.0,
                    "question": first_question,
                    "answer": [search_call, read_call],
                    "reasoning_content": [first_reasoning],
                    "resolved": False,
                    "step_count": 2,
                    "model_calls": [{"model_call_id": "model-call-1"}],
                },
                {
                    "invocation_id": "root",
                    "task_id": task_id,
                    "rollout_id": source_rollout_id,
                    "turn_no": 2,
                    "timestamp": 1_700_000_002.0,
                    "question": second_question,
                    "answer": [final_message],
                    "reasoning_content": [second_reasoning],
                    "resolved": True,
                    "step_count": 2,
                    "model_calls": [{"model_call_id": "model-call-2"}],
                },
            ],
            "model_calls": [
                {
                    "model_call_id": "model-call-1",
                    "started_at": 1_700_000_000.0,
                    "completed_at": 1_700_000_000.045,
                    "duration_ms": 45.0,
                    "request": {"input": "model-visible-input-1"},
                    "response": {"status": "completed", "output": "model-visible-output-1"},
                    "response_metadata": {
                        "response_id": "response-1",
                        "model_ref": model_ref,
                        "model": "model-a",
                        "dialect": "responses",
                        "status_code": 200,
                        "response_status": "completed",
                        "finish_reason": "tool_calls",
                        "latency_ttft_ms": 12.0,
                    },
                    "token_stats": {
                        "prompt_tokens": 101,
                        "completion_tokens": 11,
                        "reasoning_tokens": 7,
                        "total_tokens": 112,
                        "cached_tokens": 5,
                    },
                },
                {
                    "model_call_id": "model-call-2",
                    "duration_ms": 30.0,
                    "response_metadata": {
                        "response_id": "response-2",
                        "model_ref": model_ref,
                        "model": "model-a",
                        "response_status": "completed",
                    },
                    "token_stats": {
                        "prompt_tokens": 131,
                        "completion_tokens": 13,
                        "reasoning_tokens": 3,
                        "total_tokens": 144,
                        "cached_tokens": 8,
                    },
                },
            ],
            "tool_calls": [
                {
                    "invocation_id": "root",
                    "tool_call_id": "call-read",
                    "tool_name": "read",
                    "status": "completed",
                    "started_at": 1_700_000_001.0,
                    "completed_at": 1_700_000_001.1,
                    "duration_ms": 100.0,
                    "timing_source": "executor",
                    "output": "record contents",
                },
                {
                    "invocation_id": "root",
                    "tool_call_id": "call-search",
                    "tool_name": "search",
                    "status": "completed",
                    "started_at": 1_700_000_001.2,
                    "completed_at": 1_700_000_001.25,
                    "duration_ms": 50.0,
                    "timing_source": "executor",
                    "output": "search results",
                },
            ],
            "gaps": [],
        },
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8")


def test_gym_rollout_to_atif_preserves_provenance_turns_metrics_and_tool_identity() -> None:
    rollout = _canonical_rollout()
    original = copy.deepcopy(rollout)

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert rollout == original
    # Re-validating the encoded wire object catches output that is valid only as
    # an in-memory model due to assignment without validation.
    encoded = trajectory.model_dump(mode="json", exclude_none=True)
    AtifTrajectoryV1_7.model_validate(encoded)
    assert trajectory.schema_version == "ATIF-v1.7"
    assert trajectory.session_id == "evaluation-42"
    assert trajectory.trajectory_id == "evaluation-42:3:7"
    assert trajectory.agent.model_dump(exclude_none=True) == {
        "name": "simple_agent",
        "version": "2.3.1",
        "model_name": "model-a",
    }
    assert [(step.step_id, step.source) for step in trajectory.steps] == [
        (1, "system"),
        (2, "user"),
        (3, "agent"),
        (4, "agent"),
    ]
    assert trajectory.steps[0].message == "Use tools when needed."
    assert trajectory.steps[1].message == "Compare the two records."
    assert trajectory.steps[2].message == ""
    assert [part.model_dump(exclude_none=True) for part in trajectory.steps[3].message] == [
        {"type": "text", "text": "Both records contain the same value."}
    ]

    tool_step = trajectory.steps[2]
    assert tool_step.reasoning_content == "Read both records before comparing them."
    assert [call.tool_call_id for call in tool_step.tool_calls or []] == ["call-search", "call-read"]
    assert [call.extra for call in tool_step.tool_calls or []] == [
        {"nemo_gym": {"source_item_id": "item-call-search"}},
        {"nemo_gym": {"source_item_id": "item-call-read"}},
    ]
    assert [result.source_call_id for result in tool_step.observation.results] == ["call-read", "call-search"]
    assert [result.content for result in tool_step.observation.results] == ["record contents", "search results"]
    assert tool_step.observation.results[0].extra == {
        "nemo_gym": {
            "started_at": 1_700_000_001.0,
            "completed_at": 1_700_000_001.1,
            "duration_ms": 100.0,
            "timing_source": "executor",
        }
    }
    assert tool_step.timestamp == "2023-11-14T22:13:20Z"
    assert tool_step.metrics.model_dump(exclude_none=True) == {
        "prompt_tokens": 101,
        "completion_tokens": 11,
        "cached_tokens": 5,
        "extra": {"nemo_gym": {"reasoning_tokens": 7, "total_tokens": 112}},
    }
    assert tool_step.extra["nemo_gym"]["turn"] == {
        "turn_no": 1,
        "step_count": 2,
        "resolved": False,
    }
    assert tool_step.extra["nemo_gym"]["source_items"] == {"reasoning_ids": ["reason-1"]}
    source_trajectory = TrajectoryRecord.model_validate(rollout["ng_trajectory"])
    assert tool_step.extra["nemo_gym"]["model_call"] == source_trajectory.model_calls[0].model_dump(
        mode="json", exclude_none=True
    )
    assert trajectory.steps[3].reasoning_content == "The records agree."
    assert trajectory.steps[3].extra["nemo_gym"]["turn"] == {
        "turn_no": 2,
        "step_count": 2,
        "resolved": True,
    }
    assert trajectory.steps[3].extra["nemo_gym"]["source_items"] == {
        "reasoning_ids": ["reason-2"],
        "message_id": "message-1",
    }
    assert trajectory.final_metrics.model_dump(exclude_none=True) == {
        "total_prompt_tokens": 232,
        "total_completion_tokens": 24,
        "total_cached_tokens": 13,
        "total_steps": 4,
    }
    assert trajectory.extra == {
        "nemo_gym": {
            "exporter": {"name": "nemo-gym", "version": __version__},
            "source": {
                "format": "ng_trajectory",
                "schema_version": "1.0",
                "task_id": "task-3",
                "rollout_id": "rollout-3-7",
                "task_index": 3,
                "rollout_index": 7,
                "invocation_id": "root",
                "invocation_status": "completed",
            },
            "conversion": {"profile": "ng-trajectory-to-atif-v1", "status": "complete"},
        }
    }


def test_gym_rollout_to_atif_preserves_optional_function_output_item_ids() -> None:
    rollout = _canonical_rollout()
    conversation = rollout["ng_trajectory"]["invocations"][0]["conversation"]
    second_question = rollout["ng_trajectory"]["turns"][1]["question"]
    for source in (conversation, second_question):
        source[5]["id"] = "item-output-read"
        source[6]["id"] = "item-output-search"

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    results = trajectory.steps[2].observation.results
    assert [result.extra["nemo_gym"]["source_item_id"] for result in results] == [
        "item-output-read",
        "item-output-search",
    ]


def test_gym_rollout_to_atif_preserves_reasoning_and_total_tokens_without_standard_counts() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["token_stats"] = {
        "reasoning_tokens": 9,
        "total_tokens": 9,
    }

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[2].metrics.model_dump(exclude_none=True) == {
        "extra": {"nemo_gym": {"reasoning_tokens": 9, "total_tokens": 9}}
    }


def test_gym_rollout_to_atif_always_emits_exact_total_steps_without_complete_token_metrics() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["token_stats"] = {}

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.final_metrics.model_dump(exclude_none=True) == {"total_steps": 4}


@pytest.mark.parametrize(
    ("session_id", "agent_version", "message"),
    [
        (" \t", "2.3.1", "session_id: expected a non-empty string"),
        ("evaluation-42", " \n", "agent_version: expected a non-empty string"),
        (None, "2.3.1", "session_id: expected a non-empty string"),
        (7, "2.3.1", "session_id: expected a non-empty string"),
        ("evaluation-42", None, "agent_version: expected a non-empty string"),
        ("evaluation-42", 7, "agent_version: expected a non-empty string"),
    ],
)
def test_gym_rollout_to_atif_rejects_blank_direct_call_identity(
    session_id: Any,
    agent_version: Any,
    message: str,
) -> None:
    with pytest.raises(AtifExportError, match=message):
        gym_rollout_to_atif(_canonical_rollout(), session_id=session_id, agent_version=agent_version)


@pytest.mark.parametrize("field_name", ("tool_name", "output"))
def test_strict_export_accepts_absent_optional_tool_record_evidence(field_name: str) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["tool_calls"][0][field_name] = None

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[2].tool_calls[1].function_name == "read"
    assert trajectory.steps[2].observation.results[0].content == "record contents"


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("tool_name", "different-tool", "different tool name"),
        ("output", "different output", "different recorded output"),
    ],
)
def test_strict_export_rejects_conflicting_optional_tool_record_evidence(
    field_name: str,
    field_value: str,
    message: str,
) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["tool_calls"][0][field_name] = field_value

    with pytest.raises(AtifExportError, match=message):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_rejects_inconsistent_model_call_total_tokens() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["token_stats"]["total_tokens"] = 999

    with pytest.raises(AtifExportError, match=r"total_tokens does not match prompt_tokens \+ completion_tokens"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_rejects_cached_tokens_greater_than_prompt_tokens() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["token_stats"]["cached_tokens"] = 102

    with pytest.raises(AtifExportError, match="cached_tokens exceeds prompt_tokens"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize("field_name", ("prompt_tokens", "completion_tokens", "reasoning_tokens", "cached_tokens"))
def test_strict_export_rejects_partial_token_components_greater_than_total(field_name: str) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["token_stats"] = {
        field_name: 10,
        "total_tokens": 5,
    }

    with pytest.raises(AtifExportError, match=rf"{field_name} exceeds total_tokens"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_does_not_treat_an_explicit_null_as_an_absent_redundant_field() -> None:
    rollout = _canonical_rollout()
    question = rollout["ng_trajectory"]["turns"][0]["question"]
    question[0] = copy.deepcopy(question[0])
    question[0]["unrepresented_field"] = None

    with pytest.raises(AtifExportError, match="unsupported fields would be dropped: 'unrepresented_field'"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_accepts_known_optional_nulls_from_producer_model_dump() -> None:
    rollout = _canonical_rollout()
    record = TrajectoryRecord.model_validate(rollout["ng_trajectory"])
    conversation = record.invocations[0].conversation
    record.turns[0].question = conversation[:2]
    record.turns[0].reasoning_content = [conversation[2]]
    record.turns[0].answer = conversation[3:5]
    record.turns[1].question = conversation[:7]
    record.turns[1].reasoning_content = [conversation[7]]
    record.turns[1].answer = [conversation[8]]
    rollout["ng_trajectory"] = record.model_dump(mode="json")

    assert rollout["ng_trajectory"]["turns"][0]["reasoning_content"][0]["encrypted_content"] is None
    assert rollout["ng_trajectory"]["turns"][1]["answer"][0]["content"][0]["logprobs"] is None

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[2].reasoning_content == "Read both records before comparing them."
    assert trajectory.steps[3].message[0].text == "Both records contain the same value."


@pytest.mark.parametrize("location", ("agent", "captured", "turn", "invocation"))
def test_strict_export_rejects_unknown_null_fields_in_nested_server_references(location: str) -> None:
    rollout = _canonical_rollout()
    if location == "agent":
        reference = rollout["agent_ref"]
    elif location == "captured":
        reference = rollout["ng_trajectory"]["model_calls"][0]["response_metadata"]["model_ref"]
    elif location == "turn":
        reference = rollout["ng_trajectory"]["turns"][0]["model_calls"][0]
        reference["model_ref"] = {"type": "responses_api_models", "name": "policy"}
        reference = reference["model_ref"]
    else:
        reference = rollout["ng_trajectory"]["invocations"][0]["model_calls"][0]
        reference["model_ref"] = {"type": "responses_api_models", "name": "policy"}
        reference = reference["model_ref"]
    reference["unrepresented_field"] = None

    with pytest.raises(AtifExportError, match="unsupported fields would be dropped: 'unrepresented_field'"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_distinguishes_json_booleans_from_token_id_numbers_across_redundant_copies() -> None:
    rollout = _canonical_rollout()
    final_message = rollout["ng_trajectory"]["invocations"][0]["conversation"][-1]
    final_message.update(
        {
            "prompt_token_ids": [1],
            "generation_token_ids": [2],
            "generation_log_probs": [-0.1],
        }
    )
    turn_answer = copy.deepcopy(final_message)
    turn_answer["prompt_token_ids"] = [True]
    rollout["ng_trajectory"]["turns"][-1]["answer"] = [turn_answer]

    with pytest.raises(AtifExportError, match="expected a non-negative integer, not a boolean"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_gym_rollout_to_atif_preserves_training_token_metadata_independently_from_usage_counts() -> None:
    rollout = _canonical_rollout()
    final_message = rollout["ng_trajectory"]["invocations"][0]["conversation"][-1]
    final_message.update(
        {
            "prompt_token_ids": [10, 11],
            "generation_token_ids": [12, 13],
            "generation_log_probs": [-0.25, -0.5],
            "routed_experts": "nrlre1:uint16:2x1x1:AAAA",
        }
    )

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[3].metrics.model_dump(exclude_none=True) == {
        "prompt_tokens": 131,
        "completion_tokens": 13,
        "cached_tokens": 8,
        "prompt_token_ids": [10, 11],
        "completion_token_ids": [12, 13],
        "logprobs": [-0.25, -0.5],
        "extra": {
            "nemo_gym": {
                "reasoning_tokens": 3,
                "total_tokens": 144,
                "routed_experts": "nrlre1:uint16:2x1x1:AAAA",
            }
        },
    }


@pytest.mark.parametrize("owner", ("reasoning", "earlier-parallel-call"))
def test_strict_export_rejects_training_metadata_that_ingress_cannot_restore(owner: str) -> None:
    rollout = _canonical_rollout()
    conversation = rollout["ng_trajectory"]["invocations"][0]["conversation"]
    item = conversation[7] if owner == "reasoning" else conversation[3]
    item.update(
        {
            "prompt_token_ids": [10],
            "generation_token_ids": [11],
            "generation_log_probs": [-0.25],
        }
    )

    with pytest.raises(AtifExportError, match="final model-generated output item"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_gym_rollout_to_atif_accepts_training_metadata_on_the_final_parallel_call() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["conversation"][4].update(
        {
            "prompt_token_ids": [10],
            "generation_token_ids": [11, 12],
            "generation_log_probs": [-1, 0],
            "routed_experts": [[[1]], [[2]]],
        }
    )

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    metrics = trajectory.steps[2].metrics
    assert metrics.prompt_token_ids == [10]
    assert metrics.completion_token_ids == [11, 12]
    assert metrics.logprobs == [-1.0, 0.0]
    assert metrics.extra["nemo_gym"]["routed_experts"] == [[[1]], [[2]]]


def test_strict_export_rejects_multiple_training_token_payloads_for_one_model_call() -> None:
    rollout = _canonical_rollout()
    for item in rollout["ng_trajectory"]["invocations"][0]["conversation"][-2:]:
        item.update(
            {
                "prompt_token_ids": [10],
                "generation_token_ids": [11],
                "generation_log_probs": [-0.25],
            }
        )

    with pytest.raises(AtifExportError, match="more than one output item carries training token metadata"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_rejects_misaligned_training_token_metadata() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["conversation"][-1].update(
        {
            "prompt_token_ids": [10],
            "generation_token_ids": [11, 12],
            "generation_log_probs": [-0.25],
        }
    )

    with pytest.raises(AtifExportError, match="token IDs and log probabilities must have the same length"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_rejects_training_token_metadata_on_source_messages() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["conversation"][1].update(
        {
            "prompt_token_ids": [10],
            "generation_token_ids": [11],
            "generation_log_probs": [-0.25],
        }
    )

    with pytest.raises(AtifExportError, match="training token metadata is only supported on agent output"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("prompt_token_ids", [True], "expected a non-negative integer"),
        ("prompt_token_ids", ["10"], "expected a non-negative integer"),
        ("generation_token_ids", [1.0], "expected a non-negative integer"),
        ("generation_token_ids", [-1], "expected a non-negative integer"),
        ("generation_log_probs", [False], "expected a finite number"),
        ("generation_log_probs", ["-0.25"], "expected a finite number"),
        ("routed_experts", [[[True]]], "expected an integer"),
        ("routed_experts", [[["2"]]], "expected an integer"),
        ("routed_experts", [[[1.5]]], "expected an integer"),
    ],
)
def test_strict_export_rejects_coercible_training_numbers(
    field_name: str,
    invalid_value: Any,
    message: str,
) -> None:
    rollout = _canonical_rollout()
    final_message = rollout["ng_trajectory"]["invocations"][0]["conversation"][-1]
    final_message.update(
        {
            "prompt_token_ids": [10],
            "generation_token_ids": [11],
            "generation_log_probs": [-0.25],
            field_name: invalid_value,
        }
    )

    with pytest.raises(AtifExportError, match=message):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("prompt_token_ids", [True], "token IDs must be non-negative JSON integer arrays"),
        ("completion_token_ids", ["11"], "token IDs must be non-negative JSON integer arrays"),
        ("completion_token_ids", [-1], "token IDs must be non-negative JSON integer arrays"),
        ("logprobs", [False], "log probabilities must be finite JSON number arrays"),
        ("logprobs", ["-0.25"], "log probabilities must be finite JSON number arrays"),
    ],
)
def test_atif_output_metrics_do_not_coerce_training_numbers(
    field_name: str,
    invalid_value: list[Any],
    message: str,
) -> None:
    with pytest.raises(ValidationError, match=message):
        AtifStepMetrics.model_validate({field_name: invalid_value})


@pytest.mark.parametrize("item_index", (0, 2, 3, 5, 8))
@pytest.mark.parametrize("field_value", ("secret", None))
def test_strict_export_rejects_unknown_fields_on_supported_raw_conversation_items(
    item_index: int,
    field_value: Any,
) -> None:
    rollout = _canonical_rollout()
    conversation = rollout["ng_trajectory"]["invocations"][0]["conversation"]
    conversation[item_index] = copy.deepcopy(conversation[item_index])
    conversation[item_index]["future_field"] = field_value

    with pytest.raises(AtifExportError, match="unsupported fields would be dropped: 'future_field'"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_rejects_a_missing_conversation_type_before_unknown_fields_can_disappear() -> None:
    rollout = _canonical_rollout()
    conversation = rollout["ng_trajectory"]["invocations"][0]["conversation"]
    conversation[-1] = copy.deepcopy(conversation[-1])
    conversation[-1].pop("type")
    conversation[-1]["future_field"] = "secret"

    with pytest.raises(AtifExportError, match="expected a non-empty supported conversation item type"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_rejects_a_missing_content_part_type_before_unknown_fields_can_disappear() -> None:
    rollout = _canonical_rollout()
    conversation = rollout["ng_trajectory"]["invocations"][0]["conversation"]
    conversation[-1] = copy.deepcopy(conversation[-1])
    conversation[-1]["content"][0].pop("type")
    conversation[-1]["content"][0]["future_field"] = "secret"

    with pytest.raises(AtifExportError, match="expected a non-empty supported content-part type"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_rejects_status_on_scalar_source_content_before_pydantic_can_drop_it() -> None:
    rollout = _canonical_rollout()
    conversation = rollout["ng_trajectory"]["invocations"][0]["conversation"]
    conversation[0] = copy.deepcopy(conversation[0])
    conversation[0]["status"] = "incomplete"

    with pytest.raises(AtifExportError, match="status is only supported with multipart content"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize("field_value", ("secret", None))
def test_strict_export_rejects_unknown_raw_output_text_fields_before_validation(field_value: Any) -> None:
    rollout = _canonical_rollout()
    conversation = rollout["ng_trajectory"]["invocations"][0]["conversation"]
    conversation[-1] = copy.deepcopy(conversation[-1])
    conversation[-1]["content"][0]["future_field"] = field_value

    with pytest.raises(AtifExportError, match=r"conversation\[8\]\.content\[0\].*future_field"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize("part_kind", ("input-text", "reasoning-summary", "reasoning-content", "tool-output"))
def test_strict_export_rejects_unknown_fields_on_other_supported_raw_content_parts(part_kind: str) -> None:
    rollout = _canonical_rollout()
    trajectory = rollout["ng_trajectory"]
    conversation = trajectory["invocations"][0]["conversation"]

    if part_kind == "input-text":
        clean_item = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "Compare the two records."}],
        }
        conversation[1] = copy.deepcopy(clean_item)
        trajectory["turns"][0]["question"][1] = copy.deepcopy(clean_item)
        trajectory["turns"][1]["question"][1] = copy.deepcopy(clean_item)
        target = conversation[1]["content"][0]
    elif part_kind == "reasoning-summary":
        conversation[2] = copy.deepcopy(conversation[2])
        target = conversation[2]["summary"][0]
    elif part_kind == "reasoning-content":
        clean_item = {
            "id": "reason-1",
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "Read both records before comparing them."}],
        }
        conversation[2] = copy.deepcopy(clean_item)
        trajectory["turns"][0]["reasoning_content"][0] = copy.deepcopy(clean_item)
        trajectory["turns"][1]["question"][2] = copy.deepcopy(clean_item)
        target = conversation[2]["content"][0]
    else:
        clean_output = [{"type": "input_text", "text": "record contents"}]
        conversation[5] = copy.deepcopy(conversation[5])
        conversation[5]["output"] = copy.deepcopy(clean_output)
        trajectory["turns"][1]["question"][5] = copy.deepcopy(conversation[5])
        trajectory["tool_calls"][0]["output"] = copy.deepcopy(clean_output)
        target = conversation[5]["output"][0]
    target["future_field"] = "secret"

    with pytest.raises(AtifExportError, match="unsupported fields would be dropped: 'future_field'"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_gym_rollout_to_atif_preserves_text_only_multipart_tool_results() -> None:
    rollout = _canonical_rollout()
    result = [
        {"type": "input_text", "text": "first line"},
        {"type": "input_text", "text": "second line"},
    ]
    rollout["ng_trajectory"]["invocations"][0]["conversation"][5]["output"] = result
    rollout["ng_trajectory"]["tool_calls"][0]["output"] = result

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    exported = trajectory.steps[2].observation.results[0].content
    assert [part.model_dump(exclude_none=True) for part in exported] == [
        {"type": "text", "text": "first line"},
        {"type": "text", "text": "second line"},
    ]


def test_strict_export_rejects_non_text_multipart_tool_results() -> None:
    rollout = _canonical_rollout()
    result = [{"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "auto"}]
    rollout["ng_trajectory"]["invocations"][0]["conversation"][5]["output"] = result
    rollout["ng_trajectory"]["tool_calls"][0]["output"] = result

    with pytest.raises(AtifExportError, match="only text content parts are supported"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_rejects_empty_multipart_tool_results() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["conversation"][5]["output"] = []
    rollout["ng_trajectory"]["tool_calls"][0]["output"] = []

    with pytest.raises(AtifExportError, match="expected non-empty scalar or multipart text content"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_gym_rollout_to_atif_does_not_infer_unknown_step_model_from_other_turns() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][1]["response_metadata"]["model"] = None

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.agent.model_name is None
    assert [step.model_name for step in trajectory.steps if step.source == "agent"] == ["model-a", None]


def test_gym_rollout_to_atif_uses_unique_gym_indices_for_trajectory_identity() -> None:
    first = _canonical_rollout(task_index=3, rollout_index=7)
    second = _canonical_rollout(task_index=4, rollout_index=0)
    second["ng_trajectory"]["task_id"] = first["ng_trajectory"]["task_id"]
    second["ng_trajectory"]["rollout_id"] = first["ng_trajectory"]["rollout_id"]
    for turn in second["ng_trajectory"]["turns"]:
        turn["task_id"] = first["ng_trajectory"]["task_id"]
        turn["rollout_id"] = first["ng_trajectory"]["rollout_id"]

    first_atif = gym_rollout_to_atif(first, session_id="evaluation-42", agent_version="2.3.1")
    second_atif = gym_rollout_to_atif(second, session_id="evaluation-42", agent_version="2.3.1")

    assert first_atif.trajectory_id == "evaluation-42:3:7"
    assert second_atif.trajectory_id == "evaluation-42:4:0"


def test_strict_export_requires_exact_turn_and_model_call_evidence() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["turns"] = []
    rollout["ng_trajectory"]["model_calls"] = []
    rollout["ng_trajectory"]["invocations"][0]["model_calls"] = []

    with pytest.raises(AtifExportError, match="expected one turn for each of 2 agent steps, got 0"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize("field", ("prompt_tokens", "timestamp", "step_count"))
def test_strict_export_does_not_coerce_boolean_source_scalars(field: str) -> None:
    rollout = _canonical_rollout()
    if field == "prompt_tokens":
        rollout["ng_trajectory"]["model_calls"][0]["token_stats"][field] = True
    else:
        rollout["ng_trajectory"]["turns"][0][field] = True

    with pytest.raises(AtifExportError, match="missing or invalid v1.0 trajectory"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("error_category", "provider_error", "provider error evidence"),
        ("status_code", 302, "non-success status 302"),
        ("status_code", 500, "non-success status 500"),
    ],
)
def test_strict_export_rejects_model_calls_with_failure_evidence(
    field_name: str,
    field_value: Any,
    message: str,
) -> None:
    rollout = _canonical_rollout()
    metadata = rollout["ng_trajectory"]["model_calls"][0]["response_metadata"]
    metadata["response_status"] = None
    metadata[field_name] = field_value

    with pytest.raises(AtifExportError, match=message):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_allows_legacy_missing_response_status_without_failure_evidence() -> None:
    rollout = _canonical_rollout()
    metadata = rollout["ng_trajectory"]["model_calls"][0]["response_metadata"]
    metadata["response_status"] = None
    metadata["status_code"] = 204

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[2].extra["nemo_gym"]["model_call"]["response_metadata"]["status_code"] == 204


@pytest.mark.parametrize(
    ("dialect", "response"),
    [
        ("responses", {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}}),
        ("responses", {"status": "failed"}),
        ("chat", {"choices": [{"finish_reason": "length"}]}),
        ("messages", {"stop_reason": "max_tokens"}),
        ("messages", {"stop_reason": "pause_turn"}),
    ],
)
def test_strict_export_rejects_provider_native_incomplete_responses(
    dialect: str,
    response: dict[str, Any],
) -> None:
    rollout = _canonical_rollout()
    model_call = rollout["ng_trajectory"]["model_calls"][1]
    model_call["response"] = response
    model_call["response_metadata"]["dialect"] = dialect
    model_call["response_metadata"]["response_status"] = None
    model_call["response_metadata"]["finish_reason"] = None

    with pytest.raises(AtifExportError, match="not completed|indicates an incomplete model response"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    ("dialect", "response", "message"),
    [
        ("responses", {"status": 200}, "Responses status is not a string"),
        ("responses", {"status": "completed", "error": {"message": "failed"}}, "contains error evidence"),
        ("responses", {"status": "completed", "incomplete_details": "invalid"}, "is not an object"),
        (
            "responses",
            {"status": "completed", "incomplete_details": {"reason": 123}},
            "reason is not a string",
        ),
        ("responses", {"status": "completed", "incomplete_details": {}}, "contains incomplete_details"),
        ("chat", {"choices": {}}, "Chat choices is not an array"),
        ("chat", {"choices": [None]}, "Chat first choice is not an object"),
        ("chat", {"choices": [{"finish_reason": 123}]}, "Chat finish_reason is not a string"),
        ("messages", {"stop_reason": 123}, "Messages stop_reason is not a string"),
    ],
)
def test_strict_export_rejects_malformed_provider_termination_evidence(
    dialect: str,
    response: dict[str, Any],
    message: str,
) -> None:
    rollout = _canonical_rollout()
    model_call = rollout["ng_trajectory"]["model_calls"][1]
    model_call["response"] = response
    model_call["response_metadata"]["dialect"] = dialect
    model_call["response_metadata"]["response_status"] = None
    model_call["response_metadata"]["finish_reason"] = None

    with pytest.raises(AtifExportError, match=message):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    ("dialect", "response", "message"),
    [
        ("chat", {"error": {"message": "failed"}}, "chat provider response contains error evidence"),
        (
            "chat",
            {"error": {"message": "failed"}, "choices": [{"finish_reason": "stop"}]},
            "chat provider response contains error evidence",
        ),
        (
            "messages",
            {"type": "error", "error": {"message": "failed"}},
            "messages provider response contains error evidence",
        ),
        (
            "messages",
            {"type": "message", "error": {"message": "failed"}, "stop_reason": "end_turn"},
            "messages provider response contains error evidence",
        ),
        ("messages", {"type": "error", "error": None}, "messages provider response contains error evidence"),
        ("messages", {"type": 123, "stop_reason": "end_turn"}, "Messages type is not a string"),
    ],
)
def test_strict_export_rejects_provider_native_error_envelopes(
    dialect: str,
    response: dict[str, Any],
    message: str,
) -> None:
    rollout = _canonical_rollout()
    model_call = rollout["ng_trajectory"]["model_calls"][1]
    model_call["response"] = response
    model_call["response_metadata"]["dialect"] = dialect
    model_call["response_metadata"]["response_status"] = None
    model_call["response_metadata"]["finish_reason"] = None

    with pytest.raises(AtifExportError, match=message):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    ("dialect", "response", "finish_reason"),
    [
        ("chat", {"error": None, "choices": [{"finish_reason": "stop"}]}, "stop"),
        ("messages", {"type": "message", "error": None, "stop_reason": "end_turn"}, "end_turn"),
    ],
)
def test_strict_export_accepts_non_error_chat_and_messages_envelopes(
    dialect: str,
    response: dict[str, Any],
    finish_reason: str,
) -> None:
    rollout = _canonical_rollout()
    model_call = rollout["ng_trajectory"]["model_calls"][1]
    model_call["response"] = response
    model_call["response_metadata"]["dialect"] = dialect
    model_call["response_metadata"]["response_status"] = None
    model_call["response_metadata"]["finish_reason"] = finish_reason

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[3].extra["nemo_gym"]["model_call"]["response"] == response


@pytest.mark.parametrize("response", ("raw response", [], 200, True))
def test_strict_export_rejects_non_object_responses_for_known_dialects(response: Any) -> None:
    rollout = _canonical_rollout()
    model_call = rollout["ng_trajectory"]["model_calls"][1]
    model_call["response"] = response
    model_call["response_metadata"]["dialect"] = "responses"

    with pytest.raises(AtifExportError, match="responses provider response is not an object"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    ("dialect", "response", "metadata_field", "metadata_value"),
    [
        ("responses", {"status": "completed"}, "response_status", "queued"),
        ("chat", {"choices": [{"finish_reason": "stop"}]}, "finish_reason", "vendor_reason"),
        ("messages", {"stop_reason": "end_turn"}, "finish_reason", "stop_sequence"),
    ],
)
def test_strict_export_rejects_conflicting_normalized_provider_termination(
    dialect: str,
    response: dict[str, Any],
    metadata_field: str,
    metadata_value: str,
) -> None:
    rollout = _canonical_rollout()
    model_call = rollout["ng_trajectory"]["model_calls"][1]
    model_call["response"] = response
    model_call["response_metadata"]["dialect"] = dialect
    model_call["response_metadata"][metadata_field] = metadata_value

    with pytest.raises(AtifExportError, match="conflicts with the provider response|not completed"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_does_not_interpret_unknown_dialect_termination_fields() -> None:
    rollout = _canonical_rollout()
    model_call = rollout["ng_trajectory"]["model_calls"][1]
    model_call["response"] = {"status": "domain_status", "choices": [{"finish_reason": "length"}]}
    model_call["response_metadata"]["dialect"] = "custom"
    model_call["response_metadata"]["response_status"] = None
    model_call["response_metadata"]["finish_reason"] = None

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[3].extra["nemo_gym"]["model_call"]["response_metadata"]["dialect"] == "custom"


@pytest.mark.parametrize(
    ("model_call_index", "dialect", "response", "finish_reason"),
    [
        (1, "responses", {"status": "completed"}, None),
        (1, "chat", {"status": "domain_status", "choices": [{"finish_reason": "stop"}]}, "stop"),
        (1, "messages", {"stop_reason": "end_turn"}, "end_turn"),
        (1, "messages", {"stop_reason": "stop_sequence"}, "stop_sequence"),
        (0, "chat", {"choices": [{"finish_reason": "tool_calls"}]}, "tool_calls"),
        (0, "chat", {"choices": [{"finish_reason": "function_call"}]}, "function_call"),
        (0, "messages", {"stop_reason": "tool_use"}, "tool_use"),
    ],
)
def test_strict_export_accepts_provider_native_terminal_evidence_that_matches_projected_output(
    model_call_index: int,
    dialect: str,
    response: dict[str, Any],
    finish_reason: str | None,
) -> None:
    rollout = _canonical_rollout()
    model_call = rollout["ng_trajectory"]["model_calls"][model_call_index]
    model_call["response"] = response
    metadata = model_call["response_metadata"]
    metadata["dialect"] = dialect
    metadata["response_status"] = "completed" if dialect == "responses" else None
    metadata["finish_reason"] = finish_reason

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    step_index = 2 if model_call_index == 0 else 3
    assert trajectory.steps[step_index].extra["nemo_gym"]["model_call"]["response"] == response


@pytest.mark.parametrize(
    ("model_call_index", "finish_reason", "message"),
    [
        (0, "stop", "contradicts the captured tool calls"),
        (0, "end_turn", "contradicts the captured tool calls"),
        (0, "stop_sequence", "contradicts the captured tool calls"),
        (1, "tool_calls", "contradicts the captured text-only output"),
        (1, "tool_use", "contradicts the captured text-only output"),
    ],
)
def test_strict_export_rejects_finish_reasons_that_contradict_the_exported_step(
    model_call_index: int,
    finish_reason: str,
    message: str,
) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][model_call_index]["response_metadata"]["finish_reason"] = finish_reason

    with pytest.raises(AtifExportError, match=message):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    "finish_reason",
    (
        "aborted",
        "canceled",
        "cancelled",
        "content_filter",
        "failed",
        "incomplete",
        "length",
        "max_output_tokens",
        "max_tokens",
        "model_context_window_exceeded",
        "pause_turn",
        "refusal",
        "timed_out",
        "timeout",
    ),
)
def test_strict_export_rejects_known_incomplete_finish_reasons(finish_reason: str) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][1]["response_metadata"]["finish_reason"] = finish_reason

    with pytest.raises(AtifExportError, match="indicates an incomplete model response"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize("finish_reason", (None, "vendor_reason"))
def test_strict_export_preserves_noncontradictory_or_unknown_finish_reasons(finish_reason: str | None) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][1]["response_metadata"]["finish_reason"] = finish_reason

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    metadata = trajectory.steps[3].extra["nemo_gym"]["model_call"]["response_metadata"]
    if finish_reason is None:
        assert "finish_reason" not in metadata
    else:
        assert metadata["finish_reason"] == finish_reason


def test_strict_export_rejects_reasoning_tokens_greater_than_completion_tokens() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][1]["token_stats"]["reasoning_tokens"] = 14

    with pytest.raises(AtifExportError, match="reasoning_tokens exceeds completion_tokens"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_resolves_unique_model_ref_and_response_id_pairs() -> None:
    rollout = _canonical_rollout()
    model_ref = {"type": "responses_api_models", "name": "policy"}
    pair = {"model_ref": model_ref, "response_id": "response-1"}
    rollout["ng_trajectory"]["turns"][0]["model_calls"] = [pair]
    rollout["ng_trajectory"]["invocations"][0]["model_calls"][0] = pair

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[2].extra["nemo_gym"]["model_call"]["model_call_id"] == "model-call-1"


def test_strict_export_resolves_captured_model_calls_that_only_have_unique_pairs() -> None:
    rollout = _canonical_rollout()
    pairs = []
    for call in rollout["ng_trajectory"]["model_calls"]:
        call.pop("model_call_id")
        metadata = call["response_metadata"]
        pairs.append({"model_ref": metadata["model_ref"], "response_id": metadata["response_id"]})
    for turn, pair in zip(rollout["ng_trajectory"]["turns"], pairs, strict=True):
        turn["model_calls"] = [pair]
    rollout["ng_trajectory"]["invocations"][0]["model_calls"] = pairs

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    exported_call = trajectory.steps[2].extra["nemo_gym"]["model_call"]
    assert "model_call_id" not in exported_call
    assert exported_call["response_metadata"]["response_id"] == "response-1"


def test_strict_export_rejects_ambiguous_model_ref_and_response_id_pairs() -> None:
    rollout = _canonical_rollout()
    model_ref = {"type": "responses_api_models", "name": "policy"}
    pair = {"model_ref": model_ref, "response_id": "response-1"}
    rollout["ng_trajectory"]["turns"][0]["model_calls"] = [pair]
    rollout["ng_trajectory"]["model_calls"][1]["response_metadata"].update(pair)

    with pytest.raises(AtifExportError, match="match more than one captured model call"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    ("location", "field_name"),
    [
        ("captured", "model_call_id"),
        ("captured", "response_id"),
        ("turn-ref", "model_call_id"),
        ("turn-ref", "response_id"),
    ],
)
def test_strict_export_rejects_blank_model_call_identifiers(location: str, field_name: str) -> None:
    rollout = _canonical_rollout()
    if location == "captured":
        if field_name == "model_call_id":
            rollout["ng_trajectory"]["model_calls"][0][field_name] = " \t"
        else:
            rollout["ng_trajectory"]["model_calls"][0]["response_metadata"][field_name] = " \t"
    else:
        ref = rollout["ng_trajectory"]["turns"][0]["model_calls"][0]
        if field_name == "model_call_id":
            ref.update(
                {
                    "model_ref": {"type": "responses_api_models", "name": "policy"},
                    "response_id": "response-1",
                }
            )
        ref[field_name] = " \t"

    with pytest.raises(AtifExportError, match="cannot be blank"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_rejects_blank_root_invocation_identity() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["invocation_id"] = " \t"
    for turn in rollout["ng_trajectory"]["turns"]:
        turn["invocation_id"] = " \t"
    for tool_call in rollout["ng_trajectory"]["tool_calls"]:
        tool_call["invocation_id"] = " \t"

    with pytest.raises(AtifExportError, match=r"invocations\[0\]\.invocation_id: cannot be blank"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize("identity_field", ("response_id", "model_ref"))
def test_strict_export_rejects_ref_identity_missing_from_captured_model_call(identity_field: str) -> None:
    rollout = _canonical_rollout()
    metadata = rollout["ng_trajectory"]["model_calls"][0]["response_metadata"]
    rollout["ng_trajectory"]["turns"][0]["model_calls"][0][identity_field] = metadata.pop(identity_field)

    with pytest.raises(AtifExportError, match=f"{identity_field} conflicts with the captured model call"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_rejects_non_finite_provider_payload_values() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["request"]["temperature"] = float("nan")

    with pytest.raises(AtifExportError, match="non-finite numbers are not valid JSON"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_preserves_arbitrary_size_json_integers() -> None:
    rollout = _canonical_rollout()
    value = 10**100 + 123
    rollout["ng_trajectory"]["invocations"][0]["conversation"][3]["arguments"] = json.dumps({"value": value})

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[2].tool_calls[0].arguments == {"value": value}


@pytest.mark.parametrize(
    "arguments",
    ['{"value": 1, "value": 2}', '{"value": Infinity}', '{"value": 1e400}', '{"value": 1e-999}'],
)
def test_strict_export_rejects_nonstandard_or_ambiguous_tool_argument_json(arguments: str) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["conversation"][3]["arguments"] = arguments

    with pytest.raises(AtifExportError, match="expected a JSON object string"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize("arguments", ('{"value": 0e-999}', '{"value": 0e-99999999999999999999}'))
def test_strict_export_accepts_true_zero_with_extreme_json_exponents(arguments: str) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["conversation"][3]["arguments"] = arguments

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[2].tool_calls[0].arguments == {"value": 0.0}


@pytest.mark.parametrize("representation", ("encrypted", "content-and-summary"))
def test_strict_export_rejects_reasoning_representations_that_would_be_dropped(representation: str) -> None:
    rollout = _canonical_rollout()
    reasoning = rollout["ng_trajectory"]["invocations"][0]["conversation"][2]
    if representation == "encrypted":
        reasoning["encrypted_content"] = "opaque"
        message = "encrypted reasoning is not supported"
    else:
        reasoning["content"] = [{"type": "reasoning_text", "text": "full reasoning"}]
        message = "both content and summary"

    with pytest.raises(AtifExportError, match=message):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_export_writes_hash_verified_files_and_ingress_compatible_manifest(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    rows = [_canonical_rollout(), _canonical_rollout(task_index=4, rollout_index=0)]
    _write_jsonl(source, rows)
    output = tmp_path / "atif"

    result = export_rollouts_to_atif(
        ExportAtifConfig(
            rollouts_jsonl_fpath=source,
            output_dirpath=output,
            session_id="evaluation-42",
            agent_version="2.3.1",
        )
    )

    assert result.output_dirpath == output.resolve()
    assert result.trajectory_count == 2
    assert result.manifest_fpath == output.resolve() / "manifest.jsonl"
    assert sorted(path.name for path in output.iterdir()) == ["3-7.json", "4-0.json", "manifest.jsonl"]
    manifests = [json.loads(line) for line in result.manifest_fpath.read_text().splitlines()]
    assert manifests == [
        {
            "trajectory_path": "3-7.json",
            "_ng_task_index": 3,
            "_ng_rollout_index": 7,
            "expected_sha256": hashlib.sha256((output / "3-7.json").read_bytes()).hexdigest(),
        },
        {
            "trajectory_path": "4-0.json",
            "_ng_task_index": 4,
            "_ng_rollout_index": 0,
            "expected_sha256": hashlib.sha256((output / "4-0.json").read_bytes()).hexdigest(),
        },
    ]
    for manifest in manifests:
        AtifTrajectoryV1_7.model_validate_json((output / manifest["trajectory_path"]).read_text())


def test_export_validates_every_record_before_publishing_any_output(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    invalid = _canonical_rollout(task_index=4, rollout_index=0)
    invalid["ng_trajectory"]["invocations"][0]["conversation"][1]["role"] = "developer"
    _write_jsonl(source, [_canonical_rollout(), invalid])
    output = tmp_path / "atif"

    with pytest.raises(AtifExportError, match=r"line 2: .*developer messages are not supported"):
        export_rollouts_to_atif(
            ExportAtifConfig(
                rollouts_jsonl_fpath=source,
                output_dirpath=output,
                session_id="evaluation-42",
                agent_version="2.3.1",
            )
        )

    assert not output.exists()
    assert list(tmp_path.glob(".atif.tmp-*")) == []


def test_export_rejects_non_standard_json_numbers_before_conversion(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["request"]["temperature"] = float("inf")
    source.write_text(f"{json.dumps(rollout)}\n", encoding="utf-8")
    output = tmp_path / "atif"

    with pytest.raises(AtifExportError, match="invalid JSON"):
        export_rollouts_to_atif(
            ExportAtifConfig(
                rollouts_jsonl_fpath=source,
                output_dirpath=output,
                session_id="evaluation-42",
                agent_version="2.3.1",
            )
        )

    assert not output.exists()
    assert list(tmp_path.glob(".atif.tmp-*")) == []


@pytest.mark.parametrize(
    "payload",
    [
        b'{"invalid":"\xff"}\n',
        b'\xef\xbb\xbf{"value":true}\n',
        '{"value":true}\n'.encode("utf-16-be"),
        '{"value":true}\n'.encode("utf-32-be"),
    ],
)
def test_export_rejects_non_utf8_or_bom_jsonl(payload: bytes, tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    source.write_bytes(payload)
    output = tmp_path / "atif"

    with pytest.raises(AtifExportError, match=r"line 1: invalid JSON"):
        export_rollouts_to_atif(
            ExportAtifConfig(
                rollouts_jsonl_fpath=source,
                output_dirpath=output,
                session_id="evaluation-42",
                agent_version="2.3.1",
            )
        )

    assert not output.exists()
    assert list(tmp_path.glob(".atif.tmp-*")) == []


def test_export_rejects_nonzero_json_numbers_that_underflow_to_zero(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["request"]["tiny"] = "UNDERFLOW_SENTINEL"
    encoded = json.dumps(rollout).replace('"UNDERFLOW_SENTINEL"', "1e-99999999999999999999")
    source.write_text(f"{encoded}\n", encoding="utf-8")
    output = tmp_path / "atif"

    with pytest.raises(AtifExportError, match=r"line 1: invalid JSON"):
        export_rollouts_to_atif(
            ExportAtifConfig(
                rollouts_jsonl_fpath=source,
                output_dirpath=output,
                session_id="evaluation-42",
                agent_version="2.3.1",
            )
        )

    assert not output.exists()
    assert list(tmp_path.glob(".atif.tmp-*")) == []


def test_export_jsonl_preserves_arbitrary_size_integers_without_rounding(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    rollout = _canonical_rollout()
    value = 10**100 + 123
    rollout["ng_trajectory"]["model_calls"][0]["request"]["large_integer"] = value
    _write_jsonl(source, [rollout])
    output = tmp_path / "atif"

    export_rollouts_to_atif(
        ExportAtifConfig(
            rollouts_jsonl_fpath=source,
            output_dirpath=output,
            session_id="evaluation-42",
            agent_version="2.3.1",
        )
    )

    exported = json.loads((output / "3-7.json").read_text())
    assert exported["steps"][2]["extra"]["nemo_gym"]["model_call"]["request"]["large_integer"] == value


def test_export_rejects_duplicate_jsonl_object_keys(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    source.write_text('{"_ng_task_index": 1, "_ng_task_index": 2}\n', encoding="utf-8")

    with pytest.raises(AtifExportError, match="invalid JSON"):
        export_rollouts_to_atif(
            ExportAtifConfig(
                rollouts_jsonl_fpath=source,
                output_dirpath=tmp_path / "atif",
                session_id="evaluation-42",
                agent_version="2.3.1",
            )
        )


def test_export_rejects_jsonl_floats_outside_the_finite_range(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    source.write_text('{"value": 1e400}\n', encoding="utf-8")

    with pytest.raises(AtifExportError, match="invalid JSON"):
        export_rollouts_to_atif(
            ExportAtifConfig(
                rollouts_jsonl_fpath=source,
                output_dirpath=tmp_path / "atif",
                session_id="evaluation-42",
                agent_version="2.3.1",
            )
        )


def test_export_refuses_an_existing_output_path_without_modifying_it(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    _write_jsonl(source, [_canonical_rollout()])
    output = tmp_path / "atif"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(AtifExportError, match="Output path already exists"):
        export_rollouts_to_atif(
            ExportAtifConfig(
                rollouts_jsonl_fpath=source,
                output_dirpath=output,
                session_id="evaluation-42",
                agent_version="2.3.1",
            )
        )

    assert sentinel.read_text() == "do not replace"


def test_export_refuses_a_dangling_output_symlink(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    _write_jsonl(source, [_canonical_rollout()])
    output = tmp_path / "atif"
    output.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(AtifExportError, match="Output path already exists"):
        export_rollouts_to_atif(
            ExportAtifConfig(
                rollouts_jsonl_fpath=source,
                output_dirpath=output,
                session_id="evaluation-42",
                agent_version="2.3.1",
            )
        )

    assert output.is_symlink()


def test_export_rejects_a_batch_with_multiple_agent_identities(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    _write_jsonl(
        source,
        [
            _canonical_rollout(agent_name="agent-a"),
            _canonical_rollout(task_index=4, rollout_index=0, agent_name="agent-b"),
        ],
    )

    with pytest.raises(AtifExportError, match="one agent_ref.name per file"):
        export_rollouts_to_atif(
            ExportAtifConfig(
                rollouts_jsonl_fpath=source,
                output_dirpath=tmp_path / "atif",
                session_id="evaluation-42",
                agent_version="2.3.1",
            )
        )


def _with_gap(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["gaps"] = [{"code": "tool_output_unavailable", "invocation_id": "root"}]


def _with_failed_invocation(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["invocations"][0]["status"] = "failed"


def _with_subagent(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["invocations"].append(
        {
            "invocation_id": "child",
            "parent_invocation_id": "root",
            "spawned_by_tool_call_id": "call-search",
            "status": "completed",
            "conversation": [],
        }
    )


def _with_multimodal_content(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["invocations"][0]["conversation"][1]["content"] = [
        {"type": "input_image", "image_url": "data:image/png;base64,AA==", "detail": "auto"}
    ]


def _with_developer_message(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["invocations"][0]["conversation"][1]["role"] = "developer"


def _with_incomplete_input_message(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["invocations"][0]["conversation"][1] = {
        "type": "message",
        "role": "user",
        "status": "incomplete",
        "content": [{"type": "input_text", "text": "Compare the two records."}],
    }


def _with_orphan_tool_output(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["invocations"][0]["conversation"].insert(2, _function_output("orphan", "unused"))


def _with_failed_tool(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["tool_calls"][0]["status"] = "failed"


def _with_completed_tool_error(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["tool_calls"][0]["error_type"] = "latent-error"


def _with_invalid_tool_arguments(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["invocations"][0]["conversation"][3]["arguments"] = "[1, 2]"


def _with_nonstandard_tool_arguments(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["invocations"][0]["conversation"][3]["arguments"] = '{"value":NaN}'


def _with_incomplete_function_call(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["invocations"][0]["conversation"][3]["status"] = "incomplete"


def _with_incomplete_function_output(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["invocations"][0]["conversation"][5]["status"] = "incomplete"


def _with_missing_invocation_model_ref(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["invocations"][0]["model_calls"] = [{"model_call_id": "model-call-1"}]


def _with_non_agent_ref(row: dict[str, Any]) -> None:
    row["agent_ref"]["type"] = "responses_api_models"


def _with_turn_mismatch(row: dict[str, Any]) -> None:
    row["ng_trajectory"]["turns"][0]["answer"] = []


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_with_gap, "source coverage gaps are not supported"),
        (_with_failed_invocation, "expected completed, got 'failed'"),
        (_with_subagent, "expected exactly one root invocation"),
        (_with_multimodal_content, "only text content parts are supported"),
        (_with_developer_message, "developer messages are not supported"),
        (_with_incomplete_input_message, "expected completed, got 'incomplete'"),
        (_with_orphan_tool_output, "unsupported conversation item NeMoGymFunctionCallOutput"),
        (_with_failed_tool, "tool execution 'call-read' is 'failed'"),
        (_with_completed_tool_error, "completed tool execution 'call-read' contains an error"),
        (_with_invalid_tool_arguments, "expected a JSON object"),
        (_with_nonstandard_tool_arguments, "expected a JSON object string"),
        (_with_incomplete_function_call, "expected completed, got 'incomplete'"),
        (_with_incomplete_function_output, "function result 'call-read' is 'incomplete'"),
        (_with_missing_invocation_model_ref, "must reference every exported model call exactly once"),
        (_with_non_agent_ref, "expected responses_api_agents when present"),
        (_with_turn_mismatch, "does not match the exported agent answer"),
    ],
    ids=[
        "coverage-gap",
        "failed-invocation",
        "subagent",
        "multimodal",
        "developer-message",
        "incomplete-input-message",
        "orphan-tool-result",
        "failed-tool",
        "completed-tool-with-error",
        "non-object-tool-arguments",
        "non-standard-json-tool-arguments",
        "incomplete-function-call",
        "incomplete-function-result",
        "missing-invocation-model-reference",
        "wrong-agent-reference-type",
        "turn-mismatch",
    ],
)
def test_strict_export_rejects_unsupported_or_incomplete_source_shapes(
    mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    rollout = copy.deepcopy(_canonical_rollout())
    mutate(rollout)

    with pytest.raises(AtifExportError, match=message):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")
