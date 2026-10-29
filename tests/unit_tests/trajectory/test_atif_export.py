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

from nemo_gym import __version__
from nemo_gym.trajectory import AtifExportError, TrajectoryRecord, encode_atif_trajectory, transcript_to_atif
from nemo_gym.trajectory.atif_export import ExportAtifConfig, export_rollouts_to_atif, gym_rollout_to_atif
from nemo_gym.trajectory.atif_reverification import (
    index_materialized_inputs,
    load_atif_manifest,
    project_atif_manifest_entries,
)
from nemo_gym.trajectory.atif_v1_7 import AtifTrajectoryV1_7


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


def _refusal_message(item_id: str, refusal: str) -> dict[str, Any]:
    return {**_assistant_message(item_id, ""), "content": [{"type": "refusal", "refusal": refusal}]}


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
                    "response": {"status": "completed"},
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


def _export_config(source: Path, output: Path) -> ExportAtifConfig:
    return ExportAtifConfig(
        rollouts_jsonl_fpath=source,
        output_dirpath=output,
        session_id="evaluation-42",
        agent_version="2.3.1",
    )


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
        "extra": {"reasoning_tokens": 7, "total_tokens": 112},
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


def test_gym_rollout_to_atif_supports_a_complete_trajectory_without_reasoning() -> None:
    rollout = _canonical_rollout()
    trajectory = rollout["ng_trajectory"]
    trajectory["invocations"][0]["conversation"] = [
        item for item in trajectory["invocations"][0]["conversation"] if item["type"] != "reasoning"
    ]
    for turn in trajectory["turns"]:
        turn["reasoning_content"] = None
        turn["question"] = [item for item in turn["question"] if item["type"] != "reasoning"]

    exported = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    agent_steps = [step for step in exported.steps if step.source == "agent"]
    assert [step.reasoning_content for step in agent_steps] == [None, None]
    assert "source_items" not in agent_steps[0].extra["nemo_gym"]
    assert agent_steps[1].extra["nemo_gym"]["source_items"] == {"message_id": "message-1"}


def test_export_preserves_optional_usage_without_inventing_standard_counts() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["token_stats"] = {
        "reasoning_tokens": 9,
        "total_tokens": 9,
    }

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[2].metrics.model_dump(exclude_none=True) == {
        "extra": {"reasoning_tokens": 9, "total_tokens": 9}
    }
    assert trajectory.final_metrics.model_dump(exclude_none=True) == {"total_steps": 4}


def test_gym_rollout_to_atif_always_emits_exact_total_steps_without_complete_token_metrics() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][1]["token_stats"] = {}

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[2].metrics is not None
    assert trajectory.steps[3].metrics is None
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


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        (
            "annotations",
            [
                {
                    "type": "url_citation",
                    "start_index": 0,
                    "end_index": 4,
                    "title": "source",
                    "url": "https://example.com",
                }
            ],
        ),
        (
            "logprobs",
            [{"token": "Both", "bytes": [66, 111, 116, 104], "logprob": -0.1, "top_logprobs": []}],
        ),
    ],
)
def test_strict_export_rejects_output_details_not_losslessly_representable_by_initial_profile(
    field_name: str,
    field_value: list[dict[str, Any]],
) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["conversation"][-1]["content"][0][field_name] = field_value

    with pytest.raises(AtifExportError, match="annotations and log probabilities are not representable"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("incomplete-output", "expected completed, got 'incomplete'"),
        ("empty-output", "expected at least one output_text part"),
        ("empty-reasoning", "encrypted or empty reasoning cannot be represented"),
        ("multi-segment-reasoning", "multiple reasoning segments cannot be represented"),
        ("trailing-reasoning", "reasoning is not followed by an agent answer"),
        ("copied-assistant", "copied assistant context is not supported"),
        ("later-user-message", "later system or user turns are not supported"),
        ("no-agent-output", "contains no agent output"),
    ],
)
def test_strict_export_rejects_incomplete_or_lossy_conversation_boundaries(mutation: str, message: str) -> None:
    rollout = _canonical_rollout()
    conversation = rollout["ng_trajectory"]["invocations"][0]["conversation"]
    if mutation == "incomplete-output":
        conversation[-1]["status"] = "incomplete"
    elif mutation == "empty-output":
        conversation[-1]["content"] = []
    elif mutation == "empty-reasoning":
        conversation[2]["summary"] = []
    elif mutation == "multi-segment-reasoning":
        conversation[2]["summary"].append({"type": "summary_text", "text": "Then compare them."})
    elif mutation == "trailing-reasoning":
        conversation[:] = conversation[:3]
    elif mutation == "copied-assistant":
        conversation.insert(2, {"type": "message", "role": "assistant", "content": "Earlier answer."})
    elif mutation == "later-user-message":
        conversation.insert(7, {"type": "message", "role": "user", "content": "Try again."})
    else:
        conversation[:] = conversation[:2]
        rollout["ng_trajectory"]["tool_calls"] = []

    with pytest.raises(AtifExportError, match=message):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate-record", "tool_call_id values must be unique"),
        ("blank-call-id", "call_id: cannot be empty"),
        ("blank-name", "name: cannot be empty"),
        ("duplicate-call", "duplicate function call ID"),
        ("duplicate-result", "duplicate function result ID"),
        ("mismatched-results", "function results must match this turn's calls exactly"),
        ("missing-record", "missing tool execution record"),
        ("wrong-invocation", "belongs to another invocation"),
        ("unreferenced-record", "unreferenced tool executions"),
    ],
)
def test_strict_export_rejects_inconsistent_tool_identity_and_ownership(mutation: str, message: str) -> None:
    rollout = _canonical_rollout()
    trajectory = rollout["ng_trajectory"]
    conversation = trajectory["invocations"][0]["conversation"]
    if mutation == "duplicate-record":
        trajectory["tool_calls"].append(copy.deepcopy(trajectory["tool_calls"][0]))
    elif mutation == "blank-call-id":
        conversation[3]["call_id"] = " \t"
    elif mutation == "blank-name":
        conversation[3]["name"] = " \t"
    elif mutation == "duplicate-call":
        conversation[4]["call_id"] = "call-search"
    elif mutation == "duplicate-result":
        conversation.insert(6, copy.deepcopy(conversation[5]))
    elif mutation == "mismatched-results":
        del conversation[6]
    elif mutation == "missing-record":
        trajectory["tool_calls"] = [
            record for record in trajectory["tool_calls"] if record["tool_call_id"] != "call-read"
        ]
    elif mutation == "wrong-invocation":
        trajectory["tool_calls"][0]["invocation_id"] = "child"
    else:
        extra = copy.deepcopy(trajectory["tool_calls"][0])
        extra["tool_call_id"] = "call-unused"
        trajectory["tool_calls"].append(extra)

    with pytest.raises(AtifExportError, match=message):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


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


def test_strict_export_preserves_total_tokens_without_assuming_component_accounting() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["token_stats"]["total_tokens"] = 999

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[2].metrics.extra == {"reasoning_tokens": 7, "total_tokens": 999}


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


def test_strict_export_rejects_component_sum_greater_than_total_tokens() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["token_stats"] = {
        "prompt_tokens": 7,
        "completion_tokens": 5,
        "total_tokens": 10,
    }

    with pytest.raises(AtifExportError, match="prompt_tokens plus completion_tokens exceeds total_tokens"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize("field_name", ("cached_tokens", "reasoning_tokens", "total_tokens"))
def test_export_preserves_available_optional_step_usage(field_name: str) -> None:
    rollout = _canonical_rollout()
    del rollout["ng_trajectory"]["model_calls"][1]["token_stats"][field_name]

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    first_metrics = trajectory.steps[2].metrics
    second_metrics = trajectory.steps[3].metrics
    assert first_metrics is not None and second_metrics is not None
    if field_name == "cached_tokens":
        assert first_metrics.cached_tokens == 5
        assert second_metrics.cached_tokens is None
        assert trajectory.final_metrics.total_cached_tokens is None
    else:
        assert first_metrics.extra[field_name] is not None
        assert field_name not in (second_metrics.extra or {})


def test_final_metrics_aggregate_each_usage_field_only_when_every_model_call_reports_it() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["token_stats"] = {
        "prompt_tokens": 101,
        "completion_tokens": 11,
    }
    rollout["ng_trajectory"]["model_calls"][1]["token_stats"] = {"prompt_tokens": 131}

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.final_metrics.model_dump(exclude_none=True) == {
        "total_prompt_tokens": 232,
        "total_steps": 4,
    }


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


@pytest.mark.parametrize(
    ("conversation_index", "phase"),
    (
        (0, "commentary"),
        (8, "final_answer"),
    ),
)
def test_strict_export_rejects_responses_message_phase(conversation_index: int, phase: str) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["conversation"][conversation_index]["phase"] = phase

    with pytest.raises(AtifExportError, match="Responses message phase is not representable in ATIF v1.7"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_accepts_absent_responses_message_phase_and_function_namespace() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["conversation"][0]["phase"] = None
    rollout["ng_trajectory"]["invocations"][0]["conversation"][3]["namespace"] = None

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[0].message == "Use tools when needed."
    assert trajectory.steps[2].tool_calls[0].function_name == "search"


def test_strict_export_rejects_namespaced_responses_function_calls() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["conversation"][3]["namespace"] = "mcp__weather"

    with pytest.raises(
        AtifExportError, match="namespaced Responses function calls are not representable in ATIF v1.7"
    ):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    "item_type",
    (
        "mcp_call",
        "web_search_call",
        "shell_call",
        "compaction",
    ),
)
def test_strict_export_rejects_unprojected_responses_item_families(item_type: str) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["conversation"].append({"type": item_type})

    with pytest.raises(AtifExportError, match=rf"unsupported conversation item type '{item_type}'"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


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


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("prompt_token_ids", [10]),
        ("generation_token_ids", [11]),
        ("generation_log_probs", [-0.25]),
        ("routed_experts", "nrlre1:uint16:2x1x1:AAAA"),
    ],
)
def test_strict_export_rejects_training_metadata_outside_initial_profile(
    field_name: str,
    value: Any,
) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["conversation"][-1][field_name] = value

    with pytest.raises(AtifExportError, match="outside the initial ATIF export profile"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    "item_index",
    (0, 2, 3, 5, 8),
    ids=("input-message", "reasoning", "function-call", "function-output", "assistant-message"),
)
def test_strict_export_rejects_unknown_fields_on_supported_raw_conversation_items(
    item_index: int,
) -> None:
    rollout = _canonical_rollout()
    conversation = rollout["ng_trajectory"]["invocations"][0]["conversation"]
    conversation[item_index] = copy.deepcopy(conversation[item_index])
    # Explicit nulls are still present fields and must not disappear through
    # model parsing. Non-null unknowns take the same rejection branch.
    conversation[item_index]["future_field"] = None

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


def test_strict_export_rejects_unknown_raw_output_text_fields_before_validation() -> None:
    rollout = _canonical_rollout()
    conversation = rollout["ng_trajectory"]["invocations"][0]["conversation"]
    conversation[-1] = copy.deepcopy(conversation[-1])
    conversation[-1]["content"][0]["future_field"] = None

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


@pytest.mark.parametrize("second_model", (None, "model-b"), ids=("unknown-step-model", "mixed-models"))
def test_gym_rollout_to_atif_omits_agent_model_without_one_uniform_known_model(second_model: str | None) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][1]["response_metadata"]["model"] = second_model

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.agent.model_name is None
    assert [step.model_name for step in trajectory.steps if step.source == "agent"] == ["model-a", second_model]


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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("identityless-captured-call", "unknown model_call_id"),
        ("unknown-model-ref", "unknown model_call_id"),
        ("unmatched-response-ref", "model_ref and response_id do not match a captured model call"),
        ("wrong-turn-invocation", "does not match the root invocation"),
        ("unordered-turn", "turns must be ordered sequentially from 1"),
        ("question-mismatch", "does not match the model-visible conversation prefix"),
        ("reasoning-mismatch", "does not match the exported reasoning"),
        ("step-count-mismatch", "does not match the cumulative tool-call count"),
        ("multiple-model-refs", "expected exactly one captured model call"),
        ("reused-model-call", "model call is referenced by more than one turn"),
        ("blank-model", "model: cannot be blank"),
        ("blank-finish-reason", "finish_reason: cannot be blank"),
        ("unrepresentable-timestamp", "cannot be represented as an ISO 8601 timestamp"),
    ],
)
def test_strict_export_rejects_inconsistent_turn_and_model_call_ownership(mutation: str, message: str) -> None:
    rollout = _canonical_rollout()
    trajectory = rollout["ng_trajectory"]
    turns = trajectory["turns"]
    model_calls = trajectory["model_calls"]
    if mutation == "identityless-captured-call":
        model_calls[0].pop("model_call_id")
        model_calls[0]["response_metadata"].pop("model_ref")
        model_calls[0]["response_metadata"].pop("response_id")
    elif mutation == "unknown-model-ref":
        turns[0]["model_calls"] = [{"model_call_id": "missing-call"}]
    elif mutation == "unmatched-response-ref":
        turns[0]["model_calls"] = [
            {
                "model_ref": {"type": "responses_api_models", "name": "policy"},
                "response_id": "missing-response",
            }
        ]
    elif mutation == "wrong-turn-invocation":
        turns[0]["invocation_id"] = "child"
    elif mutation == "unordered-turn":
        turns[1]["turn_no"] = 3
    elif mutation == "question-mismatch":
        turns[0]["question"] = copy.deepcopy(turns[0]["question"])
        turns[0]["question"][1]["content"] = "Different question."
    elif mutation == "reasoning-mismatch":
        turns[0]["reasoning_content"] = copy.deepcopy(turns[0]["reasoning_content"])
        turns[0]["reasoning_content"][0]["summary"][0]["text"] = "Different reasoning."
    elif mutation == "step-count-mismatch":
        turns[0]["step_count"] = 1
    elif mutation == "multiple-model-refs":
        turns[0]["model_calls"].append({"model_call_id": "model-call-2"})
    elif mutation == "reused-model-call":
        turns[1]["model_calls"] = [{"model_call_id": "model-call-1"}]
    elif mutation == "blank-model":
        model_calls[0]["response_metadata"]["model"] = " \t"
    elif mutation == "blank-finish-reason":
        model_calls[0]["response_metadata"]["finish_reason"] = " \t"
    elif mutation == "unrepresentable-timestamp":
        turns[0]["timestamp"] = 1e100

    with pytest.raises(AtifExportError, match=message):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_preserves_surplus_failed_model_attempt_without_counting_it() -> None:
    rollout = _canonical_rollout()
    surplus = copy.deepcopy(rollout["ng_trajectory"]["model_calls"][1])
    surplus["model_call_id"] = "model-call-retry"
    surplus["response_metadata"].update(
        {
            "response_id": "response-retry",
            "response_status": "failed",
            "error_category": "provider_error",
        }
    )
    surplus["token_stats"] = {
        "prompt_tokens": 10_000,
        "completion_tokens": 10_000,
        "total_tokens": 20_000,
    }
    rollout["ng_trajectory"]["model_calls"].append(surplus)
    rollout["ng_trajectory"]["invocations"][0]["model_calls"].append({"model_call_id": "model-call-retry"})

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.final_metrics.model_dump(exclude_none=True) == {
        "total_prompt_tokens": 232,
        "total_completion_tokens": 24,
        "total_cached_tokens": 13,
        "total_steps": 4,
    }
    assert trajectory.extra is not None
    preserved = trajectory.extra["nemo_gym"]["surplus_model_calls"]
    assert preserved == [surplus]


@pytest.mark.parametrize(
    "identity_shape",
    ("missing", "blank-model-call-id", "blank-response-id", "duplicate-response-pair"),
)
def test_strict_export_preserves_unreferenced_surplus_calls_with_noncanonical_identity(
    identity_shape: str,
) -> None:
    rollout = _canonical_rollout()
    surplus = copy.deepcopy(rollout["ng_trajectory"]["model_calls"][1])
    surplus["model_call_id"] = "model-call-retry"
    surplus["response_metadata"].update(
        {
            "response_id": "response-retry",
            "response_status": "failed",
            "error_category": "provider_error",
        }
    )
    surplus["token_stats"] = {
        "prompt_tokens": 10_000,
        "completion_tokens": 10_000,
        "total_tokens": 20_000,
    }
    if identity_shape == "missing":
        surplus.pop("model_call_id")
        surplus["response_metadata"].pop("model_ref")
        surplus["response_metadata"].pop("response_id")
    elif identity_shape == "blank-model-call-id":
        surplus["model_call_id"] = " \t"
    elif identity_shape == "blank-response-id":
        surplus["response_metadata"]["response_id"] = " \t"
    else:
        surplus["response_metadata"]["response_id"] = "response-2"
    rollout["ng_trajectory"]["model_calls"].append(surplus)

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.final_metrics.model_dump(exclude_none=True) == {
        "total_prompt_tokens": 232,
        "total_completion_tokens": 24,
        "total_cached_tokens": 13,
        "total_steps": 4,
    }
    assert trajectory.extra is not None
    assert trajectory.extra["nemo_gym"]["surplus_model_calls"] == [surplus]


def test_strict_export_rejects_duplicate_root_invocation_model_references() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["invocations"][0]["model_calls"].append({"model_call_id": "model-call-1"})

    with pytest.raises(AtifExportError, match="must not reference a captured model call more than once"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("blank-task-id", "task_id and rollout_id cannot be blank"),
        ("blank-rollout-id", "task_id and rollout_id cannot be blank"),
        ("blank-root-invocation", r"invocations\[0\]\.invocation_id: cannot be blank"),
        ("root-parent", "subagent invocations are not supported"),
        ("completed-root-error", "completed invocation contains an error"),
        ("blank-agent-name", "expected a non-empty agent_ref.name"),
    ],
)
def test_strict_export_rejects_invalid_root_lifecycle_and_identity(mutation: str, message: str) -> None:
    rollout = _canonical_rollout()
    trajectory = rollout["ng_trajectory"]
    if mutation == "blank-task-id":
        trajectory["task_id"] = " \t"
        for turn in trajectory["turns"]:
            turn["task_id"] = " \t"
    elif mutation == "blank-rollout-id":
        trajectory["rollout_id"] = " \t"
        for turn in trajectory["turns"]:
            turn["rollout_id"] = " \t"
    elif mutation == "blank-root-invocation":
        trajectory["invocations"][0]["invocation_id"] = " \t"
        for turn in trajectory["turns"]:
            turn["invocation_id"] = " \t"
        for tool_call in trajectory["tool_calls"]:
            tool_call["invocation_id"] = " \t"
    elif mutation == "root-parent":
        trajectory["invocations"][0]["parent_invocation_id"] = "parent"
    elif mutation == "completed-root-error":
        trajectory["invocations"][0]["error_type"] = "latent-error"
    else:
        rollout["agent_ref"]["name"] = " \t"

    with pytest.raises(AtifExportError, match=message):
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


def test_raw_provider_response_is_opaque_provenance_and_does_not_change_projection() -> None:
    rollout = _canonical_rollout()
    baseline = gym_rollout_to_atif(copy.deepcopy(rollout), session_id="evaluation-42", agent_version="2.3.1")
    model_call = rollout["ng_trajectory"]["model_calls"][1]
    raw_response = {
        "status": "failed",
        "choices": [{"finish_reason": "length", "message": {"content": "different"}}],
        "usage": {"prompt_tokens": 999},
        "vendor_extension": {"future": [1, 2, 3]},
    }
    model_call["response"] = raw_response

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    baseline_step = baseline.steps[3]
    step = trajectory.steps[3]
    for field_name in (
        "source",
        "message",
        "timestamp",
        "model_name",
        "reasoning_content",
        "metrics",
        "llm_call_count",
    ):
        assert getattr(step, field_name) == getattr(baseline_step, field_name)
    assert step.extra["nemo_gym"]["model_call"]["response"] == raw_response


@pytest.mark.parametrize(
    "dialect",
    [
        "chat",
        "messages",
    ],
)
def test_strict_export_rejects_unknown_typed_finish_reason_for_known_dialect(dialect: str) -> None:
    rollout = _canonical_rollout()
    model_call = rollout["ng_trajectory"]["model_calls"][1]
    metadata = model_call["response_metadata"]
    metadata["dialect"] = dialect
    metadata["response_status"] = None
    metadata["finish_reason"] = "vendor_truncated"

    with pytest.raises(AtifExportError, match="does not prove a completed response"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_strict_export_preserves_explicit_empty_canonical_text() -> None:
    rollout = _canonical_rollout()
    final_message = rollout["ng_trajectory"]["invocations"][0]["conversation"][-1]
    final_message["content"][0]["text"] = ""

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert [part.model_dump(exclude_none=True) for part in trajectory.steps[3].message] == [
        {"type": "text", "text": ""}
    ]


def test_strict_export_does_not_use_a_request_alias_as_the_response_model() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][1]["request"] = {"model": "routing-alias"}

    trajectory = gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")

    assert trajectory.steps[3].model_name == "model-a"


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
    ("location", "field_name", "message"),
    [
        ("captured", "model_call_id", "unknown model_call_id"),
        ("captured", "response_id", "cannot be blank"),
        ("turn-ref", "model_call_id", "cannot be blank"),
        ("turn-ref", "response_id", "cannot be blank"),
    ],
)
def test_strict_export_rejects_blank_model_call_identifiers(
    location: str,
    field_name: str,
    message: str,
) -> None:
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

    with pytest.raises(AtifExportError, match=message):
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


@pytest.mark.parametrize(
    "value",
    [
        {"set"},
        {1: "integer key"},
    ],
    ids=("unsupported-value", "non-string-key"),
)
def test_direct_export_rejects_non_json_python_values(value: Any) -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["request"] = value

    with pytest.raises(AtifExportError, match="not a JSON value|JSON object keys must be strings"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


def test_direct_export_rejects_unpaired_unicode_surrogates() -> None:
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["request"] = {"prompt": "\ud800"}

    with pytest.raises(AtifExportError, match="unpaired Unicode surrogate"):
        gym_rollout_to_atif(rollout, session_id="evaluation-42", agent_version="2.3.1")


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

    result = export_rollouts_to_atif(_export_config(source, output))

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

    materialized_inputs = index_materialized_inputs(
        [
            {
                "responses_create_params": {
                    "input": [{"role": "user", "content": "Compare the two records."}],
                    "instructions": "Use tools when needed.",
                },
                "_ng_task_index": row["_ng_task_index"],
                "_ng_rollout_index": row["_ng_rollout_index"],
                "agent_ref": row["agent_ref"],
            }
            for row in rows
        ]
    )
    projected = project_atif_manifest_entries(
        load_atif_manifest(result.manifest_fpath),
        materialized_inputs,
        manifest_directory=result.manifest_fpath.parent,
    )

    assert [(item.task_index, item.rollout_index) for item in projected] == [(3, 7), (4, 0)]
    for item in projected:
        response = item.payload["response"]
        assert response["status"] == "completed"
        assert response["usage"] == {
            "input_tokens": 232,
            "input_tokens_details": {"cached_tokens": 13},
            "output_tokens": 24,
            "output_tokens_details": {"reasoning_tokens": 10},
            "total_tokens": 256,
        }
        assert [output["call_id"] for output in response["output"] if "call_id" in output] == [
            "call-search",
            "call-read",
            "call-read",
            "call-search",
        ]
        assert response["output"][-2]["summary"] == [{"text": "The records agree.", "type": "summary_text"}]
        assert response["output"][-1]["content"] == [
            {
                "annotations": [],
                "logprobs": None,
                "text": "Both records contain the same value.",
                "type": "output_text",
            }
        ]


def test_export_validates_every_record_before_publishing_any_output(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    invalid = _canonical_rollout(task_index=4, rollout_index=0)
    invalid["ng_trajectory"]["invocations"][0]["conversation"][1]["role"] = "developer"
    _write_jsonl(source, [_canonical_rollout(), invalid])
    output = tmp_path / "atif"

    with pytest.raises(AtifExportError, match=r"line 2: .*developer messages are not supported"):
        export_rollouts_to_atif(_export_config(source, output))

    assert not output.exists()
    assert list(tmp_path.glob(".atif.tmp-*")) == []


def test_export_reports_every_invalid_row_before_publishing_any_output(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    first_invalid = _canonical_rollout(task_index=3, rollout_index=7)
    first_invalid["ng_trajectory"]["invocations"][0]["conversation"][1]["role"] = "developer"
    second_invalid = _canonical_rollout(task_index=5, rollout_index=0)
    second_invalid["ng_trajectory"]["turns"][0]["timestamp"] = 1e100
    _write_jsonl(
        source,
        [
            first_invalid,
            _canonical_rollout(task_index=4, rollout_index=0),
            second_invalid,
        ],
    )
    output = tmp_path / "atif"

    with pytest.raises(AtifExportError) as captured:
        export_rollouts_to_atif(_export_config(source, output))

    message = str(captured.value)
    assert "ATIF export rejected 2 row(s)" in message
    assert "line 1" in message
    assert "developer messages are not supported" in message
    assert "line 3" in message
    assert "cannot be represented as an ISO 8601 timestamp" in message
    assert not output.exists()
    assert list(tmp_path.glob(".atif.tmp-*")) == []


def test_export_publishes_group_readable_directory(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    _write_jsonl(source, [_canonical_rollout()])
    output = tmp_path / "atif"

    export_rollouts_to_atif(_export_config(source, output))

    assert output.stat().st_mode & 0o777 == 0o755


def test_export_rejects_an_invalid_json_batch_before_conversion(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    rollout = _canonical_rollout()
    rollout["ng_trajectory"]["model_calls"][0]["request"]["temperature"] = float("inf")
    source.write_text(f"{json.dumps(rollout)}\n", encoding="utf-8")
    output = tmp_path / "atif"

    with pytest.raises(AtifExportError, match="invalid JSON"):
        export_rollouts_to_atif(_export_config(source, output))

    assert not output.exists()
    assert list(tmp_path.glob(".atif.tmp-*")) == []


def test_export_jsonl_preserves_arbitrary_size_integers_without_rounding(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    rollout = _canonical_rollout()
    value = 10**100 + 123
    rollout["ng_trajectory"]["model_calls"][0]["request"]["large_integer"] = value
    _write_jsonl(source, [rollout])
    output = tmp_path / "atif"

    export_rollouts_to_atif(_export_config(source, output))

    exported = json.loads((output / "3-7.json").read_text())
    assert exported["steps"][2]["extra"]["nemo_gym"]["model_call"]["request"]["large_integer"] == value


def test_export_refuses_an_existing_output_path_without_modifying_it(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    _write_jsonl(source, [_canonical_rollout()])
    output = tmp_path / "atif"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("do not replace", encoding="utf-8")

    with pytest.raises(AtifExportError, match="Output path already exists"):
        export_rollouts_to_atif(_export_config(source, output))

    assert sentinel.read_text() == "do not replace"


def test_export_refuses_a_dangling_output_symlink(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    _write_jsonl(source, [_canonical_rollout()])
    output = tmp_path / "atif"
    output.symlink_to(tmp_path / "missing-target", target_is_directory=True)

    with pytest.raises(AtifExportError, match="Output path already exists"):
        export_rollouts_to_atif(_export_config(source, output))

    assert output.is_symlink()


def test_export_rejects_a_missing_rollouts_file(tmp_path: Path) -> None:
    with pytest.raises(AtifExportError, match="Rollouts file not found"):
        export_rollouts_to_atif(_export_config(tmp_path / "missing.jsonl", tmp_path / "atif"))


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"", "Rollouts file contains no records"),
        (b"\n", "line 1: blank JSONL records are not supported"),
        (b"[]\n", "line 1: expected a JSON object"),
    ],
)
def test_export_rejects_empty_blank_or_non_object_batches(payload: bytes, message: str, tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    source.write_bytes(payload)

    with pytest.raises(AtifExportError, match=message):
        export_rollouts_to_atif(_export_config(source, tmp_path / "atif"))


def test_export_rejects_duplicate_rollout_keys_before_publishing(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.jsonl"
    _write_jsonl(source, [_canonical_rollout(), _canonical_rollout()])
    output = tmp_path / "atif"

    with pytest.raises(AtifExportError, match=r"line 2: duplicate Gym rollout key \(3, 7\)"):
        export_rollouts_to_atif(_export_config(source, output))

    assert not output.exists()


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
        export_rollouts_to_atif(_export_config(source, tmp_path / "atif"))


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
        (_with_missing_invocation_model_ref, "must reference every turn-selected model call"),
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


# -- live transcripts (the harbor environment server's export before every verify) -----------------


def _transcript(input_items, output_items, **kwargs) -> AtifTrajectoryV1_7:
    return transcript_to_atif(
        input_items,
        output_items,
        session_id="rollout-1",
        agent_name="hermes_agent",
        agent_version="1.2.3",
        **kwargs,
    )


def test_transcript_to_atif_groups_tool_calls_with_their_outputs_and_keeps_reasoning() -> None:
    trajectory = _transcript(
        [{"role": "system", "content": "Be careful."}, {"role": "user", "content": "Create hello.txt"}],
        [
            _reasoning("rs_1", "I will write the file."),
            _function_call("call_1", "terminal", {"cmd": "echo hi > hello.txt"}),
            _function_call("call_2", "terminal", {"cmd": "cat hello.txt"}),
            _function_output("call_1", ""),
            _function_output("call_2", "hi\n"),
            _assistant_message("msg_1", "Done."),
        ],
        model_name="qwen",
        usage={"input_tokens": 120, "output_tokens": 30, "total_tokens": 150},
    )

    assert trajectory.schema_version == "ATIF-v1.7"
    assert [(step.step_id, step.source) for step in trajectory.steps] == [
        (1, "system"),
        (2, "user"),
        (3, "agent"),
        (4, "agent"),
    ]
    tool_step = trajectory.steps[2]
    assert tool_step.reasoning_content == "I will write the file."
    assert [(call.tool_call_id, call.function_name, call.arguments) for call in tool_step.tool_calls] == [
        ("call_1", "terminal", {"cmd": "echo hi > hello.txt"}),
        ("call_2", "terminal", {"cmd": "cat hello.txt"}),
    ]
    assert [(r.source_call_id, r.content) for r in tool_step.observation.results] == [
        ("call_1", ""),
        ("call_2", "hi\n"),
    ]
    assert trajectory.steps[3].message == "Done." and trajectory.steps[3].observation is None
    assert trajectory.agent.name == "hermes_agent" and trajectory.agent.model_name == "qwen"
    assert trajectory.final_metrics.total_steps == 4
    assert trajectory.final_metrics.total_prompt_tokens == 120
    assert trajectory.final_metrics.total_completion_tokens == 30
    assert trajectory.extra["nemo_gym"]["conversion"] == {
        "profile": "ng-transcript-to-atif-v1",
        "status": "complete",
        "open_tool_calls": [],
    }
    # The bytes Harbor reads are the strict model's JSON, so they validate back into the same trajectory.
    encoded = encode_atif_trajectory(trajectory)
    assert AtifTrajectoryV1_7.model_validate_json(encoded) == trajectory


def test_transcript_to_atif_terminus_user_turns_become_the_previous_steps_observation() -> None:
    """Terminus feeds the terminal back as user turns; ATIF has no later user turns, so they are observations."""
    trajectory = _transcript(
        "Fix the build",
        [
            _assistant_message("msg_1", "```bash\nmake\n```"),
            {"role": "user", "content": [{"type": "input_text", "text": "make: *** No rule to make target"}]},
            _assistant_message("msg_2", "```bash\nls\n```"),
            {"role": "user", "content": "Makefile src"},
            _assistant_message("msg_3", "Task complete."),
        ],
    )

    sources = [step.source for step in trajectory.steps]
    assert sources == ["user", "agent", "agent", "agent"]
    first, second, last = trajectory.steps[1:]
    assert first.observation.results[0].content == "make: *** No rule to make target"
    assert first.extra["nemo_gym"]["observation_source"] == "user_turn"
    assert second.observation.results[0].content == "Makefile src"
    assert last.observation is None


def test_transcript_to_atif_maps_developer_to_system_and_keeps_pending_calls_open() -> None:
    trajectory = _transcript(
        [{"role": "developer", "content": "Use the terminal."}, {"role": "user", "content": "go"}],
        [_function_call("call_9", "terminal", {"cmd": "sleep 1000"})],
    )
    assert trajectory.steps[0].source == "system" and trajectory.steps[0].extra == {"nemo_gym": {"role": "developer"}}
    # The harness stopped before the tool returned: the call stays open, which the conversion record says.
    assert trajectory.steps[-1].observation is None
    assert trajectory.extra["nemo_gym"]["conversion"]["open_tool_calls"] == ["call_9"]


@pytest.mark.parametrize(
    ("output_items", "message"),
    [
        ([_function_output("call_404", "x")], "matches no pending call"),
        ([_assistant_message("m", "a"), {"role": "system", "content": "later"}], "later system or developer"),
        ([_reasoning("rs", "thinking"), {"role": "system", "content": "later"}], "later system or developer"),
        ([{"role": "assistant", "content": "copied"}], "copied assistant context"),
        ([_function_call("c1", "t", {}), _function_call("c1", "t", {})], "duplicate function call ID"),
        ([{"type": "mcp_list_tools", "id": "l", "server_label": "s", "tools": []}], "unsupported transcript item"),
    ],
)
def test_transcript_to_atif_refuses_what_it_cannot_represent(output_items: list[dict[str, Any]], message: str) -> None:
    with pytest.raises(AtifExportError, match=message):
        _transcript([{"role": "user", "content": "go"}], output_items)


@pytest.mark.parametrize(
    "output_items",
    [
        [
            _function_call("c1", "t", {}),
            {**_function_output("c1", ""), "output": [{"type": "refusal", "refusal": "no"}]},
        ],
        [_assistant_message("m", "a"), {"role": "user", "content": [{"type": "refusal", "refusal": "no"}]}],
    ],
    ids=["tool-output", "user-turn"],
)
def test_transcript_to_atif_accepts_a_refusal_only_in_the_models_own_message(
    output_items: list[dict[str, Any]],
) -> None:
    """A refusal is something the model says; in a tool result or a user turn it is not a Responses API shape
    (the item union may read such a turn as some other, unsupported item; either way it never exports)."""
    with pytest.raises(AtifExportError, match="not a Responses API|only text content parts|unsupported transcript"):
        _transcript([{"role": "user", "content": "go"}], output_items)


def _bad_arguments_call(call_id: str, arguments: str) -> dict[str, Any]:
    return {**_function_call(call_id, "terminal", {}), "arguments": arguments}


@pytest.mark.parametrize(
    ("output_items", "reasons", "check"),
    [
        # A reasoning policy cut off at max_output_tokens: the converter emits a reasoning-only output.
        (
            [_assistant_message("m1", "ls"), _reasoning("rs", "half a thought")],
            ["trailing_reasoning"],
            lambda t: (t.steps[-1].message, t.steps[-1].reasoning_content, t.steps[-1].extra)
            == ("", "half a thought", {"nemo_gym": {"incomplete": "max_output_tokens"}}),
        ),
        # Terminus keeps that output and goes on after compaction: the next user turn is its observation.
        (
            [
                _reasoning("r1", "plan"),
                _assistant_message("m1", "```bash\nmake\n```"),
                {"role": "user", "content": "make: error"},
                _reasoning("r2", "cut off"),
                {"role": "user", "content": "(terminal unchanged)"},
                _reasoning("r3", "retry"),
                _assistant_message("m2", "```bash\nls\n```"),
            ],
            ["trailing_reasoning"],
            lambda t: [(s.source, s.message, s.reasoning_content) for s in t.steps]
            == [
                ("user", "go", None),
                ("agent", "```bash\nmake\n```", "plan"),
                ("agent", "", "cut off"),
                ("agent", "```bash\nls\n```", "retry"),
            ]
            and t.steps[2].observation.results[0].content == "(terminal unchanged)"
            and t.steps[2].extra["nemo_gym"] == {"incomplete": "max_output_tokens", "observation_source": "user_turn"},
        ),
        # Reasoning between a call and its result (the harness returned the result after the cut-off turn).
        (
            [_function_call("c1", "t", {}), _reasoning("rs", "x"), _function_output("c1", "out")],
            ["trailing_reasoning"],
            lambda t: [s.reasoning_content for s in t.steps[1:]] == [None, "x"]
            and t.steps[1].observation.results[0].content == "out",
        ),
        # Two <think> blocks in one completion: one reasoning item per block, or one item with two parts.
        (
            [_reasoning("r1", "a"), _reasoning("r2", "b"), _assistant_message("m", "ok")],
            ["joined_reasoning_segments"],
            lambda t: t.steps[1].reasoning_content == "a\nb",
        ),
        (
            [
                {
                    **_reasoning("r1", "a"),
                    "summary": [_reasoning("r1", "a")["summary"][0], _reasoning("r1", "b")["summary"][0]],
                },
                _assistant_message("m", "ok"),
            ],
            ["joined_reasoning_segments"],
            lambda t: t.steps[1].reasoning_content == "a\nb",
        ),
        # OpenAI reasoning models answer with `summary: []` unless a summary is requested.
        (
            [{**_reasoning("r1", "a"), "summary": []}, _assistant_message("m", "ok")],
            ["empty_reasoning"],
            lambda t: t.steps[1].reasoning_content is None and t.steps[1].message == "ok",
        ),
        (
            [{**_reasoning("r1", "a"), "summary": [], "encrypted_content": "opaque"}, _assistant_message("m", "ok")],
            ["encrypted_reasoning"],
            lambda t: t.steps[1].reasoning_content is None,
        ),
        # Arguments the model got wrong: the harness survived them, the episode is still gradable.
        (
            [_bad_arguments_call("c1", ""), _function_output("c1", "error"), _assistant_message("m", "ok")],
            ["unparsable_arguments"],
            lambda t: t.steps[1].tool_calls[0].arguments == {}
            and t.steps[1].tool_calls[0].extra["nemo_gym"] == {"source_item_id": "item-c1", "raw_arguments": ""},
        ),
        (
            [_bad_arguments_call("c1", '{"cmd": "ls" '), _function_output("c1", "error")],
            ["unparsable_arguments"],
            lambda t: t.steps[1].tool_calls[0].extra["nemo_gym"]["raw_arguments"] == '{"cmd": "ls" ',
        ),
        (
            [_bad_arguments_call("c1", "[1, 2]"), _function_output("c1", "error")],
            ["unparsable_arguments"],
            lambda t: t.steps[1].tool_calls[0].arguments == {},
        ),
        # The model refused a turn; Terminus fed the terminal back and the model answered on the next one. The
        # refusal is the step's text (Harbor's Terminus keeps an empty step there) and the episode is graded.
        (
            [
                _refusal_message("m1", "I can't help with that."),
                {"role": "user", "content": "(terminal unchanged)"},
                _assistant_message("m2", "```bash\nls\n```"),
            ],
            ["refusal"],
            lambda t: t.steps[1].message == "I can't help with that."
            and t.steps[1].extra["nemo_gym"]
            == {"source_item_id": "m1", "refusal": True, "observation_source": "user_turn"}
            and t.steps[1].observation.results[0].content == "(terminal unchanged)"
            and t.steps[2].message == "```bash\nls\n```"
            and t.steps[2].extra == {"nemo_gym": {"source_item_id": "m2"}},
        ),
        # An empty refusal is exactly Harbor's own shape for a refused turn: an agent step with an empty message.
        (
            [_refusal_message("m1", ""), _assistant_message("m2", "ok")],
            ["refusal"],
            lambda t: t.steps[1].message == "" and t.steps[1].extra["nemo_gym"]["refusal"] is True,
        ),
    ],
    ids=[
        "trailing-reasoning",
        "terminus-truncated-turn",
        "reasoning-before-tool-result",
        "two-reasoning-items",
        "two-summary-parts",
        "empty-summary",
        "encrypted",
        "empty-arguments",
        "malformed-arguments",
        "non-object-arguments",
        "refusal",
        "empty-refusal",
    ],
)
def test_transcript_to_atif_exports_model_caused_shapes_as_partial(
    output_items: list[dict[str, Any]], reasons: list[str], check: Callable[[AtifTrajectoryV1_7], bool]
) -> None:
    trajectory = _transcript([{"role": "user", "content": "go"}], output_items)

    assert check(trajectory), trajectory.steps
    conversion = trajectory.extra["nemo_gym"]["conversion"]
    assert conversion["status"] == "partial" and conversion["reasons"] == reasons
    # The document round-trips through the encoder as a valid ATIF v1.7 trajectory.
    encoded = json.loads(encode_atif_trajectory(trajectory))
    assert AtifTrajectoryV1_7.model_validate(encoded).steps[-1].step_id == len(trajectory.steps)
    harbor_trajectory = pytest.importorskip("harbor.models.trajectories.trajectory", reason="Harbor is optional")
    harbor_trajectory.Trajectory.model_validate(encoded)


def test_transcript_to_atif_complete_transcripts_carry_no_reasons() -> None:
    trajectory = _transcript(
        [{"role": "user", "content": "go"}],
        [
            _reasoning("r", "plan"),
            _function_call("c1", "t", {"k": 1}),
            _function_output("c1", "o"),
            _assistant_message("m", "ok"),
        ],
    )
    assert trajectory.extra["nemo_gym"]["conversion"] == {
        "profile": "ng-transcript-to-atif-v1",
        "status": "complete",
        "open_tool_calls": [],
    }
    assert trajectory.steps[1].tool_calls[0].extra == {"nemo_gym": {"source_item_id": "item-c1"}}


def test_transcript_to_atif_needs_identity_and_at_least_one_message() -> None:
    with pytest.raises(AtifExportError, match="session_id"):
        transcript_to_atif([], [], session_id=" ", agent_name="a", agent_version="1")
    with pytest.raises(AtifExportError, match="contains no messages"):
        _transcript([], [])
