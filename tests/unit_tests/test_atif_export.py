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
from nemo_gym.atif_export import (
    AtifExportError,
    ExportAtifConfig,
    export_rollouts_to_atif,
    gym_rollout_to_atif,
)
from nemo_gym.relay_atif import AtifTrajectoryV1_7
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

    with pytest.raises(AtifExportError, match="developer messages are not supported"):
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
