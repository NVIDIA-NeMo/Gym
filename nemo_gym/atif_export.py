# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Strict offline conversion from Gym rollout trajectories to ATIF v1.7."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import orjson
from pydantic import BaseModel, Field, ValidationError, field_validator

from nemo_gym import __version__
from nemo_gym.config_types import BaseNeMoGymCLIConfig, ConfigError
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymMessage,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.relay_atif import (
    AtifAgent,
    AtifContent,
    AtifContentPart,
    AtifFinalMetrics,
    AtifObservation,
    AtifObservationResult,
    AtifStep,
    AtifStepMetrics,
    AtifToolCall,
    AtifTrajectoryV1_7,
)
from nemo_gym.rollout_observability import ModelCallRef, TrajectoryModelCall, TrajectoryRecord


class AtifExportError(ConfigError):
    """A rollout cannot be represented by Gym's supported ATIF profile."""


class ExportAtifConfig(BaseNeMoGymCLIConfig):
    """Configuration for ``gym eval export --format atif``."""

    format: Literal["atif"] = "atif"
    rollouts_jsonl_fpath: Path
    output_dirpath: Path
    session_id: str = Field(min_length=1)
    agent_version: str = Field(min_length=1)

    @field_validator("session_id", "agent_version")
    @classmethod
    def reject_blank_identity(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must contain a non-whitespace character")
        return value


@dataclass(frozen=True)
class AtifExportResult:
    output_dirpath: Path
    trajectory_count: int
    manifest_fpath: Path


@dataclass
class _AgentGroup:
    question: list[Any]
    reasoning: list[Any]
    answer: list[Any]
    step: AtifStep


def _path_error(path: str, detail: str) -> AtifExportError:
    return AtifExportError(f"{path}: {detail}")


def _json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items() if item is not None}
    return value


def _reject_non_finite_numbers(value: Any, *, path: str) -> None:
    """Reject values that JSON cannot encode without changing them."""

    if isinstance(value, float) and not math.isfinite(value):
        raise _path_error(path, "non-finite numbers are not valid JSON")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite_numbers(item, path=f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _reject_non_finite_numbers(item, path=f"{path}.{key}")


def _text_content(value: Any, *, path: str) -> AtifContent:
    if isinstance(value, str):
        return value
    if not isinstance(value, list) or not value:
        raise _path_error(path, "expected non-empty scalar or multipart text content")

    parts: list[AtifContentPart] = []
    for index, part in enumerate(value):
        raw = _json_value(part)
        if not isinstance(raw, dict) or raw.get("type") not in {"input_text", "output_text", "text"}:
            raise _path_error(f"{path}[{index}]", "only text content parts are supported")
        text = raw.get("text")
        if not isinstance(text, str):
            raise _path_error(f"{path}[{index}].text", "expected a string")
        if raw.get("annotations") not in (None, []) or raw.get("logprobs") not in (None, []):
            raise _path_error(f"{path}[{index}]", "annotations and log probabilities are not representable")
        parts.append(AtifContentPart(type="text", text=text))
    return parts


def _message_content(item: Any, *, path: str) -> AtifContent:
    if isinstance(item, (NeMoGymEasyInputMessage, NeMoGymMessage)):
        if isinstance(item, NeMoGymMessage) and item.status != "completed":
            raise _path_error(f"{path}.status", f"expected completed, got {item.status!r}")
        return _text_content(item.content, path=f"{path}.content")
    if isinstance(item, NeMoGymResponseOutputMessage):
        if item.status != "completed":
            raise _path_error(f"{path}.status", f"expected completed, got {item.status!r}")
        if not item.content:
            raise _path_error(f"{path}.content", "expected at least one output_text part")
        return _text_content(item.content, path=f"{path}.content")
    raise _path_error(path, f"unsupported message type {type(item).__name__}")


def _reasoning_text(items: list[NeMoGymResponseReasoningItem], *, path: str) -> str | None:
    if not items:
        return None

    segments: list[str] = []
    for item_index, item in enumerate(items):
        raw = item.model_dump(mode="json", exclude_none=True)
        content = raw.get("content") or []
        summary = raw.get("summary") or []
        if raw.get("encrypted_content") is not None:
            raise _path_error(f"{path}[{item_index}]", "encrypted reasoning is not supported")
        if content and summary:
            raise _path_error(
                f"{path}[{item_index}]",
                "reasoning with both content and summary cannot be represented without dropping one",
            )
        source = content if content else summary
        if not source:
            raise _path_error(
                f"{path}[{item_index}]", "encrypted or empty reasoning cannot be represented as ATIF text"
            )
        for segment_index, segment in enumerate(source):
            text = segment.get("text") if isinstance(segment, dict) else None
            if not isinstance(text, str):
                raise _path_error(f"{path}[{item_index}][{segment_index}]", "reasoning segment does not contain text")
            segments.append(text)
    if len(segments) != 1:
        raise _path_error(path, "multiple reasoning segments cannot be represented without changing boundaries")
    return segments[0]


def _parse_arguments(call: NeMoGymResponseFunctionToolCall, *, path: str) -> dict[str, Any]:
    try:
        arguments = orjson.loads(call.arguments)
    except (TypeError, orjson.JSONDecodeError) as exc:
        raise _path_error(f"{path}.arguments", "expected a JSON object string") from exc
    if not isinstance(arguments, dict):
        raise _path_error(f"{path}.arguments", "expected a JSON object")
    return arguments


def _tool_record_extra(record: Any) -> dict[str, Any] | None:
    source = record.model_dump(
        mode="json",
        include={"sandbox_id", "started_at", "completed_at", "duration_ms", "timing_source"},
        exclude_none=True,
    )
    return {"nemo_gym": source} if source else None


def _build_groups(trajectory: TrajectoryRecord, invocation: Any) -> tuple[list[AtifStep], list[_AgentGroup]]:
    steps: list[AtifStep] = []
    groups: list[_AgentGroup] = []
    conversation = invocation.conversation
    tool_records = {record.tool_call_id: record for record in trajectory.tool_calls}
    if len(tool_records) != len(trajectory.tool_calls):
        raise _path_error("ng_trajectory.tool_calls", "tool_call_id values must be unique")

    used_tool_ids: set[str] = set()
    prefix: list[Any] = []
    index = 0
    seen_agent_output = False
    while index < len(conversation):
        item = conversation[index]
        item_path = f"ng_trajectory.invocations[0].conversation[{index}]"

        if isinstance(item, (NeMoGymEasyInputMessage, NeMoGymMessage)):
            if item.role == "developer":
                raise _path_error(f"{item_path}.role", "developer messages are not supported")
            if item.role == "assistant":
                raise _path_error(f"{item_path}.role", "copied assistant context is not supported")
            if seen_agent_output:
                raise _path_error(item_path, "later system or user turns are not supported")
            steps.append(
                AtifStep(
                    step_id=len(steps) + 1,
                    source=item.role,
                    message=_message_content(item, path=item_path),
                )
            )
            prefix.append(item)
            index += 1
            continue

        seen_agent_output = True
        question = list(prefix)
        reasoning_items: list[NeMoGymResponseReasoningItem] = []
        while index < len(conversation) and isinstance(conversation[index], NeMoGymResponseReasoningItem):
            reasoning_items.append(conversation[index])
            prefix.append(conversation[index])
            index += 1

        if index >= len(conversation):
            raise _path_error(item_path, "reasoning is not followed by an agent answer")

        answer_items: list[Any] = []
        tool_calls: list[NeMoGymResponseFunctionToolCall] = []
        while index < len(conversation) and isinstance(conversation[index], NeMoGymResponseFunctionToolCall):
            call = conversation[index]
            if not call.call_id.strip():
                raise _path_error(f"ng_trajectory.invocations[0].conversation[{index}].call_id", "cannot be empty")
            if not call.name.strip():
                raise _path_error(f"ng_trajectory.invocations[0].conversation[{index}].name", "cannot be empty")
            if call.status not in (None, "completed"):
                raise _path_error(
                    f"ng_trajectory.invocations[0].conversation[{index}].status",
                    f"expected completed, got {call.status!r}",
                )
            if call.call_id in used_tool_ids:
                raise _path_error(item_path, f"duplicate function call ID {call.call_id!r}")
            tool_calls.append(call)
            answer_items.append(call)
            prefix.append(call)
            used_tool_ids.add(call.call_id)
            index += 1

        if tool_calls:
            outputs: dict[str, NeMoGymFunctionCallOutput] = {}
            output_sequence: list[NeMoGymFunctionCallOutput] = []
            while index < len(conversation) and isinstance(conversation[index], NeMoGymFunctionCallOutput):
                output = conversation[index]
                if output.call_id in outputs:
                    raise _path_error(item_path, f"duplicate function result ID {output.call_id!r}")
                if output.status not in (None, "completed"):
                    raise _path_error(item_path, f"function result {output.call_id!r} is {output.status!r}")
                outputs[output.call_id] = output
                output_sequence.append(output)
                prefix.append(output)
                index += 1

            expected_ids = {call.call_id for call in tool_calls}
            if set(outputs) != expected_ids:
                raise _path_error(
                    item_path,
                    f"function results must match this turn's calls exactly; expected {sorted(expected_ids)}, "
                    f"got {sorted(outputs)}",
                )

            atif_calls: list[AtifToolCall] = []
            records_by_id: dict[str, Any] = {}
            for call_index, call in enumerate(tool_calls):
                record = tool_records.get(call.call_id)
                if record is None:
                    raise _path_error(item_path, f"missing tool execution record for {call.call_id!r}")
                if record.invocation_id != invocation.invocation_id:
                    raise _path_error(item_path, f"tool execution {call.call_id!r} belongs to another invocation")
                if record.status != "completed":
                    raise _path_error(
                        item_path, f"tool execution {call.call_id!r} is {record.status!r}, not completed"
                    )
                if record.error_type is not None:
                    raise _path_error(item_path, f"completed tool execution {call.call_id!r} contains an error")
                if record.tool_name != call.name:
                    raise _path_error(item_path, f"tool execution {call.call_id!r} has a different tool name")
                output = outputs[call.call_id]
                if not isinstance(output.output, str):
                    raise _path_error(item_path, f"tool result {call.call_id!r} is not scalar text")
                if record.output != output.output:
                    raise _path_error(item_path, f"tool execution {call.call_id!r} has a different recorded output")
                records_by_id[call.call_id] = record
                atif_calls.append(
                    AtifToolCall(
                        tool_call_id=call.call_id,
                        function_name=call.name,
                        arguments=_parse_arguments(call, path=f"{item_path}.tool_calls[{call_index}]"),
                    )
                )
            results = [
                AtifObservationResult(
                    source_call_id=output.call_id,
                    content=output.output,
                    extra=_tool_record_extra(records_by_id[output.call_id]),
                )
                for output in output_sequence
            ]

            step = AtifStep(
                step_id=len(steps) + 1,
                source="agent",
                message="",
                reasoning_content=_reasoning_text(reasoning_items, path=f"{item_path}.reasoning"),
                tool_calls=atif_calls,
                observation=AtifObservation(results=results),
            )
        elif isinstance(conversation[index], NeMoGymResponseOutputMessage):
            message = conversation[index]
            answer_items.append(message)
            prefix.append(message)
            index += 1
            step = AtifStep(
                step_id=len(steps) + 1,
                source="agent",
                message=_message_content(message, path=item_path),
                reasoning_content=_reasoning_text(reasoning_items, path=f"{item_path}.reasoning"),
            )
        else:
            raise _path_error(item_path, f"unsupported conversation item {type(conversation[index]).__name__}")

        steps.append(step)
        groups.append(_AgentGroup(question=question, reasoning=reasoning_items, answer=answer_items, step=step))

    unused_tool_ids = set(tool_records) - used_tool_ids
    if unused_tool_ids:
        raise _path_error("ng_trajectory.tool_calls", f"unreferenced tool executions: {sorted(unused_tool_ids)}")
    if not groups:
        raise _path_error("ng_trajectory.invocations[0].conversation", "contains no agent output")
    return steps, groups


def _resolve_model_call(ref: ModelCallRef, calls: dict[str, TrajectoryModelCall], *, path: str) -> TrajectoryModelCall:
    if not ref.model_call_id or not ref.model_call_id.strip():
        raise _path_error(path, "model_call_id was not resolved")
    call = calls.get(ref.model_call_id)
    if call is None:
        raise _path_error(path, f"unknown model_call_id {ref.model_call_id!r}")
    if ref.response_id is not None and ref.response_id != call.response_metadata.response_id:
        raise _path_error(path, "response_id conflicts with the captured model call")
    if ref.model_ref is not None and ref.model_ref != call.response_metadata.model_ref:
        raise _path_error(path, "model_ref conflicts with the captured model call")
    return call


def _apply_turns(trajectory: TrajectoryRecord, groups: list[_AgentGroup]) -> None:
    if len(trajectory.turns) != len(groups):
        raise _path_error(
            "ng_trajectory.turns",
            f"expected one turn for each of {len(groups)} agent steps, got {len(trajectory.turns)}",
        )

    calls = {call.model_call_id: call for call in trajectory.model_calls if call.model_call_id}
    if len(calls) != len(trajectory.model_calls):
        raise _path_error("ng_trajectory.model_calls", "every captured model call must have a unique model_call_id")
    used_calls: set[str] = set()
    previous_turn_no = 0
    total_tool_calls = 0
    for index, (turn, group) in enumerate(zip(trajectory.turns, groups, strict=True)):
        path = f"ng_trajectory.turns[{index}]"
        if turn.invocation_id != trajectory.invocations[0].invocation_id:
            raise _path_error(f"{path}.invocation_id", "does not match the root invocation")
        if turn.turn_no != previous_turn_no + 1:
            raise _path_error(f"{path}.turn_no", "turns must be ordered sequentially from 1")
        previous_turn_no = turn.turn_no
        if _json_value(turn.question) != _json_value(group.question):
            raise _path_error(f"{path}.question", "does not match the model-visible conversation prefix")
        if _json_value(turn.answer) != _json_value(group.answer):
            raise _path_error(f"{path}.answer", "does not match the exported agent answer")
        if _json_value(turn.reasoning_content) != (_json_value(group.reasoning) or None):
            raise _path_error(f"{path}.reasoning_content", "does not match the exported reasoning")
        total_tool_calls += len(group.step.tool_calls or [])
        if turn.step_count != total_tool_calls:
            raise _path_error(f"{path}.step_count", "does not match the cumulative tool-call count")
        if len(turn.model_calls) != 1:
            raise _path_error(f"{path}.model_calls", "expected exactly one captured model call")

        call = _resolve_model_call(turn.model_calls[0], calls, path=f"{path}.model_calls[0]")
        assert call.model_call_id is not None
        if call.model_call_id in used_calls:
            raise _path_error(f"{path}.model_calls[0]", "model call is referenced by more than one turn")
        used_calls.add(call.model_call_id)
        if call.response_metadata.response_status not in (None, "completed"):
            raise _path_error(path, f"model call is {call.response_metadata.response_status!r}, not completed")

        try:
            group.step.timestamp = datetime.fromtimestamp(turn.timestamp, UTC).isoformat().replace("+00:00", "Z")
        except (OverflowError, OSError, ValueError) as exc:
            raise _path_error(f"{path}.timestamp", "cannot be represented as an ISO 8601 timestamp") from exc
        group.step.model_name = call.response_metadata.model
        group.step.llm_call_count = 1
        stats = call.token_stats
        if any(
            value is not None
            for value in (
                stats.prompt_tokens,
                stats.completion_tokens,
                stats.reasoning_tokens,
                stats.total_tokens,
                stats.cached_tokens,
            )
        ):
            extra = {
                key: value
                for key, value in {
                    "reasoning_tokens": stats.reasoning_tokens,
                    "total_tokens": stats.total_tokens,
                }.items()
                if value is not None
            }
            group.step.metrics = AtifStepMetrics(
                prompt_tokens=stats.prompt_tokens,
                completion_tokens=stats.completion_tokens,
                cached_tokens=stats.cached_tokens,
                extra={"nemo_gym": extra} if extra else None,
            )
        group.step.extra = {
            "nemo_gym": {
                "turn": {
                    "turn_no": turn.turn_no,
                    "step_count": turn.step_count,
                    **({"resolved": turn.resolved} if turn.resolved is not None else {}),
                },
                "model_call": call.model_dump(mode="json", exclude_none=True),
            }
        }

    if set(calls) != used_calls:
        raise _path_error("ng_trajectory.model_calls", f"unreferenced model calls: {sorted(set(calls) - used_calls)}")

    invocation_call_ids: list[str] = []
    for index, ref in enumerate(trajectory.invocations[0].model_calls):
        call = _resolve_model_call(
            ref,
            calls,
            path=f"ng_trajectory.invocations[0].model_calls[{index}]",
        )
        assert call.model_call_id is not None
        invocation_call_ids.append(call.model_call_id)
    if len(invocation_call_ids) != len(set(invocation_call_ids)) or set(invocation_call_ids) != used_calls:
        raise _path_error(
            "ng_trajectory.invocations[0].model_calls",
            "must reference every exported model call exactly once",
        )


def _final_metrics(groups: list[_AgentGroup], *, total_steps: int) -> AtifFinalMetrics | None:
    metrics = [group.step.metrics for group in groups]
    if not metrics or any(metric is None for metric in metrics):
        return None
    typed_metrics = [metric for metric in metrics if metric is not None]

    def total(field_name: str) -> int | None:
        values = [getattr(metric, field_name) for metric in typed_metrics]
        return sum(values) if all(value is not None for value in values) else None

    return AtifFinalMetrics(
        total_prompt_tokens=total("prompt_tokens"),
        total_completion_tokens=total("completion_tokens"),
        total_cached_tokens=total("cached_tokens"),
        total_steps=total_steps,
    )


def gym_rollout_to_atif(rollout: dict[str, Any], *, session_id: str, agent_version: str) -> AtifTrajectoryV1_7:
    """Convert one rollout row, rejecting any structure Gym cannot represent completely."""

    _reject_non_finite_numbers(rollout, path="rollout")
    try:
        trajectory = TrajectoryRecord.model_validate(rollout.get("ng_trajectory"))
    except (TypeError, ValidationError) as exc:
        raise _path_error("ng_trajectory", "missing or invalid v1.0 trajectory") from exc
    if trajectory.gaps:
        raise _path_error("ng_trajectory.gaps", "source coverage gaps are not supported by strict export")
    if not trajectory.task_id.strip() or not trajectory.rollout_id.strip():
        raise _path_error("ng_trajectory", "task_id and rollout_id cannot be blank")
    if len(trajectory.invocations) != 1:
        raise _path_error("ng_trajectory.invocations", "expected exactly one root invocation")
    invocation = trajectory.invocations[0]
    if invocation.parent_invocation_id is not None or invocation.spawned_by_tool_call_id is not None:
        raise _path_error("ng_trajectory.invocations[0]", "subagent invocations are not supported")
    if invocation.status != "completed":
        raise _path_error("ng_trajectory.invocations[0].status", f"expected completed, got {invocation.status!r}")
    if invocation.error_type is not None:
        raise _path_error("ng_trajectory.invocations[0].error_type", "completed invocation contains an error")

    agent_ref = rollout.get(AGENT_REF_KEY_NAME)
    agent_name = agent_ref.get("name") if isinstance(agent_ref, dict) else None
    if not isinstance(agent_name, str) or not agent_name.strip():
        raise _path_error(AGENT_REF_KEY_NAME, "expected a non-empty agent_ref.name")
    agent_type = agent_ref.get("type")
    if agent_type not in (None, "responses_api_agents"):
        raise _path_error(f"{AGENT_REF_KEY_NAME}.type", "expected responses_api_agents when present")

    task_index = _index(rollout, TASK_INDEX_KEY_NAME, path=TASK_INDEX_KEY_NAME)
    rollout_index = _index(rollout, ROLLOUT_INDEX_KEY_NAME, path=ROLLOUT_INDEX_KEY_NAME)

    steps, groups = _build_groups(trajectory, invocation)
    _apply_turns(trajectory, groups)
    step_model_names = [group.step.model_name for group in groups]
    known_model_names = {name for name in step_model_names if name is not None}
    model_name = (
        next(iter(known_model_names)) if len(known_model_names) == 1 and None not in step_model_names else None
    )
    return AtifTrajectoryV1_7(
        schema_version="ATIF-v1.7",
        session_id=session_id,
        trajectory_id=f"{session_id}:{task_index}:{rollout_index}",
        agent=AtifAgent(name=agent_name, version=agent_version, model_name=model_name),
        steps=steps,
        final_metrics=_final_metrics(groups, total_steps=len(steps)),
        extra={
            "nemo_gym": {
                "exporter": {"name": "nemo-gym", "version": __version__},
                "source": {
                    "format": "ng_trajectory",
                    "schema_version": trajectory.schema_version,
                    "task_id": trajectory.task_id,
                    "rollout_id": trajectory.rollout_id,
                    "task_index": task_index,
                    "rollout_index": rollout_index,
                    "invocation_id": invocation.invocation_id,
                    "invocation_status": invocation.status,
                    **(
                        {"invocation_duration_ms": invocation.duration_ms}
                        if invocation.duration_ms is not None
                        else {}
                    ),
                },
                "conversion": {"profile": "ng-trajectory-to-atif-v1", "status": "complete"},
            }
        },
    )


def _index(row: dict[str, Any], key: str, *, path: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _path_error(path, "expected a non-negative integer")
    return value


def _encoded_trajectory(trajectory: AtifTrajectoryV1_7) -> bytes:
    return (trajectory.model_dump_json(indent=2, exclude_none=True) + "\n").encode()


def export_rollouts_to_atif(config: ExportAtifConfig) -> AtifExportResult:
    """Validate a rollout JSONL completely, then atomically publish ATIF files and a manifest."""

    source = config.rollouts_jsonl_fpath.expanduser().resolve()
    output = config.output_dirpath.expanduser().absolute()
    if not source.is_file():
        raise AtifExportError(f"Rollouts file not found: {source}")
    if os.path.lexists(output):
        raise AtifExportError(f"Output path already exists: {output}")

    keys: set[tuple[int, int]] = set()
    agent_names: set[str] = set()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    try:
        manifest_lines: list[str] = []
        with source.open(encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    raise _path_error(f"line {line_no}", "blank JSONL records are not supported")
                try:
                    row = orjson.loads(line)
                except orjson.JSONDecodeError as exc:
                    raise _path_error(f"line {line_no}", "invalid JSON") from exc
                if not isinstance(row, dict):
                    raise _path_error(f"line {line_no}", "expected a JSON object")
                task_index = _index(row, TASK_INDEX_KEY_NAME, path=f"line {line_no}.{TASK_INDEX_KEY_NAME}")
                rollout_index = _index(row, ROLLOUT_INDEX_KEY_NAME, path=f"line {line_no}.{ROLLOUT_INDEX_KEY_NAME}")
                key = (task_index, rollout_index)
                if key in keys:
                    raise _path_error(f"line {line_no}", f"duplicate Gym rollout key {key}")
                keys.add(key)
                agent_ref = row.get(AGENT_REF_KEY_NAME)
                if isinstance(agent_ref, dict) and isinstance(agent_ref.get("name"), str):
                    agent_names.add(agent_ref["name"])
                trajectory = gym_rollout_to_atif(
                    row,
                    session_id=config.session_id,
                    agent_version=config.agent_version,
                )
                encoded = _encoded_trajectory(trajectory)
                digest = hashlib.sha256(encoded).hexdigest()
                filename = f"{task_index}-{rollout_index}.json"
                manifest = {
                    "trajectory_path": filename,
                    TASK_INDEX_KEY_NAME: task_index,
                    ROLLOUT_INDEX_KEY_NAME: rollout_index,
                    "expected_sha256": digest,
                }
                (staging / filename).write_bytes(encoded)
                manifest_lines.append(json.dumps(manifest, sort_keys=True, separators=(",", ":")))

        if not manifest_lines:
            raise AtifExportError("Rollouts file contains no records")
        if len(agent_names) != 1:
            raise AtifExportError(
                "Strict ATIF export requires one agent_ref.name per file because --agent-version is batch-scoped"
            )
        (staging / "manifest.jsonl").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
        if os.path.lexists(output):
            raise AtifExportError(f"Output path was created during export: {output}")
        os.replace(staging, output)
    finally:
        if staging.exists():
            for child in staging.iterdir():
                child.unlink()
            staging.rmdir()

    return AtifExportResult(
        output_dirpath=output,
        trajectory_count=len(manifest_lines),
        manifest_fpath=output / "manifest.jsonl",
    )
