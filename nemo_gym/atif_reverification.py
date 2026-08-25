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
"""Strict ATIF projection for external rollout reverification.

The adapter validates a deliberately bounded Relay-produced ATIF v1.7 subset,
joins each trajectory to an explicit materialized Gym task, and constructs the
Responses-shaped payload used by Gym's existing stateless verifier path.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from openai.types.responses.response_function_call_output_item_list_param import (
    ResponseFunctionCallOutputItemListParam,
)
from openai.types.responses.response_input_text_content_param import ResponseInputTextContentParam
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymFunctionCallOutput,
    NeMoGymResponse,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseReasoningItem,
    NeMoGymResponseUsage,
    NeMoGymSummary,
)
from nemo_gym.relay_atif import AtifTrajectoryV1_7


SUPPORTED_ATIF_VERSION = "ATIF-v1.7"


# A native Gym response carries these request fields back alongside the model's
# output.  Keep the materialized task authoritative rather than attempting to
# reconstruct them from ATIF's intentionally smaller agent description.
_RESPONSE_REQUEST_FIELDS = frozenset(
    {
        "background",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "reasoning",
        "service_tier",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
    }
)


class AtifProjectionError(ConfigError):
    """Raised when an ATIF trajectory cannot be projected without changing what is scored."""


@dataclass(frozen=True)
class LoadedAtifTrajectory:
    """A validated trajectory together with immutable source provenance."""

    trajectory: AtifTrajectoryV1_7
    source_path: Path
    source_sha256: str


class AtifReverifyManifestEntry(BaseModel):
    """Explicit join between one external trajectory and one Gym rollout."""

    trajectory_path: Path
    task_index: int = Field(alias="_ng_task_index", ge=0, strict=True)
    rollout_index: int = Field(alias="_ng_rollout_index", ge=0, strict=True)
    expected_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    model_config = ConfigDict(extra="forbid", populate_by_name=True)


@dataclass(frozen=True)
class ProjectedAtifVerifyPayload:
    """Verifier payload plus provenance that must survive result persistence."""

    payload: dict[str, Any]
    trajectory_id: str | None
    session_id: str | None
    source_path: Path
    source_sha256: str
    task_index: int
    rollout_index: int
    schema_version: str
    projection_status: Literal["complete"] = "complete"


def load_atif_trajectory(path: Path) -> LoadedAtifTrajectory:
    """Read and validate one ATIF document without involving the Relay runtime."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AtifProjectionError(f"could not read ATIF trajectory {path}: {exc}") from exc
    try:
        trajectory = AtifTrajectoryV1_7.model_validate_json(payload)
    except ValueError as exc:
        raise AtifProjectionError(f"invalid ATIF trajectory {path}: {exc}") from exc
    _validate_supported_trajectory(trajectory)
    return LoadedAtifTrajectory(
        trajectory=trajectory,
        source_path=path,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


def load_atif_manifest(path: Path) -> list[AtifReverifyManifestEntry]:
    """Load an explicit trajectory-to-rollout manifest from JSONL."""

    entries: list[AtifReverifyManifestEntry] = []
    try:
        manifest = path.open("rb")
    except OSError as exc:
        raise AtifProjectionError(f"could not read ATIF manifest {path}: {exc}") from exc
    with manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                entries.append(AtifReverifyManifestEntry.model_validate_json(line))
            except ValueError as exc:
                raise AtifProjectionError(f"invalid ATIF manifest row {line_number} in {path}: {exc}") from exc
    if not entries:
        raise AtifProjectionError(f"ATIF manifest {path} contains no entries")
    return entries


def index_materialized_inputs(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[int, int], dict[str, Any]]:
    """Index materialized tasks without guessing identity from row order."""

    indexed: dict[tuple[int, int], dict[str, Any]] = {}
    for row_number, row in enumerate(rows, start=1):
        task_index = row.get(TASK_INDEX_KEY_NAME)
        rollout_index = row.get(ROLLOUT_INDEX_KEY_NAME)
        if type(task_index) is not int or task_index < 0:
            raise AtifProjectionError(
                f"materialized input row {row_number} has invalid {TASK_INDEX_KEY_NAME}: {task_index!r}"
            )
        if type(rollout_index) is not int or rollout_index < 0:
            raise AtifProjectionError(
                f"materialized input row {row_number} has invalid {ROLLOUT_INDEX_KEY_NAME}: {rollout_index!r}"
            )
        key = (task_index, rollout_index)
        if key in indexed:
            raise AtifProjectionError(f"duplicate materialized input key {key}")
        indexed[key] = dict(row)
    return indexed


def project_atif_manifest_entry(
    entry: AtifReverifyManifestEntry,
    materialized_inputs: Mapping[tuple[int, int], Mapping[str, Any]],
    *,
    manifest_directory: Path,
) -> ProjectedAtifVerifyPayload:
    """Resolve one manifest entry without relying on filenames or session IDs."""

    source_path = entry.trajectory_path
    if not source_path.is_absolute():
        source_path = manifest_directory / source_path
    source_path = source_path.resolve()
    loaded = load_atif_trajectory(source_path)

    if entry.expected_sha256 is not None and loaded.source_sha256 != entry.expected_sha256:
        raise AtifProjectionError(
            f"ATIF source hash mismatch for {source_path}: expected {entry.expected_sha256}, "
            f"got {loaded.source_sha256}"
        )

    key = (entry.task_index, entry.rollout_index)
    materialized_input = materialized_inputs.get(key)
    if materialized_input is None:
        raise AtifProjectionError(f"manifest entry {source_path} has no matching materialized input {key}")

    return ProjectedAtifVerifyPayload(
        payload=build_atif_verify_payload(dict(materialized_input), loaded.trajectory),
        trajectory_id=loaded.trajectory.trajectory_id,
        session_id=loaded.trajectory.session_id,
        source_path=source_path,
        source_sha256=loaded.source_sha256,
        task_index=entry.task_index,
        rollout_index=entry.rollout_index,
        schema_version=loaded.trajectory.schema_version,
    )


def project_atif_manifest_entries(
    entries: Iterable[AtifReverifyManifestEntry],
    materialized_inputs: Mapping[tuple[int, int], Mapping[str, Any]],
    *,
    manifest_directory: Path,
) -> list[ProjectedAtifVerifyPayload]:
    """Project a one-to-one manifest while rejecting ambiguous batch joins."""

    projected_payloads: list[ProjectedAtifVerifyPayload] = []
    seen_rollouts: set[tuple[int, int]] = set()
    seen_source_paths: set[Path] = set()
    seen_source_hashes: set[str] = set()
    seen_trajectory_identities: set[tuple[str, str]] = set()

    for entry in entries:
        rollout_key = (entry.task_index, entry.rollout_index)
        if rollout_key in seen_rollouts:
            raise AtifProjectionError(f"manifest maps materialized rollout {rollout_key} more than once")
        seen_rollouts.add(rollout_key)

        projected = project_atif_manifest_entry(
            entry,
            materialized_inputs,
            manifest_directory=manifest_directory,
        )
        if projected.source_path in seen_source_paths:
            raise AtifProjectionError(f"manifest uses ATIF source path {projected.source_path} more than once")
        seen_source_paths.add(projected.source_path)
        if projected.source_sha256 in seen_source_hashes:
            raise AtifProjectionError(f"manifest uses ATIF source content {projected.source_sha256} more than once")
        seen_source_hashes.add(projected.source_sha256)

        trajectory_id = (projected.trajectory_id or "").strip()
        trajectory_identity = (
            ("trajectory_id", trajectory_id) if trajectory_id else ("content_sha256", projected.source_sha256)
        )
        if trajectory_identity in seen_trajectory_identities:
            raise AtifProjectionError(f"manifest repeats ATIF trajectory identity {trajectory_identity}")
        seen_trajectory_identities.add(trajectory_identity)
        projected_payloads.append(projected)

    return projected_payloads


def build_atif_verify_payload(materialized_input: dict[str, Any], trajectory: AtifTrajectoryV1_7) -> dict[str, Any]:
    """Replace a materialized task's native response with its ATIF projection.

    The materialized task remains authoritative for the system/user input,
    verifier metadata, agent routing, and rollout correlation.  ATIF contributes
    only the response produced by the external agent.
    """

    response = atif_trajectory_to_response(trajectory)
    response = _apply_materialized_request_context(response, materialized_input)
    return materialized_input | {"response": response.model_dump(mode="json")}


def atif_trajectory_to_response(trajectory: AtifTrajectoryV1_7) -> NeMoGymResponse:
    """Project one complete text-only ATIF v1.7 trajectory into a Gym response."""

    _validate_supported_trajectory(trajectory)

    output: list[Any] = []
    agent_steps = [step for step in trajectory.steps if step.source == "agent"]
    if not agent_steps:
        raise AtifProjectionError("ATIF trajectory has no agent steps to score")
    seen_call_ids: set[str] = set()
    seen_result_call_ids: set[str] = set()

    for step in agent_steps:
        step_prefix = f"step {step.step_id}"

        raw_reasoning_items = _responses_reasoning_items(trajectory, step)
        if raw_reasoning_items:
            output.extend(raw_reasoning_items)
        elif step.reasoning_content is not None:
            output.append(
                NeMoGymResponseReasoningItem(
                    id=_item_id(trajectory, "reasoning", str(step.step_id)),
                    summary=[NeMoGymSummary(type="summary_text", text=step.reasoning_content)],
                )
            )
        tool_calls = step.tool_calls or []
        message_parts = _text_parts(step.message, f"{step_prefix} message")
        if message_parts or not tool_calls:
            output.append(
                NeMoGymResponseOutputMessage(
                    id=_item_id(trajectory, "message", str(step.step_id)),
                    content=[
                        NeMoGymResponseOutputText(annotations=[], text=text, logprobs=None) for text in message_parts
                    ]
                    or [NeMoGymResponseOutputText(annotations=[], text="", logprobs=None)],
                )
            )

        call_ids: set[str] = set()
        for call in tool_calls:
            if not call.tool_call_id:
                raise AtifProjectionError(f"{step_prefix} contains a tool call with an empty tool_call_id")
            if not call.function_name:
                raise AtifProjectionError(f"{step_prefix} tool call {call.tool_call_id!r} has an empty function_name")
            if call.tool_call_id in call_ids:
                raise AtifProjectionError(f"{step_prefix} repeats tool_call_id {call.tool_call_id!r}")
            if call.tool_call_id in seen_call_ids:
                raise AtifProjectionError(f"ATIF trajectory repeats tool_call_id {call.tool_call_id!r} across steps")
            call_ids.add(call.tool_call_id)
            seen_call_ids.add(call.tool_call_id)
            output.append(
                NeMoGymResponseFunctionToolCall(
                    arguments=json.dumps(call.arguments, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
                    call_id=call.tool_call_id,
                    name=call.function_name,
                    id=_item_id(trajectory, "function-call", call.tool_call_id),
                    status="completed",
                )
            )

        step_result_call_ids: set[str] = set()
        if step.observation is not None:
            for result_index, result in enumerate(step.observation.results):
                if result.subagent_trajectory_ref:
                    raise AtifProjectionError(
                        f"{step_prefix} observation {result_index} references a subagent trajectory"
                    )
                if result.source_call_id is None:
                    raise AtifProjectionError(f"{step_prefix} observation {result_index} has no source_call_id")
                if result.source_call_id not in call_ids:
                    # The official model currently enforces this too.  Keep the
                    # application check so the projection's contract is explicit.
                    raise AtifProjectionError(
                        f"{step_prefix} observation {result_index} references unknown tool call "
                        f"{result.source_call_id!r}"
                    )
                if result.source_call_id in seen_result_call_ids:
                    raise AtifProjectionError(
                        f"ATIF trajectory contains multiple outputs for tool call {result.source_call_id!r}"
                    )
                seen_result_call_ids.add(result.source_call_id)
                step_result_call_ids.add(result.source_call_id)
                result_output = _observation_output(result, f"{step_prefix} observation {result_index}")

                output.append(
                    NeMoGymFunctionCallOutput(
                        call_id=result.source_call_id,
                        output=result_output,
                        id=_item_id(
                            trajectory,
                            "function-output",
                            result.source_call_id,
                            str(result_index),
                        ),
                        status="completed",
                    )
                )

        unresolved_call_ids = call_ids - step_result_call_ids
        if unresolved_call_ids:
            unresolved = ", ".join(sorted(repr(call_id) for call_id in unresolved_call_ids))
            raise AtifProjectionError(f"{step_prefix} has no observation result for tool call(s) {unresolved}")

    return NeMoGymResponse(
        id=_item_id(trajectory, "response"),
        created_at=_created_at(agent_steps),
        model=_model_name(trajectory),
        object="response",
        output=output,
        parallel_tool_calls=any(len(step.tool_calls or []) > 1 for step in agent_steps),
        tool_choice="auto",
        tools=[],
        status="completed",
        usage=_usage(trajectory),
    )


def _validate_supported_trajectory(trajectory: AtifTrajectoryV1_7) -> None:
    if trajectory.schema_version != SUPPORTED_ATIF_VERSION:
        raise AtifProjectionError(
            f"unsupported ATIF schema version {trajectory.schema_version!r}; expected {SUPPORTED_ATIF_VERSION!r}"
        )
    if trajectory.continued_trajectory_ref is not None:
        raise AtifProjectionError("continued ATIF trajectories are not supported by the initial reverify adapter")
    if trajectory.subagent_trajectories:
        raise AtifProjectionError("embedded subagent trajectories are not supported by the initial reverify adapter")
    if _has_multimodal_content(trajectory):
        raise AtifProjectionError("multimodal ATIF content cannot be projected losslessly for reverification")
    copied_steps = [step.step_id for step in trajectory.steps if step.is_copied_context]
    if copied_steps:
        raise AtifProjectionError(f"copied continuation context is not supported; copied step IDs: {copied_steps}")

    saw_agent_step = False
    for step in trajectory.steps:
        _reject_known_incomplete_or_failed_step(step)
        _reject_unprojected_provider_outputs(step)
        if step.source == "agent":
            if step.llm_call_count is not None and step.llm_call_count > 1:
                raise AtifProjectionError(
                    f"agent step {step.step_id} aggregates {step.llm_call_count} LLM calls and cannot be "
                    "projected into one ordered Responses sequence"
                )
            saw_agent_step = True
        elif saw_agent_step:
            raise AtifProjectionError(
                f"non-agent step {step.step_id} appears after agent output and would change the scored conversation"
            )
        if step.source != "agent" and step.observation is not None:
            raise AtifProjectionError(
                f"non-agent step {step.step_id} contains an observation that cannot be projected into a response"
            )
        if step.observation is not None:
            for result_index, result in enumerate(step.observation.results):
                if result.subagent_trajectory_ref:
                    raise AtifProjectionError(
                        f"step {step.step_id} observation {result_index} references a subagent trajectory"
                    )


def _has_multimodal_content(trajectory: AtifTrajectoryV1_7) -> bool:
    for step in trajectory.steps:
        if isinstance(step.message, list) and any(part.type != "text" for part in step.message):
            return True
        if step.observation is None:
            continue
        for result in step.observation.results:
            if isinstance(result.content, list) and any(part.type != "text" for part in result.content):
                return True
    return False


def _text_parts(value: Any, field_name: str) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []

    parts: list[str] = []
    for index, part in enumerate(value):
        if part.type != "text" or part.text is None:
            raise AtifProjectionError(f"{field_name} part {index} is not losslessly projectable text")
        parts.append(part.text)
    return parts


def _observation_output(result: Any, field_name: str) -> str | ResponseFunctionCallOutputItemListParam:
    """Project both standard ATIF content and Relay's structured result extension."""

    if result.content is not None:
        result_parts = _text_parts(result.content, field_name)
        if isinstance(result.content, str):
            return result.content
        return [ResponseInputTextContentParam(type="input_text", text=text) for text in result_parts]

    # Current Relay writes non-string JSON tool returns here because ATIF's
    # `content` field accepts only text or text/image parts.  This is a public
    # producer behavior, not a malformed fallback.
    if result.extra is not None and "tool_result" in result.extra:
        tool_result = result.extra["tool_result"]
        if isinstance(tool_result, str):
            return tool_result
        return json.dumps(tool_result, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    raise AtifProjectionError(f"{field_name} has neither content nor Relay extra.tool_result")


def _apply_materialized_request_context(
    response: NeMoGymResponse,
    materialized_input: Mapping[str, Any],
) -> NeMoGymResponse:
    params = materialized_input.get("responses_create_params")
    if not isinstance(params, Mapping):
        raise AtifProjectionError("materialized input has no responses_create_params object")

    response_payload = response.model_dump(mode="json")
    response_payload.update({field: params[field] for field in _RESPONSE_REQUEST_FIELDS if field in params})
    return NeMoGymResponse.model_validate(response_payload)


def _item_id(trajectory: AtifTrajectoryV1_7, *components: str) -> str:
    trajectory_key = _trajectory_identity_seed(trajectory)
    digest = hashlib.sha256("\0".join((trajectory_key, *components)).encode()).hexdigest()[:20]
    return f"atif_{digest}"


def _trajectory_identity_seed(trajectory: AtifTrajectoryV1_7) -> str:
    if trajectory.trajectory_id is not None and trajectory.trajectory_id.strip():
        return trajectory.trajectory_id

    canonical = trajectory.model_dump(mode="json")
    canonical["trajectory_id"] = None
    encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return f"content:{hashlib.sha256(encoded.encode()).hexdigest()}"


def _reject_known_incomplete_or_failed_step(step: Any) -> None:
    if not isinstance(step.extra, Mapping):
        return

    invocation = step.extra.get("invocation")
    if isinstance(invocation, Mapping):
        status = invocation.get("status")
        if status is not None and (not isinstance(status, str) or status.lower() != "completed"):
            raise AtifProjectionError(f"step {step.step_id} has non-completed Relay invocation status {status!r}")

    tool_invocations = step.extra.get("tool_invocations")
    if isinstance(tool_invocations, list):
        for index, tool_invocation in enumerate(tool_invocations):
            if not isinstance(tool_invocation, Mapping):
                continue
            status = tool_invocation.get("status")
            if status is not None and (not isinstance(status, str) or status.lower() != "completed"):
                raise AtifProjectionError(
                    f"step {step.step_id} tool invocation {index} has non-completed status {status!r}"
                )

    llm_response = step.extra.get("llm_response")
    if not isinstance(llm_response, Mapping):
        return
    provider_response = llm_response.get("raw_response")
    if not isinstance(provider_response, Mapping):
        provider_response = llm_response
    response_status = provider_response.get("status")
    if response_status is not None and (
        not isinstance(response_status, str) or response_status.lower() != "completed"
    ):
        raise AtifProjectionError(
            f"step {step.step_id} has non-completed provider response status {response_status!r}"
        )
    if provider_response.get("error") is not None or provider_response.get("incomplete_details") is not None:
        raise AtifProjectionError(f"step {step.step_id} contains a failed or incomplete provider response")

    choices = provider_response.get("choices")
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, Mapping) and choice.get("finish_reason") in {"content_filter", "length"}:
                raise AtifProjectionError(
                    f"step {step.step_id} has incomplete finish_reason {choice['finish_reason']!r}"
                )
    if provider_response.get("stop_reason") == "max_tokens":
        raise AtifProjectionError(f"step {step.step_id} stopped because the provider token limit was reached")


def _responses_reasoning_items(trajectory: AtifTrajectoryV1_7, step: Any) -> list[NeMoGymResponseReasoningItem]:
    """Preserve reasoning items carried in Relay's raw OpenAI Responses extension."""

    llm_response = step.extra.get("llm_response") if isinstance(step.extra, Mapping) else None
    if isinstance(llm_response, Mapping) and isinstance(llm_response.get("raw_response"), Mapping):
        llm_response = llm_response["raw_response"]
    output = llm_response.get("output") if isinstance(llm_response, Mapping) else None
    if not isinstance(output, list):
        return []

    reasoning_items: list[NeMoGymResponseReasoningItem] = []
    for index, item in enumerate(output):
        if not isinstance(item, Mapping) or item.get("type") != "reasoning":
            continue
        summary = _reasoning_text_parts(item.get("summary"), "summary_text", step.step_id, index, "summary")
        content = _reasoning_text_parts(item.get("content"), "reasoning_text", step.step_id, index, "content")
        encrypted_content = item.get("encrypted_content")
        if encrypted_content is not None and not isinstance(encrypted_content, str):
            raise AtifProjectionError(f"step {step.step_id} reasoning item {index} has invalid encrypted_content")
        raw_id = item.get("id")
        item_id = (
            raw_id
            if isinstance(raw_id, str) and raw_id
            else _item_id(trajectory, "raw-reasoning", str(step.step_id), str(index))
        )
        reasoning_items.append(
            NeMoGymResponseReasoningItem.model_validate(
                {
                    "id": item_id,
                    "summary": summary,
                    "content": content or None,
                    "encrypted_content": encrypted_content,
                    "type": "reasoning",
                }
            )
        )
    return reasoning_items


def _reasoning_text_parts(
    value: Any,
    part_type: Literal["summary_text", "reasoning_text"],
    step_id: int,
    item_index: int,
    field_name: str,
) -> list[dict[str, str]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise AtifProjectionError(f"step {step_id} reasoning item {item_index} has non-list {field_name}")
    parts: list[dict[str, str]] = []
    for part_index, part in enumerate(value):
        if not isinstance(part, Mapping) or not isinstance(part.get("text"), str):
            raise AtifProjectionError(
                f"step {step_id} reasoning item {item_index} {field_name} part {part_index} is invalid"
            )
        declared_type = part.get("type")
        if declared_type not in (None, part_type):
            raise AtifProjectionError(
                f"step {step_id} reasoning item {item_index} {field_name} part {part_index} "
                f"has unsupported type {declared_type!r}"
            )
        parts.append({"type": part_type, "text": part["text"]})
    return parts


def _reject_unprojected_provider_outputs(step: Any) -> None:
    """Reject known raw provider items that the initial Responses projection would drop."""

    llm_response = step.extra.get("llm_response") if isinstance(step.extra, Mapping) else None
    if not isinstance(llm_response, Mapping):
        return

    provider_response = llm_response.get("raw_response")
    if isinstance(provider_response, Mapping):
        llm_response = provider_response

    output = llm_response.get("output")
    if isinstance(output, list):
        supported_item_types = {"reasoning", "function_call", "message", "output_text"}
        for item_index, item in enumerate(output):
            item_type = item.get("type") if isinstance(item, Mapping) else None
            if item_type not in supported_item_types:
                raise AtifProjectionError(
                    f"step {step.step_id} contains unsupported Responses output item {item_type!r} at index {item_index}"
                )
            if item_type == "message":
                content = item.get("content")
                if not isinstance(content, list):
                    raise AtifProjectionError(
                        f"step {step.step_id} Responses message {item_index} has invalid content"
                    )
                unsupported_parts = [
                    part.get("type") if isinstance(part, Mapping) else None
                    for part in content
                    if not isinstance(part, Mapping) or part.get("type") != "output_text"
                ]
                if unsupported_parts:
                    raise AtifProjectionError(
                        f"step {step.step_id} Responses message {item_index} contains unsupported content "
                        f"{unsupported_parts!r}"
                    )
        _validate_responses_raw_coverage(step, output)
        return

    output_text = llm_response.get("output_text")
    if isinstance(output_text, str):
        _validate_raw_coverage(step, raw_message=output_text, raw_tool_calls=[])
        return

    choices = llm_response.get("choices")
    if isinstance(choices, list):
        if len(choices) != 1:
            raise AtifProjectionError(
                f"step {step.step_id} contains {len(choices)} chat choices; expected exactly one"
            )
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, Mapping) else None
        if isinstance(message, Mapping) and any(
            message.get(field) is not None for field in ("audio", "function_call")
        ):
            raise AtifProjectionError(f"step {step.step_id} contains unsupported chat response fields")
        if isinstance(message, Mapping) and message.get("refusal"):
            raise AtifProjectionError(f"step {step.step_id} contains an unsupported chat refusal")
        _validate_chat_raw_coverage(step, choice)
        return

    content = llm_response.get("content")
    if isinstance(content, list):
        supported_content_types = {"text", "tool_use"}
        unsupported_types = [
            block.get("type") if isinstance(block, Mapping) else None
            for block in content
            if not isinstance(block, Mapping) or block.get("type") not in supported_content_types
        ]
        if unsupported_types:
            raise AtifProjectionError(
                f"step {step.step_id} contains unsupported Anthropic content {unsupported_types!r}"
            )
        _validate_anthropic_raw_coverage(step, content)
        return

    raise AtifProjectionError(
        f"step {step.step_id} contains an unrecognized raw provider response shape; complete projection is unknown"
    )


def _validate_responses_raw_coverage(step: Any, output: list[Any]) -> None:
    raw_message_parts: list[str] = []
    raw_tool_calls: list[tuple[str, str, dict[str, Any]]] = []
    for item_index, item in enumerate(output):
        if not isinstance(item, Mapping):
            raise AtifProjectionError(f"step {step.step_id} Responses output item {item_index} is not an object")
        item_type = item.get("type")
        if item_type == "function_call":
            raw_tool_calls.append(
                _raw_tool_call(
                    call_id=item.get("call_id"),
                    name=item.get("name"),
                    arguments=item.get("arguments"),
                    field_name=f"step {step.step_id} Responses function call {item_index}",
                )
            )
        elif item_type == "message":
            for part_index, part in enumerate(item.get("content", [])):
                if not isinstance(part, Mapping) or not isinstance(part.get("text"), str):
                    raise AtifProjectionError(
                        f"step {step.step_id} Responses message {item_index} part {part_index} is invalid"
                    )
                raw_message_parts.append(part["text"])
        elif item_type == "output_text":
            text = item.get("text")
            if not isinstance(text, str):
                raise AtifProjectionError(f"step {step.step_id} Responses output_text {item_index} is invalid")
            raw_message_parts.append(text)

    _validate_raw_coverage(step, raw_message="\n".join(raw_message_parts), raw_tool_calls=raw_tool_calls)


def _validate_chat_raw_coverage(step: Any, choice: Mapping[str, Any]) -> None:
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise AtifProjectionError(f"step {step.step_id} chat choice has no message object")

    content = message.get("content")
    if content is None:
        raw_message = ""
    elif isinstance(content, str):
        raw_message = content
    elif isinstance(content, list):
        raw_parts: list[str] = []
        for part_index, part in enumerate(content):
            if not isinstance(part, Mapping) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise AtifProjectionError(
                    f"step {step.step_id} chat message content part {part_index} is not lossless text"
                )
            raw_parts.append(part["text"])
        raw_message = "\n".join(raw_parts)
    else:
        raise AtifProjectionError(f"step {step.step_id} chat message content has an unsupported type")

    raw_tool_calls: list[tuple[str, str, dict[str, Any]]] = []
    tool_calls = message.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise AtifProjectionError(f"step {step.step_id} chat tool_calls is not a list")
        for call_index, call in enumerate(tool_calls):
            function = call.get("function") if isinstance(call, Mapping) else None
            if not isinstance(function, Mapping):
                raise AtifProjectionError(f"step {step.step_id} chat tool call {call_index} is invalid")
            raw_tool_calls.append(
                _raw_tool_call(
                    call_id=call.get("id"),
                    name=function.get("name"),
                    arguments=function.get("arguments"),
                    field_name=f"step {step.step_id} chat tool call {call_index}",
                )
            )

    _validate_raw_coverage(step, raw_message=raw_message, raw_tool_calls=raw_tool_calls)


def _validate_anthropic_raw_coverage(step: Any, content: list[Any]) -> None:
    raw_message_parts: list[str] = []
    raw_tool_calls: list[tuple[str, str, dict[str, Any]]] = []
    for block_index, block in enumerate(content):
        if not isinstance(block, Mapping):
            raise AtifProjectionError(f"step {step.step_id} Anthropic content block {block_index} is invalid")
        if block.get("type") == "text":
            text = block.get("text")
            if not isinstance(text, str):
                raise AtifProjectionError(f"step {step.step_id} Anthropic text block {block_index} is invalid")
            raw_message_parts.append(text)
        elif block.get("type") == "tool_use":
            raw_tool_calls.append(
                _raw_tool_call(
                    call_id=block.get("id"),
                    name=block.get("name"),
                    arguments=block.get("input"),
                    field_name=f"step {step.step_id} Anthropic tool_use block {block_index}",
                )
            )

    _validate_raw_coverage(step, raw_message="\n".join(raw_message_parts), raw_tool_calls=raw_tool_calls)


def _raw_tool_call(
    *,
    call_id: Any,
    name: Any,
    arguments: Any,
    field_name: str,
) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(call_id, str) or not call_id:
        raise AtifProjectionError(f"{field_name} has no non-empty invocation ID")
    if not isinstance(name, str) or not name:
        raise AtifProjectionError(f"{field_name} has no non-empty function name")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise AtifProjectionError(f"{field_name} contains invalid JSON arguments: {exc}") from exc
    if not isinstance(arguments, dict):
        raise AtifProjectionError(f"{field_name} arguments are not a JSON object")
    return call_id, name, arguments


def _validate_raw_coverage(
    step: Any,
    *,
    raw_message: str,
    raw_tool_calls: list[tuple[str, str, dict[str, Any]]],
) -> None:
    canonical_message = "\n".join(_text_parts(step.message, f"step {step.step_id} message"))
    if raw_message != canonical_message:
        raise AtifProjectionError(
            f"step {step.step_id} raw provider message does not match the canonical ATIF message"
        )

    canonical_tool_calls = [(call.tool_call_id, call.function_name, call.arguments) for call in step.tool_calls or []]
    if raw_tool_calls != canonical_tool_calls:
        raise AtifProjectionError(
            f"step {step.step_id} raw provider tool calls do not match the canonical ATIF tool calls"
        )


def _created_at(agent_steps: list[Any]) -> float:
    for step in reversed(agent_steps):
        if step.timestamp is not None:
            timestamp = datetime.fromisoformat(step.timestamp.replace("Z", "+00:00"))
            if timestamp.tzinfo is None:
                raise AtifProjectionError(
                    f"agent step {step.step_id} timestamp has no timezone and cannot produce a stable created_at"
                )
            return timestamp.timestamp()
    return 0.0


def _model_name(trajectory: AtifTrajectoryV1_7) -> str:
    for step in reversed(trajectory.steps):
        if step.source == "agent" and step.model_name:
            return step.model_name
    return trajectory.agent.model_name or "unknown"


def _usage(trajectory: AtifTrajectoryV1_7) -> NeMoGymResponseUsage | None:
    metrics = trajectory.final_metrics
    if metrics is None or metrics.total_prompt_tokens is None or metrics.total_completion_tokens is None:
        return None
    return NeMoGymResponseUsage(
        input_tokens=metrics.total_prompt_tokens,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=metrics.total_cached_tokens),
        output_tokens=metrics.total_completion_tokens,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=None),
        total_tokens=metrics.total_prompt_tokens + metrics.total_completion_tokens,
    )
