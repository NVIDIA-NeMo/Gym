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
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

from openai.types.responses.response_function_call_output_item_list_param import (
    ResponseFunctionCallOutputItemListParam,
)
from openai.types.responses.response_input_text_content_param import ResponseInputTextContentParam
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.atif_json import json_values_equal, strict_json_loads
from nemo_gym.atif_v1_7 import AtifTrajectoryV1_7
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
    TokenIDLogProbMixin,
    training_variant_of,
)


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
    trajectory_content_sha256: str
    task_index: int
    rollout_index: int
    schema_version: str
    projection_status: Literal["complete"] = "complete"


@dataclass(frozen=True)
class _ResolvedProviderResponse:
    """A provider payload and its optional Relay/Hermes response envelope."""

    provider: Mapping[str, Any]
    envelope: Mapping[str, Any] | None = None


def load_atif_trajectory(path: Path) -> LoadedAtifTrajectory:
    """Read and validate one ATIF document without involving the Relay runtime."""

    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise AtifProjectionError(f"could not read ATIF trajectory {path}: {exc}") from exc
    try:
        trajectory = AtifTrajectoryV1_7.model_validate(strict_json_loads(payload))
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
                entries.append(AtifReverifyManifestEntry.model_validate(strict_json_loads(line)))
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
    _validate_declared_gym_source_identity(loaded.trajectory, key)
    materialized_input = materialized_inputs.get(key)
    if materialized_input is None:
        raise AtifProjectionError(f"manifest entry {source_path} has no matching materialized input {key}")

    return ProjectedAtifVerifyPayload(
        payload=build_atif_verify_payload(dict(materialized_input), loaded.trajectory),
        trajectory_id=loaded.trajectory.trajectory_id,
        session_id=loaded.trajectory.session_id,
        source_path=source_path,
        source_sha256=loaded.source_sha256,
        trajectory_content_sha256=_trajectory_content_sha256(loaded.trajectory),
        task_index=entry.task_index,
        rollout_index=entry.rollout_index,
        schema_version=loaded.trajectory.schema_version,
    )


def _validate_declared_gym_source_identity(
    trajectory: AtifTrajectoryV1_7,
    manifest_key: tuple[int, int],
) -> None:
    if not isinstance(trajectory.extra, Mapping):
        return
    nemo_gym = trajectory.extra.get("nemo_gym")
    if not isinstance(nemo_gym, Mapping):
        return
    source = nemo_gym.get("source")
    if not isinstance(source, Mapping):
        return

    for field_name, expected in zip(("task_index", "rollout_index"), manifest_key, strict=True):
        if field_name not in source:
            continue
        value = source[field_name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AtifProjectionError(f"Gym ATIF source {field_name} is invalid: {value!r}")
        if value != expected:
            raise AtifProjectionError(f"Gym ATIF source {field_name} {value} conflicts with manifest value {expected}")


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
            ("trajectory_id", trajectory_id)
            if trajectory_id
            else ("content_sha256", projected.trajectory_content_sha256)
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
            if not call.tool_call_id.strip():
                raise AtifProjectionError(f"{step_prefix} contains a tool call with a blank tool_call_id")
            if not call.function_name.strip():
                raise AtifProjectionError(f"{step_prefix} tool call {call.tool_call_id!r} has a blank function_name")
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

        training_metadata = _step_training_metadata(step)
        if training_metadata is not None:
            last_model_output = output[-1]
            training_variant = training_variant_of(last_model_output.__class__)
            output[-1] = training_variant(
                **last_model_output.model_dump(),
                **training_metadata.model_dump(exclude_none=True),
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

    _validate_projected_output_ids(output)

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
    _reject_nonfinite_json_numbers(trajectory.model_dump(mode="python"), "ATIF trajectory")
    _reject_declared_incomplete_gym_conversion(trajectory)
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
        _step_training_metadata(step)
        if step.source == "agent":
            if step.tool_calls and _text_parts(step.message, f"step {step.step_id} message"):
                raise AtifProjectionError(
                    f"agent step {step.step_id} contains both message text and tool calls; "
                    "ATIF does not preserve their provider output-item ordering"
                )
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


def _reject_declared_incomplete_gym_conversion(trajectory: AtifTrajectoryV1_7) -> None:
    """Honor completion evidence emitted by Gym's ATIF converter when present."""

    if not isinstance(trajectory.extra, Mapping) or "nemo_gym" not in trajectory.extra:
        return
    nemo_gym = trajectory.extra["nemo_gym"]
    if not isinstance(nemo_gym, Mapping):
        raise AtifProjectionError("ATIF extra.nemo_gym must be an object")

    if "source" in nemo_gym:
        source = nemo_gym["source"]
        if not isinstance(source, Mapping):
            raise AtifProjectionError("ATIF extra.nemo_gym.source must be an object")
        if "invocation_status" in source:
            status = source["invocation_status"]
            if not isinstance(status, str) or status.lower() != "completed":
                raise AtifProjectionError(f"Gym source invocation is not completed: {status!r}")

    if "conversion" in nemo_gym:
        conversion = nemo_gym["conversion"]
        if not isinstance(conversion, Mapping):
            raise AtifProjectionError("ATIF extra.nemo_gym.conversion must be an object")
        if "status" in conversion:
            status = conversion["status"]
            if not isinstance(status, str) or status.lower() != "complete":
                raise AtifProjectionError(f"Gym ATIF conversion is not complete: {status!r}")


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
    if not value:
        raise AtifProjectionError(f"{field_name} contains an empty content-part list")

    parts: list[str] = []
    for index, part in enumerate(value):
        if part.type != "text" or part.text is None:
            raise AtifProjectionError(f"{field_name} part {index} is not losslessly projectable text")
        parts.append(part.text)
    return parts


def _observation_output(result: Any, field_name: str) -> str | ResponseFunctionCallOutputItemListParam:
    """Project both standard ATIF content and Relay's structured result extension."""

    if result.content is not None and isinstance(result.extra, Mapping) and "tool_result" in result.extra:
        raise AtifProjectionError(f"{field_name} contains both content and Relay extra.tool_result")

    if result.content is not None:
        result_parts = _text_parts(result.content, field_name)
        if isinstance(result.content, str):
            return result.content
        if not result_parts:
            raise AtifProjectionError(
                f"{field_name} contains an empty content-part list; it cannot be distinguished from a lost structured result"
            )
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

    return f"content:{_trajectory_content_sha256(trajectory)}"


def _trajectory_content_sha256(trajectory: AtifTrajectoryV1_7) -> str:
    """Hash semantic ATIF content independently of source JSON formatting."""

    canonical = trajectory.model_dump(mode="json")
    canonical["trajectory_id"] = None
    encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _step_training_metadata(step: Any) -> TokenIDLogProbMixin | None:
    """Map Gym's complete ATIF token metadata extension back to a training item."""

    metrics = step.metrics
    if metrics is None:
        return None

    required_values = {
        "prompt_token_ids": metrics.prompt_token_ids,
        "generation_token_ids": metrics.completion_token_ids,
        "generation_log_probs": metrics.logprobs,
    }
    nemo_gym_extra = metrics.extra.get("nemo_gym") if isinstance(metrics.extra, Mapping) else None
    has_routed_experts = isinstance(nemo_gym_extra, Mapping) and "routed_experts" in nemo_gym_extra
    if not any(value is not None for value in required_values.values()) and not has_routed_experts:
        return None

    missing = [field for field, value in required_values.items() if value is None]
    if missing:
        raise AtifProjectionError(
            f"step {step.step_id} training token metadata is incomplete; missing: {', '.join(sorted(missing))}"
        )

    generation_token_ids = required_values["generation_token_ids"]
    generation_log_probs = required_values["generation_log_probs"]
    assert generation_token_ids is not None
    assert generation_log_probs is not None
    if len(generation_token_ids) != len(generation_log_probs):
        raise AtifProjectionError(
            f"step {step.step_id} completion token IDs and log probabilities must have the same length"
        )

    payload = dict(required_values)
    if has_routed_experts:
        routed_experts = nemo_gym_extra["routed_experts"]
        _validate_routed_expert_types(routed_experts, f"step {step.step_id} routed_experts")
        payload["routed_experts"] = routed_experts
    try:
        return TokenIDLogProbMixin.model_validate(payload)
    except ValueError as exc:
        raise AtifProjectionError(f"step {step.step_id} contains invalid training token metadata: {exc}") from exc


def _validate_routed_expert_types(value: Any, field_name: str) -> None:
    """Reject boolean/coercible routed-expert indices before Pydantic normalization."""

    if isinstance(value, str):
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_routed_expert_types(item, f"{field_name}[{index}]")
        return
    if isinstance(value, bool) or not isinstance(value, int):
        raise AtifProjectionError(f"{field_name} must contain only JSON integer indices")


def _resolve_provider_response(step: Any) -> _ResolvedProviderResponse | None:
    if not isinstance(step.extra, Mapping) or "llm_response" not in step.extra:
        return None
    llm_response = step.extra["llm_response"]
    if not isinstance(llm_response, Mapping):
        raise AtifProjectionError(
            f"step {step.step_id} Relay llm_response is not a supported provider response object"
        )
    if "raw_response" not in llm_response:
        return _ResolvedProviderResponse(provider=llm_response)
    raw_response = llm_response["raw_response"]
    if not isinstance(raw_response, Mapping):
        raise AtifProjectionError(f"step {step.step_id} Relay raw_response is not a provider response object")
    return _ResolvedProviderResponse(provider=raw_response, envelope=llm_response)


def _provider_response_family(step: Any, response: Mapping[str, Any]) -> Literal["responses", "chat", "anthropic"]:
    families: list[Literal["responses", "chat", "anthropic"]] = []
    if "output" in response or "output_text" in response:
        families.append("responses")
    if "choices" in response:
        families.append("chat")
    if "content" in response:
        families.append("anthropic")
    if len(families) > 1:
        raise AtifProjectionError(
            f"step {step.step_id} provider response mixes incompatible output families: {', '.join(families)}"
        )
    if not families:
        raise AtifProjectionError(
            f"step {step.step_id} contains an unrecognized raw provider response shape; complete projection is unknown"
        )

    hidden_generic_fields = {
        "actions",
        "answer",
        "assistant_message",
        "message",
        "tool_calls",
    }
    hidden = sorted(field for field in hidden_generic_fields if field in response)
    if hidden:
        raise AtifProjectionError(
            f"step {step.step_id} provider response contains additional output fields: "
            f"{', '.join(repr(field) for field in hidden)}"
        )
    return families[0]


def _validate_relay_hermes_wrapper(step: Any, resolved: _ResolvedProviderResponse) -> str | None:
    """Validate the exact response envelope emitted by Relay's Hermes integration."""

    envelope = resolved.envelope
    if envelope is None:
        return None
    _reject_unknown_provider_fields(
        envelope,
        {"assistant_message", "finish_reason", "raw_response", "usage"},
        f"step {step.step_id} Relay response envelope",
    )

    assistant_message = envelope.get("assistant_message")
    if not isinstance(assistant_message, Mapping):
        raise AtifProjectionError(f"step {step.step_id} Relay assistant_message is not an object")
    _reject_unknown_provider_fields(
        assistant_message,
        {"content", "role", "tool_calls"},
        f"step {step.step_id} Relay assistant_message",
    )
    if assistant_message.get("role") != "assistant":
        raise AtifProjectionError(f"step {step.step_id} Relay assistant_message has a non-assistant role")

    content = assistant_message.get("content")
    if content is None:
        message_parts: list[str] = []
    elif isinstance(content, str):
        message_parts = [content] if content else []
    else:
        raise AtifProjectionError(f"step {step.step_id} Relay assistant_message content is not scalar text")

    raw_tool_calls: list[tuple[str, str, dict[str, Any]]] = []
    tool_calls = assistant_message.get("tool_calls")
    if tool_calls is not None:
        if not isinstance(tool_calls, list):
            raise AtifProjectionError(f"step {step.step_id} Relay assistant_message tool_calls is not a list")
        for index, call in enumerate(tool_calls):
            if not isinstance(call, Mapping):
                raise AtifProjectionError(
                    f"step {step.step_id} Relay assistant_message tool call {index} is not an object"
                )
            _reject_unknown_provider_fields(
                call,
                {"arguments", "id", "name", "provider_data"},
                f"step {step.step_id} Relay assistant_message tool call {index}",
            )
            raw_tool_calls.append(
                _raw_tool_call(
                    call_id=call.get("id"),
                    name=call.get("name"),
                    arguments=call.get("arguments"),
                    field_name=f"step {step.step_id} Relay assistant_message tool call {index}",
                )
            )
    _validate_raw_coverage(step, raw_message_parts=message_parts, raw_tool_calls=raw_tool_calls)

    envelope_finish = envelope.get("finish_reason")
    if envelope_finish is not None:
        if envelope_finish not in {"stop", "tool_calls"}:
            raise AtifProjectionError(
                f"step {step.step_id} Relay response envelope has non-terminal finish_reason {envelope_finish!r}"
            )
        if raw_tool_calls and envelope_finish != "tool_calls":
            raise AtifProjectionError(
                f"step {step.step_id} Relay assistant_message tool calls conflict with "
                f"finish_reason {envelope_finish!r}"
            )
        if not raw_tool_calls and envelope_finish == "tool_calls":
            raise AtifProjectionError(
                f"step {step.step_id} Relay response envelope has finish_reason='tool_calls' without tool calls"
            )
    return envelope_finish


def _relay_wrapper_proves_completed_tool_call(step: Any) -> bool:
    invocation = step.extra.get("invocation") if isinstance(step.extra, Mapping) else None
    status = invocation.get("status") if isinstance(invocation, Mapping) else None
    if not isinstance(status, str) or status.lower() != "completed":
        return False
    call_ids = {call.tool_call_id for call in step.tool_calls or []}
    if not call_ids or step.observation is None:
        return False
    result_ids = [result.source_call_id for result in step.observation.results]
    return len(result_ids) == len(call_ids) and set(result_ids) == call_ids


def _reject_known_incomplete_or_failed_step(step: Any) -> None:
    for index, tool_call in enumerate(step.tool_calls or []):
        if not isinstance(tool_call.extra, Mapping) or "status" not in tool_call.extra:
            continue
        status = tool_call.extra["status"]
        if not isinstance(status, str) or status.lower() != "completed":
            raise AtifProjectionError(f"step {step.step_id} tool call {index} has non-completed status {status!r}")

    if not isinstance(step.extra, Mapping):
        return

    if "invocation" in step.extra:
        invocation = step.extra["invocation"]
        if not isinstance(invocation, Mapping):
            raise AtifProjectionError(f"step {step.step_id} Relay invocation metadata is not an object")
        if "status" in invocation:
            status = invocation["status"]
            if not isinstance(status, str) or status.lower() != "completed":
                raise AtifProjectionError(f"step {step.step_id} has non-completed Relay invocation status {status!r}")

    if "tool_invocations" in step.extra:
        tool_invocations = step.extra["tool_invocations"]
        if not isinstance(tool_invocations, list):
            raise AtifProjectionError(f"step {step.step_id} Relay tool_invocations metadata is not a list")
        tool_calls = step.tool_calls or []
        if len(tool_invocations) != len(tool_calls):
            raise AtifProjectionError(
                f"step {step.step_id} Relay tool_invocations metadata has {len(tool_invocations)} entries "
                f"for {len(tool_calls)} canonical tool calls"
            )
        for index, tool_invocation in enumerate(tool_invocations):
            if not isinstance(tool_invocation, Mapping):
                raise AtifProjectionError(
                    f"step {step.step_id} Relay tool invocation {index} metadata is not an object"
                )
            invocation_id = tool_invocation.get("invocation_id")
            if not isinstance(invocation_id, str) or not invocation_id.strip():
                raise AtifProjectionError(
                    f"step {step.step_id} Relay tool invocation {index} has a blank or invalid invocation_id"
                )
            if invocation_id != tool_calls[index].tool_call_id:
                raise AtifProjectionError(
                    f"step {step.step_id} Relay tool invocation {index} invocation_id {invocation_id!r} "
                    f"does not match canonical tool call {tool_calls[index].tool_call_id!r}"
                )
            if "status" in tool_invocation:
                status = tool_invocation["status"]
                if not isinstance(status, str) or status.lower() != "completed":
                    raise AtifProjectionError(
                        f"step {step.step_id} tool invocation {index} has non-completed status {status!r}"
                    )

    resolved = _resolve_provider_response(step)
    if resolved is None:
        return
    response_layers = [resolved.provider]
    if resolved.envelope is not None:
        response_layers.insert(0, resolved.envelope)
    for response_layer in response_layers:
        response_status = response_layer.get("status")
        if response_status is not None and (
            not isinstance(response_status, str) or response_status.lower() != "completed"
        ):
            raise AtifProjectionError(
                f"step {step.step_id} has non-completed provider response status {response_status!r}"
            )
        if response_layer.get("error") is not None or response_layer.get("incomplete_details") is not None:
            raise AtifProjectionError(f"step {step.step_id} contains a failed or incomplete provider response")


def _responses_reasoning_items(trajectory: AtifTrajectoryV1_7, step: Any) -> list[NeMoGymResponseReasoningItem]:
    """Preserve reasoning items carried in Relay's raw OpenAI Responses extension."""

    resolved = _resolve_provider_response(step)
    output = resolved.provider.get("output") if resolved is not None else None
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


def _reject_unknown_provider_fields(value: Mapping[str, Any], allowed: set[str], field_name: str) -> None:
    unknown = sorted(repr(key) for key in value if key not in allowed)
    if unknown:
        raise AtifProjectionError(f"{field_name} contains unsupported fields: {', '.join(unknown)}")


def _reject_unknown_provider_parts(value: Any, allowed: set[str], field_name: str) -> None:
    if value is None or not isinstance(value, list):
        return
    for index, part in enumerate(value):
        if isinstance(part, Mapping):
            _reject_unknown_provider_fields(part, allowed, f"{field_name} part {index}")


def _reject_unprojected_provider_outputs(step: Any) -> None:
    """Reject known raw provider items that the initial Responses projection would drop."""

    resolved = _resolve_provider_response(step)
    if resolved is None:
        return
    llm_response = resolved.provider
    wrapper_finish_reason = _validate_relay_hermes_wrapper(step, resolved)
    family = _provider_response_family(step, llm_response)

    if family == "responses" and "output" in llm_response:
        output = llm_response["output"]
        if not isinstance(output, list):
            raise AtifProjectionError(f"step {step.step_id} Responses output is not a list")
        supported_item_types = {"reasoning", "function_call", "message", "output_text"}
        saw_non_reasoning_item = False
        for item_index, item in enumerate(output):
            item_type = item.get("type") if isinstance(item, Mapping) else None
            if item_type not in supported_item_types:
                raise AtifProjectionError(
                    f"step {step.step_id} contains unsupported Responses output item {item_type!r} at index {item_index}"
                )
            if item_type == "reasoning":
                if saw_non_reasoning_item:
                    raise AtifProjectionError(
                        f"step {step.step_id} Responses reasoning item {item_index} appears after output; "
                        "the projection would reorder it"
                    )
            else:
                saw_non_reasoning_item = True
            if item_type == "message":
                role = item.get("role")
                if role != "assistant":
                    raise AtifProjectionError(
                        f"step {step.step_id} Responses message {item_index} has non-assistant role {role!r}"
                    )
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
                for part_index, part in enumerate(content):
                    _reject_dropped_text_metadata(
                        part,
                        f"step {step.step_id} Responses message {item_index} part {part_index}",
                    )
            elif item_type == "output_text":
                _reject_dropped_text_metadata(
                    item,
                    f"step {step.step_id} Responses output_text {item_index}",
                )

            if item_type in {"reasoning", "message", "function_call"}:
                item_status = item.get("status")
                if item_status is not None and (
                    not isinstance(item_status, str) or item_status.lower() != "completed"
                ):
                    raise AtifProjectionError(
                        f"step {step.step_id} Responses {item_type} {item_index} "
                        f"has non-completed status {item_status!r}"
                    )
        response_status = llm_response.get("status")
        if not isinstance(response_status, str) or response_status.lower() != "completed":
            raise AtifProjectionError(
                f"step {step.step_id} Responses provider status is not completed: {response_status!r}"
            )
        _validate_responses_raw_coverage(step, output)
        return

    if family == "responses":
        output_text = llm_response["output_text"]
        if not isinstance(output_text, str):
            raise AtifProjectionError(f"step {step.step_id} Responses output_text is not scalar text")
        response_status = llm_response.get("status")
        if not isinstance(response_status, str) or response_status.lower() != "completed":
            raise AtifProjectionError(
                f"step {step.step_id} Responses provider status is not completed: {response_status!r}"
            )
        _validate_raw_coverage(
            step,
            raw_message_parts=[output_text] if output_text else [],
            raw_tool_calls=[],
        )
        return

    if family == "chat":
        choices = llm_response["choices"]
        if not isinstance(choices, list):
            raise AtifProjectionError(f"step {step.step_id} chat choices is not a list")
        if len(choices) != 1:
            raise AtifProjectionError(
                f"step {step.step_id} contains {len(choices)} chat choices; expected exactly one"
            )
        choice = choices[0]
        if isinstance(choice, Mapping):
            _reject_unknown_provider_fields(
                choice,
                {"index", "message", "finish_reason", "logprobs"},
                f"step {step.step_id} chat choice",
            )
            choice_index = choice.get("index")
            if choice_index is not None and (
                isinstance(choice_index, bool) or not isinstance(choice_index, int) or choice_index != 0
            ):
                raise AtifProjectionError(f"step {step.step_id} chat choice has inconsistent index {choice_index!r}")
        message = choice.get("message") if isinstance(choice, Mapping) else None
        if isinstance(message, Mapping) and any(
            message.get(field) is not None for field in ("audio", "function_call")
        ):
            raise AtifProjectionError(f"step {step.step_id} contains unsupported chat response fields")
        if isinstance(message, Mapping) and message.get("refusal"):
            raise AtifProjectionError(f"step {step.step_id} contains an unsupported chat refusal")
        if isinstance(message, Mapping) and message.get("annotations"):
            raise AtifProjectionError(f"step {step.step_id} chat message contains non-empty annotations")
        if isinstance(choice, Mapping) and choice.get("logprobs"):
            raise AtifProjectionError(f"step {step.step_id} chat choice contains non-empty logprobs")
        _validate_chat_raw_coverage(
            step,
            choice,
            relay_hermes_wrapper=resolved.envelope is not None,
            wrapper_finish_reason=wrapper_finish_reason,
        )
        return

    if family == "anthropic":
        content = llm_response["content"]
        if not isinstance(content, list):
            raise AtifProjectionError(f"step {step.step_id} Anthropic content is not a list")
        role = llm_response.get("role")
        if role != "assistant":
            raise AtifProjectionError(f"step {step.step_id} Anthropic message has non-assistant role {role!r}")
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
        _validate_anthropic_raw_coverage(step, content, stop_reason=llm_response.get("stop_reason"))
        return

    raise AssertionError(f"unhandled provider response family: {family}")


def _validate_responses_raw_coverage(step: Any, output: list[Any]) -> None:
    if step.reasoning_content is not None and any(
        isinstance(item, Mapping) and item.get("type") == "reasoning" for item in output
    ):
        raise AtifProjectionError(
            f"step {step.step_id} contains both canonical reasoning_content and raw Responses reasoning items"
        )

    raw_message_parts: list[str] = []
    raw_tool_calls: list[tuple[str, str, dict[str, Any]]] = []
    message_item_count = 0
    for item_index, item in enumerate(output):
        if not isinstance(item, Mapping):
            raise AtifProjectionError(f"step {step.step_id} Responses output item {item_index} is not an object")
        item_type = item.get("type")
        allowed_fields = {
            "reasoning": {"id", "type", "status", "summary", "content", "encrypted_content"},
            "message": {"id", "type", "role", "status", "content"},
            "output_text": {"type", "text", "annotations", "logprobs"},
            "function_call": {"id", "call_id", "type", "name", "arguments", "status"},
        }
        _reject_unknown_provider_fields(
            item,
            allowed_fields[item_type],
            f"step {step.step_id} Responses {item_type} {item_index}",
        )
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
            message_item_count += 1
            _reject_unknown_provider_parts(
                item.get("content"),
                {"type", "text", "annotations", "logprobs"},
                f"step {step.step_id} Responses message {item_index}",
            )
            for part_index, part in enumerate(item.get("content", [])):
                if not isinstance(part, Mapping) or not isinstance(part.get("text"), str):
                    raise AtifProjectionError(
                        f"step {step.step_id} Responses message {item_index} part {part_index} is invalid"
                    )
                raw_message_parts.append(part["text"])
        elif item_type == "output_text":
            message_item_count += 1
            text = item.get("text")
            if not isinstance(text, str):
                raise AtifProjectionError(f"step {step.step_id} Responses output_text {item_index} is invalid")
            raw_message_parts.append(text)
        elif item_type == "reasoning":
            _reject_unknown_provider_parts(
                item.get("summary"),
                {"type", "text"},
                f"step {step.step_id} Responses reasoning {item_index} summary",
            )
            _reject_unknown_provider_parts(
                item.get("content"),
                {"type", "text"},
                f"step {step.step_id} Responses reasoning {item_index} content",
            )

    if message_item_count > 1:
        raise AtifProjectionError(
            f"step {step.step_id} contains {message_item_count} Responses message/output_text items; "
            "ATIF cannot preserve their item boundaries"
        )

    _validate_raw_coverage(step, raw_message_parts=raw_message_parts, raw_tool_calls=raw_tool_calls)


def _validate_chat_raw_coverage(
    step: Any,
    choice: Mapping[str, Any],
    *,
    relay_hermes_wrapper: bool = False,
    wrapper_finish_reason: str | None = None,
) -> None:
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise AtifProjectionError(f"step {step.step_id} chat choice has no message object")
    role = message.get("role")
    if role is None and relay_hermes_wrapper:
        role = "assistant"
    if role != "assistant":
        raise AtifProjectionError(f"step {step.step_id} chat message has non-assistant role {role!r}")
    _reject_unknown_provider_fields(
        message,
        {
            "role",
            "content",
            "refusal",
            "audio",
            "function_call",
            "tool_calls",
            "annotations",
            "reasoning_content",
            "reasoning",
        },
        f"step {step.step_id} chat message",
    )

    raw_reasoning = [
        (field_name, message[field_name])
        for field_name in ("reasoning_content", "reasoning")
        if field_name in message and message[field_name] is not None
    ]
    for field_name, value in raw_reasoning:
        if not isinstance(value, str):
            raise AtifProjectionError(f"step {step.step_id} chat {field_name} is not scalar text")
    if len(raw_reasoning) == 2 and raw_reasoning[0][1] != raw_reasoning[1][1]:
        raise AtifProjectionError(f"step {step.step_id} chat reasoning aliases conflict")
    if raw_reasoning and raw_reasoning[0][1] != step.reasoning_content:
        raise AtifProjectionError(f"step {step.step_id} raw chat reasoning does not match canonical ATIF reasoning")

    content = message.get("content")
    if content is None:
        raw_message_parts: list[str] = []
    elif isinstance(content, str):
        raw_message_parts = [content] if content else []
    elif isinstance(content, list):
        raw_message_parts = []
        for part_index, part in enumerate(content):
            if not isinstance(part, Mapping) or part.get("type") != "text" or not isinstance(part.get("text"), str):
                raise AtifProjectionError(
                    f"step {step.step_id} chat message content part {part_index} is not lossless text"
                )
            _reject_unknown_provider_fields(
                part,
                {"type", "text"},
                f"step {step.step_id} chat message content part {part_index}",
            )
            raw_message_parts.append(part["text"])
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
            _reject_unknown_provider_fields(
                call,
                {"id", "type", "function", "index"},
                f"step {step.step_id} chat tool call {call_index}",
            )
            call_type = call.get("type")
            if call_type not in (None, "function"):
                raise AtifProjectionError(
                    f"step {step.step_id} chat tool call {call_index} has unsupported type {call_type!r}"
                )
            declared_index = call.get("index")
            if declared_index is not None and (
                isinstance(declared_index, bool) or not isinstance(declared_index, int) or declared_index != call_index
            ):
                raise AtifProjectionError(
                    f"step {step.step_id} chat tool call {call_index} has inconsistent index {declared_index!r}"
                )
            _reject_unknown_provider_fields(
                function,
                {"name", "arguments"},
                f"step {step.step_id} chat tool call {call_index} function",
            )
            raw_tool_calls.append(
                _raw_tool_call(
                    call_id=call.get("id"),
                    name=function.get("name"),
                    arguments=function.get("arguments"),
                    field_name=f"step {step.step_id} chat tool call {call_index}",
                )
            )

    finish_reason = choice.get("finish_reason")
    if finish_reason is None:
        finish_reason = wrapper_finish_reason
    if (
        finish_reason is None
        and relay_hermes_wrapper
        and raw_tool_calls
        and _relay_wrapper_proves_completed_tool_call(step)
    ):
        finish_reason = "tool_calls"
    if finish_reason not in {"stop", "tool_calls"}:
        raise AtifProjectionError(f"step {step.step_id} has non-terminal chat finish_reason {finish_reason!r}")
    if raw_tool_calls and finish_reason != "tool_calls":
        raise AtifProjectionError(
            f"step {step.step_id} chat tool calls have inconsistent finish_reason {finish_reason!r}"
        )
    if not raw_tool_calls and finish_reason == "tool_calls":
        raise AtifProjectionError(f"step {step.step_id} chat finish_reason='tool_calls' has no tool calls")

    _validate_raw_coverage(step, raw_message_parts=raw_message_parts, raw_tool_calls=raw_tool_calls)


def _validate_anthropic_raw_coverage(step: Any, content: list[Any], *, stop_reason: Any) -> None:
    if stop_reason not in {"end_turn", "stop_sequence", "tool_use"}:
        raise AtifProjectionError(f"step {step.step_id} has non-terminal Anthropic stop_reason {stop_reason!r}")
    raw_message_parts: list[str] = []
    raw_tool_calls: list[tuple[str, str, dict[str, Any]]] = []
    for block_index, block in enumerate(content):
        if not isinstance(block, Mapping):
            raise AtifProjectionError(f"step {step.step_id} Anthropic content block {block_index} is invalid")
        if block.get("type") == "text":
            _reject_unknown_provider_fields(
                block,
                {"type", "text", "citations"},
                f"step {step.step_id} Anthropic text block {block_index}",
            )
            text = block.get("text")
            if not isinstance(text, str):
                raise AtifProjectionError(f"step {step.step_id} Anthropic text block {block_index} is invalid")
            if block.get("citations"):
                raise AtifProjectionError(
                    f"step {step.step_id} Anthropic text block {block_index} contains non-empty citations"
                )
            raw_message_parts.append(text)
        elif block.get("type") == "tool_use":
            _reject_unknown_provider_fields(
                block,
                {"type", "id", "name", "input", "caller"},
                f"step {step.step_id} Anthropic tool_use block {block_index}",
            )
            caller = block.get("caller")
            if caller is not None:
                if not isinstance(caller, Mapping) or caller.get("type") != "direct":
                    raise AtifProjectionError(
                        f"step {step.step_id} Anthropic tool_use block {block_index} "
                        "has an unsupported server-tool caller"
                    )
                _reject_unknown_provider_fields(
                    caller,
                    {"type"},
                    f"step {step.step_id} Anthropic tool_use block {block_index} caller",
                )
            raw_tool_calls.append(
                _raw_tool_call(
                    call_id=block.get("id"),
                    name=block.get("name"),
                    arguments=block.get("input"),
                    field_name=f"step {step.step_id} Anthropic tool_use block {block_index}",
                )
            )

    has_tool_use = bool(raw_tool_calls)
    if (has_tool_use and stop_reason != "tool_use") or (not has_tool_use and stop_reason == "tool_use"):
        raise AtifProjectionError(
            f"step {step.step_id} Anthropic stop_reason {stop_reason!r} is inconsistent with tool_use content"
        )

    _validate_raw_coverage(step, raw_message_parts=raw_message_parts, raw_tool_calls=raw_tool_calls)


def _reject_dropped_text_metadata(part: Mapping[str, Any], field_name: str) -> None:
    for metadata_field in ("annotations", "logprobs"):
        value = part.get(metadata_field)
        if value:
            raise AtifProjectionError(
                f"{field_name} contains non-empty {metadata_field} that the ATIF projection cannot preserve"
            )


def _validate_projected_output_ids(output: list[Any]) -> None:
    seen_ids: set[str] = set()
    for index, item in enumerate(output):
        item_id = getattr(item, "id", None)
        if not isinstance(item_id, str) or not item_id.strip():
            raise AtifProjectionError(f"projected response item {index} has a blank or invalid id")
        if item_id in seen_ids:
            raise AtifProjectionError(f"projected response repeats item id {item_id!r}")
        seen_ids.add(item_id)


def _raw_tool_call(
    *,
    call_id: Any,
    name: Any,
    arguments: Any,
    field_name: str,
) -> tuple[str, str, dict[str, Any]]:
    if not isinstance(call_id, str) or not call_id.strip():
        raise AtifProjectionError(f"{field_name} has no non-blank invocation ID")
    if not isinstance(name, str) or not name.strip():
        raise AtifProjectionError(f"{field_name} has no non-blank function name")
    if isinstance(arguments, str):
        try:
            arguments = strict_json_loads(arguments)
        except ValueError as exc:
            raise AtifProjectionError(f"{field_name} contains invalid JSON arguments: {exc}") from exc
    if not isinstance(arguments, dict):
        raise AtifProjectionError(f"{field_name} arguments are not a JSON object")
    return call_id, name, arguments


def _validate_raw_coverage(
    step: Any,
    *,
    raw_message_parts: list[str],
    raw_tool_calls: list[tuple[str, str, dict[str, Any]]],
) -> None:
    canonical_message_parts = _text_parts(step.message, f"step {step.step_id} message")
    if raw_message_parts != canonical_message_parts:
        raise AtifProjectionError(
            f"step {step.step_id} raw provider message does not match the canonical ATIF message"
        )

    canonical_tool_calls = [(call.tool_call_id, call.function_name, call.arguments) for call in step.tool_calls or []]
    if not _json_tool_calls_equal(raw_tool_calls, canonical_tool_calls):
        raise AtifProjectionError(
            f"step {step.step_id} raw provider tool calls do not match the canonical ATIF tool calls"
        )


def _json_tool_calls_equal(
    raw_calls: list[tuple[str, str, dict[str, Any]]],
    canonical_calls: list[tuple[str, str, dict[str, Any]]],
) -> bool:
    if len(raw_calls) != len(canonical_calls):
        return False
    return all(
        raw_id == canonical_id and raw_name == canonical_name and json_values_equal(raw_arguments, canonical_arguments)
        for (raw_id, raw_name, raw_arguments), (canonical_id, canonical_name, canonical_arguments) in zip(
            raw_calls, canonical_calls, strict=True
        )
    )


def _reject_nonfinite_json_numbers(value: Any, field_name: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise AtifProjectionError(f"{field_name} contains non-finite JSON number {value!r}")
    if isinstance(value, Mapping):
        for key, nested_value in value.items():
            _reject_nonfinite_json_numbers(nested_value, f"{field_name}.{key}")
    elif isinstance(value, list):
        for index, nested_value in enumerate(value):
            _reject_nonfinite_json_numbers(nested_value, f"{field_name}[{index}]")


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
    agent_steps = [step for step in trajectory.steps if step.source == "agent"]
    resolved_model_names = [step.model_name or trajectory.agent.model_name for step in agent_steps]
    known_model_names = {model_name for model_name in resolved_model_names if model_name is not None}
    if len(known_model_names) > 1:
        raise AtifProjectionError(
            "ATIF agent steps resolve to multiple model names and cannot be represented by one verifier response"
        )
    if known_model_names and any(model_name is None for model_name in resolved_model_names):
        raise AtifProjectionError(
            "ATIF agent steps mix known and unknown model identity and cannot be represented by one verifier response"
        )
    return next(iter(known_model_names)) if known_model_names else "unknown"


def _usage(trajectory: AtifTrajectoryV1_7) -> NeMoGymResponseUsage | None:
    model_steps = [step for step in trajectory.steps if step.source == "agent" and step.llm_call_count != 0]
    for step in model_steps:
        if (
            step.metrics is not None
            and step.metrics.prompt_tokens is not None
            and step.metrics.cached_tokens is not None
            and step.metrics.cached_tokens > step.metrics.prompt_tokens
        ):
            raise AtifProjectionError(
                f"step {step.step_id} cached_tokens exceeds prompt_tokens; cached tokens must be a subset"
            )
    has_step_usage = any(_step_has_usage_counts(step) for step in model_steps)
    has_complete_step_totals = bool(model_steps) and all(
        step.metrics is not None
        and step.metrics.prompt_tokens is not None
        and step.metrics.completion_tokens is not None
        for step in model_steps
    )
    reasoning_tokens, reported_total_tokens = _nemo_gym_usage_details(model_steps)
    metrics = trajectory.final_metrics
    if metrics is None:
        if has_step_usage and has_complete_step_totals:
            raise AtifProjectionError("ATIF contains per-step token metrics without final_metrics")
        return None

    if (
        metrics.total_prompt_tokens is not None
        and metrics.total_cached_tokens is not None
        and metrics.total_cached_tokens > metrics.total_prompt_tokens
    ):
        raise AtifProjectionError(
            "ATIF final total_cached_tokens exceeds total_prompt_tokens; cached tokens must be a subset"
        )

    if metrics.total_prompt_tokens is None and metrics.total_completion_tokens is None:
        if has_step_usage and has_complete_step_totals:
            raise AtifProjectionError("ATIF final_metrics omits token totals despite per-step token metrics")
        return None
    if metrics.total_prompt_tokens is None or metrics.total_completion_tokens is None:
        if has_step_usage and not has_complete_step_totals:
            return None
        raise AtifProjectionError("ATIF final_metrics must provide both prompt and completion token totals")

    step_metrics = [step.metrics for step in model_steps]
    if has_step_usage:
        if not has_complete_step_totals:
            return None
        typed_step_metrics = [step_metric for step_metric in step_metrics if step_metric is not None]
        prompt_total = sum(step_metric.prompt_tokens or 0 for step_metric in typed_step_metrics)
        completion_total = sum(step_metric.completion_tokens or 0 for step_metric in typed_step_metrics)
        if prompt_total != metrics.total_prompt_tokens or completion_total != metrics.total_completion_tokens:
            raise AtifProjectionError("ATIF final token totals do not match the complete per-model-step metrics")

        cached_values = [step_metric.cached_tokens for step_metric in typed_step_metrics]
        cached_tokens = (
            None
            if any(value is None for value in cached_values)
            else sum(value for value in cached_values if value is not None)
        )
        if (
            cached_tokens is not None
            and metrics.total_cached_tokens is not None
            and cached_tokens != metrics.total_cached_tokens
        ):
            raise AtifProjectionError("ATIF final cached-token total does not match the per-model-step metrics")
    else:
        # Relay may emit only the final aggregate. With no partial step evidence,
        # the aggregate is the sole authoritative usage measurement.
        prompt_total = metrics.total_prompt_tokens
        completion_total = metrics.total_completion_tokens
        cached_tokens = metrics.total_cached_tokens

    computed_total_tokens = prompt_total + completion_total
    if reported_total_tokens is not None and (
        reported_total_tokens < prompt_total or reported_total_tokens < completion_total
    ):
        raise AtifProjectionError("Gym ATIF total_tokens metadata is smaller than a component token count")
    if reasoning_tokens is not None and reasoning_tokens > completion_total:
        raise AtifProjectionError("Gym ATIF reasoning_tokens metadata exceeds total completion tokens")
    return NeMoGymResponseUsage(
        input_tokens=prompt_total,
        input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=cached_tokens),
        output_tokens=completion_total,
        output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=reasoning_tokens),
        total_tokens=reported_total_tokens if reported_total_tokens is not None else computed_total_tokens,
    )


def _step_has_usage_counts(step: Any) -> bool:
    metrics = step.metrics
    if metrics is None:
        return False
    if any(value is not None for value in (metrics.prompt_tokens, metrics.completion_tokens, metrics.cached_tokens)):
        return True
    if not isinstance(metrics.extra, Mapping):
        return False
    if "total_tokens" in metrics.extra or "output_tokens_details" in metrics.extra:
        return True
    nemo_gym = metrics.extra.get("nemo_gym")
    return isinstance(nemo_gym, Mapping) and any(
        field_name in nemo_gym for field_name in ("reasoning_tokens", "total_tokens")
    )


def _nemo_gym_usage_details(agent_steps: list[Any]) -> tuple[int | None, int | None]:
    """Aggregate optional Gym token details without turning unknown values into zero."""

    reasoning_values: list[int | None] = []
    total_values: list[int | None] = []
    saw_reasoning = False
    saw_total = False
    for step in agent_steps:
        metrics = step.metrics
        metric_extra = metrics.extra if metrics is not None and isinstance(metrics.extra, Mapping) else {}
        namespace = metric_extra.get("nemo_gym")
        if "nemo_gym" in metric_extra and namespace is not None and not isinstance(namespace, Mapping):
            raise AtifProjectionError(f"step {step.step_id} metrics.extra.nemo_gym is not an object")
        namespace = namespace if isinstance(namespace, Mapping) else {}
        output_details = metric_extra.get("output_tokens_details")
        if (
            "output_tokens_details" in metric_extra
            and output_details is not None
            and not isinstance(output_details, Mapping)
        ):
            raise AtifProjectionError(f"step {step.step_id} metrics.extra.output_tokens_details is not an object")
        output_details = output_details if isinstance(output_details, Mapping) else {}

        gym_reasoning = _optional_nonnegative_count(namespace, "reasoning_tokens", step.step_id)
        relay_reasoning = _optional_nonnegative_count(output_details, "reasoning_tokens", step.step_id)
        reasoning = _coalesce_metric_aliases(
            gym_reasoning,
            relay_reasoning,
            field_name="reasoning_tokens",
            step_id=step.step_id,
        )
        gym_total = _optional_nonnegative_count(namespace, "total_tokens", step.step_id)
        relay_total = _optional_nonnegative_count(metric_extra, "total_tokens", step.step_id)
        reported_total = _coalesce_metric_aliases(
            gym_total,
            relay_total,
            field_name="total_tokens",
            step_id=step.step_id,
        )
        if metrics is not None:
            if reported_total is not None and any(
                value is not None and value > reported_total
                for value in (metrics.prompt_tokens, metrics.completion_tokens)
            ):
                raise AtifProjectionError(
                    f"step {step.step_id} ATIF total_tokens metadata is smaller than a component token count"
                )
            if (
                reasoning is not None
                and metrics.completion_tokens is not None
                and reasoning > metrics.completion_tokens
            ):
                raise AtifProjectionError(
                    f"step {step.step_id} ATIF reasoning_tokens metadata exceeds completion_tokens"
                )
        reasoning_values.append(reasoning)
        total_values.append(reported_total)
        saw_reasoning = saw_reasoning or "reasoning_tokens" in namespace or "reasoning_tokens" in output_details
        saw_total = saw_total or "total_tokens" in namespace or "total_tokens" in metric_extra

    reasoning_total = sum(value for value in reasoning_values if value is not None)
    if not saw_reasoning or any(value is None for value in reasoning_values):
        reasoning_total = None
    reported_total = sum(value for value in total_values if value is not None)
    if not saw_total or any(value is None for value in total_values):
        reported_total = None
    return reasoning_total, reported_total


def _optional_nonnegative_count(namespace: Mapping[str, Any], field_name: str, step_id: int) -> int | None:
    if field_name not in namespace:
        return None
    value = namespace[field_name]
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AtifProjectionError(f"step {step_id} ATIF {field_name} metadata is not a non-negative integer")
    return value


def _coalesce_metric_aliases(
    first: int | None,
    second: int | None,
    *,
    field_name: str,
    step_id: int,
) -> int | None:
    if first is not None and second is not None and first != second:
        raise AtifProjectionError(f"step {step_id} contains conflicting {field_name} metadata")
    return first if first is not None else second
