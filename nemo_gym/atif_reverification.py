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

The adapter validates a deliberately bounded canonical ATIF v1.7 subset, joins
each trajectory to an explicit materialized Gym task, and constructs the
Responses-shaped payload used by Gym's existing stateless verifier path.

Provider-native payloads and producer-private status records carried in
``extra`` are producer evidence, not part of the ATIF interchange contract.
Exporters own their projection into the canonical fields consumed here.
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

from nemo_gym.atif_json import strict_json_loads
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
    """Verifier payload plus provenance that must survive result persistence.

    ``projection_status`` covers canonical ATIF fields only. It does not attest
    that a producer faithfully converted provider-native metadata in ``extra``.
    """

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

        if step.reasoning_content is not None:
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
        _reject_unsupported_training_metadata(step)
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
    """Hash semantic ATIF content without ignored producer metadata.

    Provider payloads and status records carried in ``extra`` do not affect the
    projected verifier response.  Exclude them from the content identity so
    they cannot perturb generated response or item IDs when ``trajectory_id``
    is absent.  Retain the few extension fields that this adapter consumes.
    """

    canonical = trajectory.model_dump(mode="json")
    canonical["trajectory_id"] = None
    agent = canonical["agent"]
    agent.pop("extra", None)

    root_extra = canonical.get("extra")
    if isinstance(root_extra, dict):
        nemo_gym = root_extra.get("nemo_gym")
        source = nemo_gym.get("source") if isinstance(nemo_gym, dict) else None
        canonical["extra"] = {"nemo_gym": {"source": source}} if isinstance(source, dict) else None

    final_metrics = canonical.get("final_metrics")
    if isinstance(final_metrics, dict):
        final_metrics.pop("extra", None)

    for step in canonical["steps"]:
        step.pop("extra", None)
        for tool_call in step.get("tool_calls") or []:
            tool_call.pop("extra", None)

        observation = step.get("observation")
        if isinstance(observation, dict):
            for result in observation.get("results") or []:
                result_extra = result.get("extra")
                result["extra"] = (
                    {"tool_result": result_extra["tool_result"]}
                    if isinstance(result_extra, dict) and "tool_result" in result_extra
                    else None
                )

        metrics = step.get("metrics")
        if not isinstance(metrics, dict):
            continue
        metrics_extra = metrics.get("extra")
        retained_metrics_extra: dict[str, Any] = {}
        if isinstance(metrics_extra, dict):
            if "total_tokens" in metrics_extra:
                retained_metrics_extra["total_tokens"] = metrics_extra["total_tokens"]
            output_details = metrics_extra.get("output_tokens_details")
            if isinstance(output_details, dict) and "reasoning_tokens" in output_details:
                retained_metrics_extra["output_tokens_details"] = {
                    "reasoning_tokens": output_details["reasoning_tokens"]
                }
        metrics["extra"] = retained_metrics_extra or None

    encoded = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _reject_unsupported_training_metadata(step: Any) -> None:
    metrics = step.metrics
    if metrics is None:
        return
    if any(value is not None for value in (metrics.prompt_token_ids, metrics.completion_token_ids, metrics.logprobs)):
        raise AtifProjectionError(
            f"step {step.step_id} contains training token metadata that the initial reverify adapter does not support"
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
    reasoning_tokens, reported_total_tokens = _relay_usage_details(model_steps)
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
        raise AtifProjectionError("ATIF total_tokens metadata is smaller than a component token count")
    if reasoning_tokens is not None and reasoning_tokens > completion_total:
        raise AtifProjectionError("ATIF reasoning_tokens metadata exceeds total completion tokens")
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
    return "total_tokens" in metrics.extra or "output_tokens_details" in metrics.extra


def _relay_usage_details(agent_steps: list[Any]) -> tuple[int | None, int | None]:
    """Aggregate optional Relay token details without turning unknown values into zero."""

    reasoning_values: list[int | None] = []
    total_values: list[int | None] = []
    saw_reasoning = False
    saw_total = False
    for step in agent_steps:
        metrics = step.metrics
        metric_extra = metrics.extra if metrics is not None and isinstance(metrics.extra, Mapping) else {}
        output_details = metric_extra.get("output_tokens_details")
        if (
            "output_tokens_details" in metric_extra
            and output_details is not None
            and not isinstance(output_details, Mapping)
        ):
            raise AtifProjectionError(f"step {step.step_id} metrics.extra.output_tokens_details is not an object")
        output_details = output_details if isinstance(output_details, Mapping) else {}

        reasoning = _optional_nonnegative_count(output_details, "reasoning_tokens", step.step_id)
        reported_total = _optional_nonnegative_count(metric_extra, "total_tokens", step.step_id)
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
        saw_reasoning = saw_reasoning or "reasoning_tokens" in output_details
        saw_total = saw_total or "total_tokens" in metric_extra

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
