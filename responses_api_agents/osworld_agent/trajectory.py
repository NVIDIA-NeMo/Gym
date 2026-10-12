# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-independent OSWorld trajectory contracts.

Every OSWorld run has an environment trajectory, regardless of whether its
model endpoint exposes tokenizer-level evidence.  This module records that
semantic trajectory first and describes exact model-call evidence as an
optional capability.  Training is a consumer decision; Gym never changes the
agent's prompting policy merely because a caller may later train on the run.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


_CALLER_IDENTITY_FIELDS = (
    "context_compaction_rollout_id",
    "context_compaction_group_id",
    "context_compaction_task_id",
    "context_compaction_rollout_index",
    "context_compaction_attempt_index",
)
_TRAJECTORY_IDENTITY_FIELDS = (
    "rollout_id",
    "group_id",
    "task_id",
    "rollout_index",
    "attempt_index",
)
_EVENT_IDENTITY_FIELDS = (
    "sampling_event_id",
    "source_group_id",
)
_MEDIA_PART_TYPES = frozenset({"image_url", "input_image", "image"})


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_digest(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-compatible evidence."""

    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def stable_id(prefix: str, *parts: Any) -> str:
    """Return a deterministic, bounded identifier."""

    return f"{prefix}-{canonical_digest(parts)[:24]}"


def _nonnegative_int(value: Any, *, fallback: int, field: str) -> int:
    if value is None:
        return fallback
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def resolve_trajectory_identity(
    *,
    request_extra: Mapping[str, Any],
    verifier_metadata: Mapping[str, Any],
    model_name: str,
) -> dict[str, Any]:
    """Resolve the one logical identity shared by logs and trace evidence.

    Training callers own the identity and stamp it before Gym dispatch.
    Standalone benchmark callers may omit it, in which case Gym derives a
    stable but training-ineligible identity.  Resolve this before launching the
    OSWorld child so every service boundary can log the same values later used
    by :func:`build_trajectory_envelope`.
    """

    metadata_task = verifier_metadata.get("osworld_task")
    metadata_task_id = verifier_metadata.get("task_id")
    if not metadata_task_id and isinstance(metadata_task, Mapping):
        metadata_task_id = metadata_task.get("id")
    if metadata_task_id is not None and (not isinstance(metadata_task_id, str) or not metadata_task_id):
        raise ValueError("verifier_metadata task_id must be a non-empty string")

    generic_identity = request_extra.get("trajectory_identity")
    caller_values = {field: request_extra.get(field) for field in _CALLER_IDENTITY_FIELDS}
    legacy_identity_present = request_extra.get("context_compaction_contract_version") is not None or any(
        value is not None for value in caller_values.values()
    )
    if generic_identity is not None and legacy_identity_present:
        raise ValueError(
            "OSWorld request must use trajectory_identity or legacy context_compaction identity fields, not both"
        )
    if generic_identity is not None:
        if not isinstance(generic_identity, Mapping):
            raise TypeError("trajectory_identity must be a mapping")
        if generic_identity.get("schema_version") != 1:
            raise ValueError("Unsupported trajectory_identity schema_version")
        missing = [field for field in _TRAJECTORY_IDENTITY_FIELDS if generic_identity.get(field) is None]
        if missing:
            raise ValueError("Caller-stamped trajectory_identity is incomplete: " + ", ".join(missing))
        rollout_id = generic_identity["rollout_id"]
        group_id = generic_identity["group_id"]
        task_id = generic_identity["task_id"]
        rollout_index = _nonnegative_int(
            generic_identity["rollout_index"],
            fallback=0,
            field="trajectory_identity.rollout_index",
        )
        attempt_index = _nonnegative_int(
            generic_identity["attempt_index"],
            fallback=0,
            field="trajectory_identity.attempt_index",
        )
        for field, value in (
            ("trajectory_identity.rollout_id", rollout_id),
            ("trajectory_identity.group_id", group_id),
            ("trajectory_identity.task_id", task_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        if metadata_task_id is not None and task_id != metadata_task_id:
            raise ValueError("trajectory_identity.task_id must match verifier_metadata task_id")
        event_values = {field: generic_identity.get(field) for field in _EVENT_IDENTITY_FIELDS}
        if (event_values["sampling_event_id"] is None) != (event_values["source_group_id"] is None):
            raise ValueError("trajectory_identity sampling_event_id and source_group_id must be present together")
        for field, value in event_values.items():
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"trajectory_identity.{field} must be a non-empty string")
        identity_source = "caller"
    elif legacy_identity_present:
        if request_extra.get("context_compaction_contract_version") != 2:
            raise ValueError(
                "Caller-stamped OSWorld trajectory identity requires context_compaction_contract_version=2"
            )
        missing = [field for field, value in caller_values.items() if value is None]
        if missing:
            raise ValueError("Caller-stamped OSWorld trajectory identity is incomplete: " + ", ".join(missing))
        rollout_id = caller_values["context_compaction_rollout_id"]
        group_id = caller_values["context_compaction_group_id"]
        task_id = caller_values["context_compaction_task_id"]
        rollout_index = _nonnegative_int(
            caller_values["context_compaction_rollout_index"],
            fallback=0,
            field="context_compaction_rollout_index",
        )
        attempt_index = _nonnegative_int(
            caller_values["context_compaction_attempt_index"],
            fallback=0,
            field="context_compaction_attempt_index",
        )
        for field, value in (
            ("context_compaction_rollout_id", rollout_id),
            ("context_compaction_group_id", group_id),
            ("context_compaction_task_id", task_id),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        if metadata_task_id is not None and task_id != metadata_task_id:
            raise ValueError("context_compaction_task_id must match verifier_metadata task_id")
        identity_source = "caller"
    else:
        task_id = metadata_task_id
        if not isinstance(task_id, str) or not task_id:
            task_id = "unknown-task"
        rollout_index = _nonnegative_int(
            request_extra.get("_ng_rollout_index"),
            fallback=0,
            field="_ng_rollout_index",
        )
        attempt_index = _nonnegative_int(
            request_extra.get("_ng_attempt_index"),
            fallback=0,
            field="_ng_attempt_index",
        )
        group_id = stable_id("trajectory-group", task_id)
        rollout_id = stable_id(
            "rollout",
            task_id,
            group_id,
            rollout_index,
            attempt_index,
            model_name,
        )
        identity_source = "derived"

    identity = {
        "rollout_id": rollout_id,
        "group_id": group_id,
        "task_id": task_id,
        "rollout_index": rollout_index,
        "attempt_index": attempt_index,
        "identity_source": identity_source,
    }
    if generic_identity is not None:
        identity.update(
            {
                field: generic_identity[field]
                for field in _EVENT_IDENTITY_FIELDS
                if generic_identity.get(field) is not None
            }
        )
    return identity


def _image_source(part: Mapping[str, Any]) -> dict[str, Any] | None:
    """Normalize an OpenAI-style image part without changing its bytes."""

    if part.get("type") not in _MEDIA_PART_TYPES:
        return None
    raw_source = part.get("image_url") or part.get("image") or part.get("url")
    detail = part.get("detail") or "high"
    if isinstance(raw_source, Mapping):
        detail = raw_source.get("detail") or detail
        raw_source = raw_source.get("url")
    if not isinstance(raw_source, str) or not raw_source:
        raise ValueError("OSWorld trajectory encountered an image without a source URL")
    return {
        "type": "input_image",
        "image_url": raw_source,
        "detail": str(detail),
    }


def register_media_asset(
    source_part: Mapping[str, Any],
    *,
    media_assets: dict[str, dict[str, Any]],
) -> str:
    """Store one immutable media asset in the trajectory-level arena."""

    content_digest = canonical_digest(source_part)
    media_id = f"media-{content_digest[:24]}"
    asset = {
        "media_id": media_id,
        "content_digest": content_digest,
        "source_part": dict(source_part),
        "original_dimensions": None,
        "color_mode": None,
        "source_format": None,
    }
    previous = media_assets.setdefault(media_id, asset)
    if previous.get("content_digest") != content_digest:
        raise RuntimeError(f"OSWorld trajectory media ID collision for {media_id}")
    return media_id


def project_prompt_messages(
    prompt_messages: Sequence[Mapping[str, Any]],
    *,
    media_assets: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Factor raw media out of a materialized prompt into stable references."""

    projected_messages: list[dict[str, Any]] = []
    media_ids: list[str] = []
    for message_index, message in enumerate(prompt_messages):
        if not isinstance(message, Mapping):
            raise ValueError(f"OSWorld prompt message {message_index} must be a mapping")
        projected_message = dict(message)
        content = message.get("content")
        if isinstance(content, str) or content is None:
            projected_messages.append(projected_message)
            continue
        if not isinstance(content, (list, tuple)):
            raise ValueError(f"OSWorld prompt message {message_index} has invalid content")
        projected_content: list[Any] = []
        for part in content:
            if not isinstance(part, Mapping):
                projected_content.append(part)
                continue
            source = _image_source(part)
            if source is None:
                projected_content.append(dict(part))
                continue
            media_id = register_media_asset(source, media_assets=media_assets)
            media_ids.append(media_id)
            projected_content.append(
                {
                    "type": "input_image",
                    "media_id": media_id,
                    "detail": source["detail"],
                }
            )
        projected_message["content"] = projected_content
        projected_messages.append(projected_message)
    return projected_messages, media_ids


def _exact_generation_arrays(
    response: Mapping[str, Any],
) -> tuple[dict[str, list[Any]] | None, list[str]]:
    reasons: list[str] = []
    prompt_ids = response.get("prompt_token_ids")
    generation_ids = response.get("generation_token_ids")
    generation_logprobs = response.get("generation_log_probs")
    if not isinstance(prompt_ids, (list, tuple)) or not prompt_ids:
        reasons.append("exact_prompt_token_ids_unavailable")
    elif any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in prompt_ids):
        reasons.append("exact_prompt_token_ids_invalid")
    if not isinstance(generation_ids, (list, tuple)) or not generation_ids:
        reasons.append("exact_sampled_token_ids_unavailable")
    elif any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in generation_ids):
        reasons.append("exact_sampled_token_ids_invalid")
    if not isinstance(generation_logprobs, (list, tuple)):
        reasons.append("exact_sampled_logprobs_unavailable")
    elif any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
        for value in generation_logprobs
    ):
        reasons.append("exact_sampled_logprobs_invalid")
    if not reasons and len(generation_ids) != len(generation_logprobs):
        reasons.append("sampled_token_logprob_length_mismatch")
    if reasons:
        return None, reasons
    return {
        "prompt_token_ids": [int(value) for value in prompt_ids],
        "sampled_token_ids": [int(value) for value in generation_ids],
        "sampled_logprobs": [float(value) for value in generation_logprobs],
    }, []


def collect_model_calls(
    steps: Sequence[Mapping[str, Any]],
    *,
    trajectory_id: str,
    runtime_eligible: bool,
    media_assets: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Collect model-call records without assuming one call per env step."""

    model_calls: list[dict[str, Any]] = []
    incomplete_reasons: set[str] = set()
    for step_position, step in enumerate(steps):
        info = step.get("info")
        agent_info = info.get("agent") if isinstance(info, Mapping) else None
        raw_calls = agent_info.get("model_calls") if isinstance(agent_info, Mapping) else None
        if not isinstance(raw_calls, list) or not raw_calls:
            incomplete_reasons.add("model_call_evidence_unavailable")
            continue
        for local_call_index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, Mapping):
                incomplete_reasons.add("model_call_evidence_invalid")
                continue
            parse_attempt = raw_call.get("parse_attempt")
            if isinstance(parse_attempt, bool) or not isinstance(parse_attempt, int) or parse_attempt <= 0:
                parse_attempt = local_call_index + 1
                incomplete_reasons.add("model_call_parse_attempt_invalid")
            model_call_id = stable_id(
                "model-call",
                trajectory_id,
                step_position,
                parse_attempt,
            )
            prompt_messages = raw_call.get("prompt_messages")
            response = raw_call.get("response")
            call_reasons: list[str] = []
            if not isinstance(prompt_messages, list) or not prompt_messages:
                call_reasons.append("exact_prompt_messages_unavailable")
                projected_prompt: list[dict[str, Any]] = []
                media_ids: list[str] = []
            else:
                projected_prompt, media_ids = project_prompt_messages(
                    prompt_messages,
                    media_assets=media_assets,
                )
            if not isinstance(response, Mapping):
                call_reasons.append("structured_generation_response_unavailable")
                arrays = None
            else:
                arrays, array_reasons = _exact_generation_arrays(response)
                call_reasons.extend(array_reasons)
            partial_generation_evidence = {
                "prompt_token_ids": (
                    list(response["prompt_token_ids"])
                    if isinstance(response, Mapping) and isinstance(response.get("prompt_token_ids"), (list, tuple))
                    else None
                ),
                "generation_token_ids": (
                    list(response["generation_token_ids"])
                    if isinstance(response, Mapping)
                    and isinstance(response.get("generation_token_ids"), (list, tuple))
                    else None
                ),
                "generation_log_probs": (
                    list(response["generation_log_probs"])
                    if isinstance(response, Mapping)
                    and isinstance(response.get("generation_log_probs"), (list, tuple))
                    else None
                ),
                "finish_reason": (response.get("finish_reason") if isinstance(response, Mapping) else None),
            }
            incomplete_reasons.update(call_reasons)
            reward = step.get("reward", 0.0)
            if isinstance(reward, bool) or not isinstance(reward, (int, float)):
                raise TypeError(f"OSWorld step {step_position} reward must be numeric")
            reward = float(reward)
            if not math.isfinite(reward):
                raise ValueError(f"OSWorld step {step_position} reward must be finite")
            parsed_actions = raw_call.get("parsed_actions")
            if not isinstance(parsed_actions, list):
                parsed_actions = []
            model_calls.append(
                {
                    "model_call_id": model_call_id,
                    "turn_id": len(model_calls) + 1,
                    "environment_step": step.get("step", step_position),
                    "step_position": step_position,
                    "parse_attempt": parse_attempt,
                    "prompt_messages": prompt_messages,
                    "projected_prompt_messages": projected_prompt,
                    "media_ids": media_ids,
                    "response": dict(response) if isinstance(response, Mapping) else None,
                    "exact_generation_arrays": arrays,
                    "partial_generation_evidence": partial_generation_evidence,
                    "exact_evidence": not call_reasons,
                    "accepted": raw_call.get("accepted") is True,
                    "parse_error": raw_call.get("parse_error"),
                    "parsed_actions": list(parsed_actions),
                    "reward": reward,
                    "done": bool(step.get("done", False)),
                    # ``eligible`` is the legacy wire carrier. It mirrors only
                    # Gym runtime admission; exact-trace/loss admission is a
                    # separate trainer decision.
                    "eligible": runtime_eligible,
                    "runtime_eligible": runtime_eligible,
                }
            )
    return model_calls, sorted(incomplete_reasons)


def build_trajectory_envelope(
    *,
    steps: Sequence[Mapping[str, Any]],
    request_extra: Mapping[str, Any],
    verifier_metadata: Mapping[str, Any],
    model_name: str,
    runtime_eligible: bool,
    runtime_admission_reason: str,
    runtime_admission_policy_id: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build the universal semantic trajectory and evidence capability report."""

    if not isinstance(runtime_eligible, bool):
        raise TypeError("runtime_eligible must be boolean")
    for field, value in (
        ("runtime_admission_reason", runtime_admission_reason),
        ("runtime_admission_policy_id", runtime_admission_policy_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} must be a non-empty string")

    identity = resolve_trajectory_identity(
        request_extra=request_extra,
        verifier_metadata=verifier_metadata,
        model_name=model_name,
    )
    trajectory_id = stable_id("trajectory", identity, model_name)
    media_assets: dict[str, dict[str, Any]] = {}
    model_calls, evidence_reasons = collect_model_calls(
        steps,
        trajectory_id=trajectory_id,
        runtime_eligible=runtime_eligible,
        media_assets=media_assets,
    )
    calls_by_step: dict[int, list[dict[str, Any]]] = {}
    for model_call in model_calls:
        calls_by_step.setdefault(model_call["step_position"], []).append(model_call)

    transitions: list[dict[str, Any]] = []
    for step_position, step in enumerate(steps):
        reward = step.get("reward", 0.0)
        if isinstance(reward, bool) or not isinstance(reward, (int, float)):
            raise TypeError(f"OSWorld step {step_position} reward must be numeric")
        reward = float(reward)
        if not math.isfinite(reward):
            raise ValueError(f"OSWorld step {step_position} reward must be finite")
        actions = step.get("actions") or []
        if not isinstance(actions, list):
            raise TypeError(f"OSWorld step {step_position} actions must be a list")
        step_calls = calls_by_step.get(step_position, [])
        transition_id = stable_id("transition", trajectory_id, step_position)
        transitions.append(
            {
                "transition_id": transition_id,
                "turn_id": step_position + 1,
                "state": {
                    "observation": dict(step.get("state") or {}),
                    "model_call_ids": [call["model_call_id"] for call in step_calls],
                },
                "action": {
                    "raw_completion": str(step.get("model_text") or ""),
                    "parsed_actions": list(actions),
                    "accepted_model_call_id": next(
                        (call["model_call_id"] for call in reversed(step_calls) if call["accepted"]),
                        None,
                    ),
                },
                "reward": reward,
                "next_state": {
                    "observation": dict(step.get("next_state") or {}),
                },
                "done": bool(step.get("done", False)),
                "eligible": runtime_eligible,
                "runtime_eligible": runtime_eligible,
            }
        )

    exact_model_call_evidence = bool(model_calls) and not evidence_reasons
    exact_trace_reasons = list(evidence_reasons)
    if identity["identity_source"] != "caller":
        exact_trace_reasons.append("caller_owned_rollout_identity_unavailable")
    eligibility_reasons = list(exact_trace_reasons)
    if not runtime_eligible:
        eligibility_reasons.append("rollout_sample_masked")
    status = (
        "requires_runtime_admission"
        if exact_model_call_evidence and runtime_eligible and identity["identity_source"] == "caller"
        else "ineligible"
    )
    contract_without_id = {
        "schema_version": 2,
        "mode": "osworld_semantic_trajectory",
        **identity,
        "trajectory_id": trajectory_id,
        "model_name": model_name,
        "transition_count": len(transitions),
        "model_call_count": len(model_calls),
        "capabilities": {
            "semantic_trajectory": True,
            "exact_model_call_evidence": exact_model_call_evidence,
            "arbitrary_prompt_rewrites": exact_model_call_evidence,
            "trainable_token_reconstruction": exact_model_call_evidence,
        },
        "runtime_admission": {
            "eligible": runtime_eligible,
            "reason": runtime_admission_reason,
            "policy_id": runtime_admission_policy_id,
        },
        "exact_trace_admission": {
            "decision_owner": "training_consumer",
            "evidence_complete": exact_model_call_evidence,
            "caller_identity_available": identity["identity_source"] == "caller",
            "incomplete_reasons": sorted(set(exact_trace_reasons)),
        },
        # Compatibility view for existing exact-trace consumers. New code
        # should combine runtime_admission with exact_trace_admission instead
        # of treating this field as a Gym-owned algorithm decision.
        "training_eligibility": {
            "status": status,
            "incomplete_reasons": sorted(set(eligibility_reasons)),
        },
    }
    trajectory_contract = {
        **contract_without_id,
        "trajectory_contract_id": stable_id(
            "trajectory-contract",
            contract_without_id,
        ),
    }
    trajectory_model_calls = [
        {
            "model_call_id": call["model_call_id"],
            "turn_id": call["turn_id"],
            "environment_step": call["environment_step"],
            "parse_attempt": call["parse_attempt"],
            "state": {
                "prompt_messages": call["projected_prompt_messages"],
                "media_ids": call["media_ids"],
            },
            "action": {
                "raw_completion": str((call["response"] or {}).get("raw_content") or ""),
                "parsed_actions": call["parsed_actions"],
            },
            "reward": call["reward"],
            "done": call["done"],
            "eligible": call["eligible"],
            "runtime_eligible": call["runtime_eligible"],
            "accepted": call["accepted"],
            "parse_error": call["parse_error"],
            "generation_evidence": {
                **call["partial_generation_evidence"],
                "exact": call["exact_evidence"],
            },
        }
        for call in model_calls
    ]
    # Kept as a compact compatibility view for older consumers. New consumers
    # should use trajectory_model_calls, which contains the actual prompt,
    # action, reward, and optional generation evidence for each policy call.
    model_call_summaries = [
        {
            "model_call_id": call["model_call_id"],
            "turn_id": call["turn_id"],
            "environment_step": call["environment_step"],
            "parse_attempt": call["parse_attempt"],
            "accepted": call["accepted"],
            "parse_error": call["parse_error"],
            "exact_evidence": call["exact_evidence"],
        }
        for call in model_calls
    ]
    return (
        {
            "trajectory_contract": trajectory_contract,
            "trajectory_transitions": transitions,
            "trajectory_model_calls": trajectory_model_calls,
            "model_call_summaries": model_call_summaries,
            "media_assets": media_assets,
        },
        model_calls,
    )
