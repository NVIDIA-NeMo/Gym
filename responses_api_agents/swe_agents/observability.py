# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize completion artifacts emitted by the SWE agent harnesses."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from nemo_gym.config_types import ModelServerRef
from nemo_gym.responses_converter import VLLMConverter
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    SandboxObservation,
    model_visible_tool_calls,
)


OBSERVATIONS_FILENAME = "agent_observations.json"


def _value(metrics: Any, name: str) -> Any:
    return metrics.get(name) if isinstance(metrics, dict) else getattr(metrics, name, None)


def sandbox_observations_from_metrics(metrics: Any, sandbox_id: str | None = None) -> list[SandboxObservation]:
    """Map only resource and outcome facts reported by the SWE runner."""
    observations: list[SandboxObservation] = []
    phases = (
        (
            "agent",
            _value(metrics, "openhands_run_time"),
            _value(metrics, "agent_peak_rss_mb"),
            _value(metrics, "agent_timed_out"),
            _value(metrics, "oom_killed"),
        ),
        (
            "verifier",
            _value(metrics, "final_eval_time"),
            _value(metrics, "eval_peak_rss_mb"),
            _value(metrics, "eval_timed_out"),
            _value(metrics, "eval_oom_killed"),
        ),
    )
    for role, wall_time_s, peak_memory_mib, timed_out, oom_killed in phases:
        if wall_time_s is None and peak_memory_mib is None and not timed_out and not oom_killed:
            continue
        outcome = "oom" if oom_killed else "timeout" if timed_out else "unknown"
        observations.append(
            SandboxObservation(
                role=role,
                provider="apptainer",
                sandbox_id=sandbox_id,
                outcome=outcome,
                wall_time_s=wall_time_s,
                peak_memory_mib=peak_memory_mib,
                resource_usage_source="proc_tree_watchdog" if peak_memory_mib is not None else None,
            )
        )
    return observations


def materialize_completion(data: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Any]]:
    """Return the cumulative chat history and tools stored in one artifact."""
    messages = [dict(message) for message in data.get("messages") or [] if isinstance(message, dict)]
    tools = list((data.get("kwargs") or {}).get("tools") or [])
    try:
        final_assistant_message = dict(data["response"]["choices"][0]["message"])
    except (KeyError, IndexError, TypeError):
        return messages, tools

    provider_fields = data.get("provider_specific_fields") or {}
    for key in (
        "prompt_token_ids",
        "generation_token_ids",
        "generation_log_probs",
        "routed_experts",
    ):
        if key in provider_fields:
            final_assistant_message[key] = provider_fields[key]

    if final_assistant_message.get("content") or final_assistant_message.get("tool_calls"):
        messages.append(final_assistant_message)
    return messages, tools


def _artifact_order(path: Path, data: dict[str, Any]) -> tuple[int, float, str]:
    turn = data.get("turn")
    if isinstance(turn, int) and not isinstance(turn, bool):
        return (2, float(turn), path.name)
    timestamp = data.get("timestamp")
    if isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        return (1, float(timestamp), path.name)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return (0, mtime, path.name)


def latest_completion_paths(completion_paths: Iterable[Path]) -> tuple[Path, ...]:
    """Select the cumulative artifact retained by the existing SWE path."""
    latest: dict[str, tuple[Path, float]] = {}
    for path in map(Path, completion_paths):
        data: dict[str, Any] = {}
        try:
            parsed = json.loads(path.read_text())
            if isinstance(parsed, dict):
                data = parsed
        except (OSError, UnicodeError, json.JSONDecodeError):
            pass
        session_id = data.get("session_id")
        key = session_id if isinstance(session_id, str) and session_id else "main"
        modified_at = path.stat().st_mtime
        if key not in latest or modified_at > latest[key][1]:
            latest[key] = (path, modified_at)
    return tuple(path for path, _ in latest.values())


def _gap(_source: str, code: str, *, invocation_id: str | None = None, detail: str | None = None) -> ObservationGap:
    return ObservationGap(code=code, invocation_id=invocation_id, detail=detail)


def build_swe_observations(
    completion_paths: Iterable[Path],
    *,
    framework: Literal["openhands", "opencode"],
    model_ref: ModelServerRef,
) -> AgentObservationBundle:
    """Build one compact bundle while retaining every exact model response ID.

    Completion files carry cumulative histories. Only the latest file per
    invocation supplies the conversation; all files contribute their response ID.
    """
    source = f"swe_{framework}"
    gaps: list[ObservationGap] = []
    artifacts: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)

    paths = tuple(sorted(Path(path) for path in completion_paths))
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, UnicodeError, json.JSONDecodeError):
            gaps.append(_gap(source, "agent_artifact_parse_failed", detail=path.name))
            continue
        if not isinstance(data, dict):
            gaps.append(_gap(source, "agent_artifact_parse_failed", detail=path.name))
            continue

        if framework == "openhands":
            invocation_id = "root"
        else:
            invocation_id = data.get("session_id")
            if not isinstance(invocation_id, str) or not invocation_id:
                gaps.append(_gap(source, "agent_session_id_missing", detail=path.name))
                continue
        artifacts[invocation_id].append((path, data))

    if framework == "openhands" and "root" not in artifacts:
        artifacts["root"] = []
    if not paths:
        gaps.append(_gap(source, "agent_artifact_unavailable"))

    converter = VLLMConverter(return_token_id_information=True)
    invocations: list[AgentInvocation] = []
    parents: dict[str, str | None] = {}
    response_owners: dict[str, str] = {}

    for invocation_id, entries in artifacts.items():
        entries.sort(key=lambda item: _artifact_order(*item))
        response_ids: list[str] = []
        seen_response_ids: set[str] = set()
        parent_values: set[str | None] = set()

        for path, data in entries:
            response = data.get("response")
            response_id = response.get("id") if isinstance(response, dict) else None
            if isinstance(response_id, str) and response_id and response_id != "unknown":
                owner = response_owners.setdefault(response_id, invocation_id)
                if owner != invocation_id:
                    gaps.append(
                        _gap(
                            source,
                            "model_response_owner_conflict",
                            invocation_id=invocation_id,
                            detail=response_id,
                        )
                    )
                elif response_id not in seen_response_ids:
                    response_ids.append(response_id)
                    seen_response_ids.add(response_id)
            else:
                gaps.append(_gap(source, "model_response_id_missing", invocation_id=invocation_id, detail=path.name))

            if framework == "opencode":
                parent = data.get("parent_session_id")
                if parent in (None, ""):
                    parent_values.add(None)
                elif isinstance(parent, str):
                    parent_values.add(parent)
                else:
                    gaps.append(
                        _gap(source, "parent_invocation_id_invalid", invocation_id=invocation_id, detail=path.name)
                    )

        parent: str | None = None
        if framework == "opencode":
            if len(parent_values) == 1:
                parent = next(iter(parent_values))
            elif len(parent_values) > 1:
                gaps.append(_gap(source, "parent_invocation_conflict", invocation_id=invocation_id))
        parents[invocation_id] = parent

        conversation = []
        if entries:
            _, latest = entries[-1]
            try:
                messages, _ = materialize_completion(latest)
                conversation = converter.chat_completions_messages_to_responses_items(messages)
            except Exception as exc:
                gaps.append(
                    _gap(
                        source,
                        "conversation_conversion_failed",
                        invocation_id=invocation_id,
                        detail=type(exc).__name__,
                    )
                )

        for tool_call_id, _, _ in model_visible_tool_calls(conversation):
            gaps.append(
                _gap(
                    source,
                    "tool_timing_unavailable",
                    invocation_id=invocation_id,
                    detail=tool_call_id,
                )
            )
            gaps.append(
                _gap(
                    source,
                    "tool_outcome_unavailable",
                    invocation_id=invocation_id,
                    detail=tool_call_id,
                )
            )

        invocations.append(
            AgentInvocation(
                invocation_id=invocation_id,
                parent_invocation_id=parent,
                model_calls=[
                    ModelCallRef(model_ref=model_ref, response_id=response_id) for response_id in response_ids
                ],
                conversation=conversation,
            )
        )

    invocation_ids = set(parents)
    for invocation_id, parent in parents.items():
        if parent is not None and parent not in invocation_ids:
            gaps.append(_gap(source, "parent_invocation_missing", invocation_id=invocation_id, detail=parent))
        if parent is not None:
            gaps.append(_gap(source, "subagent_spawn_tool_unavailable", invocation_id=invocation_id))

    if framework == "openhands":
        gaps.append(_gap(source, "subagent_hierarchy_unavailable", invocation_id="root"))
    gaps.append(_gap(source, "context_compaction_unavailable"))

    invocations.sort(key=lambda invocation: (invocation.parent_invocation_id is not None, invocation.invocation_id))
    return AgentObservationBundle(
        source=source,
        records=invocations,
        gaps=gaps,
    )
