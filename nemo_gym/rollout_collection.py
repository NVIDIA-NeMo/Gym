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
import asyncio
import glob as glob_module
import json
import logging
import os
import warnings
from asyncio import Future, Semaphore
from collections import Counter, defaultdict
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import timedelta
from difflib import get_close_matches
from itertools import repeat
from pathlib import Path
from time import time
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import orjson
from aiohttp import ClientError
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, Field, field_validator, model_validator
from tqdm.asyncio import tqdm

from nemo_gym import _resolve_under_cwd_or_install
from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.base_responses_api_model import (
    clear_model_call_captures_for_rollouts,
    merge_model_call_capture_into_record,
    model_call_capture_dirs_from_config,
    observability_enabled_from_config,
)
from nemo_gym.config_types import (
    BaseNeMoGymCLIConfig,
    BaseServerConfig,
    ConfigError,
    ConfigPathNotFoundError,
    UploadRolloutsConfigMixin,
)
from nemo_gym.exporters import export_metrics, export_rollouts, get_exporters
from nemo_gym.global_config import (
    AGENT_REF_KEY_NAME,
    AGENT_SERVER_TYPE_KEY_NAME,
    ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    RESPONSES_CREATE_PARAMS_KEY_NAME,
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    SKILLS_REF_KEY_NAME,
    TASK_INDEX_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
    allowed_agents_for,
    dataset_agent_pins,
    get_global_config_dict,
    pairing_override_enabled,
    resolve_dataset_agent,
)
from nemo_gym.path_utils import aggregate_metrics_path_for, failures_path_for
from nemo_gym.prompt import apply_prompt_to_row, load_prompt_config, validate_prompt_compatibility
from nemo_gym.rollout_correlation import maybe_rollout_id_from_run_body
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ModelCallRef,
    ObservationGap,
    ToolCallObservation,
    TrajectoryModelCall,
    TrajectoryRecord,
    TrajectoryTokenStats,
    TrajectoryToolCall,
    TrajectoryTurn,
)
from nemo_gym.telemetry._fallbacks import is_span_group_enabled, managed_span
from nemo_gym.telemetry.span_groups import GymSpanGroup


_failures_path_for = failures_path_for  # Backwards-compatible alias
from nemo_gym.server_utils import (
    ServerClient,
    get_response_json,
    raise_for_status,
)
from nemo_gym.server_utils import (
    setup_server_client as setup_server_client_utils,
)
from nemo_gym.skills import SkillsConfig, load_skill_directory
from nemo_gym.token_id_capture import (
    TokenCaptureStore,
    TokenIdCaptureConfig,
    clear_token_captures_for_rollouts,
    installed_token_source,
    token_id_capture_dirs_from_config,
)
from nemo_gym.token_id_capture.config import token_id_capture_enabled_for_agent
from nemo_gym.token_id_capture.delivery import (
    MASK_SAMPLE_KEY,
    capture_build_can_retire,
    finalize_rollout_token_capture,
    retire_rollout_token_capture,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Failure-routing sentinels (set by agent servers, read by the dispatcher).
#
# Background:
#   The historical contract was "every dispatched task produces one row in
#   the main rollouts jsonl, succeeded or failed." That contract broke
#   resume-after-walltime: synthetic ``-failed`` rows written during a
#   SIGTERM grace window look identical to real successes to the dedup in
#   ``_load_from_cache`` (which keys only on (task_index, rollout_index)),
#   so chain-hop 2 thinks failed tasks are done and never retries them.
#
# New contract:
#   - Successes go to the main jsonl (``output_jsonl_fpath``).
#   - Failures go to a sidecar (``<output_stem>_failures.jsonl``), one row
#     per attempt, with ``_ng_failure_class`` set. An ``agent_run_error`` or
#     ``agent_request_failed`` row holds no reward and no response: there was
#     no rollout. The two differ in whether the agent answered at all.
#   - ``kill_shaped`` failures (Slurm SIGTERM, Ray actor died, OOM, ...) go
#     NOWHERE: the absence of a row is the canonical signal. Resume's
#     set-difference re-dispatches them naturally; per-task timeout bounds
#     the chain-hop wallclock.
#   - On resume, ``_load_from_cache`` reads BOTH files: main jsonl tells
#     it what's permanently done, sidecar tells it how many attempts each
#     non-success has consumed (capped at NEMO_GYM_MAX_ROLLOUT_ATTEMPTS,
#     default 3). Rows flagged ``_ng_failure_terminal=True`` are never
#     retried regardless of attempt count.
# ---------------------------------------------------------------------------

NG_FAILURE_CLASS_KEY = "_ng_failure_class"
NG_NO_PERSIST_KEY = "_ng_no_persist"
NG_TERMINAL_KEY = "_ng_failure_terminal"
AGENT_REQUEST_FAILED_FAILURE_CLASS = "agent_request_failed"
AGENT_RUN_ERROR_FAILURE_CLASS = "agent_run_error"
_NO_RESULT_FAILURE_CLASSES = frozenset({AGENT_REQUEST_FAILED_FAILURE_CLASS, AGENT_RUN_ERROR_FAILURE_CLASS})
NG_TRAJECTORY_KEY = "ng_trajectory"
NG_PERF_KEY = "ng_perf"
_MODEL_CALL_PAYLOAD_KEYS = ("request", "response", "request_raw", "response_raw")

_DEFAULT_MAX_ROLLOUT_ATTEMPTS = 3


@dataclass(frozen=True)
class _CompletedRollout:
    """A finished ``/run`` dispatch, with timing carried alongside (not inside) the raw result."""

    row: Dict[str, Any]
    result: Dict[str, Any]
    rollout_latency_ms: Optional[float]


def _nonnegative_int(value: Any) -> Optional[int]:
    return value if type(value) is int and value >= 0 else None


def _has_observation_gap(result: dict[str, Any], code: str) -> bool:
    for key in (NG_TRAJECTORY_KEY, "ng_agent_observations"):
        observations = result.get(key)
        gaps = observations.get("gaps") if isinstance(observations, dict) else None
        if isinstance(gaps, list) and any(isinstance(gap, dict) and gap.get("code") == code for gap in gaps):
            return True
    return False


def _trajectory_identity(row: dict[str, Any]) -> tuple[str, str]:
    task_id = next(
        (str(row[key]) for key in ("task_id", "problem_id", "instance_id") if row.get(key) is not None),
        str(row[TASK_INDEX_KEY_NAME]),
    )
    rollout_id = maybe_rollout_id_from_run_body(row) or f"{row[TASK_INDEX_KEY_NAME]}-{row[ROLLOUT_INDEX_KEY_NAME]}"
    return task_id, rollout_id


def _build_trajectory_record(row: dict[str, Any], result: dict[str, Any]) -> TrajectoryRecord:
    task_id, rollout_id = _trajectory_identity(row)
    gaps: list[ObservationGap] = []
    invocations: list[AgentInvocation] = []
    turns: list[TrajectoryTurn] = []
    tools: list[TrajectoryToolCall] = []
    model_calls: list[TrajectoryModelCall] = []

    raw_trajectory = result.get(NG_TRAJECTORY_KEY)
    if isinstance(raw_trajectory, dict):
        try:
            trajectory = TrajectoryRecord.model_validate(raw_trajectory)
            mismatches = [
                field
                for field, producer, canonical in (
                    ("task_id", trajectory.task_id, task_id),
                    ("rollout_id", trajectory.rollout_id, rollout_id),
                )
                if producer != canonical
            ]
            if mismatches:
                gaps.append(ObservationGap(code="producer_trajectory_identity_mismatch", detail=",".join(mismatches)))
                turns = [
                    turn.model_copy(update={"task_id": task_id, "rollout_id": rollout_id}) for turn in trajectory.turns
                ]
            else:
                turns = trajectory.turns
            gaps.extend(trajectory.gaps)
            invocations = trajectory.invocations
            tools = trajectory.tool_calls
            model_calls = trajectory.model_calls
        except Exception as exc:
            gaps.append(ObservationGap(code="producer_trajectory_invalid", detail=type(exc).__name__))

    raw_observations = result.get("ng_agent_observations")
    if raw_observations is not None:
        try:
            observations = AgentObservationBundle.model_validate(raw_observations)
            gaps.extend(observations.gaps)
            observed_invocations = [record for record in observations.records if isinstance(record, AgentInvocation)]
            producer_invocation_ids = {record.invocation_id for record in invocations}
            invocations.extend(
                record for record in observed_invocations if record.invocation_id not in producer_invocation_ids
            )
            observed_tools = [record for record in observations.records if isinstance(record, ToolCallObservation)]
            if observed_tools:
                outputs = {
                    (invocation.invocation_id, item.call_id): item.output
                    for invocation in invocations
                    for item in invocation.conversation
                    if getattr(item, "type", None) == "function_call_output"
                }
                positions = {(tool.invocation_id, tool.tool_call_id): index for index, tool in enumerate(tools)}
                for observed in observed_tools:
                    key = (observed.invocation_id, observed.tool_call_id)
                    position = positions.get(key)
                    existing = tools[position] if position is not None else None
                    merged = existing.model_dump(mode="json") if existing is not None else {}
                    update = observed.model_dump(mode="json", exclude_none=True)
                    if existing is not None and observed.status == "unknown" and existing.status != "unknown":
                        update.pop("status", None)
                    merged.update(update)
                    if key in outputs:
                        merged["output"] = outputs[key]
                    projected = TrajectoryToolCall.model_validate(merged)
                    if position is None:
                        positions[key] = len(tools)
                        tools.append(projected)
                    else:
                        tools[position] = projected
        except Exception as exc:
            gaps.append(ObservationGap(code="agent_observations_invalid", detail=type(exc).__name__))

    turns.sort(key=lambda turn: (turn.timestamp, turn.invocation_id, turn.turn_no))

    capture = result.get("ng_model_call_capture")
    capture = capture if isinstance(capture, dict) else {}
    raw_calls = capture.get("calls") or []
    model_call_positions = {
        call.model_call_id: index for index, call in enumerate(model_calls) if call.model_call_id is not None
    }
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue
        model_call_id = raw_call.get("model_call_id")
        metadata = {
            key: raw_call[key]
            for key in (
                "response_id",
                "model_ref",
                "model",
                "dialect",
                "status_code",
                "response_status",
                "finish_reason",
                "error_category",
                "latency_ttft_ms",
            )
            if raw_call.get(key) is not None
        }
        response = raw_call.get("response")
        if isinstance(response, dict) and isinstance(response.get("status"), str):
            metadata.setdefault("response_status", response["status"])
        projected = TrajectoryModelCall(
            model_call_id=model_call_id,
            started_at=raw_call.get("started_at"),
            completed_at=raw_call.get("completed_at"),
            duration_ms=raw_call.get("latency_total_ms"),
            request=raw_call.get("request") if raw_call.get("request") is not None else raw_call.get("request_raw"),
            response=raw_call.get("response")
            if raw_call.get("response") is not None
            else raw_call.get("response_raw"),
            response_metadata=metadata,
            token_stats=TrajectoryTokenStats(
                prompt_tokens=_nonnegative_int(raw_call.get("tokens_in")),
                completion_tokens=_nonnegative_int(raw_call.get("tokens_out")),
                reasoning_tokens=_nonnegative_int(raw_call.get("tokens_reasoning")),
                total_tokens=_nonnegative_int(raw_call.get("tokens_total")),
                cached_tokens=_nonnegative_int(raw_call.get("cached_tokens")),
            ),
        )
        position = model_call_positions.pop(model_call_id, None) if model_call_id is not None else None
        if position is None:
            model_calls.append(projected)
        else:
            merged = model_calls[position].model_dump(mode="json")
            update = projected.model_dump(mode="json", exclude_none=True)
            for key in ("response_metadata", "token_stats"):
                merged[key].update(update.pop(key))
            merged.update(update)
            projected = TrajectoryModelCall.model_validate(merged)
            model_calls[position] = projected

    for raw_gap in capture.get("gaps") or []:
        if isinstance(raw_gap, dict):
            try:
                gaps.append(ObservationGap.model_validate(raw_gap))
            except Exception:
                gaps.append(ObservationGap(code="model_call_capture_gap_invalid"))
    if not model_calls:
        gaps.append(ObservationGap(code="model_calls_unavailable"))
    if not turns:
        gaps.append(ObservationGap(code="turns_unavailable"))
    if not any(invocation.conversation for invocation in invocations):
        gaps.append(ObservationGap(code="conversation_unavailable"))

    return TrajectoryRecord(
        task_id=task_id,
        rollout_id=rollout_id,
        invocations=invocations,
        turns=turns,
        model_calls=model_calls,
        tool_calls=tools,
        gaps=list({(gap.code, gap.invocation_id, gap.detail): gap for gap in gaps}.values()),
    )


def _strip_capture_payloads(result: dict[str, Any]) -> None:
    capture = result.get("ng_model_call_capture")
    calls = capture.get("calls") if isinstance(capture, dict) else None
    for call in calls if isinstance(calls, list) else []:
        if isinstance(call, dict):
            for key in _MODEL_CALL_PAYLOAD_KEYS:
                call.pop(key, None)


def _rollout_for_export(result: dict[str, Any]) -> dict[str, Any]:
    """Return an exporter view without the complete trajectory or raw capture payloads."""
    sanitized = dict(result)
    sanitized.pop(NG_TRAJECTORY_KEY, None)
    sanitized.pop("ng_model_call_capture", None)
    capture = result.get("ng_model_call_capture")
    if isinstance(capture, dict):
        sanitized_capture = dict(capture)
        calls = capture.get("calls")
        if isinstance(calls, list):
            sanitized_capture["calls"] = [
                {key: value for key, value in call.items() if key not in _MODEL_CALL_PAYLOAD_KEYS}
                for call in calls
                if isinstance(call, dict)
            ]
        else:
            sanitized_capture.pop("calls", None)
        sanitized["ng_model_call_capture"] = sanitized_capture
    return sanitized


def _attach_trajectory_record(row: dict[str, Any], result: dict[str, Any]) -> None:
    try:
        result[NG_TRAJECTORY_KEY] = _build_trajectory_record(row, result).model_dump(mode="json")
    except Exception as exc:
        result.pop(NG_TRAJECTORY_KEY, None)
        logger.warning("Could not project standardized trajectory evidence.", exc_info=True)
        gap = ObservationGap(code="trajectory_projection_failed", detail=type(exc).__name__).model_dump(
            mode="json", exclude_none=True
        )
        target = result.get("ng_model_call_capture")
        if not isinstance(target, dict):
            target = result.get("ng_agent_observations")
        gap_attached = False
        if isinstance(target, dict):
            gaps = target.setdefault("gaps", [])
            if isinstance(gaps, list):
                gaps.append(gap)
                gap_attached = True
        if not gap_attached:
            try:
                task_id, rollout_id = _trajectory_identity(row)
                result[NG_TRAJECTORY_KEY] = TrajectoryRecord(
                    task_id=task_id,
                    rollout_id=rollout_id,
                    gaps=[ObservationGap.model_validate(gap)],
                ).model_dump(mode="json")
            except Exception:
                logger.warning("Could not retain the trajectory projection failure gap.", exc_info=True)
    else:
        # Raw capture payloads remain as a fallback on failure. After success,
        # ng_trajectory owns them, so remove only the duplicate copies.
        _strip_capture_payloads(result)


def _build_ng_perf(result: dict[str, Any], *, rollout_latency_ms: Optional[float]) -> Optional[dict[str, Any]]:
    """Assemble the per-rollout ``ng_perf`` summary from ``ng_trajectory``.

    Returns ``None`` (``ng_perf`` stays absent) unless at least one reasoning turn was
    observed: per-turn evidence is needed rather than just raw model-call capture,
    so a rollout collected with observability disabled produces no ``ng_perf`` at all.

    Token fields are summed only over model calls owned by a reasoning-turn ``AgentInvocation``,
    excluding compaction calls -- mixing in compaction overhead would skew the token efficiency signal.

    ``num_turns`` counts reasoning turns summed across all invocations (an ``AgentInvocation``
    is one root-agent or subagent conversation that may span many turns). Each invocation
    contributes its explicit ``TrajectoryTurn`` count when the harness emits turn records,
    falling back to its owned model-call count (one assistant response per turn), then to 1
    (an invocation that ran had at least one turn) -- so hybrid trajectories where only some
    invocations report turns still count every conversation.

    ``token_observability_coverage`` reports what fraction of those turns actually resolved to a
    captured call: a turn whose ``ModelCallRef`` was unmatched or ambiguous silently loses its
    tokens from the sums below, and this is the only signal that it happened.
    """
    raw_trajectory = result.get(NG_TRAJECTORY_KEY)
    if not isinstance(raw_trajectory, dict):
        return None
    try:
        trajectory = TrajectoryRecord.model_validate(raw_trajectory)
    except Exception:
        return None
    if not trajectory.invocations:
        return None

    # Index captured calls by both identities ModelCallRef supports, mirroring
    # join_model_call_observations: a ref may carry model_call_id, or the exact
    # (model_ref, response_id) pair.
    calls_by_id: dict[str, list[int]] = {}
    calls_by_response: dict[tuple[str, str, str], list[int]] = {}
    for index, call in enumerate(trajectory.model_calls):
        if call.model_call_id:
            calls_by_id.setdefault(call.model_call_id, []).append(index)
        call_model_ref = call.response_metadata.model_ref
        call_response_id = call.response_metadata.response_id
        if call_model_ref is not None and call_response_id:
            calls_by_response.setdefault((call_model_ref.type, call_model_ref.name, call_response_id), []).append(
                index
            )

    def _match_call_index(ref: ModelCallRef) -> Optional[int]:
        if ref.model_call_id:
            candidates = [
                index
                for index in calls_by_id.get(ref.model_call_id, [])
                if (
                    ref.model_ref is None or ref.model_ref == trajectory.model_calls[index].response_metadata.model_ref
                )
                and (
                    ref.response_id is None
                    or ref.response_id == trajectory.model_calls[index].response_metadata.response_id
                )
            ]
        elif ref.model_ref is not None and ref.response_id:
            candidates = calls_by_response.get((ref.model_ref.type, ref.model_ref.name, ref.response_id), [])
        else:
            candidates = []
        return candidates[0] if len(candidates) == 1 else None

    # De-duplicate by matched call *position* rather than model_call_id; the id is
    # optional because a (model_ref, response_id)-only match has none. One physical call
    # can't be claimed twice even if two invocations' refs both resolve to it (e.g. a
    # join_model_call_observations conflict that left a ref attached to its losing
    # invocation, unremoved, just gap-flagged).
    seen_positions: set[int] = set()
    owned_calls = []
    owned_calls_by_invocation: Counter = Counter()
    for invocation in trajectory.invocations:
        for ref in invocation.model_calls:
            index = _match_call_index(ref)
            if index is None or index in seen_positions:
                continue
            seen_positions.add(index)
            owned_calls.append(trajectory.model_calls[index])
            owned_calls_by_invocation[invocation.invocation_id] += 1
    tool_calls_by_invocation = Counter(tool.invocation_id for tool in trajectory.tool_calls)
    num_tool_calls = sum(
        tool_calls_by_invocation.get(invocation.invocation_id, 0) for invocation in trajectory.invocations
    )

    def _sum_tokens(attr: str) -> Optional[int]:
        values = [
            getattr(call.token_stats, attr) for call in owned_calls if getattr(call.token_stats, attr) is not None
        ]
        return sum(values) if values else None

    turns_by_invocation = Counter(turn.invocation_id for turn in trajectory.turns)
    num_turns = sum(
        turns_by_invocation.get(invocation.invocation_id, 0)
        or owned_calls_by_invocation.get(invocation.invocation_id, 0)
        or 1
        for invocation in trajectory.invocations
    )
    ng_perf: dict[str, Any] = {
        "num_turns": num_turns,
        "num_tool_calls": num_tool_calls,
        "token_observability_coverage": min(1.0, len(owned_calls) / num_turns),
    }
    for ng_perf_key, token_stats_attr in (
        ("prompt_tokens", "prompt_tokens"),
        ("cached_prompt_tokens", "cached_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("reasoning_tokens", "reasoning_tokens"),
    ):
        value = _sum_tokens(token_stats_attr)
        if value is not None:
            ng_perf[ng_perf_key] = value

    if isinstance(rollout_latency_ms, (int, float)):
        ng_perf["total_latency_ms"] = rollout_latency_ms

    return ng_perf


def _attach_ng_perf(
    result: dict[str, Any], *, observability_enabled: bool, rollout_latency_ms: Optional[float] = None
) -> None:
    if not observability_enabled:
        # ng_perf stays absent entirely when observability is off (OQ4): a caller who
        # disabled it only wants the final score, not partial/best-effort perf evidence.
        return
    try:
        ng_perf = _build_ng_perf(result, rollout_latency_ms=rollout_latency_ms)
    except Exception:
        logger.warning("Could not assemble ng_perf for a rollout.", exc_info=True)
        return
    if ng_perf is not None:
        result[NG_PERF_KEY] = ng_perf


def _get_max_rollout_attempts() -> int:
    """Read ``NEMO_GYM_MAX_ROLLOUT_ATTEMPTS`` (positive int) or default to 3."""
    raw = os.environ.get("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS")
    if raw is None or raw == "":
        return _DEFAULT_MAX_ROLLOUT_ATTEMPTS
    try:
        n = int(raw)
        if n < 1:
            raise ValueError(f"must be >= 1, got {n}")
        return n
    except (TypeError, ValueError) as e:
        print(
            f"WARNING: could not parse NEMO_GYM_MAX_ROLLOUT_ATTEMPTS={raw!r} ({e}); "
            f"falling back to default {_DEFAULT_MAX_ROLLOUT_ATTEMPTS}.",
            flush=True,
        )
        return _DEFAULT_MAX_ROLLOUT_ATTEMPTS


def _normalize_health_check_ignored_checks(value) -> List[str]:
    from nemo_gym.rollout_health import normalize_ignored_checks

    return list(normalize_ignored_checks(value))


class SharedRolloutCollectionConfig(UploadRolloutsConfigMixin, BaseNeMoGymCLIConfig):
    output_jsonl_fpath: str = Field(description="The output data jsonl file path.")
    num_samples_in_parallel: Optional[int] = Field(
        default=None, description="Limit the number of concurrent samples running at once."
    )
    responses_create_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Overrides for the responses_create_params e.g. temperature, max_output_tokens, etc.",
    )
    disable_aggregation: bool = Field(
        default=False,
        description=(
            "Skip the post-rollout aggregate-metrics computation and file write. "
            "Used when sharding rollouts across multiple jobs that will be aggregated together "
            "afterward by `gym eval aggregate`."
        ),
    )
    disable_health_check: bool = Field(
        default=False,
        description="Skip post-aggregation rollout quality verification and report writing.",
    )
    health_check_workers: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of rollout-health worker processes (defaults to min(cpus, 8)).",
    )
    health_check_ignored_checks: List[str] = Field(
        default_factory=list,
        description="Health-check IDs to exclude from execution and verdict derivation.",
    )

    @field_validator("health_check_ignored_checks", mode="before")
    @classmethod
    def _validate_health_check_ignored_checks(cls, value):
        return _normalize_health_check_ignored_checks(value)

    count_failure_classes_as_zero: List[str] = Field(
        default_factory=list,
        description=(
            "Failure classes from the failures sidecar to count in aggregate metrics, e.g. "
            "['agent_run_error'], so a failed rollout lands in the denominator. A row that carries "
            "no reward is scored zero for the metrics only; no artifact is changed."
        ),
    )

    route_failures_to_sidecar: bool = Field(
        default=False,
        description=(
            "Record a failed agent /run as a failures-sidecar row and keep going, instead of ending "
            "the run. The failed rollouts are then absent from the rollouts jsonl and from the "
            "score, which is reported over fewer rollouts than were dispatched."
        ),
    )
    rollout_collection_driver: Optional[str] = Field(
        default=None,
        description=(
            "Optional dotted ``module.path:function`` to run rollout collection instead of the "
            "built-in helper. Lets a benchmark plug in a custom procedure (e.g. an adaptive, "
            "multi-pass run) while still producing the standard rollout + aggregate-metrics "
            "artifacts. The function is awaited with (rollout_collection_config, global_config_dict). "
            "When unset, the standard single-pass collection runs."
        ),
    )


class E2ERolloutCollectionConfig(SharedRolloutCollectionConfig):
    """
    Spin up all necessary servers and perform a batch of rollout collection using each dataset inside the provided configs.

    Examples:

    ```bash
    gym eval run \
        +output_jsonl_fpath=weather_rollouts.jsonl \
        +num_samples_in_parallel=10
    ```
    """

    split: Union[Literal["train"], Literal["validation"], Literal["benchmark"]]
    reuse_existing_data_preparation: bool = False

    @model_validator(mode="before")
    @classmethod
    def _reject_input_jsonl_fpath(cls, data):
        # This config has no input_jsonl_fpath field, so pydantic would silently drop it and
        # e2e collection would overwrite it with the prepared split path — the user's file
        # would be ignored without any indication. Match on Mapping, not dict: the CLI passes
        # an OmegaConf DictConfig, which is a Mapping but not a dict.
        if isinstance(data, Mapping) and "input_jsonl_fpath" in data:
            raise ConfigError(
                "`input_jsonl_fpath` (-i/--input) is not supported when serving end-to-end: the input is "
                "always the prepared dataset for the requested split. Either add --no-serve to collect "
                "rollouts from your own input file against already-running servers, or drop -i/--input "
                "to use the prepared data."
            )
        return data

    @model_validator(mode="before")
    @classmethod
    def _reject_example_split(cls, data):
        # `example` is a real dataset type but deliberately not a runnable split: example
        # datasets are the committed smoke-test samples the PR data gate validates, and they
        # are excluded from prepared splits so they never leak into training or eval data.
        # Catch it before the Literal check so the user gets the documented recipe instead of
        # a bare "Input should be 'train'".
        if isinstance(data, Mapping) and data.get("split") == "example":
            raise ConfigError(
                "`--split example` is not runnable end-to-end: example datasets are committed "
                "smoke-test samples, not prepared train/validation/benchmark splits. To run one, "
                "start the servers and point at the example file directly:\n"
                "  gym env start --resources-server <server> ...\n"
                "  gym eval run --no-serve --agent <agent> --input <server_dir>/data/example.jsonl --output <out>.jsonl\n"
                "See the Quickstart: https://docs.nvidia.com/nemo/gym/latest/get-started/quickstart"
            )
        return data


class RolloutCollectionConfig(SharedRolloutCollectionConfig):
    """
    Perform a batch of rollout collection.

    Examples:

    ```bash
    gym eval run --no-serve \
        +agent_name=example_single_tool_call_simple_agent \
        +input_jsonl_fpath=weather_query.jsonl \
        +output_jsonl_fpath=weather_rollouts.jsonl \
        +limit=100 \
        +num_repeats=4 \
        +num_samples_in_parallel=10
    ```
    """

    agent_name: Optional[str] = Field(
        default=None,
        description=(
            "The agent to collect rollouts from. Routes every row to this agent, overriding any "
            "agent_ref already present in the data (a warning lists overridden values). "
            "Shorthand for agent_map={_default: <agent_name>}."
        ),
    )
    agent_map: Optional[Dict[str, str]] = Field(
        default=None,
        description=(
            "Explicit per-agent re-routing, keyed by the agent_ref.name or task_source found in "
            "the data (e.g. {old_agent: new_agent}). The special key '_default' applies to every "
            "row with no specific entry, including rows with no agent_ref at all. "
            "Precedence: agent_map[<row value>] > agent_map._default > row agent_ref > task_source resolution."
        ),
    )
    fan_out: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description=(
            "Run each matching row once per listed agent (cross-product), keyed by the row's "
            "agent_ref.name or task_source (e.g. {genrm_compare_resources_server: [agent_a, agent_b]}). "
            "Each copy gets its own rollout index; outputs are tagged with the agent that produced them, "
            "so per-agent metrics separate naturally. Composes with num_repeats (repeats apply per agent)."
        ),
    )
    input_jsonl_fpath: str = Field(
        description="The input data source to use to collect rollouts, in the form of a file path to a jsonl file."
    )
    limit: Optional[int] = Field(
        default=None, description="Maximum number of examples to load and take from the input dataset."
    )
    num_repeats: Union[int, Dict[str, int]] = Field(
        default=1,
        description=(
            "How many times to repeat each example. Either an int (applied to every row) or a "
            "dict keyed by the dispatched agent name or by the row's own routing key (its "
            "agent_ref.name or task_source as written in the data, before any agent_map/fan_out "
            "re-route); the dispatched agent wins when both have entries. In dict form, every row "
            "must match an entry, unless a special '_default' key is provided as a fallback. "
            "Useful for mean@k."
        ),
    )
    num_repeats_add_seed: bool = Field(
        default=False,
        description='When num_repeats > 1, pass a per-rollout "seed" via metadata.extra_body (honored by vLLM model servers).',
    )
    resume_from_cache: bool = Field(
        default=False,
        description="If the same command is run multiple times, check the materialized inputs and current outputs and remove the inputs that have already been run",
    )
    prompt_config: Optional[str] = Field(
        default=None,
        description="Path to a prompt YAML file. Builds responses_create_params.input from the template at rollout time. Mutually exclusive with pre-populated responses_create_params.input in the JSONL data.",
    )
    skills: Optional[SkillsConfig] = Field(
        default=None,
        description="Run-level skills config (skills.path). Makes a directory of Agent Skills standard skills available to the agent at rollout time and stamps each result with a skills_ref. Applied to a skill-agnostic dataset; not a dataset-row field.",
    )

    @field_validator("num_repeats", mode="before")
    @classmethod
    def _coerce_null_num_repeats(cls, v):
        # default to 1 if num_repeats is None
        # for backwards compatibility
        return 1 if v is None else v

    @model_validator(mode="after")
    def _fold_agent_name_into_agent_map(self) -> "RolloutCollectionConfig":
        # agent_name is sugar for agent_map._default; fold it so downstream code has one knob.
        if self.agent_name is None:
            return self
        existing_default = (self.agent_map or {}).get("_default")
        if existing_default is not None and existing_default != self.agent_name:
            raise ValueError(
                f"agent_name={self.agent_name!r} conflicts with agent_map._default={existing_default!r}. "
                "Set only one of them."
            )
        self.agent_map = {**(self.agent_map or {}), "_default": self.agent_name}
        return self

    @model_validator(mode="after")
    def _validate_num_repeats(self) -> "RolloutCollectionConfig":
        nr = self.num_repeats
        if isinstance(nr, int):
            if nr < 1:
                raise ValueError(f"num_repeats must be >= 1, got {nr}")
        else:
            bad = {name: n for name, n in nr.items() if n < 1}
            if bad:
                raise ValueError(f"num_repeats dict values must be >= 1, got {bad}")
        return self

    @model_validator(mode="after")
    def _validate_fan_out(self) -> "RolloutCollectionConfig":
        for key, agents in (self.fan_out or {}).items():
            if not agents:
                raise ValueError(
                    f"fan_out[{key!r}] is an empty list, which would silently drop every matching row "
                    "(zero rollouts collected). Remove the key, or list at least one agent."
                )
            duplicates = sorted({a for a in agents if list(agents).count(a) > 1})
            if duplicates:
                raise ValueError(
                    f"fan_out[{key!r}] lists the same agent more than once: {duplicates}. Each listed "
                    "agent already runs every matching row; use num_repeats for repetition."
                )
        return self

    @property
    def materialized_jsonl_fpath(self) -> Path:
        output_fpath = Path(self.output_jsonl_fpath)
        return output_fpath.with_stem(output_fpath.stem + "_materialized_inputs").with_suffix(".jsonl")


def _rollout_request_debug_summary(row: Dict[str, Any]) -> Dict[str, Any]:
    agent_ref = row.get(AGENT_REF_KEY_NAME) or {}
    summary = {
        TASK_INDEX_KEY_NAME: row.get(TASK_INDEX_KEY_NAME),
        ROLLOUT_INDEX_KEY_NAME: row.get(ROLLOUT_INDEX_KEY_NAME),
        "agent_name": agent_ref.get("name") if isinstance(agent_ref, dict) else None,
    }
    return {k: v for k, v in summary.items() if v is not None}


# Request failures that are data, not bugs. Anything else still propagates.
_RUN_FAILURE_ERRORS = (ClientError, orjson.JSONDecodeError, TimeoutError)
# Statuses something in front of the agent answers with; the agent itself returns 500.
_AGENT_DID_NOT_RUN_STATUSES = frozenset({429, 502, 503, 504})
_MAX_FAILURE_BODY_CHARS = 2000


def _agent_request_failure_row(exc: BaseException, status: Optional[int]) -> Dict[str, Any]:
    """One sidecar row for a `/run` call that came back without a result.

    No reward and no response: an infrastructure failure is not a verifier score of zero, and a
    placeholder would read as real generation data to token capture, aggregation and trainers.
    The class says whether the rollout ran. A NeMo Gym agent answers 500 when its own handler
    raises, so any status it answered with means the agent ran and broke, which is also how a
    model server rejecting the model's own output arrives here. A gateway status, or no reply to
    take a status from, says nothing about the rollout. Neither class carries a reward; an evaluation that wants the
    first counted names it in ``count_failure_classes_as_zero``.
    """
    agent_ran = status is not None and status not in _AGENT_DID_NOT_RUN_STATUSES
    body = getattr(exc, "response_content", None)
    return {
        NG_FAILURE_CLASS_KEY: (AGENT_RUN_ERROR_FAILURE_CLASS if agent_ran else AGENT_REQUEST_FAILED_FAILURE_CLASS),
        "_ng_failure_type": type(exc).__name__,
        "_ng_failure_message": str(exc) or repr(exc),
        "_ng_failure_http_status": status,
        "_ng_failure_response_body": _truncated_body(body),
    }


def _truncated_body(body: Optional[bytes]) -> Optional[str]:
    """Decode at most the kept prefix, so a huge error page is never decoded in full."""
    if not body:
        return None
    text = body[: _MAX_FAILURE_BODY_CHARS * 4].decode("utf-8", "replace")
    return text[:_MAX_FAILURE_BODY_CHARS] + ("…" if len(body) > _MAX_FAILURE_BODY_CHARS else "")


def _latest_failure_rows(failures_fpaths: List[Path]) -> Dict[Tuple[Any, Any], Dict[str, Any]]:
    """The last attempt recorded for each rollout across the failures sidecars."""
    latest_by_key: Dict[Tuple[Any, Any], Dict[str, Any]] = {}
    for fpath in failures_fpaths:
        if not fpath.exists():
            continue
        with fpath.open("rb") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                row = loads_jsonl_line(line, fpath, line_no)
                latest_by_key[(row.get(TASK_INDEX_KEY_NAME), row.get(ROLLOUT_INDEX_KEY_NAME))] = row
    return latest_by_key


def _failure_rows_counted_as_zero(
    failures_fpaths: List[Path], failure_classes: List[str], scored_keys: set
) -> List[Dict[str, Any]]:
    """Sidecar rows the caller opted to count in the metrics denominator.

    The last attempt of a rollout is the one that stands, so it is selected across every failure
    class before the wanted classes are picked out. Selecting the other way round would let a
    stale attempt be counted after a later one landed in a class the caller did not ask for.

    A row that already carries a ``reward`` is counted as it stands. A row that carries none
    records that no rollout happened, so it is counted as a zero here and only here: the score
    enters the metric input, never the sidecar or the rollouts jsonl, which keeps the artifacts
    free of a verdict no verifier gave. A rollout that also succeeded is never counted.
    """
    if not failure_classes:
        return []

    latest_by_key = _latest_failure_rows(failures_fpaths)
    wanted = set(failure_classes)
    counted = []
    for key, row in latest_by_key.items():
        if key in scored_keys or row.get(NG_FAILURE_CLASS_KEY) not in wanted:
            continue
        # Diagnostics stay in the sidecar: an HTTP status is a number, and the aggregator
        # averages every number it is handed.
        scored = {k: v for k, v in row.items() if not k.startswith("_ng_failure_")}
        scored.setdefault("reward", 0.0)
        counted.append(scored)
    return counted


def _coverage_report(expected: int, scored: int, failure_counts: Counter, failures_hint: Any) -> str:
    """State how much of the input the score covers, for the runs where it is not all of it.

    Silence here is what makes a partial run look complete, so this reports against the
    materialized input rather than the rollouts one hop happened to dispatch.
    """
    missing = expected - scored
    if missing <= 0:
        return ""
    routed = ", ".join(f"{count} {name}" for name, count in sorted(failure_counts.items()))
    routed = f"{routed} routed this run; " if routed else ""
    return (
        f"\nRollouts missing from the score: {missing} of {expected} materialized ({routed}see {failures_hint})"
        f"\nMetrics cover: {scored} of {expected} rollouts"
    )


class RolloutCollectionHelper(BaseModel):
    def _preprocess_rows_from_config(self, config: RolloutCollectionConfig) -> List[Dict]:
        range_iterator = repeat(0)
        if config.limit:
            range_iterator = range(config.limit)
            print(f"Limiting the number of rows to {config.limit}")

        # Load prompt config if specified
        prompt_cfg = None
        if config.prompt_config:
            prompt_cfg = load_prompt_config(config.prompt_config)
            print(f"Using prompt config: {config.prompt_config}")

        # Search NEMO_GYM_EXTRA_ROOTS, cwd, then the install root.
        _input_path = _resolve_under_cwd_or_install(config.input_jsonl_fpath)
        if not _input_path.exists():
            raise ConfigPathNotFoundError(
                f"Input file not found: '{config.input_jsonl_fpath}' (--input). Check the path is spelled correctly."
            )
        with open(_input_path) as input_file:
            rows_iterator: Iterator[str] = tqdm(input_file, desc="Reading rows")
            rows_iterator: Iterator[tuple[int, str]] = zip(range_iterator, rows_iterator)
            raw_rows = [
                (row_idx, row_str, loads_jsonl_line(row_str, _input_path, line_no))
                for line_no, (row_idx, row_str) in enumerate(rows_iterator, 1)
            ]

        # Validate and apply prompt config before per-row processing
        if prompt_cfg is not None:
            validate_prompt_compatibility([row for _, _, row in raw_rows], prompt_cfg)
            raw_rows = [(idx, s, apply_prompt_to_row(row, prompt_cfg)) for idx, s, row in raw_rows]

        return RolloutCollectionHelper._preprocess_raw_rows(raw_rows, config)

    def preprocess_examples(
        self,
        examples: List[Dict],
        *,
        agent_map: Optional[Dict[str, str]] = None,
        fan_out: Optional[Dict[str, List[str]]] = None,
        num_repeats: Union[int, Dict[str, int]] = 1,
        num_repeats_add_seed: bool = False,
        global_config_dict: Optional[DictConfig] = None,
    ) -> List[Dict]:
        """Apply run-level routing and repetition to caller-held rows.

        Public entry point for direct ``run_examples`` callers (e.g. trainer integrations that
        drive dispatch themselves): ``run_examples`` resolves task_sources and validates agent
        names, but ``agent_map``, ``fan_out`` and ``num_repeats`` are applied only during
        preprocessing. Call this first, then pass the returned rows to ``run_examples``.

        Pass ``global_config_dict`` (the merged config) to also resolve task_source-only rows to
        their agents here; leave it None to defer that to ``run_examples``, which does it against
        the head server's config. Input rows are not mutated; the expanded, stamped copies are
        returned.
        """
        config = RolloutCollectionConfig(
            input_jsonl_fpath="<in-memory>",
            output_jsonl_fpath="<in-memory>",
            agent_map=agent_map,
            fan_out=fan_out,
            num_repeats=num_repeats,
            num_repeats_add_seed=num_repeats_add_seed,
        )
        raw_rows = [
            (idx, orjson.dumps(row, option=orjson.OPT_SORT_KEYS).decode(), row.copy())
            for idx, row in enumerate(examples)
        ]
        rows = self._preprocess_raw_rows(raw_rows, config)
        if global_config_dict is not None:
            self.resolve_task_sources(rows, global_config_dict)
        return rows

    @staticmethod
    def _preprocess_raw_rows(raw_rows: List[Tuple[int, str, Dict]], config: RolloutCollectionConfig) -> List[Dict]:
        if config.num_repeats_add_seed:
            print(
                "Adding unique `seed` values to each input via metadata.extra_body (only honored by vLLM model servers)"
            )

        if config.agent_map:
            print(f"Routing rows via agent_map {config.agent_map}")

        if config.responses_create_params:
            print(f"Overriding responses_create_params fields with {config.responses_create_params}")
            responses_create_params_overrides = OmegaConf.to_container(
                OmegaConf.create(config.responses_create_params), resolve=True
            )
        else:
            responses_create_params_overrides = dict()

        if isinstance(config.num_repeats, int):
            fixed_num_repeats: Optional[int] = config.num_repeats
            per_agent_repeats: Dict[str, int] = {}
            default_repeats: Optional[int] = None
            print(f"Repeating rows {fixed_num_repeats} times (in a pattern of abc to aabbcc)!")
        else:
            fixed_num_repeats = None
            per_agent_repeats = {k: v for k, v in config.num_repeats.items() if k != "_default"}
            default_repeats = config.num_repeats.get("_default")
            print(f"Per-agent num_repeats: {dict(config.num_repeats)}")
        agents_seen: set[str] = set()

        # Resolve skills once for the whole run (hash is content-derived, computed at startup).
        skills_ref_dict = None
        if config.skills:
            skills_ref = load_skill_directory(config.skills.path)
            skills_ref_dict = skills_ref.model_dump()
            print(
                f"Using skills from {config.skills.path} "
                f"(hash={skills_ref.hash}, {len(skills_ref.skills)} skill(s): "
                f"{', '.join(s.name for s in skills_ref.skills)})"
            )

        # For gym eval profile to match rollouts to tasks
        row_to_task_idx: Dict[str, int] = dict()
        task_idx_to_rollout_idx: Dict[int, int] = Counter()
        row_idxs_missing_agent_ref: List[int] = []
        agents_missing_from_num_repeats: set[str] = set()
        rows: List[Dict] = []
        overridden_agents: set[Tuple[str, str]] = set()
        for row_idx, row_str, row in tqdm(raw_rows, desc="Preprocessing and repeating rows"):
            # Routing basis: the name this row routes by — its agent_ref.name when present, else
            # its task_source (resolved to an agent by resolve_task_sources once the merged config
            # is in hand). agent_map[<basis>] > agent_map._default > row agent_ref > task_source.
            agent_name = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
            basis = agent_name if agent_name is not None else row.get(TASK_SOURCE_KEY_NAME)
            if config.agent_map:
                # A row may carry both an agent_ref and a task_source (derived artifacts do);
                # a map entry for either re-routes it, the agent name taking precedence.
                mapped = next(
                    (
                        config.agent_map[key]
                        for key in (agent_name, row.get(TASK_SOURCE_KEY_NAME))
                        if key is not None and key in config.agent_map
                    ),
                    None,
                )
                if mapped is None:
                    mapped = config.agent_map.get("_default")
                if mapped is not None:
                    if agent_name is not None and agent_name != mapped:
                        overridden_agents.add((agent_name, mapped))
                    agent_name = mapped
                    row[AGENT_REF_KEY_NAME] = {"name": agent_name}

            # Fan-out: run this row once per listed agent (cross-product). Otherwise a single
            # target — the row's agent when known, else deferred to task_source resolution.
            targets: List[Optional[str]]
            if config.fan_out and basis is not None and basis in config.fan_out:
                targets = list(config.fan_out[basis])
            elif agent_name is not None:
                targets = [agent_name]
            elif row.get(TASK_SOURCE_KEY_NAME) is not None:
                targets = [None]
            else:
                row_idxs_missing_agent_ref.append(row_idx)
                continue

            # Responses create params
            row[RESPONSES_CREATE_PARAMS_KEY_NAME] = (
                row[RESPONSES_CREATE_PARAMS_KEY_NAME] | responses_create_params_overrides
            )

            # Stamp the run-level skills_ref onto the row so it is sent to the agent in the
            # /run request body and propagated to results. The source dataset stays untouched.
            if skills_ref_dict is not None:
                row[SKILLS_REF_KEY_NAME] = skills_ref_dict

            # Resolve task index. Honor a caller-provided value when present (e.g. when an
            # upstream slicer has stamped a globally-stable index across chunks so that
            # subsequent /aggregate_metrics groupby unions chunks correctly); otherwise dedupe
            # identical input rows to the same task index as before.
            if TASK_INDEX_KEY_NAME not in row:
                row[TASK_INDEX_KEY_NAME] = row_to_task_idx.setdefault(row_str, len(row_to_task_idx))

            base_row = row
            for target in targets:
                # num_repeats keys match either side of a re-route: the dispatched agent (the
                # fan-out/agent_map target) or the row's original routing key (its agent_ref.name
                # or task_source as written in the data). The dispatched agent wins when both have
                # entries, so `agent_map={source: agent}` composes with `num_repeats={source: k}`.
                # Dict-form misses batch into one consolidated raise after the loop.
                repeat_keys = [k for k in dict.fromkeys((target, basis)) if k is not None]
                agents_seen.update(repeat_keys)
                if fixed_num_repeats is not None:
                    row_num_repeats = fixed_num_repeats
                elif (matched := next((k for k in repeat_keys if k in per_agent_repeats), None)) is not None:
                    row_num_repeats = per_agent_repeats[matched]
                elif default_repeats is not None:
                    row_num_repeats = default_repeats
                else:
                    agents_missing_from_num_repeats.add(" / ".join(repeat_keys))
                    continue

                for _ in range(row_num_repeats):
                    row = base_row.copy()
                    # Restamp only when fan-out routes this copy somewhere else; otherwise keep
                    # the row's agent_ref dict byte-for-byte (it may carry extra fields like type).
                    if target is not None and (row.get(AGENT_REF_KEY_NAME) or {}).get("name") != target:
                        row[AGENT_REF_KEY_NAME] = {"name": target}

                    # Resolve rollout index
                    row[ROLLOUT_INDEX_KEY_NAME] = task_idx_to_rollout_idx[row[TASK_INDEX_KEY_NAME]]
                    task_idx_to_rollout_idx[row[TASK_INDEX_KEY_NAME]] += 1

                    if config.num_repeats_add_seed:
                        row[RESPONSES_CREATE_PARAMS_KEY_NAME] = row[RESPONSES_CREATE_PARAMS_KEY_NAME].copy()
                        metadata = row[RESPONSES_CREATE_PARAMS_KEY_NAME].setdefault("metadata", {})
                        extra_body = json.loads(metadata.get("extra_body", "{}"))
                        extra_body["seed"] = row[ROLLOUT_INDEX_KEY_NAME]
                        metadata["extra_body"] = json.dumps(extra_body)

                    rows.append(row)

        if overridden_agents:
            warnings.warn(
                "agent_map overrode agent_ref values already present in the data: "
                f"{sorted(overridden_agents)}. Prior to this release, +agent_name only filled rows "
                "missing an agent_ref; it now re-routes every row.",
                stacklevel=2,
            )

        if row_idxs_missing_agent_ref:
            raise ValueError(
                f"No agent specified for rows {row_idxs_missing_agent_ref}. Provide +agent_name (or "
                "+agent_map with a _default entry), or include agent_ref or task_source in the data."
            )

        if agents_missing_from_num_repeats:
            raise ValueError(
                f"num_repeats dict has no entry for routing keys {sorted(agents_missing_from_num_repeats)} "
                f"and no '_default' fallback. Listed keys: {sorted(per_agent_repeats)}"
            )

        unknown_agents = set(per_agent_repeats) - agents_seen
        if unknown_agents:
            warnings.warn(
                f"num_repeats dict contains agent names that never appeared in input rows "
                f"(possible typo?): {sorted(unknown_agents)}",
                stacklevel=2,
            )

        return rows

    def _load_from_cache(
        self, config: RolloutCollectionConfig
    ) -> Tuple[List[Dict], List[Dict], List[Dict], List[List[str]]]:
        with config.materialized_jsonl_fpath.open() as f:
            original_input_rows = list(map(orjson.loads, tqdm(f, desc="Reading materialized input rows")))
        with Path(config.output_jsonl_fpath).open("rb") as f:
            result_strs = [[line.strip()] for line in tqdm(f, desc="Reading existing output rows")]
        results = [orjson.loads(p[0]) for p in result_strs]

        get_key = lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])

        # Successes (and any legacy '-failed' rows written by pre-fix Gym
        # builds) live in the main jsonl. They short-circuit dispatch.
        successes_seen = set(map(get_key, results))

        # Sidecar: one row per non-kill_shaped failure attempt. Count attempts
        # per key + flag terminal rows so chain-hop 2 retries the right ones.
        failures_fpath = failures_path_for(Path(config.output_jsonl_fpath))
        attempts_by_key: Counter = Counter()
        terminal_keys: set = set()
        if failures_fpath.exists():
            with failures_fpath.open("rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    fr = orjson.loads(line)
                    if TASK_INDEX_KEY_NAME not in fr or ROLLOUT_INDEX_KEY_NAME not in fr:
                        continue
                    k = (fr[TASK_INDEX_KEY_NAME], fr[ROLLOUT_INDEX_KEY_NAME])
                    attempts_by_key[k] += 1
                    if fr.get(NG_TERMINAL_KEY):
                        terminal_keys.add(k)

        max_attempts = _get_max_rollout_attempts()
        maxed_out = {k for k, n in attempts_by_key.items() if n >= max_attempts}
        gated = successes_seen | terminal_keys | maxed_out

        input_rows = [row for row in original_input_rows if get_key(row) not in gated]

        # Stamp the resume attempt (count of prior failures for this key) on actual retries so their
        # captured model calls are keyed separately from the prior attempt's (see
        # maybe_rollout_id_from_run_body). The first attempt (0) is left unstamped -> bare rollout id.
        for row in input_rows:
            attempt = attempts_by_key.get(get_key(row), 0)
            if attempt > 0:
                row[ATTEMPT_INDEX_KEY_NAME] = attempt

        key_to_row = dict(zip(map(get_key, original_input_rows), original_input_rows))
        rows = [key_to_row[get_key(result)] for result in results]

        print(
            f"""Resumed from cache. Found:
- {len(original_input_rows)} original input rows
- {len(rows)} rows already done (in main jsonl)
- {sum(attempts_by_key.values())} prior failure attempts ({len(attempts_by_key)} unique tasks) in sidecar
- {len(terminal_keys)} sidecar-terminal (timeout_exceeded / skipped) → not retried
- {len(maxed_out)} hit max_attempts={max_attempts} → not retried
- {len(input_rows)} rows that still need to be run"""
        )

        return input_rows, rows, results, result_strs

    async def run_from_config(self, config: RolloutCollectionConfig) -> Tuple[List[Dict]]:
        """Collect rollouts for a whole config. Wrapped in the run-scoped `job` span.

        This is the driver side of an evaluation run and the outermost span Gym produces,
        so every rollout it dispatches is a descendant of it. `job` is in the `default`
        preset but deliberately not in `per_rollout`, where each rollout is meant to be
        its own bounded root trace.
        """
        if not is_span_group_enabled(GymSpanGroup.JOB):
            return await self._run_from_config(config)
        with managed_span(GymSpanGroup.JOB, "gym.job"):
            return await self._run_from_config(config)

    async def _run_from_config(self, config: RolloutCollectionConfig) -> Tuple[List[Dict]]:
        output_fpath = Path(config.output_jsonl_fpath)
        failures_fpath = failures_path_for(output_fpath)

        # Create the output directory up front: every artifact this run writes (materialized inputs,
        # rollouts, failures sidecar, aggregate metrics) is derived from output_fpath and keeps its
        # parent, and the materialized-inputs write below is the first one. Keep this above that
        # write -- a user pointing --output at a not-yet-existing directory is the common case
        # outside a git clone.
        output_fpath.parent.mkdir(parents=True, exist_ok=True)

        if config.resume_from_cache and config.materialized_jsonl_fpath.exists() and output_fpath.exists():
            (
                input_rows,
                rows,
                results,
                result_strs,
            ) = self._load_from_cache(config)
            persisted_rows = list(rows)
            persisted_results = list(results)
        else:
            if config.resume_from_cache:
                if not output_fpath.exists():
                    print(f"Skipping resume_from_cache because output_fpath {output_fpath} doesn't exist!")
                if not config.materialized_jsonl_fpath.exists():
                    print(
                        f"Skipping resume_from_cache because materialized_jsonl_fpath {config.materialized_jsonl_fpath} doesn't exist!"
                    )
            else:
                print("Clearing output fpath since `resume_from_cache=False`!")

            rows: List[Dict] = []
            results: List[Dict] = []
            persisted_rows: List[Dict] = []
            persisted_results: List[Dict] = []

            input_rows = self._preprocess_rows_from_config(config)
            # Returned rows are sorted by (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME])

            # Resolve task_source rows to agents BEFORE the materialized write: materialized
            # inputs are the run-scoped artifact and must carry the resolved agent_ref (custom
            # drivers, e.g. gdpval's multistage orchestrator, read it from there). Guarded so
            # legacy agent_ref-only runs never need the head server at this point.
            if any(
                (r.get(AGENT_REF_KEY_NAME) or {}).get("name") is None and r.get(TASK_SOURCE_KEY_NAME) is not None
                for r in input_rows
            ):
                self.resolve_task_sources(input_rows, self.setup_server_client().global_config_dict)

            with config.materialized_jsonl_fpath.open("wb") as f:
                for row in tqdm(input_rows, desc="Writing materialized rows"):
                    f.write(orjson.dumps(row) + b"\n")

            output_fpath.unlink(missing_ok=True)
            failures_fpath.unlink(missing_ok=True)

        semaphore = nullcontext()
        if config.num_samples_in_parallel:
            print(f"Querying with {config.num_samples_in_parallel} concurrent requests")
            semaphore = Semaphore(config.num_samples_in_parallel)

        # Resolve capture dirs once so each rollout's captured model calls can be folded
        # into its record below (uniform across agents; no-op when capture is off / dirs absent).
        global_config = get_global_config_dict()
        capture_dirs = model_call_capture_dirs_from_config(global_config)
        observability_enabled = observability_enabled_from_config(global_config)
        # Resolve the training-token store directory once.
        # Training capture is independent of evaluation capture.
        # An empty result disables training-token capture.
        token_capture_dirs = token_id_capture_dirs_from_config(global_config)
        # The finalizer reads and freezes records through this source.
        # The source is absent when capture or response rebuilding is disabled.
        # A framework-owned transport may rebuild through its own source.
        # The sink still records captures when Gym does not rebuild.
        # Reruns still clear deterministic rollout ids before dispatch.
        token_source = None
        owned_token_source = None
        token_capture_config = TokenIdCaptureConfig.model_validate(global_config)
        if token_capture_config.enabled and token_capture_config.token_id_capture.rebuild_response:
            token_source = installed_token_source()
            if token_source is None and token_capture_dirs:
                token_source = TokenCaptureStore(token_capture_dirs[0])
            if isinstance(token_source, TokenCaptureStore):
                owned_token_source = token_source

        # Clear only rows about to be dispatched, after resume has assigned retry suffixes. This also
        # removes a kill-shaped attempt's partial capture when its rollout-attempt id is reused.
        if capture_dirs:
            print("Clearing existing model-call captures for rollouts being dispatched")
            clear_model_call_captures_for_rollouts(input_rows, capture_dirs)
        token_capture_rows = [
            row
            for row in input_rows
            if token_id_capture_enabled_for_agent(global_config, (row.get(AGENT_REF_KEY_NAME) or {}).get("name"))
        ]
        if token_capture_config.token_id_capture.rebuild_response and token_capture_rows and token_source is None:
            raise ValueError(
                "Token capture response rebuilding requires a TokenSource in the rollout-collector process. "
                "Call install_token_source before starting collection or configure token_id_capture.dir."
            )
        if token_capture_dirs and token_capture_rows:
            # Token stores append under deterministic rollout ids.
            # Clear stale records to avoid merging different attempts.
            print("Clearing existing token captures for rollouts being dispatched")
            clear_token_captures_for_rollouts(token_capture_rows, token_capture_dirs)

        # Stop a run that produces mostly masked captures.
        finalized_count = 0
        masked_count = 0
        mask_reasons: Counter = Counter()
        warned_malformed_rollout_id = False

        # Intermediate status printing
        pcts_to_print = list(range(1, 100)) + [99.5, 100]
        agent_name_to_metrics = defaultdict(Counter)
        agent_name_to_counts = defaultdict(int)
        counts_left = Counter(r[AGENT_REF_KEY_NAME]["name"] for r in input_rows)
        dispatched_per_agent = Counter(counts_left)
        start_time = time()

        if config.route_failures_to_sidecar:
            print(
                "route_failures_to_sidecar is on: a failed agent /run becomes a sidecar row instead of "
                "ending the run, and its rollout leaves the score.",
                flush=True,
            )

        results_file = output_fpath.open("ab")
        failures_file = failures_fpath.open("ab")
        failure_counts: Counter = Counter()
        for future in self._run_examples_with_metadata(
            input_rows,
            semaphore=semaphore,
            route_failures_to_sidecar=config.route_failures_to_sidecar,
        ):
            completed = await future
            row, result, rollout_latency_ms = completed.row, completed.result, completed.rollout_latency_ms

            result[TASK_INDEX_KEY_NAME] = row[TASK_INDEX_KEY_NAME]
            result[ROLLOUT_INDEX_KEY_NAME] = row[ROLLOUT_INDEX_KEY_NAME]
            result[AGENT_REF_KEY_NAME] = row[AGENT_REF_KEY_NAME]
            if TASK_SOURCE_KEY_NAME in row:
                result[TASK_SOURCE_KEY_NAME] = row[TASK_SOURCE_KEY_NAME]
            if SKILLS_REF_KEY_NAME in row:
                result[SKILLS_REF_KEY_NAME] = row[SKILLS_REF_KEY_NAME]
            if ATTEMPT_INDEX_KEY_NAME in row:
                result[ATTEMPT_INDEX_KEY_NAME] = row[ATTEMPT_INDEX_KEY_NAME]
            if ROLLOUT_ID_KEY_NAME in row:
                # Capture readback recomputes the id from the finished record.
                # Preserve an explicit id on the result just like the indices.
                result[ROLLOUT_ID_KEY_NAME] = row[ROLLOUT_ID_KEY_NAME]

            no_persist = bool(result.get(NG_NO_PERSIST_KEY))
            failure_class = result.get(NG_FAILURE_CLASS_KEY)
            # No rollout happened, so there is nothing to capture, tokenize or average.
            no_result = failure_class in _NO_RESULT_FAILURE_CLASSES

            # Fold this rollout's captured model calls into its record (uniform across agents; no-op
            # when capture is off). Never alters the harness output/reward already in `result`.
            if capture_dirs and not no_result:
                merge_model_call_capture_into_record(
                    result,
                    capture_dirs,
                    include_payloads=not _has_observation_gap(result, "multimodal_history_redacted"),
                )

            if "ng_model_call_capture" in result or "ng_agent_observations" in result or NG_TRAJECTORY_KEY in result:
                _attach_trajectory_record(row, result)

            # Assembles ng_perf from ng_trajectory when observability is enabled.
            _attach_ng_perf(result, observability_enabled=observability_enabled, rollout_latency_ms=rollout_latency_ms)

            # Freeze and rebuild tokens only for participating agents.
            # This step does not retire the frozen snapshot.
            # It leaves harness output and reward unchanged.
            # Direct callers of run_examples finalize each record themselves.
            token_capture_build = None
            if not no_result and token_id_capture_enabled_for_agent(
                global_config,
                (row.get(AGENT_REF_KEY_NAME) or {}).get("name"),
            ):
                token_capture_build = await finalize_rollout_token_capture(result, token_source)
                if token_capture_build is not None:
                    finalized_count += 1
                    if token_capture_build.get(MASK_SAMPLE_KEY):
                        masked_count += 1
                        # Aggregate available reasons for the abort message.
                        build_metrics = token_capture_build.get("metrics") or {}
                        if build_metrics.get("capture_incomplete"):
                            mask_reasons["capture_incomplete"] += 1
                        if build_metrics.get("unresolved_parent_calls"):
                            mask_reasons["unresolved_parent_calls"] += 1
                        build_error = token_capture_build.get("error") or build_metrics.get("error")
                        if build_error:
                            mask_reasons[str(build_error)] += 1
                    settings = token_capture_config.token_id_capture
                    if (
                        settings.max_mask_fraction is not None
                        and finalized_count >= settings.mask_fraction_min_samples
                        and masked_count / finalized_count > settings.max_mask_fraction
                    ):
                        raise RuntimeError(
                            f"{masked_count}/{finalized_count} finalized rollouts "
                            f"({masked_count / finalized_count:.1%}) are masked, exceeding "
                            f"token_id_capture.max_mask_fraction={settings.max_mask_fraction}. "
                            f"Mask reasons: {dict(mask_reasons)}. Aborting instead of collecting "
                            "mostly token-less data."
                        )

            rows.append(row)
            results.append(result)
            serialized = orjson.dumps(result)

            if no_persist:
                # kill_shaped: don't write anywhere. Set-difference on resume
                # naturally re-dispatches; per-task timeout bounds wallclock.
                pass
            elif failure_class is not None:
                # Non-kill_shaped failure → sidecar. The aggregator only reads
                # the main jsonl, so this keeps win-rate uncontaminated.
                failure_counts[failure_class] += 1
                # Every dropped rollout says so as it happens, whichever layer classified it.
                # tqdm.write keeps the line off the progress bar it would otherwise collide with.
                detail = str(result.get("_ng_failure_message") or result.get("error") or "")[:200]
                tqdm.write(
                    "🚨 [rollout_collection] rollout dropped from the score: "
                    f"row={json.dumps(_rollout_request_debug_summary(row), sort_keys=True)} "
                    f"class={failure_class} error={detail}"
                )
                failures_file.write(serialized + b"\n")
                failures_file.flush()
            else:
                # Success → main jsonl.
                results_file.write(serialized + b"\n")
                results_file.flush()
                persisted_rows.append(row)
                persisted_results.append(result)
                try:
                    rollout_id = maybe_rollout_id_from_run_body(result)
                except (TypeError, ValueError) as error:
                    # Preserve capture evidence when the rollout id is invalid.
                    rollout_id = None
                    if not warned_malformed_rollout_id:
                        warned_malformed_rollout_id = True
                        warnings.warn(
                            f"a result carries a malformed rollout id ({error}); "
                            "its token capture will not be retired.",
                            stacklevel=2,
                        )
                if rollout_id is not None and capture_build_can_retire(token_capture_build):
                    os.fsync(results_file.fileno())
                    await retire_rollout_token_capture(rollout_id, token_source, token_capture_build)

            counts_left[row[AGENT_REF_KEY_NAME]["name"]] -= 1
            if counts_left[row[AGENT_REF_KEY_NAME]["name"]] <= 0:
                counts_left.pop(row[AGENT_REF_KEY_NAME]["name"])

            agent_name = result["agent_ref"]["name"]
            if not no_result:
                # An infrastructure failure is not a score of zero, and not a sample either.
                metrics = agent_name_to_metrics[agent_name]
                metrics.update(
                    {k: v for k, v in result.items() if isinstance(v, (int, float)) and not k.startswith("_")}
                )
                agent_name_to_counts[agent_name] += 1

            current_pct = 100 * len(results) / len(input_rows)
            if pcts_to_print and current_pct >= pcts_to_print[0]:
                while pcts_to_print and current_pct >= pcts_to_print[0]:
                    pcts_to_print.pop(0)

                time_taken_s = time() - start_time
                time_taken = timedelta(seconds=int(time_taken_s))
                rollouts_per_min = len(results) / (time_taken_s / 60)
                print_str = f"Finished {len(results)} / {len(input_rows)} rollouts ({int(current_pct)}%) in {time_taken} ({rollouts_per_min:.2f} rollouts/min). "

                top_left = counts_left.most_common()
                top_left_str = "\n".join(f"{i + 1}. {k}: {v}" for i, (k, v) in enumerate(top_left))
                print_str += f"""Examples left:
{top_left_str}
"""
                for agent_name in sorted(agent_name_to_metrics):
                    metrics = agent_name_to_metrics[agent_name]
                    agent_total_samples = dispatched_per_agent[agent_name]
                    agent_sample_pct = 100 * agent_name_to_counts[agent_name] / agent_total_samples
                    avg_metrics = {k: v / agent_name_to_counts[agent_name] for k, v in metrics.items()}
                    print_str += f"""Found {agent_name_to_counts[agent_name]} / {agent_total_samples} ({agent_sample_pct:.2f}%) rollouts for `{agent_name}`.
{json.dumps(avg_metrics, indent=4)}
"""
                # Use tqdm.write here so we can print properly with tqdm being used.
                tqdm.write(print_str)

                if get_exporters():
                    step_metrics = {"progress/total/rollouts_per_min": rollouts_per_min}
                    for agent_name, metrics in agent_name_to_metrics.items():
                        step_metrics[f"progress/{agent_name}/reward"] = round(
                            100 * metrics["reward"] / agent_name_to_counts[agent_name], 2
                        )
                        step_metrics[f"progress/{agent_name}/reward_lower_bound"] = round(
                            100 * metrics["reward"] / (counts_left[agent_name] + agent_name_to_counts[agent_name]), 2
                        )

                    export_metrics(step_metrics, step=int(current_pct))

        results_file.close()
        failures_file.close()

        if input_rows and not persisted_results:
            raise RuntimeError(
                f"None of the {len(input_rows)} dispatched rollouts produced a result "
                f"{dict(failure_counts)}. Inspect {failures_fpath}; the run has no score to report."
            )
        if owned_token_source is not None:
            await owned_token_source.close()

        if config.upload_rollouts and get_exporters():  # pragma: no cover
            print("Uploading rollouts. This may take a few minutes if your data is large.")
            export_rollouts([_rollout_for_export(result) for result in results])

        print("Sorting results to ensure consistent ordering")
        rows.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
        results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
        persisted_rows.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))
        persisted_results.sort(key=lambda r: (r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]))

        # Compute and write aggregate metrics via /aggregate_metrics using only the
        # rows written to the main rollouts jsonl so runtime aggregation matches
        # `gym eval aggregate`.
        counted: List[Dict] = []
        if config.disable_aggregation:
            print(
                "Skipping aggregate-metrics computation because disable_aggregation=True. "
                "Run `gym eval aggregate` after all shards finish to compute the global metrics."
            )
            aggregate_metrics_fpath = None
        else:
            print("Computing aggregate metrics")
            counted[:] = _failure_rows_counted_as_zero(
                [failures_fpath],
                config.count_failure_classes_as_zero,
                {(r[TASK_INDEX_KEY_NAME], r[ROLLOUT_INDEX_KEY_NAME]) for r in persisted_results},
            )
            if config.count_failure_classes_as_zero:
                print(
                    f"Counting {len(counted)} failure row(s) as scored zeros: {config.count_failure_classes_as_zero}"
                )
            aggregate_metrics_fpath = await self._call_aggregate_metrics(
                persisted_results + counted, persisted_rows + counted, output_fpath
            )

        expected_rollouts = (
            sum(1 for _ in config.materialized_jsonl_fpath.open("rb"))
            if config.materialized_jsonl_fpath.exists()
            else len(input_rows) + len(persisted_results)
        )
        scored_rollouts = len(persisted_results) + len(counted)
        coverage = _coverage_report(expected_rollouts, scored_rollouts, failure_counts, failures_fpath)
        if get_exporters():  # pragma: no cover
            export_metrics(
                {
                    "coverage/expected": expected_rollouts,
                    "coverage/scored": scored_rollouts,
                    "coverage/missing": expected_rollouts - scored_rollouts,
                }
            )

        print(f"""Finished rollout collection! View results at:
Fully materialized inputs: {config.materialized_jsonl_fpath}
Rollouts: {output_fpath}
Aggregate metrics: {aggregate_metrics_fpath}{coverage}""")

        if not config.disable_aggregation and not config.disable_health_check:
            from nemo_gym.rollout_health import format_health_report, run_health_checks

            try:
                health_result = await asyncio.to_thread(
                    run_health_checks,
                    output_fpath,
                    workers=config.health_check_workers,
                    ignored_checks=config.health_check_ignored_checks,
                )
            except Exception:
                logger.exception(
                    "Rollout health checks failed after collection; rollout artifacts are still available."
                )
            else:
                print(format_health_report(health_result))

        return results

    async def _call_aggregate_metrics(
        self,
        results: List[Dict],
        rows: List[Dict],
        output_fpath: Path,
    ) -> Optional[Path]:
        """Call /aggregate_metrics on each agent server after rollouts complete.

        Writes a single _aggregate_metrics.json with one entry per agent (same shape
        as the old _agent_metrics.json). Returns the file path.
        """
        if not results:
            return None

        # Group results by agent name
        agent_results: Dict[str, List[Dict]] = {}
        for row, result in zip(rows, results):
            agent_name = (row.get(AGENT_REF_KEY_NAME) or {}).get("name")
            if not agent_name:
                continue
            agent_results.setdefault(agent_name, []).append(result)

        server_client = self.setup_server_client()

        async def _fetch_agent_metrics(agent_name: str, agent_result_list: List[Dict]) -> Dict:
            # Strip heavyweight fields before sending, but preserve response.usage
            stripped = []
            for r in agent_result_list:
                entry = {
                    k: v
                    for k, v in r.items()
                    if k
                    not in (
                        "response",
                        "responses_create_params",
                        "ng_agent_observations",
                        "ng_model_call_capture",
                        NG_TRAJECTORY_KEY,
                    )
                }
                usage = (r.get("response") or {}).get("usage")
                if usage:
                    entry["response"] = {"usage": usage}
                stripped.append(entry)

            agg_request = AggregateMetricsRequest(verify_responses=stripped)
            agg_response = await server_client.post(
                server_name=agent_name,
                url_path="/aggregate_metrics",
                json=agg_request,
            )
            await raise_for_status(agg_response)
            agg_result = AggregateMetrics.model_validate(await get_response_json(agg_response))

            agent_entry = {
                AGENT_REF_KEY_NAME: {"name": agent_name},
                "agent_metrics": agg_result.agent_metrics,
                "key_metrics": agg_result.key_metrics,
                "group_level_metrics": agg_result.group_level_metrics,
                "repeat_level_metrics": agg_result.repeat_level_metrics,
            }
            if agg_result.perf_summary is not None:
                agent_entry["perf_summary"] = agg_result.perf_summary
            return agent_entry

        all_agent_metrics: List[Dict] = []
        tasks = [_fetch_agent_metrics(name, results_list) for name, results_list in agent_results.items()]
        for coro in asyncio.as_completed(tasks):
            agent_entry = await coro
            all_agent_metrics.append(agent_entry)

            agent_name = agent_entry[AGENT_REF_KEY_NAME]["name"]
            key_metrics = agent_entry.get("key_metrics", {})
            print(f"\nKey metrics for {agent_name}:\n" + json.dumps(key_metrics, indent=4))

        primitive_types = (bool, int, float, str, type(None))
        metrics_to_log = dict()
        for agent_entry in all_agent_metrics:
            agent_name = agent_entry[AGENT_REF_KEY_NAME]["name"]
            metrics_to_log.update(
                {
                    f"{agent_name}/{k}": v
                    for k, v in agent_entry["agent_metrics"].items()
                    if isinstance(v, primitive_types)
                }
            )
            metrics_to_log.update(
                {
                    f"key_metrics/{agent_name}/{k}": v
                    for k, v in agent_entry["key_metrics"].items()
                    if isinstance(v, primitive_types)
                }
            )

        export_metrics(metrics_to_log)

        # Write single file with all agents
        metrics_fpath = aggregate_metrics_path_for(output_fpath)
        metrics_fpath.write_bytes(orjson.dumps(all_agent_metrics, option=orjson.OPT_INDENT_2))

        return metrics_fpath

    @staticmethod
    def resolve_task_sources(examples: List[Dict], global_config_dict: DictConfig) -> None:
        """Stamp an agent_ref onto every row that carries only a task_source.

        task_source names the config instance that declared the row's dataset. Resolution is
        :func:`~nemo_gym.global_config.resolve_dataset_agent` — the same rules benchmark
        discovery uses, so dispatch can never disagree with the listing. Conflicting `agent:`
        pins across one instance's datasets are a hard error (rows carry only the instance
        name), as are unknown/non-routable instances; +agent_map is the disambiguator.

        Rows that already have an agent_ref are left untouched, so this is a no-op on legacy
        datasets and on already-resolved (materialized) rows. Runs before any dispatch.
        """
        legacy_routed = sum(
            1
            for row in examples
            if (row.get(AGENT_REF_KEY_NAME) or {}).get("name") is not None and row.get(TASK_SOURCE_KEY_NAME) is None
        )
        if legacy_routed:
            warnings.warn(
                f"{legacy_routed} rows routed via their baked-in agent_ref (no task_source). This "
                "legacy path is deprecated: re-collate the dataset with current Gym to produce "
                "task_source-routed rows, or re-route explicitly with +agent_map.",
                DeprecationWarning,
                stacklevel=2,
            )

        unresolved = {
            ts
            for row in examples
            if (row.get(AGENT_REF_KEY_NAME) or {}).get("name") is None
            and (ts := row.get(TASK_SOURCE_KEY_NAME)) is not None
        }
        if not unresolved:
            return

        resolution: Dict[str, str] = {}
        errors: List[str] = []
        for ts in sorted(unresolved):
            block = global_config_dict.get(ts)
            if block is None:
                close = get_close_matches(ts, list(global_config_dict.keys()), n=1)
                errors.append(
                    f"{ts!r}: not in the running config" + (f" (did you mean {close[0]!r}?)" if close else "")
                )
            elif not isinstance(block, DictConfig):
                errors.append(f"{ts!r}: not a server instance")
            elif "responses_api_agents" in block or "resources_servers" in block:
                pins = dataset_agent_pins(global_config_dict, ts)
                if len(pins) > 1:
                    errors.append(
                        f"{ts!r}: its datasets pin conflicting agents ({sorted(pins)}), which task_source "
                        f"alone cannot tell apart; pass +agent_map={{{ts}: <agent>}} to pick one"
                    )
                    continue
                try:
                    resolution[ts] = resolve_dataset_agent(global_config_dict, ts, pin=pins[0] if pins else None)
                except ConfigError as e:
                    errors.append(f"{ts!r}: {e}")
            else:
                errors.append(f"{ts!r}: instance is not an agent or resources server (datasets cannot route here)")

        if errors:
            raise ValueError("Cannot resolve task_source to an agent: " + "; ".join(errors))

        for row in examples:
            if (row.get(AGENT_REF_KEY_NAME) or {}).get("name") is None:
                ts = row.get(TASK_SOURCE_KEY_NAME)
                if ts is not None:
                    row[AGENT_REF_KEY_NAME] = {"name": resolution[ts]}

    @staticmethod
    def _validate_agent_names(examples: List[Dict], global_config_dict: DictConfig) -> None:
        """Fail before any dispatch when a row names an agent absent from the running config.

        Without this, the first bad row dies mid-collection with a raw omegaconf ConfigKeyError
        after valid rows have already been dispatched.
        """
        requested = {name for row in examples if (name := (row.get(AGENT_REF_KEY_NAME) or {}).get("name")) is not None}
        available = {
            str(name)
            for name, block in global_config_dict.items()
            if isinstance(block, DictConfig) and "responses_api_agents" in block
        }
        unknown = sorted(requested - available)
        if not unknown:
            return
        hints = []
        for name in unknown:
            # Naming a non-agent instance (e.g. a resources server via agent_map) is as fatal as a
            # typo: /run only exists on agent servers.
            if name in global_config_dict:
                hints.append(f"{name!r} (exists but is not an agent instance)")
                continue
            close = get_close_matches(name, available, n=1)
            hints.append(f"{name!r}" + (f" (did you mean {close[0]!r}?)" if close else ""))
        raise ValueError(
            f"Rows reference agents not present in the running config: {', '.join(hints)}. "
            "Include the agent's config in the run, or re-route with +agent_map/+agent_name."
        )

    @staticmethod
    def _validate_agent_pairings(examples: List[Dict], global_config_dict: DictConfig) -> None:
        """Fail before dispatch when a row points to agent incompatible with the resources server it runs on."""
        if pairing_override_enabled(global_config_dict):
            return
        routes = {
            (name, row.get(TASK_SOURCE_KEY_NAME))
            for row in examples
            if (name := (row.get(AGENT_REF_KEY_NAME) or {}).get("name")) is not None
        }
        rejected: Dict[Tuple[str, str], str] = {}
        for agent, task_source in routes:
            # indexing by name is safe because we already validated the agent names
            agents = global_config_dict[agent][AGENT_SERVER_TYPE_KEY_NAME]
            if not agents:
                continue
            # this is only one agent type per agent instance and the check it elsewhere
            agent_type = str(next(iter(agents)))
            reference = OmegaConf.select(agents[agent_type], "resources_server")
            bound = reference.get("name") if isinstance(reference, DictConfig) else None
            for verifier, relation in (
                (task_source, "their task_source"),
                (bound, "the resources server it runs against"),
            ):
                allowed = allowed_agents_for(global_config_dict, verifier)
                if allowed is None or agent_type in allowed:
                    continue
                rejected[(agent, str(verifier))] = (
                    f"  - rows routed to '{agent}' run '{agent_type}', but {relation} "
                    f"'{verifier}' accepts only: {', '.join(allowed)}"
                )
        if not rejected:
            return
        raise ValueError(
            "Rows would be scored by a verifier that does not accept the agent running them:\n"
            + "\n".join(line for _, line in sorted(rejected.items()))
            + "\n\nRoute these rows to an agent running one of the accepted types, or pass "
            f"--allow-unsupported-pairing (or set {ALLOW_UNSUPPORTED_PAIRING_ENV_VAR_NAME}=1) to bypass the check."
        )

    def _run_examples_with_metadata(
        self,
        examples: List[Dict],
        head_server_config: Optional[BaseServerConfig] = None,
        semaphore: Optional[Semaphore] = None,
        route_failures_to_sidecar: bool = False,
    ) -> Iterator[Future]:  # pragma: no cover
        """
        Internal dispatch shared by ``run_examples`` and Gym's own collection paths.

        Identical contract to ``run_examples``, but each future resolves to a ``_CompletedRollout``
        that carries ``rollout_latency_ms`` alongside the raw ``/run`` result instead of inside it,
        so internal-only timing never has to be smuggled through (and stripped back out of) a dict
        that a direct caller of ``run_examples`` could also observe.
        """
        server_client = self.setup_server_client(head_server_config)
        self.resolve_task_sources(examples, server_client.global_config_dict)
        self._validate_agent_names(examples, server_client.global_config_dict)
        self._validate_agent_pairings(examples, server_client.global_config_dict)
        semaphore = semaphore or nullcontext()

        async def _post_subroutine(row: Dict) -> _CompletedRollout:
            async with semaphore:
                started_at = time()
                res = None
                try:
                    res = await server_client.post(server_name=row["agent_ref"]["name"], url_path="/run", json=row)
                    await raise_for_status(res)
                    result = await get_response_json(res)
                    # Independently-measured task wall-clock (ng_perf.total_latency_ms), not derived
                    # from summed model-call/tool latencies to account for additional overhead.
                    rollout_latency_ms = (time() - started_at) * 1000
                    return _CompletedRollout(row=row, result=result, rollout_latency_ms=rollout_latency_ms)
                except Exception as e:
                    print(
                        "[rollout_collection] /run failed "
                        f"status={getattr(res, 'status', None)} "
                        f"row={json.dumps(_rollout_request_debug_summary(row), sort_keys=True)}",
                        flush=True,
                    )
                    if not route_failures_to_sidecar or not isinstance(e, _RUN_FAILURE_ERRORS):
                        raise
                    if res is not None:
                        res.release()
                    # The status comes from the error when it carries one, and from the response
                    # when the body was the part that failed.
                    status = getattr(e, "status", None) or getattr(res, "status", None)
                    return _CompletedRollout(
                        row=row, result=_agent_request_failure_row(e, status), rollout_latency_ms=None
                    )

        return tqdm.as_completed(
            map(_post_subroutine, examples),
            desc="Collecting rollouts",
            miniters=10,
            total=len(examples),
            maxinterval=60,
        )

    def run_examples(
        self,
        examples: List[Dict],
        head_server_config: Optional[BaseServerConfig] = None,
        semaphore: Optional[Semaphore] = None,
        route_failures_to_sidecar: bool = False,
    ) -> Iterator[Future]:  # pragma: no cover
        """
        We provide this function as a lower level interface for running rollout collection.

        Rows are dispatched as given: task_sources are resolved and agent names validated here,
        but run-level knobs (``agent_map``, ``fan_out``, ``num_repeats``) are NOT applied — call
        ``preprocess_examples`` first if you need them.

        ``route_failures_to_sidecar`` makes a failed `/run` a failure row instead of an exception
        that ends every rollout still in flight. It defaults off because those rollouts then leave
        the score.

        Every future resolves to exactly the ``(row, result)`` pair Gym's own `/run` endpoint
        returned — no Gym-private fields are ever added to ``result``.
        """

        async def _without_metadata(future: Future) -> Tuple[Dict, Dict]:
            completed = await future
            return completed.row, completed.result

        return map(
            _without_metadata,
            self._run_examples_with_metadata(
                examples,
                head_server_config=head_server_config,
                semaphore=semaphore,
                route_failures_to_sidecar=route_failures_to_sidecar,
            ),
        )

    def setup_server_client(
        self, head_server_config: Optional[BaseServerConfig] = None
    ) -> ServerClient:  # pragma: no cover
        return setup_server_client_utils(head_server_config)


class RolloutAggregationConfig(BaseNeMoGymCLIConfig):
    """
    Aggregate metrics across rollout shards produced by `gym eval run --no-serve +disable_aggregation=true`.

    Reads every JSONL file matching `input_glob`, computes aggregate metrics by POSTing to each
    agent server's `/aggregate_metrics` endpoint over the global union of records, and writes a
    single `<output_jsonl_fpath stem>_aggregate_metrics.json` next to the rollouts. By default
    also concatenates all shards into `output_jsonl_fpath`.

    Examples:

    ```bash
    gym eval aggregate \
        "+config_paths=[benchmarks/aime24/config.yaml,responses_api_models/vllm_model/configs/vllm_model.yaml]" \
        +input_glob='results/rollouts-rs*-chunk*.jsonl' \
        +output_jsonl_fpath=results/rollouts.jsonl
    ```
    """

    input_glob: str = Field(
        description=(
            "Glob pattern or comma-separated list of glob patterns matching the rollout shards "
            "to aggregate (e.g. 'results/rollouts-rs*-chunk*.jsonl' or "
            "'results/run1/rollouts.jsonl,results/run2/rollouts.jsonl'). Whitespace around "
            "commas is stripped. Duplicate matches across patterns are deduplicated."
        )
    )
    output_jsonl_fpath: str = Field(
        description=(
            "Path used to derive the aggregate-metrics output location "
            "('<stem>_aggregate_metrics.json' next to this path) and, when merge_shards=True, "
            "the merged-rollouts file."
        ),
    )
    merge_shards: bool = Field(
        default=True,
        description="Concatenate the matched shard JSONLs into output_jsonl_fpath alongside the metrics file.",
    )
    count_failure_classes_as_zero: List[str] = Field(
        default_factory=list,
        description=(
            "Failure classes from the failures sidecar to count in aggregate metrics, e.g. "
            "['agent_run_error'], so a failed rollout lands in the denominator. A row that carries "
            "no reward is scored zero for the metrics only; no artifact is changed."
        ),
    )
    disable_health_check: bool = Field(
        default=False,
        description="Skip post-aggregation rollout quality verification and report writing.",
    )
    health_check_workers: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of rollout-health worker processes (defaults to min(cpus, 8)).",
    )
    health_check_ignored_checks: List[str] = Field(
        default_factory=list,
        description="Health-check IDs to exclude from execution and verdict derivation.",
    )

    @field_validator("health_check_ignored_checks", mode="before")
    @classmethod
    def _validate_health_check_ignored_checks(cls, value):
        return _normalize_health_check_ignored_checks(value)


def loads_jsonl_line(raw, fpath, line_no: int):
    """Parse one JSONL line, raising a clean `ConfigError` (naming file + line) on malformed JSON."""
    try:
        return orjson.loads(raw)
    except orjson.JSONDecodeError as e:
        raise ConfigError(f"Malformed JSON in '{fpath}' at line {line_no}: {e}") from e


def _expand_input_glob(input_glob: str) -> List[str]:
    """Expand a glob-or-comma-separated-globs string into a sorted, deduplicated list of paths.

    Examples:
      'results/rollouts.jsonl' -> ['results/rollouts.jsonl'] (if it exists)
      'a/*.jsonl, b/*.jsonl'   -> matches of both patterns, deduplicated
    """
    patterns = [p.strip() for p in input_glob.split(",") if p.strip()]
    seen: Dict[str, None] = {}  # preserve insertion order while deduping
    for pattern in patterns:
        for path in sorted(glob_module.glob(pattern)):
            seen.setdefault(path, None)
    return list(seen)


class RolloutAggregationHelper(BaseModel):
    async def run_from_config(self, config: RolloutAggregationConfig) -> Optional[Path]:
        input_paths = _expand_input_glob(config.input_glob)
        if not input_paths:
            raise ConfigPathNotFoundError(f"No shards matched input_glob={config.input_glob!r}")
        print(f"Aggregating {len(input_paths)} shard(s):")
        for p in input_paths:
            print(f"  - {p}")

        results: List[Dict] = []
        for shard_path in input_paths:
            with open(shard_path, "rb") as f:
                for line_no, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    results.append(loads_jsonl_line(line, shard_path, line_no))
        print(f"Loaded {len(results)} rollout record(s) from {len(input_paths)} shard(s)")

        # Sort for deterministic aggregation ordering (matches run_from_config's post-collection sort)
        results.sort(key=lambda r: (r.get(TASK_INDEX_KEY_NAME), r.get(ROLLOUT_INDEX_KEY_NAME)))

        output_fpath = Path(config.output_jsonl_fpath)
        output_fpath.parent.mkdir(parents=True, exist_ok=True)

        if config.merge_shards:
            print(f"Merging shards into {output_fpath}")
            with output_fpath.open("wb") as out:
                for r in results:
                    out.write(orjson.dumps(r) + b"\n")

        failures_fpaths = [failures_path_for(Path(path)) for path in input_paths]
        scored_keys = {(r.get(TASK_INDEX_KEY_NAME), r.get(ROLLOUT_INDEX_KEY_NAME)) for r in results}
        counted = _failure_rows_counted_as_zero(
            failures_fpaths,
            config.count_failure_classes_as_zero,
            scored_keys,
        )
        if config.count_failure_classes_as_zero:
            print(f"Counting {len(counted)} failure row(s) as scored zeros: {config.count_failure_classes_as_zero}")

        # `_call_aggregate_metrics` only inspects each row's AGENT_REF_KEY_NAME, which results already carry.
        helper = RolloutCollectionHelper()
        scored = results + counted
        aggregate_metrics_fpath = await helper._call_aggregate_metrics(scored, scored, output_fpath)

        # The shards' own sidecars say which rollouts never made it into the files just scored.
        counted_keys = {(r.get(TASK_INDEX_KEY_NAME), r.get(ROLLOUT_INDEX_KEY_NAME)) for r in counted}
        dropped = Counter(
            row.get(NG_FAILURE_CLASS_KEY) or "unknown"
            for key, row in _latest_failure_rows(failures_fpaths).items()
            if key not in scored_keys and key not in counted_keys
        )
        scored_rollouts = len(results) + len(counted)
        coverage = _coverage_report(
            scored_rollouts + sum(dropped.values()),
            scored_rollouts,
            dropped,
            "the shards' _failures.jsonl sidecars",
        )
        if get_exporters():  # pragma: no cover
            export_metrics(
                {
                    "coverage/expected": scored_rollouts + sum(dropped.values()),
                    "coverage/scored": scored_rollouts,
                    "coverage/missing": sum(dropped.values()),
                }
            )

        print(f"""Finished rollout aggregation! View results at:
Merged rollouts: {output_fpath if config.merge_shards else "<not merged>"}
Aggregate metrics: {aggregate_metrics_fpath}{coverage}""")

        if not config.disable_health_check:
            from nemo_gym.rollout_health import format_health_report, run_health_checks

            try:
                health_result = await asyncio.to_thread(
                    run_health_checks,
                    output_fpath if config.merge_shards else [Path(path) for path in input_paths],
                    output_dir=output_fpath.parent,
                    workers=config.health_check_workers,
                    ignored_checks=config.health_check_ignored_checks,
                )
            except Exception:
                logger.exception(
                    "Rollout health checks failed after aggregation; aggregate artifacts are still available."
                )
            else:
                print(format_health_report(health_result))

        return aggregate_metrics_fpath


# Backward-compatibility shims (CLI refactor): these CLI entry points moved to `nemo_gym.cli.eval`.
# Re-exported lazily to avoid a circular import; accessing them emits a DeprecationWarning.
from nemo_gym.cli._compat import moved_attr_getter  # noqa: E402


__getattr__ = moved_attr_getter(
    __name__,
    {
        "collect_rollouts": "nemo_gym.cli.eval",
        "aggregate_rollouts": "nemo_gym.cli.eval",
    },
)
