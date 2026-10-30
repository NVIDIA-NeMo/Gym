# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data contracts shared by rollout-health checks and orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from nemo_gym.config.global_config import (
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.health.timeouts import RolloutTiming


TASK_INDEX_KEY = TASK_INDEX_KEY_NAME
ROLLOUT_INDEX_KEY = ROLLOUT_INDEX_KEY_NAME
ROLLOUT_ID_KEY = ROLLOUT_ID_KEY_NAME
QUALITY_SUMMARY_FILENAME = "quality_summary.json"
ROLLOUT_VERDICTS_FILENAME = "rollout_verdicts.jsonl"
# The key under the quality summary's ``run`` that names the rollouts file(s) it was computed from, relative to
# the folder it is written in: the reports keep one name per folder whatever the rollouts file is called.
QUALITY_SUMMARY_SOURCES_KEY = "rollouts"

Verdict = Literal["healthy", "unhealthy", "unobserved"]
# Where each limit a summary was judged by came from: the run's config, the run's own earlier summary, or an
# explicit override when it was rechecked.
LimitsSource = Literal["config", "persisted", "override"]


class RunHealthLimits(BaseModel):
    """The run-level failure limits, as a share of expected rollouts; ``None`` is no limit, the default."""

    model_config = ConfigDict(frozen=True)

    max_not_final_failure_fraction: float | None = Field(default=None, ge=0, le=1)
    max_final_failure_fraction: float | None = Field(default=None, ge=0, le=1)

    def summary(self, source: LimitsSource | Mapping[str, LimitsSource]) -> dict[str, Any]:
        """The limits as written to ``quality_summary.json``, so the verdict can be reproduced.

        ``source`` maps each limit to where it came from; a single source applies to every limit.
        """
        sources = {name: source if isinstance(source, str) else source[name] for name in type(self).model_fields}
        return {**self.model_dump(), "source": sources}


class RunHealthConfig(BaseModel):
    """The health-check settings every command that judges a run shares: one place for names and defaults."""

    health_check_max_not_final_failure_fraction: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Run health is unhealthy when not final (retryable) failures reach this share of expected rollouts. "
            "Unset by default: no limit."
        ),
    )
    health_check_max_final_failure_fraction: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description=(
            "Run health is unhealthy when final (not retried) failures reach this share of expected rollouts. "
            "Unset by default: no limit."
        ),
    )

    @property
    def run_health_limits(self) -> RunHealthLimits:
        return RunHealthLimits(
            max_not_final_failure_fraction=self.health_check_max_not_final_failure_fraction,
            max_final_failure_fraction=self.health_check_max_final_failure_fraction,
        )


class CheckScope(str, Enum):
    ROLLOUT = "rollout"
    TASK = "task"
    RUN = "run"


class CheckSubject(str, Enum):
    CHECK_EXECUTION = "check_execution"
    RECORD = "record"
    ROLLOUT = "rollout"
    AGENT_TURN = "agent_turn"
    MODEL_CALL = "model_call"
    TRAJECTORY_CAPTURE = "trajectory_capture"
    TASK = "task"
    RUN = "run"


class CheckInput(str, Enum):
    """Persisted or derived evidence required to evaluate a check."""

    # One parsed object from a rollout JSONL file.
    RECORD = "record"
    # The canonical TrajectoryRecord persisted under the rollout's ng_trajectory key.
    TRAJECTORY = "trajectory"
    # Canonical TrajectoryTurn objects from TrajectoryRecord.turns.
    AGENT_TURNS = "agent_turns"
    # Canonical model calls joined in memory to explicit TrajectoryTurn references.
    BOUND_CALLS = "bound_calls"
    # Canonical model calls joined to explicit AgentInvocation or TrajectoryTurn references.
    OWNED_MODEL_CALLS = "owned_model_calls"
    # Canonical model calls as captured, whether or not anything references them.
    # A call that failed carries no response id, so it can never be bound.
    OBSERVED_MODEL_CALLS = "observed_model_calls"
    # Runner-derived rollout verdicts grouped by task for task-level reduction.
    REPEAT_VERDICTS = "repeat_verdicts"
    # Runner-derived RolloutDigest objects grouped by task for task-level reduction.
    REPEAT_DIGESTS = "repeat_digests"
    # The run accounting: the _materialized_inputs.jsonl inventory, the _failures.jsonl sidecar and the identities
    # of the saved records, which a stopped run still holds in memory.
    RUN_ACCOUNTING = "run_accounting"


CALL_BINDING_INPUTS = frozenset({CheckInput.BOUND_CALLS, CheckInput.OWNED_MODEL_CALLS})


class CheckSpec(BaseModel):
    """Stable, self-describing health-check contract."""

    model_config = ConfigDict(frozen=True)

    id: str
    evaluation_scope: CheckScope
    subject: CheckSubject
    reads: frozenset[CheckInput]

    @field_validator("reads")
    @classmethod
    def _require_one_call_binding_view(cls, reads: frozenset[CheckInput]) -> frozenset[CheckInput]:
        if len(reads & CALL_BINDING_INPUTS) > 1:
            raise ValueError("a health check cannot read both turn-bound and invocation-owned call bindings")
        return reads


class Finding(BaseModel):
    """Evidence emitted by a check. Checks never emit verdicts."""

    check: str
    subject: dict[str, int | str]
    locator: dict[str, int | str] | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class RolloutDigest(BaseModel):
    task_index: int | str
    rollout_index: int | str
    # Multi-stage drivers (e.g. gdpval) persist one row per (task, rollout) per stage.
    stage_index: int | str | None = None
    rollout_id: str
    verdict: Verdict
    findings: list[Finding]
    unobserved: list[str]
    capture_observed: bool
    policy_calls_observed: bool = False
    model_calls: int = 0
    successful_model_calls: int = 0
    model_call_errors: int = 0
    model_call_errors_usage_unknown: int = 0
    errors_by_status: dict[str, int] = Field(default_factory=dict)
    ended_on_error: bool = False
    duplicated_calls: int = 0
    transcript_prompt_tokens: int = 0
    transcript_completion_tokens: int = 0
    capture_prompt_tokens: int = 0
    capture_completion_tokens: int = 0
    # What this rollout contributes to `run.stats.timeouts`.
    timing: RolloutTiming | None = None


class HealthCheckResult(BaseModel):
    summary: dict[str, Any]
    rollouts: list[RolloutDigest]
    summary_path: Path
    verdicts_path: Path


@dataclass(frozen=True, slots=True)
class _LineSlice:
    path: str
    offset: int
    length: int
    ordinal: int
    source_index: int
    line_number: int


@dataclass(frozen=True, slots=True)
class _WorkerInput:
    line: _LineSlice
    ignored_checks: frozenset[str]


@dataclass(frozen=True, slots=True)
class _AgentStep:
    locator: dict[str, int | str]
    has_message: bool
    has_tool_calls: bool
    model_call_refs: tuple[str, ...]

    @property
    def has_model_activity(self) -> bool:
        return self.has_message or self.has_tool_calls or bool(self.model_call_refs)


@dataclass(frozen=True, slots=True)
class _CallBindings:
    references: tuple[str, ...]
    matched_calls: tuple[dict[str, Any], ...]
    missing_references: tuple[str, ...]
    duplicated_references: tuple[tuple[str, int], ...]
    # What each reference resolved to, by `_call_reference_signature`: one call, none, or several.
    resolutions: Mapping[tuple[str, str, str, str], tuple[dict[str, Any], ...]] = field(default_factory=dict)

    @property
    def observed(self) -> bool:
        return bool(self.references)

    @property
    def complete(self) -> bool:
        return self.observed and not self.missing_references and not self.duplicated_references


@dataclass(frozen=True, slots=True)
class _TaskRepeat:
    rollout_index: int | str
    verdict: Verdict
    policy_calls_observed: bool
    successful_model_calls: int
