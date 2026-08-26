# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data contracts shared by rollout-health checks and orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.global_config import (
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)


TASK_INDEX_KEY = TASK_INDEX_KEY_NAME
ROLLOUT_INDEX_KEY = ROLLOUT_INDEX_KEY_NAME
ROLLOUT_ID_KEY = ROLLOUT_ID_KEY_NAME
QUALITY_SUMMARY_FILENAME = "quality_summary.json"
ROLLOUT_VERDICTS_FILENAME = "rollout_verdicts.jsonl"

Verdict = Literal["healthy", "unhealthy", "unobserved"]


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
    # Runner-derived rollout verdicts grouped by task for task-level reduction.
    REPEAT_VERDICTS = "repeat_verdicts"
    # Runner-derived RolloutDigest objects grouped by task for task-level reduction.
    REPEAT_DIGESTS = "repeat_digests"


class CheckSpec(BaseModel):
    """Stable, self-describing health-check contract."""

    model_config = ConfigDict(frozen=True)

    id: str
    evaluation_scope: CheckScope
    subject: CheckSubject
    reads: frozenset[CheckInput]


class Finding(BaseModel):
    """Evidence emitted by a check. Checks never emit verdicts."""

    check: str
    subject: dict[str, int | str]
    locator: dict[str, int | str] | None = None
    detail: dict[str, Any] = Field(default_factory=dict)


class RolloutDigest(BaseModel):
    task_index: int | str
    rollout_index: int | str
    rollout_id: str
    verdict: Verdict
    findings: list[Finding]
    unobserved: list[str]
    capture_observed: bool
    policy_calls_observed: bool = False
    model_calls: int = 0
    successful_model_calls: int = 0
    model_call_errors: int = 0
    errors_by_status: dict[str, int] = Field(default_factory=dict)
    ended_on_error: bool = False
    duplicated_calls: int = 0
    transcript_prompt_tokens: int = 0
    transcript_completion_tokens: int = 0
    capture_prompt_tokens: int = 0
    capture_completion_tokens: int = 0


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
