# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-episode artifacts: the record verify persists, from which a session-less verify re-grades.

Rollout data: keep the directory private.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from resources_servers.agentic_if.episode.judging import JudgeCall
from resources_servers.agentic_if.episode.outcome import ExecutionStatus
from resources_servers.agentic_if.if_grading.contracts import SessionEvidence


EPISODE_RECORD = "evidence.json"
RESULT_RECORD = "result.json"


def save_json(path: Path, payload: JsonValue) -> None:
    """Write ``payload`` atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    # JavaScript string slicing may leave a lone surrogate in a raw receipt.
    # JSON escapes preserve it losslessly even when the response view is invalid.
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


class BackendVerdict(BaseModel):
    """The task backend's own verify reply, or why there is none."""

    model_config = ConfigDict(extra="forbid")

    server: str
    result: dict[str, JsonValue] | None = None
    error: str | None = None


class PersistedVerdict(BaseModel):
    """The reward and failure fields verify reported for the episode."""

    model_config = ConfigDict(extra="forbid")

    reward: float
    mask_sample: bool
    evaluation_completed: bool
    failure_kind: str | None = None
    failure_reason: str | None = None


class EpisodeRecord(BaseModel):
    """Everything a session-less verify needs to re-grade instruction following without the sandbox."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["agentic_if.episode/1"] = "agentic_if.episode/1"
    run_id: str
    # The sealed task as seeded: the backend's own fields and ``agentic_if``, without the actor's request.
    row: dict[str, JsonValue]
    evidence: SessionEvidence
    execution_status: ExecutionStatus | None
    backend: BackendVerdict
    verdict: PersistedVerdict
    # Seed facts (working directory, baseline, replay and bundle receipts) and collection gaps, for audit.
    seed: dict[str, JsonValue]
    collection: dict[str, JsonValue]
    # Every call the live session's grades made to the judge, in order: a grade the judge's outage stopped, then the
    # grade of the retried verify. A session-less re-grade reports its own calls in its result and adds none here.
    judge_calls: list[JudgeCall] = Field(default_factory=list)


def load_episode_record(directory: Path) -> EpisodeRecord:
    """Read the record a live verify persisted in ``directory``."""
    # The standard parser, not Pydantic's: a lone surrogate escape saved from a plugin's receipt must load back.
    return EpisodeRecord.model_validate(json.loads((directory / EPISODE_RECORD).read_text(encoding="utf-8")))
