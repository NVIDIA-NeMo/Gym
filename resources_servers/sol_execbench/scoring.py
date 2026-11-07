# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SHA-pinned workload anchors and SOL scores within the published scoring domain.

Formula: https://github.com/NVIDIA/SOL-ExecBench/blob/a9fa0804c793d438e70850c33fe34426e66d53dd/src/sol_execbench/sol_score.py
Domain and audit policy: https://arxiv.org/html/2603.19173v1#S4.SS3
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from resources_servers.sol_execbench.problem_store import ProblemManifest, Sha256


NonBlank = Annotated[str, Field(pattern=r"\S")]
Hardware = Literal["B200", "LOCAL"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class WorkloadAnchor(_StrictModel):
    """Milliseconds for a reviewed scoring baseline and its hardware SOL bound."""

    baseline_ms: float = Field(gt=0, allow_inf_nan=False)
    sol_ms: float = Field(ge=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def positive_headroom(self) -> "WorkloadAnchor":
        if self.baseline_ms <= self.sol_ms:
            raise ValueError("Scoring baseline must be greater than the SOL bound")
        return self


class AnchorProvenance(_StrictModel):
    """Source and immutable revision or reviewed snapshot identifying the anchors."""

    source: NonBlank
    revision: NonBlank


class TaskAnchors(_StrictModel):
    """Anchors bound to one trusted problem and keyed by exact workload UUID."""

    problem_digest: Sha256
    workloads: dict[NonBlank, WorkloadAnchor] = Field(min_length=1)


class AnchorManifest(_StrictModel):
    """Server-owned anchors; request rows cannot supply or override these values."""

    schema_version: int = Field(strict=True, ge=1, le=1)
    problem_manifest_sha256: Sha256
    target_hardware: Hardware
    provenance: AnchorProvenance
    tasks: dict[NonBlank, TaskAnchors] = Field(min_length=1)


def _unique_object(pairs: list[tuple[str, JsonValue]]) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate anchor JSON key: {key!r}")
        result[key] = value
    return result


def load_anchor_manifest(
    path: Path,
    sha256: str,
    *,
    problem_manifest: ProblemManifest,
    problem_manifest_sha256: str,
    target_hardware: Hardware,
) -> AnchorManifest:
    """Check exact bytes, provenance bindings, and complete coverage before GPU work."""
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError("Anchor manifest SHA256 mismatch")
    anchors = AnchorManifest.model_validate(json.loads(data, object_pairs_hook=_unique_object))
    if anchors.problem_manifest_sha256 != problem_manifest_sha256:
        raise ValueError("Anchor problem manifest SHA256 mismatch")
    if anchors.target_hardware != target_hardware:
        raise ValueError("Anchor target hardware mismatch")
    problems = {problem.task_id: problem for problem in problem_manifest.problems}
    if set(anchors.tasks) != set(problems):
        raise ValueError("Anchor tasks must exactly match the selected problem manifest")
    for task_id, problem in problems.items():
        task = anchors.tasks[task_id]
        if task.problem_digest != problem.problem_digest:
            raise ValueError(f"Anchor problem digest mismatch: {task_id}")
        if set(task.workloads) != {workload["uuid"] for workload in problem.workloads}:
            raise ValueError(f"Anchor workload UUIDs must exactly match the selected problem: {task_id}")
    return anchors


def score_workload(candidate_ms: float, anchor: WorkloadAnchor) -> float:
    """Return the unclipped native formula; out-of-domain measurements need an audit."""
    if isinstance(candidate_ms, bool) or not isinstance(candidate_ms, (float, int)):
        raise ValueError("Candidate latency must be a finite positive number")
    if not math.isfinite(candidate_ms) or candidate_ms <= 0:
        raise ValueError("Candidate latency must be a finite positive number")
    if candidate_ms < anchor.sol_ms:
        raise ValueError("Candidate latency is below the SOL bound; scoring requires an audit")
    return 1.0 / (1.0 + (candidate_ms - anchor.sol_ms) / (anchor.baseline_ms - anchor.sol_ms))
