# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read the terminal publication from a RoboLab condition owned by the environment server."""

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, StrictBool

from benchmarks.robolab.prepare import Manifest


SHA256 = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
Score = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class Identity(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    arm: Literal["sdk_v15", "open_model"]
    task: str
    run_index: int
    seed: Annotated[int, Field(ge=0, le=0)]


class TerminalResult(BaseModel):
    model_config = ConfigDict(strict=True)

    schema_id: Literal["robolab.full-suite-episode.v2"] = Field(alias="schema")
    identity: Identity
    attempted: StrictBool
    scored: StrictBool
    native_complete: StrictBool
    native_task: str | None
    native_score: Score | None
    native_horizon: int
    success: StrictBool | None
    outcome_class: str
    bindings: dict[str, SHA256]
    evidence: dict[str, JsonValue]


class TerminalPublication(BaseModel):
    schema_id: Literal["robolab.sealed-terminal.v1"] = Field(alias="schema")
    derived_episode_result: TerminalResult


class NativeProcess(BaseModel):
    model_config = ConfigDict(strict=True)

    returncode: int | None
    forced: bool
    live_owned_after_cleanup: list[int]


class ConditionResult(BaseModel):
    model_config = ConfigDict(strict=True)

    task: str
    arm: str
    requested_runs: list[int]
    attempted_runs: list[int]
    process: NativeProcess
    error: str | None
    interrupted_signal: int | None


def read_condition(condition: Path, manifest: Manifest, digest: str, source_sha256: str) -> dict[str, JsonValue]:
    """Project an episode after the runner exited and the selected condition published its result."""
    case, plan = manifest.case, manifest.native_run_plan
    episode = condition / "runs" / f"run-{case.run:03d}"
    terminal_bytes = (episode / "terminal-record.json").read_bytes()
    row = TerminalPublication.model_validate_json(terminal_bytes).derived_episode_result
    published = json.loads((episode / "episode-result.json").read_bytes())
    TerminalResult.model_validate(published)
    if published != json.loads(terminal_bytes)["derived_episode_result"]:
        raise ValueError("RoboLab sealed and final episode results differ")
    if (
        row.identity != Identity(arm=plan.arm, task=case.task, run_index=case.run, seed=0)
        or row.native_horizon != case.native_horizon
        or row.bindings.get("source_manifest_sha256") != source_sha256
    ):
        raise ValueError("RoboLab terminal identity, horizon or executed source differs from the assigned condition")

    receipt = ConditionResult.model_validate_json((condition / "condition-result.json").read_bytes())
    if (
        receipt.task != case.task
        or receipt.arm != plan.arm
        or receipt.requested_runs != [case.run]
        or receipt.attempted_runs != [case.run]
        or receipt.error is not None
        or receipt.interrupted_signal is not None
    ):
        raise ValueError("RoboLab condition did not finish the assigned run")
    process = receipt.process
    if process.returncode != 0 or process.forced or process.live_owned_after_cleanup:
        raise ValueError("RoboLab simulator cleanup was not confirmed")
    result: dict[str, JsonValue] = {
        "case_id": case.id,
        "seed": row.identity.seed,
        "manifest_sha256": digest,
        "official_manifest_sha256": manifest.official_manifest_sha256,
        "native_source_sha256": source_sha256,
        "terminal_sha256": hashlib.sha256(terminal_bytes).hexdigest(),
        "native_score": None,
        "native_success": None,
        "native_outcome": row.outcome_class,
        "cleanup_confirmed": True,
    }
    if not row.scored:
        if row.success is not None or row.outcome_class == "scored":
            raise ValueError("RoboLab unscored run has a conflicting outcome")
        return result
    if (
        not row.attempted
        or not row.native_complete
        or row.outcome_class != "scored"
        or row.native_task != case.task
        or row.native_score is None
        or row.success is None
        or row.evidence.get("archive_audit_passed") is not True
    ):
        raise ValueError("RoboLab scored run lacks an audited native outcome")
    summaries = []
    with (condition / "official" / "episode_results.jsonl").open() as file:
        for line in file:
            summary = json.loads(line)
            if not isinstance(summary, dict) or type(summary.get("run")) is not int:
                raise ValueError("RoboLab official summary has an invalid run index")
            if summary["run"] == case.run:
                summaries.append(summary)
    if len(summaries) != 1:
        raise ValueError("RoboLab requires one official summary for the assigned run")
    summary = summaries[0]
    summary_sha256 = hashlib.sha256(json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (
        summary_sha256 != row.evidence.get("official_summary_sha256")
        or type(summary.get("score")) not in (int, float)
        or summary["score"] != row.native_score
    ):
        raise ValueError("RoboLab terminal score differs from the sealed official summary")
    result.update(reward=row.native_score, native_score=row.native_score, native_success=row.success)
    return result
