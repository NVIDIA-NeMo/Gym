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
"""Workspace orchestration for iterative agent skill improvement (#1495).

v0 is manual / coding-agent-in-the-loop: snapshot skills, run eval externally,
record aggregate metrics, keep or revert. See the "Improve Skills" evaluation
docs page (``fern/versions/latest/pages/evaluation/improve-skills.mdx``).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import orjson
import rich
from pydantic import BaseModel, Field

from nemo_gym.config_types import BaseNeMoGymCLIConfig, ConfigError
from nemo_gym.skills import SkillsRef, hash_skill_dir, load_skill_directory


class RoundStatus(str, Enum):
    PENDING = "pending"
    KEPT = "kept"
    REVERTED = "reverted"
    SUPERSEDED = "superseded"


class MetricComparison(BaseModel):
    name: str
    baseline: Optional[float]
    candidate: Optional[float]
    delta: Optional[float]


class RoundComparison(BaseModel):
    baseline_round_id: Optional[str] = None
    baseline_skills_hash: Optional[str] = None
    candidate_skills_hash: str
    metrics: List[MetricComparison] = Field(default_factory=list)


class RoundManifest(BaseModel):
    round_id: str
    status: RoundStatus = RoundStatus.PENDING
    created_at: str
    skills_ref: SkillsRef
    aggregate_metrics_fpath: Optional[str] = None
    rollouts_jsonl_fpath: Optional[str] = None
    materialized_inputs_jsonl_fpath: Optional[str] = None
    comparison: Optional[RoundComparison] = None
    notes: Optional[str] = None


class ImproveWorkspaceConfig(BaseModel):
    """Persisted workspace metadata (``improve.yaml``)."""

    name: str
    skills_path: str
    eval_command_hint: Optional[str] = None
    primary_metric: str = Field(
        default="mean/reward",
        description="Headline metric from aggregate key_metrics used for keep/revert summaries.",
    )


class ImproveInitConfig(BaseNeMoGymCLIConfig):
    """
    Initialize a skill-improvement workspace from a measured baseline.

    Examples:

    ```bash
    gym eval improve-init \\
        --workspace .gym/improve/my-skills \\
        --skills .claude/skills \\
        --baseline-metrics results/baseline_aggregate_metrics.json
    ```
    """

    workspace: str = Field(description="Directory to create for this improvement session.")
    skills: str = Field(description="Live skills directory (Agent Skills layout).")
    name: str = Field(default="skill-improvement", description="Human-readable workspace name.")
    baseline_metrics: str = Field(description="Path to baseline *_aggregate_metrics.json from gym eval run.")
    baseline_skills: Optional[str] = Field(
        default=None,
        description="Optional skills tree for the baseline; defaults to --skills at init time.",
    )
    eval_command_hint: Optional[str] = Field(
        default=None,
        description="Optional template command printed by improve-round (e.g. your gym eval run invocation).",
    )
    primary_metric: str = Field(default="mean/reward", description="Metric key for before/after summaries.")


class ImproveRoundConfig(BaseNeMoGymCLIConfig):
    """
    Start a new improve round: snapshot the current live skills directory.

    Examples:

    ```bash
    gym eval improve-round --workspace .gym/improve/my-skills
    ```
    """

    workspace: str = Field(description="Improvement workspace directory.")
    notes: Optional[str] = Field(default=None, description="Optional note stored on the round manifest.")


class ImproveRecordConfig(BaseNeMoGymCLIConfig):
    """
    Record eval results for the latest (or specified) round and compare to accepted baseline.

    Examples:

    ```bash
    gym eval improve-record \\
        --workspace .gym/improve/my-skills \\
        --aggregate-metrics results/round3_aggregate_metrics.json \\
        --rollouts results/round3_rollouts.jsonl
    ```
    """

    workspace: str
    aggregate_metrics: str
    round_id: Optional[str] = Field(default=None, description="Round to record; default: latest pending round.")
    rollouts: Optional[str] = None
    materialized_inputs: Optional[str] = None


class ImproveKeepConfig(BaseNeMoGymCLIConfig):
    """
    Accept a round: update accepted baseline and optionally copy skills back to the live directory.

    Examples:

    ```bash
    gym eval improve-keep --workspace .gym/improve/my-skills --round 0003
    gym eval improve-keep --workspace .gym/improve/my-skills --round 0003 --no-copy-live
    ```
    """

    workspace: str
    round_id: str = Field(description="Round id to keep (e.g. 0003).")
    copy_live: bool = Field(
        default=True,
        description="Copy accepted skills back to the live skills.path directory.",
    )
    force: bool = Field(default=False, description="Overwrite live skills even if they changed since the round.")


class ImproveRevertConfig(BaseNeMoGymCLIConfig):
    """
    Restore the accepted skills snapshot to the live skills directory.

    Examples:

    ```bash
    gym eval improve-revert --workspace .gym/improve/my-skills
    ```
    """

    workspace: str
    force: bool = Field(default=False, description="Overwrite live skills even if they differ from expected.")


class ImproveHistoryConfig(BaseNeMoGymCLIConfig):
    """List rounds and decisions for a workspace."""

    workspace: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _workspace_paths(workspace: Path) -> Dict[str, Path]:
    return {
        "root": workspace,
        "config": workspace / "improve.yaml",
        "accepted": workspace / "accepted",
        "accepted_skills": workspace / "accepted" / "skills",
        "accepted_skills_ref": workspace / "accepted" / "skills_ref.json",
        "accepted_metrics": workspace / "accepted" / "aggregate_metrics.json",
        "rounds": workspace / "rounds",
        "history": workspace / "history.jsonl",
    }


def _copy_skills_tree(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)


def _load_workspace_config(workspace: Path) -> ImproveWorkspaceConfig:
    paths = _workspace_paths(workspace)
    if not paths["config"].exists():
        raise ConfigError(f"Workspace not initialized: {workspace} (missing improve.yaml)")
    return ImproveWorkspaceConfig.model_validate(json.loads(paths["config"].read_text()))


def _append_history(workspace: Path, event: str, payload: Dict[str, Any]) -> None:
    paths = _workspace_paths(workspace)
    paths["history"].parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": _utc_now(), "event": event, **payload}
    with paths["history"].open("ab") as f:
        f.write(orjson.dumps(record) + b"\n")


def _next_round_id(workspace: Path) -> str:
    rounds_dir = _workspace_paths(workspace)["rounds"]
    rounds_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(p.name for p in rounds_dir.iterdir() if p.is_dir() and p.name.isdigit())
    if not existing:
        return "0001"
    return f"{int(existing[-1]) + 1:04d}"


def _round_dir(workspace: Path, round_id: str) -> Path:
    return _workspace_paths(workspace)["rounds"] / round_id


def _load_round_manifest(workspace: Path, round_id: str) -> RoundManifest:
    manifest_path = _round_dir(workspace, round_id) / "manifest.json"
    if not manifest_path.exists():
        raise ConfigError(f"Round {round_id} not found in {workspace}")
    return RoundManifest.model_validate(json.loads(manifest_path.read_text()))


def _save_round_manifest(workspace: Path, manifest: RoundManifest) -> None:
    round_path = _round_dir(workspace, manifest.round_id)
    round_path.mkdir(parents=True, exist_ok=True)
    (round_path / "manifest.json").write_bytes(orjson.dumps(manifest.model_dump(), option=orjson.OPT_INDENT_2))


def load_aggregate_metrics_key_metrics(aggregate_metrics_fpath: Path) -> Dict[str, float]:
    """Extract merged key_metrics across agents from an aggregate metrics file."""
    if not aggregate_metrics_fpath.exists():
        raise ConfigError(f"Aggregate metrics file not found: {aggregate_metrics_fpath}")

    data = json.loads(aggregate_metrics_fpath.read_text())
    if not isinstance(data, list):
        raise ConfigError(f"Expected aggregate metrics JSON array in {aggregate_metrics_fpath}")

    merged: Dict[str, float] = {}
    for entry in data:
        key_metrics = entry.get("key_metrics") or {}
        for k, v in key_metrics.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                merged[k] = float(v)
    return merged


def compare_key_metrics(
    baseline: Dict[str, float],
    candidate: Dict[str, float],
    *,
    metric_names: Optional[List[str]] = None,
) -> List[MetricComparison]:
    names = metric_names or sorted(set(baseline) | set(candidate))
    comparisons: List[MetricComparison] = []
    for name in names:
        b = baseline.get(name)
        c = candidate.get(name)
        delta = (c - b) if b is not None and c is not None else None
        comparisons.append(MetricComparison(name=name, baseline=b, candidate=c, delta=delta))
    return comparisons


def init_workspace(config: ImproveInitConfig) -> Path:
    workspace = Path(config.workspace)
    if workspace.exists() and any(workspace.iterdir()):
        raise ConfigError(f"Workspace {workspace} already exists and is not empty.")

    paths = _workspace_paths(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    paths["rounds"].mkdir(parents=True, exist_ok=True)

    baseline_skills_src = Path(config.baseline_skills or config.skills)
    if not baseline_skills_src.is_dir():
        raise ConfigError(f"Baseline skills directory not found: {baseline_skills_src}")

    baseline_metrics_src = Path(config.baseline_metrics)
    if not baseline_metrics_src.exists():
        raise ConfigError(f"Baseline aggregate metrics not found: {baseline_metrics_src}")

    ws_config = ImproveWorkspaceConfig(
        name=config.name,
        skills_path=config.skills,
        eval_command_hint=config.eval_command_hint,
        primary_metric=config.primary_metric,
    )
    paths["config"].write_bytes(orjson.dumps(ws_config.model_dump(), option=orjson.OPT_INDENT_2))

    _copy_skills_tree(baseline_skills_src, paths["accepted_skills"])
    shutil.copy2(baseline_metrics_src, paths["accepted_metrics"])

    skills_ref = load_skill_directory(str(paths["accepted_skills"]))
    paths["accepted_skills_ref"].write_bytes(orjson.dumps(skills_ref.model_dump(), option=orjson.OPT_INDENT_2))

    _append_history(
        workspace,
        "init",
        {
            "skills_path": config.skills,
            "baseline_metrics": str(baseline_metrics_src),
            "accepted_skills_hash": skills_ref.hash,
        },
    )

    rich.print(f"[green]Initialized improve workspace at {workspace}[/green]")
    rich.print(f"Accepted baseline skills hash: {skills_ref.hash}")
    return workspace


def start_round(config: ImproveRoundConfig) -> Tuple[str, SkillsRef]:
    workspace = Path(config.workspace)
    ws_config = _load_workspace_config(workspace)
    live_skills = Path(ws_config.skills_path)
    if not live_skills.is_dir():
        raise ConfigError(f"Live skills directory not found: {live_skills}")

    round_id = _next_round_id(workspace)
    round_path = _round_dir(workspace, round_id)
    candidate_skills = round_path / "candidate_skills"
    _copy_skills_tree(live_skills, candidate_skills)

    skills_ref = load_skill_directory(str(candidate_skills))
    manifest = RoundManifest(
        round_id=round_id,
        status=RoundStatus.PENDING,
        created_at=_utc_now(),
        skills_ref=skills_ref,
        notes=config.notes,
    )
    _save_round_manifest(workspace, manifest)
    _append_history(workspace, "round_start", {"round_id": round_id, "skills_hash": skills_ref.hash})

    rich.print(f"[green]Started round {round_id}[/green] (skills hash={skills_ref.hash})")
    skills_path_override = f"+skills.path={candidate_skills}"
    if ws_config.eval_command_hint:
        rich.print("\nSuggested eval command:")
        rich.print(f"  {ws_config.eval_command_hint} {skills_path_override}")
    else:
        rich.print("\nRun eval with skills staged for this round, e.g.:")
        rich.print(
            f"  gym eval run --no-serve ... --output results/round{round_id}_rollouts.jsonl {skills_path_override}"
        )
    rich.print(f"\nThen record results:\n  gym eval improve-record --workspace {workspace} --aggregate-metrics <path>")

    return round_id, skills_ref


def record_round(config: ImproveRecordConfig) -> RoundManifest:
    workspace = Path(config.workspace)
    ws_config = _load_workspace_config(workspace)

    round_id = config.round_id
    if round_id is None:
        rounds_dir = _workspace_paths(workspace)["rounds"]
        pending = []
        for p in sorted(rounds_dir.iterdir()):
            if not p.is_dir():
                continue
            manifest = _load_round_manifest(workspace, p.name)
            if manifest.status == RoundStatus.PENDING:
                pending.append(manifest.round_id)
        if not pending:
            raise ConfigError("No pending round to record. Run gym eval improve-round first.")
        round_id = pending[-1]

    manifest = _load_round_manifest(workspace, round_id)
    if manifest.status != RoundStatus.PENDING:
        raise ConfigError(f"Round {round_id} is {manifest.status.value}, expected pending.")

    agg_src = Path(config.aggregate_metrics)
    round_path = _round_dir(workspace, round_id)
    agg_dest = round_path / "aggregate_metrics.json"
    shutil.copy2(agg_src, agg_dest)

    if config.rollouts:
        shutil.copy2(config.rollouts, round_path / "rollouts.jsonl")
        manifest.rollouts_jsonl_fpath = str(round_path / "rollouts.jsonl")
    if config.materialized_inputs:
        shutil.copy2(config.materialized_inputs, round_path / "materialized_inputs.jsonl")
        manifest.materialized_inputs_jsonl_fpath = str(round_path / "materialized_inputs.jsonl")

    manifest.aggregate_metrics_fpath = str(agg_dest)

    accepted_metrics = load_aggregate_metrics_key_metrics(_workspace_paths(workspace)["accepted_metrics"])
    candidate_metrics = load_aggregate_metrics_key_metrics(agg_dest)

    accepted_ref_path = _workspace_paths(workspace)["accepted_skills_ref"]
    accepted_hash = None
    if accepted_ref_path.exists():
        accepted_hash = json.loads(accepted_ref_path.read_text()).get("hash")

    comparison = RoundComparison(
        baseline_skills_hash=accepted_hash,
        candidate_skills_hash=manifest.skills_ref.hash,
        metrics=compare_key_metrics(accepted_metrics, candidate_metrics),
    )
    manifest.comparison = comparison
    _save_round_manifest(workspace, manifest)
    (round_path / "comparison.json").write_bytes(orjson.dumps(comparison.model_dump(), option=orjson.OPT_INDENT_2))

    _append_history(
        workspace,
        "round_record",
        {"round_id": round_id, "skills_hash": manifest.skills_ref.hash, "aggregate_metrics": str(agg_dest)},
    )

    _print_comparison_table(comparison, primary_metric=ws_config.primary_metric)
    rich.print(f"\nKeep:  gym eval improve-keep --workspace {workspace} --round {round_id}")
    rich.print(f"Revert live skills: gym eval improve-revert --workspace {workspace}")

    return manifest


def keep_round(config: ImproveKeepConfig) -> None:
    workspace = Path(config.workspace)
    ws_config = _load_workspace_config(workspace)
    manifest = _load_round_manifest(workspace, config.round_id)

    if manifest.status != RoundStatus.PENDING:
        raise ConfigError(f"Round {config.round_id} is {manifest.status.value}, expected pending.")
    if not manifest.aggregate_metrics_fpath:
        raise ConfigError(f"Round {config.round_id} has no recorded metrics. Run improve-record first.")

    round_path = _round_dir(workspace, config.round_id)
    paths = _workspace_paths(workspace)

    for other in sorted(paths["rounds"].iterdir()):
        if not other.is_dir() or other.name == config.round_id:
            continue
        other_manifest = _load_round_manifest(workspace, other.name)
        if other_manifest.status == RoundStatus.KEPT:
            other_manifest.status = RoundStatus.SUPERSEDED
            _save_round_manifest(workspace, other_manifest)

    _copy_skills_tree(round_path / "candidate_skills", paths["accepted_skills"])
    shutil.copy2(Path(manifest.aggregate_metrics_fpath), paths["accepted_metrics"])
    paths["accepted_skills_ref"].write_bytes(
        orjson.dumps(manifest.skills_ref.model_dump(), option=orjson.OPT_INDENT_2)
    )

    manifest.status = RoundStatus.KEPT
    _save_round_manifest(workspace, manifest)

    if config.copy_live:
        live = Path(ws_config.skills_path)
        expected_hash = manifest.skills_ref.hash
        if live.is_dir():
            live_hash = hash_skill_dir(live)
            if live_hash != expected_hash and not config.force:
                raise ConfigError(
                    f"Live skills at {live} (hash={live_hash}) differ from round {config.round_id} "
                    f"(hash={expected_hash}). Re-run with --force to overwrite."
                )
        _copy_skills_tree(paths["accepted_skills"], live)

    _append_history(workspace, "round_keep", {"round_id": config.round_id, "skills_hash": manifest.skills_ref.hash})
    rich.print(f"[green]Kept round {config.round_id}[/green] (accepted hash={manifest.skills_ref.hash})")


def revert_live_skills(config: ImproveRevertConfig) -> None:
    workspace = Path(config.workspace)
    ws_config = _load_workspace_config(workspace)
    paths = _workspace_paths(workspace)

    if not paths["accepted_skills"].is_dir():
        raise ConfigError(f"No accepted skills snapshot in {workspace}")

    accepted_ref = SkillsRef.model_validate(json.loads(paths["accepted_skills_ref"].read_text()))
    live = Path(ws_config.skills_path)

    if live.is_dir() and not config.force:
        live_hash = hash_skill_dir(live)
        if live_hash != accepted_ref.hash:
            rich.print(f"[yellow]Live skills hash {live_hash} differs from accepted {accepted_ref.hash}.[/yellow]")
            raise ConfigError("Refusing to revert without --force.")

    live.parent.mkdir(parents=True, exist_ok=True)
    _copy_skills_tree(paths["accepted_skills"], live)
    _append_history(workspace, "revert_live", {"skills_hash": accepted_ref.hash})
    rich.print(f"[green]Restored live skills from accepted baseline[/green] (hash={accepted_ref.hash})")


def list_history(config: ImproveHistoryConfig, *, json_output: bool = False) -> List[RoundManifest]:
    workspace = Path(config.workspace)
    _load_workspace_config(workspace)
    rounds_dir = _workspace_paths(workspace)["rounds"]

    manifests: List[RoundManifest] = []
    if rounds_dir.exists():
        for p in sorted(rounds_dir.iterdir()):
            if p.is_dir() and (p / "manifest.json").exists():
                manifests.append(_load_round_manifest(workspace, p.name))

    if json_output:
        print(json.dumps([m.model_dump() for m in manifests], indent=2))
        return manifests

    if not manifests:
        rich.print("[yellow]No rounds yet.[/yellow]")
        return manifests

    from rich.table import Table

    ws_config = _load_workspace_config(workspace)
    table = Table(title=f"Improve rounds — {workspace}")
    table.add_column("Round")
    table.add_column("Status")
    table.add_column("Skills hash")
    table.add_column(ws_config.primary_metric)
    table.add_column("Recorded")

    for m in manifests:
        primary_val = ""
        if m.comparison:
            for mc in m.comparison.metrics:
                if mc.name == ws_config.primary_metric:
                    parts = []
                    if mc.baseline is not None:
                        parts.append(f"{mc.baseline:.4g}")
                    if mc.candidate is not None:
                        parts.append(f"→ {mc.candidate:.4g}")
                    if mc.delta is not None:
                        sign = "+" if mc.delta >= 0 else ""
                        parts.append(f"({sign}{mc.delta:.4g})")
                    primary_val = " ".join(parts)
                    break
        table.add_row(
            m.round_id,
            m.status.value,
            m.skills_ref.hash,
            primary_val,
            "yes" if m.aggregate_metrics_fpath else "no",
        )

    rich.print(table)
    return manifests


def _print_comparison_table(comparison: RoundComparison, *, primary_metric: str) -> None:
    from rich.table import Table

    table = Table(title="Before / after (accepted baseline → this round)")
    table.add_column("Metric")
    table.add_column("Baseline")
    table.add_column("Candidate")
    table.add_column("Delta")

    for mc in comparison.metrics:
        style = None
        if mc.name == primary_metric and mc.delta is not None:
            style = "green" if mc.delta > 0 else "red" if mc.delta < 0 else None
        table.add_row(
            mc.name,
            f"{mc.baseline:.6g}" if mc.baseline is not None else "—",
            f"{mc.candidate:.6g}" if mc.candidate is not None else "—",
            f"{mc.delta:+.6g}" if mc.delta is not None else "—",
            style=style,
        )

    rich.print(table)
