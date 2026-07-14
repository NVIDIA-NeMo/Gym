# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run a locked, paired Agent Skill A/B experiment through the Gym CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any

import psutil
import yaml
from pydantic import BaseModel, Field, model_validator

from benchmarks.agent_skills.scripts.compare_variants import (
    compare_rollouts,
    load_jsonl,
    render_report,
)
from nemo_gym.skills import load_skill_directory


REPO_ROOT = Path(__file__).resolve().parents[3]
AGENT_CONFIG_PATH = "agent_skills_create_environment_claude_code_agent.responses_api_agents.claude_code_agent"


class DatasetConfig(BaseModel):
    path: str
    split: str = "validation"


class AgentConfig(BaseModel):
    name: str
    sandbox_image: str


class ModelConfig(BaseModel):
    name: str
    temperature: float = 0.2
    max_output_tokens: int = Field(default=16_384, ge=1)


class SamplingConfig(BaseModel):
    repeats: int = Field(default=3, ge=1)
    seed: int = 1234
    concurrency: int = Field(default=4, ge=1)


class ArmConfig(BaseModel):
    bare: bool
    skills: list[str] = Field(default_factory=list)


class ComparisonConfig(BaseModel):
    control: str = "discovery_control"
    treatment: str = "treatment"
    bootstrap_samples: int = Field(default=2000, ge=100)


class ExperimentManifest(BaseModel):
    name: str
    benchmark: str = "agent_skills"
    dataset: DatasetConfig
    agent: AgentConfig
    model: ModelConfig
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    arms: dict[str, ArmConfig]
    comparison: ComparisonConfig = Field(default_factory=ComparisonConfig)

    @model_validator(mode="after")
    def validate_comparison_arms(self) -> "ExperimentManifest":
        missing = {
            self.comparison.control,
            self.comparison.treatment,
        } - set(self.arms)
        if missing:
            raise ValueError(f"Comparison references unknown arms: {', '.join(sorted(missing))}")
        if not self.arms[self.comparison.treatment].skills:
            raise ValueError("Treatment arm must include at least one skill")
        return self


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repo_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    cwd_path = Path.cwd() / candidate
    return cwd_path if cwd_path.exists() else REPO_ROOT / candidate


def load_manifest(path: Path) -> ExperimentManifest:
    return ExperimentManifest.model_validate(yaml.safe_load(path.read_text()))


def unsafe_uv_project_ancestor() -> bool:
    """Return whether this process inherits a project-aware ``uv run`` wrapper."""

    for parent in psutil.Process().parents():
        try:
            command = parent.cmdline()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            continue
        if (
            len(command) > 1
            and os.path.basename(command[0]) == "uv"
            and command[1] == "run"
            and "--no-project" not in command[2:]
        ):
            return True
    return False


def _jsonl_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "jsonl_fpath" and isinstance(item, str):
                paths.append(item)
            else:
                paths.extend(_jsonl_paths(item))
    elif isinstance(value, list):
        for item in value:
            paths.extend(_jsonl_paths(item))
    return paths


def validate_dataset_binding(manifest: ExperimentManifest) -> Path:
    benchmark_config = REPO_ROOT / "benchmarks" / manifest.benchmark / "config.yaml"
    if not benchmark_config.is_file():
        raise ValueError(f"Benchmark config does not exist: {benchmark_config}")
    config = yaml.safe_load(benchmark_config.read_text())
    configured_paths = {str(resolve_repo_path(path).resolve()) for path in _jsonl_paths(config)}
    manifest_dataset = str(resolve_repo_path(manifest.dataset.path).resolve())
    if manifest_dataset not in configured_paths:
        raise ValueError(
            f"Manifest dataset {manifest.dataset.path!r} is not bound by {benchmark_config}; "
            f"configured datasets: {sorted(configured_paths)}"
        )
    return benchmark_config


def git_state() -> dict[str, Any]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    return {"revision": revision, "dirty": bool(status.strip())}


def resolve_image_digest(image: str) -> str | None:
    docker = shutil.which("docker")
    if docker is None:
        return None
    result = subprocess.run(
        [docker, "image", "inspect", "--format", "{{.Id}}", image],
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def resolve_execution_image(image: str) -> str:
    if image.startswith("sha256:") or "@sha256:" in image:
        return image
    digest = resolve_image_digest(image)
    if not digest:
        raise ValueError(f"Sandbox image {image!r} is mutable and no local immutable digest could be resolved")
    return digest


def snapshot_arm_skills(
    manifest: ExperimentManifest,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    bundles_root = output_dir / "skill-bundles"
    for arm_name, arm in manifest.arms.items():
        if not arm.skills:
            continue
        bundle = bundles_root / arm_name
        if bundle.exists():
            shutil.rmtree(bundle)
        bundle.mkdir(parents=True)
        source_paths = []
        for skill_path_value in arm.skills:
            source = resolve_repo_path(skill_path_value)
            skill_md = source / "SKILL.md"
            if not skill_md.is_file():
                raise ValueError(f"Skill path must contain SKILL.md: {source}")
            destination = bundle / source.name
            shutil.copytree(source, destination)
            source_paths.append(str(source))
        skills_ref = load_skill_directory(str(bundle))
        snapshots[arm_name] = {
            "bundle_path": str(bundle),
            "source_paths": source_paths,
            "skills_ref": skills_ref.model_dump(),
        }
    return snapshots


def build_arm_command(
    manifest: ExperimentManifest,
    *,
    arm_name: str,
    arm: ArmConfig,
    output_dir: Path,
    skill_snapshot: dict[str, Any] | None,
    execution_image: str,
) -> list[str]:
    rollout_path = output_dir / arm_name / "rollouts.jsonl"
    rollout_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "gym",
        "eval",
        "run",
        "--benchmark",
        manifest.benchmark,
        "--agent",
        manifest.agent.name,
        "--split",
        manifest.dataset.split,
        "--output",
        str(rollout_path),
        "--num-repeats",
        str(manifest.sampling.repeats),
        "--concurrency",
        str(manifest.sampling.concurrency),
        "--temperature",
        str(manifest.model.temperature),
        "--max-output-tokens",
        str(manifest.model.max_output_tokens),
        f"+agent_skills_sandbox_image={execution_image}",
        f"+{AGENT_CONFIG_PATH}.bare={str(arm.bare).lower()}",
        f"+{AGENT_CONFIG_PATH}.model={manifest.model.name}",
    ]
    if skill_snapshot is not None:
        command.append(f"+skills.path={skill_snapshot['bundle_path']}")
    return command


def write_lock(
    manifest: ExperimentManifest,
    *,
    manifest_path: Path,
    output_dir: Path,
    arm_order: list[str],
    snapshots: dict[str, dict[str, Any]],
    commands: dict[str, list[str]],
    execution_image: str,
    benchmark_config: Path,
    allow_dirty: bool,
) -> dict[str, Any]:
    state = git_state()
    if state["dirty"] and not allow_dirty:
        raise RuntimeError(
            "Refusing to run a locked experiment from a dirty Git checkout; pass --allow-dirty to override"
        )
    dataset_path = resolve_repo_path(manifest.dataset.path)
    if not dataset_path.is_file():
        raise ValueError(f"Dataset does not exist: {dataset_path}")
    lock = {
        "manifest": manifest.model_dump(),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "git": state,
        "benchmark_config_path": str(benchmark_config),
        "benchmark_config_sha256": sha256_file(benchmark_config),
        "sandbox_image": {
            "reference": manifest.agent.sandbox_image,
            "execution_reference": execution_image,
        },
        "arm_order": arm_order,
        "skill_snapshots": snapshots,
        "commands": commands,
    }
    (output_dir / "experiment.lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return lock


def validate_skills_provenance(
    rows: list[dict[str, Any]],
    *,
    arm_name: str,
    snapshot: dict[str, Any] | None,
) -> None:
    expected_hash = (snapshot or {}).get("skills_ref", {}).get("hash")
    if expected_hash is None:
        contaminated = [index for index, row in enumerate(rows) if "skills_ref" in row]
        if contaminated:
            raise ValueError(f"Arm {arm_name!r} unexpectedly contains skills_ref on rows: {contaminated}")
        return
    observed_hashes = [(row.get("skills_ref") or {}).get("hash") for row in rows]
    if any(value != expected_hash for value in observed_hashes):
        raise ValueError(
            f"Arm {arm_name!r} skills provenance mismatch: expected {expected_hash}, observed {observed_hashes}"
        )


def run_experiment(
    manifest: ExperimentManifest,
    *,
    manifest_path: Path,
    output_dir: Path,
    dry_run: bool,
    allow_dirty: bool,
    resume: bool = False,
) -> dict[str, Any] | None:
    if not resume and output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty experiment output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = output_dir / "experiment.lock.json"
    if resume:
        if not lock_path.is_file():
            raise RuntimeError(f"Cannot resume without an experiment lock: {lock_path}")
        lock = json.loads(lock_path.read_text())
        if lock["manifest_sha256"] != sha256_file(manifest_path):
            raise RuntimeError("Cannot resume: experiment manifest changed")
        dataset_path = resolve_repo_path(manifest.dataset.path)
        if lock["dataset_sha256"] != sha256_file(dataset_path):
            raise RuntimeError("Cannot resume: experiment dataset changed")
        benchmark_config = validate_dataset_binding(manifest)
        if lock["benchmark_config_sha256"] != sha256_file(benchmark_config):
            raise RuntimeError("Cannot resume: benchmark configuration changed")
        current_git = git_state()
        if current_git["revision"] != lock["git"]["revision"]:
            raise RuntimeError("Cannot resume: Git revision changed")
        if current_git["dirty"] and not allow_dirty:
            raise RuntimeError("Cannot resume from a dirty Git checkout without --allow-dirty")
        arm_order = list(lock["arm_order"])
        snapshots = dict(lock["skill_snapshots"])
        commands = {name: list(command) for name, command in lock["commands"].items()}
    else:
        if lock_path.exists():
            raise RuntimeError(f"Experiment output already exists: {output_dir}")
        benchmark_config = validate_dataset_binding(manifest)
        execution_image = resolve_execution_image(manifest.agent.sandbox_image)
        snapshots = snapshot_arm_skills(manifest, output_dir)
        arm_order = list(manifest.arms)
        random.Random(manifest.sampling.seed).shuffle(arm_order)
        commands = {
            arm_name: build_arm_command(
                manifest,
                arm_name=arm_name,
                arm=manifest.arms[arm_name],
                output_dir=output_dir,
                skill_snapshot=snapshots.get(arm_name),
                execution_image=execution_image,
            )
            for arm_name in arm_order
        }
        write_lock(
            manifest,
            manifest_path=manifest_path,
            output_dir=output_dir,
            arm_order=arm_order,
            snapshots=snapshots,
            commands=commands,
            execution_image=execution_image,
            benchmark_config=benchmark_config,
            allow_dirty=allow_dirty,
        )

    for arm_name in arm_order:
        rollout_path = output_dir / arm_name / "rollouts.jsonl"
        command = list(commands[arm_name])
        if resume and rollout_path.exists() and "--resume" not in command:
            command.append("--resume")
        print(f"[{arm_name}] {' '.join(command)}")
        if not dry_run:
            subprocess.run(command, cwd=REPO_ROOT, check=True)
    if dry_run:
        return None

    control_path = output_dir / manifest.comparison.control / "rollouts.jsonl"
    treatment_path = output_dir / manifest.comparison.treatment / "rollouts.jsonl"
    control_rows = load_jsonl(control_path)
    treatment_rows = load_jsonl(treatment_path)
    validate_skills_provenance(
        control_rows,
        arm_name=manifest.comparison.control,
        snapshot=snapshots.get(manifest.comparison.control),
    )
    validate_skills_provenance(
        treatment_rows,
        arm_name=manifest.comparison.treatment,
        snapshot=snapshots.get(manifest.comparison.treatment),
    )
    comparison = compare_rollouts(
        control_rows,
        treatment_rows,
        control_arm=manifest.comparison.control,
        treatment_arm=manifest.comparison.treatment,
        seed=manifest.sampling.seed,
        bootstrap_samples=manifest.comparison.bootstrap_samples,
    )
    (output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    (output_dir / "report.md").write_text(render_report(comparison))
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if unsafe_uv_project_ancestor():
        raise RuntimeError(
            "Do not launch this driver with project-aware `uv run`: Ray server subprocesses "
            "use server-specific working directories. Use `.venv/bin/python "
            "benchmarks/agent_skills/scripts/run_experiment.py ...` instead."
        )

    manifest_path = args.config.resolve()
    manifest = load_manifest(manifest_path)
    output_dir = (args.output_dir or Path("results/agent-skills") / manifest.name).resolve()
    run_experiment(
        manifest,
        manifest_path=manifest_path,
        output_dir=output_dir,
        dry_run=args.dry_run,
        allow_dirty=args.allow_dirty,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
