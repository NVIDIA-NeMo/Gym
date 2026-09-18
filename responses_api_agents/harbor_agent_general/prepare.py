# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve Harbor tasks and prepare local Gym inputs without starting trials."""

import argparse
import asyncio
import hashlib
import json
import shutil
import tempfile
from importlib.metadata import version
from pathlib import Path

import yaml
from harbor.agents.factory import AgentFactory
from harbor.environments.factory import EnvironmentFactory
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, EnvironmentConfig, TaskConfig
from harbor.models.trial.paths import TrialPaths
from harbor.tasks.client import TaskClient
from pydantic import BaseModel, Field


class CompatibilityReport(BaseModel):
    """Static findings; successful inspection does not establish runtime readiness."""

    task_name: str
    steps: list[str] = Field(default_factory=list)
    compose: bool = False
    mcp_servers: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def inspect_task(task: Task, *, agent: AgentConfig, environment: EnvironmentConfig) -> CompatibilityReport:
    """Ask Harbor to validate the environment and inspect declared agent capabilities."""
    report = CompatibilityReport(
        task_name=task.name,
        steps=[step.name for step in task.config.steps or []],
        compose=(task.paths.environment_dir / "docker-compose.yaml").is_file()
        or bool(environment.extra_docker_compose),
        mcp_servers=[server.name for server in [*task.config.environment.mcp_servers, *agent.mcp_servers]],
    )
    # Constructors validate task requirements but do not start containers.
    try:
        with tempfile.TemporaryDirectory(prefix="gym-harbor-inspect-") as scratch:
            env = EnvironmentFactory.create_environment_from_config(
                config=environment,
                environment_dir=task.paths.environment_dir,
                environment_name="gym-harbor-inspect",
                session_id="gym-harbor-inspect",
                trial_paths=TrialPaths(Path(scratch)),
                task_env_config=task.config.environment,
            )
            if report.compose and not env.capabilities.docker_compose:
                report.errors.append("Selected environment does not support Docker Compose tasks.")
        capabilities = AgentFactory.get_agent_class_from_config(agent).capabilities
        if not capabilities.atif:
            report.errors.append("Selected agent does not declare ATIF output, which this Gym adapter requires.")
        if report.steps and agent.resume_trajectory and not capabilities.resume:
            report.errors.append("Selected agent cannot resume its session between task steps.")
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        report.errors.append(f"Harbor compatibility inspection failed: {exc}")
    if report.mcp_servers:
        report.warnings.append("MCP configuration is preserved; agent connectivity/tool use requires a smoke rollout.")
    if report.steps:
        report.warnings.append(
            "Gym projects the final step into response; harbor_steps preserves all steps and Harbor supplies trial reward."
        )
    if task.config.verifier.environment is not None:
        report.warnings.append("Separate verifier environment requires a representative smoke rollout.")
    return report


def load_job_config(path: Path) -> JobConfig:
    """Read the supported Harbor job subset and reject settings that would be lost."""
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Harbor job config must be a YAML/JSON object")
    supported = {"datasets", "tasks", "agents", "environment", "verifier"}
    unsupported = set(data) - supported
    if unsupported:
        raise ValueError(
            f"Unsupported job settings: {sorted(unsupported)}. This preparation command accepts "
            "datasets, tasks, agents, environment and verifier; Gym controls repeats, concurrency and retries."
        )
    # Match Harbor's cwd-relative paths; callers should run from the same directory as harbor run.
    return JobConfig.model_validate(data)


async def prepare(job: JobConfig, *, output_dir: Path, reward_key: str = "reward") -> bool:
    """Snapshot resolved tasks and emit inputs only when every task passes static inspection.

    Returns whether runnable artifacts were written. Incompatible tasks remain in
    manifest.json with their errors; no tasks are silently dropped.
    """
    if len(job.agents) != 1:
        raise ValueError("Prepare one Harbor agent at a time; the job must contain exactly one agent")
    if not reward_key:
        raise ValueError("reward_key must not be empty")
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}; choose a new directory")
    tasks = list(job.tasks)
    for dataset in job.datasets:
        tasks.extend(await dataset.get_task_configs(disable_verification=job.verifier.disable))
    if not tasks:
        raise ValueError("Harbor resolved no tasks; check the dataset and task filters")

    output_dir.mkdir(parents=True)
    rows = []
    entries = []
    for index, source in enumerate(tasks):
        downloaded = await TaskClient().download_tasks(
            task_ids=[source.get_task_id()], overwrite=source.overwrite, output_dir=source.download_dir
        )
        download = downloaded.results[0]
        # Legacy task names come from directory basenames. Keep the original name
        # inside each numbered slot so Harbor reports the same task at execution.
        named_snapshot = output_dir / "tasks" / f"{index:06d}" / download.path.name
        shutil.copytree(download.path, named_snapshot)
        task = Task(named_snapshot, disable_verification=job.verifier.disable)
        report = inspect_task(task, agent=job.agents[0], environment=job.environment)
        digest = hashlib.sha256()
        for file in sorted(named_snapshot.rglob("*")):
            if file.is_file():
                digest.update(file.relative_to(named_snapshot).as_posix().encode() + b"\0")
                digest.update(hashlib.sha256(file.read_bytes()).digest())
        local_task = TaskConfig(path=named_snapshot, source=source.source)
        rows.append(
            {
                "task_name": task.name,
                "harbor_task": local_task.model_dump(mode="json", exclude_none=True),
                "responses_create_params": {"input": []},
            }
        )
        entries.append(
            {
                "source": source.model_dump(mode="json", exclude_none=True),
                "download": download.model_dump(mode="json"),
                "snapshot": str(named_snapshot),
                "snapshot_sha256": digest.hexdigest(),
                "compatibility": report.model_dump(),
            }
        )
    compatible = all(not entry["compatibility"]["errors"] for entry in entries)
    manifest = {
        "harbor_version": version("harbor"),
        "scalar_reward_key": reward_key,
        "ready_for_smoke": compatible,
        "task_count": len(entries),
        "incompatible_count": sum(bool(entry["compatibility"]["errors"]) for entry in entries),
        "warnings": [
            "Static inspection only; model, credentials, service readiness and verifier behavior are not tested.",
            "Dataset-level custom metrics are not imported into Gym aggregation; retain Harbor scoring semantics.",
            "Task snapshots must be accessible at these absolute paths on every rollout worker.",
        ],
        "datasets": [dataset.model_dump(mode="json", exclude_none=True) for dataset in job.datasets],
        "tasks": entries,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not compatible:
        return False
    config = {
        "harbor_agent_general": {
            "responses_api_agents": {
                "harbor_agent_general": {
                    "entrypoint": "app.py",
                    "domain": "agent",
                    "harbor_jobs_dir": str(output_dir / "jobs"),
                    "harbor_reward_key": reward_key,
                    "harbor_agent": job.agents[0].model_dump(mode="json", exclude_none=True),
                    "harbor_environment": job.environment.model_dump(mode="json", exclude_none=True),
                    "harbor_verifier": job.verifier.model_dump(mode="json", exclude_none=True),
                }
            }
        }
    }
    (output_dir / "gym.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output_dir / "input.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return True


def main() -> None:
    """Prepare a registry/local dataset or the supported subset of a Harbor job config."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset", help="Harbor registry dataset name[@version]")
    source.add_argument("--path", type=Path, help="Local directory of Harbor tasks")
    source.add_argument(
        "--job-config", type=Path, help="Harbor YAML/JSON: datasets/tasks, agents, environment, verifier"
    )
    parser.add_argument("--agent", help="Harbor agent name (required unless --job-config is supplied)")
    parser.add_argument("--model", help="Model name understood by the Harbor agent")
    parser.add_argument("--environment", help="Harbor execution backend (default: docker)")
    parser.add_argument("--task", action="append", help="Include task name/glob; repeat for multiple filters")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-key", default="reward", help="Harbor objective used as Gym's scalar reward")
    args = parser.parse_args()
    if args.job_config:
        if any([args.agent, args.model, args.environment, args.task]):
            parser.error("Put agent, model, environment and task filters inside --job-config")
        job = load_job_config(args.job_config)
    else:
        if not args.agent:
            parser.error("--agent is required without --job-config")
        dataset_data = {"path": args.path} if args.path else {"name": args.dataset}
        if args.dataset and "@" in args.dataset:
            dataset_data["name"], dataset_data["version"] = args.dataset.rsplit("@", 1)
        job = JobConfig(
            datasets=[DatasetConfig(**dataset_data, task_names=args.task)],
            agents=[AgentConfig(name=args.agent, model_name=args.model)],
            environment=EnvironmentConfig(type=args.environment or "docker"),
        )
    compatible = asyncio.run(prepare(job, output_dir=args.output_dir, reward_key=args.reward_key))
    print(f"Compatibility report: {args.output_dir / 'manifest.json'}")
    if not compatible:
        parser.exit(1, "Incompatible tasks found; no runnable config or input JSONL was emitted.\n")
    print(f"Prepared Gym config: {args.output_dir / 'gym.yaml'}")


if __name__ == "__main__":
    main()
