# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
import yaml
from harbor.models.job.config import DatasetConfig, JobConfig
from harbor.models.task.task import Task
from harbor.models.trial.config import AgentConfig, EnvironmentConfig

from responses_api_agents.harbor_agent_general.app import HarborAgentConfig
from responses_api_agents.harbor_agent_general.prepare import inspect_task, load_job_config, prepare


def make_task(root: Path, name: str, *, compose: bool = False, multi_step: bool = False) -> Path:
    task = root / name
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.13-slim\nWORKDIR /app\n")
    config = ""
    if compose:
        config += '\n[[environment.mcp_servers]]\nname = "service"\nurl = "http://service:8000/mcp"\n'
        (task / "environment" / "docker-compose.yaml").write_text(
            "services:\n  main:\n    depends_on: [service]\n  service:\n    image: python:3.13-slim\n"
        )
    if multi_step:
        config += '\n[[steps]]\nname = "first"\n\n[[steps]]\nname = "second"\n'
    (task / "task.toml").write_text(config)
    for step in [task / "steps" / "first", task / "steps" / "second"] if multi_step else [task]:
        (step / "tests").mkdir(parents=True)
        (step / "instruction.md").write_text(f"Solve {step.name}")
        (step / "tests" / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    return task


@pytest.mark.asyncio
async def test_prepare_resolves_filters_snapshots_and_emits_runnable_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    selected = make_task(dataset, "selected", compose=True, multi_step=True)
    make_task(dataset, "excluded")
    job = JobConfig(
        datasets=[DatasetConfig(path=dataset, task_names=["sel*"])],
        agents=[AgentConfig(name="opencode", model_name="provider/model")],
    )
    output = tmp_path / "prepared"
    assert await prepare(job, output_dir=output)
    rows = [json.loads(line) for line in (output / "input.jsonl").read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["task_name"] == "selected"
    snapshot = Path(rows[0]["harbor_task"]["path"])
    assert (snapshot / "environment" / "docker-compose.yaml").read_bytes() == (
        selected / "environment" / "docker-compose.yaml"
    ).read_bytes()
    (selected / "steps" / "first" / "instruction.md").write_text("changed source")
    assert (snapshot / "steps" / "first" / "instruction.md").read_text() == "Solve first"
    manifest = json.loads((output / "manifest.json").read_text())
    report = manifest["tasks"][0]["compatibility"]
    assert report["compose"]
    assert report["mcp_servers"] == ["service"]
    assert report["steps"] == ["first", "second"]
    assert report["warnings"]
    config = yaml.safe_load((output / "gym.yaml").read_text())["harbor_agent_general"]["responses_api_agents"][
        "harbor_agent_general"
    ]
    config.update(name="harbor_agent_general", host="127.0.0.1", port=8080)
    agent_config = HarborAgentConfig.model_validate(config)
    from harbor.models.trial.config import TaskConfig

    run = agent_config.build_job_config("selected", "job", task=TaskConfig.model_validate(rows[0]["harbor_task"]))
    assert not run.datasets
    assert run.tasks[0].path == snapshot
    assert run.agents[0].model_name == "provider/model"


@pytest.mark.asyncio
async def test_incompatible_agent_reports_all_tasks_without_runnable_output(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    make_task(dataset, "a")
    make_task(dataset, "b")
    output = tmp_path / "prepared"
    assert not await prepare(JobConfig(datasets=[DatasetConfig(path=dataset)]), output_dir=output)
    report = json.loads((output / "manifest.json").read_text())
    assert report["task_count"] == report["incompatible_count"] == 2
    assert "ATIF" in report["tasks"][0]["compatibility"]["errors"][0]
    assert not (output / "input.jsonl").exists()
    assert not (output / "gym.yaml").exists()


def test_resume_requirement_is_checked(tmp_path: Path) -> None:
    task = Task(make_task(tmp_path, "multi", multi_step=True))
    report = inspect_task(
        task,
        agent=AgentConfig(name="terminus-2", resume_trajectory=True),
        environment=EnvironmentConfig(type="docker"),
    )
    assert any("resume" in error for error in report.errors)


def test_compose_is_rejected_for_singularity(tmp_path: Path) -> None:
    task = Task(make_task(tmp_path, "sidecar", compose=True))
    # Singularity requires a prebuilt image even for single-container tasks.
    task.config.environment.docker_image = "python:3.13-slim"
    report = inspect_task(
        task,
        agent=AgentConfig(name="opencode"),
        environment=EnvironmentConfig(
            type="singularity", kwargs={"singularity_image_cache_dir": str(tmp_path / "image-cache")}
        ),
    )
    assert report.compose
    assert any("does not support Docker Compose" in error for error in report.errors)


@pytest.mark.asyncio
async def test_prepare_refuses_overwrite_and_empty_dataset(tmp_path: Path) -> None:
    job = JobConfig(datasets=[DatasetConfig(path=tmp_path)])
    with pytest.raises(FileExistsError):
        await prepare(job, output_dir=tmp_path)
    with pytest.raises(ValueError, match="no tasks"):
        await prepare(job, output_dir=tmp_path / "output")
    assert not (tmp_path / "output").exists()


def test_job_config_preserves_mcp_and_rejects_unmapped_settings(tmp_path: Path) -> None:
    path = tmp_path / "job.yaml"
    path.write_text(
        'datasets: [{name: "org/dataset", version: "1"}]\n'
        'agents: [{name: opencode, mcp_servers: [{name: tools, url: "http://tools/mcp"}]}]\n'
        "environment: {type: docker, kwargs: {keep_containers: true}}\n"
    )
    job = load_job_config(path)
    assert job.agents[0].mcp_servers[0].name == "tools"
    assert job.environment.kwargs == {"keep_containers": True}
    path.write_text(path.read_text() + "timeout_multiplier: 2\n")
    with pytest.raises(ValueError, match="timeout_multiplier"):
        load_job_config(path)
