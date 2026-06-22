# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest


HARBOR_AGENT_ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER = HARBOR_AGENT_ROOT / "scripts" / "materialize_biomnibench_da.py"


def _write_minimal_source_task(task_root: Path, task_id: str) -> None:
    task_dir = task_root / task_id
    (task_dir / "environment" / "data").mkdir(parents=True)
    (task_dir / "environment" / "data" / "sample.txt").write_text("hello\n", encoding="utf-8")
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "tests" / "rubric.txt").write_text("Score the agent.\n", encoding="utf-8")
    (task_dir / "instruction.md").write_text(f"Task {task_id}\n", encoding="utf-8")
    (task_dir / "task.toml").write_text(
        """
version = "1.0"

[metadata]
task_type = "analysis"
category = "test"
difficulty = "easy"

[agent]
timeout_sec = 60.0

[verifier]
timeout_sec = 30.0

[environment]
storage_mb = 1024
memory_mb = 1024
cpus = 1
gpus = 0
allow_internet = false
""".strip()
        + "\n",
        encoding="utf-8",
    )


@pytest.fixture()
def minimal_source(tmp_path: Path) -> Path:
    for task_id in ("da-1-3", "da-1-4"):
        _write_minimal_source_task(tmp_path, task_id)
    return tmp_path


def _run_materializer(
    source: Path,
    output: Path,
    environment_type: str,
    *,
    overwrite: bool = True,
) -> None:
    cmd = [
        sys.executable,
        str(MATERIALIZER),
        "--local-dir",
        str(source),
        "--output-dir",
        str(output),
        "--environment-type",
        environment_type,
        "--tasks",
        "da-1-3",
        "--n-repeats",
        "1",
        "--partition",
        "all",
        "--include-singletons",
        "--include-uncovered",
    ]
    if overwrite:
        cmd.append("--overwrite")
    subprocess.run(cmd, check=True, cwd=HARBOR_AGENT_ROOT.parents[1])


def test_materialize_docker_bind_compose(minimal_source: Path, tmp_path: Path) -> None:
    output = tmp_path / "docker_tasks"
    _run_materializer(minimal_source, output, "docker")

    task_dir = output / "da-1-3-r001"
    compose = task_dir / "environment" / "docker-compose.yaml"
    assert compose.is_file()
    text = compose.read_text(encoding="utf-8")
    assert "/app/data" in text
    assert str((minimal_source / "da-1-3" / "environment" / "data").resolve()) in text

    toml = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    assert toml["environment"]["docker_image"] == "biomnibench-da-runtime:smoke"
    assert (task_dir / "tests" / "llm_judge.py").is_file()


def test_materialize_singularity_staging(minimal_source: Path, tmp_path: Path) -> None:
    output = tmp_path / "sing_tasks"
    _run_materializer(minimal_source, output, "singularity")

    task_dir = output / "da-1-3-r001"
    env = task_dir / "environment"
    assert not (env / "data").exists()
    assert (env / "files" / "data" / "sample.txt").read_text(encoding="utf-8") == "hello\n"
    setup = (env / "files" / "setup.sh").read_text(encoding="utf-8")
    assert "HARBOR_STAGING" in setup
    assert "/app/data" in setup
    assert not (env / "docker-compose.yaml").exists()

    toml = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    assert toml["environment"]["docker_image"] == "biomnibench-da-runtime:smoke"
