# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from environments.biomnibench_da import prepare


ENV_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = ENV_ROOT.parents[1]
PREPARE = ENV_ROOT / "prepare.py"


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


def _run_prepare(
    source: Path,
    output: Path,
    environment_type: str,
    *,
    overwrite: bool = True,
) -> None:
    cmd = [
        sys.executable,
        str(PREPARE),
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
    subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def test_prepare_docker_bind_compose(minimal_source: Path, tmp_path: Path) -> None:
    output = tmp_path / "docker_tasks"
    _run_prepare(minimal_source, output, "docker")

    task_dir = output / "da-1-3-r001"
    compose = task_dir / "environment" / "docker-compose.yaml"
    assert compose.is_file()
    text = compose.read_text(encoding="utf-8")
    assert "/app/data" in text
    assert "pull_policy: never" in text
    assert str((minimal_source / "da-1-3" / "environment" / "data").resolve()) in text

    toml = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8"))
    assert toml["environment"]["docker_image"] == "biomnibench-da-runtime:smoke"
    assert (task_dir / "tests" / "llm_judge.py").is_file()

    rollout_input = output / "rollout_input.jsonl"
    assert rollout_input.is_file()
    rows = [json.loads(line) for line in rollout_input.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 1
    assert rows[0]["instance_id"] == "biomnibench_da::da-1-3-r001"
    assert rows[0]["agent_ref"] == {"name": "harbor_agent"}


def test_prepare_singularity_staging(minimal_source: Path, tmp_path: Path) -> None:
    output = tmp_path / "sing_tasks"
    _run_prepare(minimal_source, output, "singularity")

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


def test_prepare_requires_local_dir_or_download(tmp_path: Path) -> None:
    missing_local_dir = tmp_path / "does_not_exist"
    output = tmp_path / "out"
    result = subprocess.run(
        [
            sys.executable,
            str(PREPARE),
            "--local-dir",
            str(missing_local_dir),
            "--output-dir",
            str(output),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "--download" in result.stderr


def test_requested_download_patterns() -> None:
    args = type("Args", (), {"tasks": ["da-10-1"], "papers": ["da-11"]})()
    assert prepare.requested_download_patterns(args) == ["da-10-1/**", "da-11-*/**"]


def test_build_docker_image_uses_requested_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(prepare.subprocess, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    prepare.build_docker_image("custom-runtime:test")

    assert calls[0][0][0] == [
        "docker",
        "build",
        "-t",
        "custom-runtime:test",
        "-f",
        prepare.DEFAULT_DOCKERFILE,
        str(prepare.ENV_ROOT),
    ]
    assert calls[0][1]["check"] is True
