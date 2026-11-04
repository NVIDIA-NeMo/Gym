# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
# The resources server and the environment ship the same dataset script; both
# read ../../ARC-AGI{,-2} relative to their own directory.
SCRIPTS = [
    "resources_servers/arc_agi/create_dataset.py",
    "environments/arc_agi/prepare.py",
]


def _write_task(split_dir: Path, task_id: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    task = {
        "train": [{"input": [[0]], "output": [[1]]}],
        "test": [{"input": [[2]], "output": [[3]]}],
    }
    (split_dir / f"{task_id}.json").write_text(json.dumps(task))


def _gym_tree(tmp_path: Path, script: str) -> Path:
    """Return the script's directory inside a Gym-shaped tree, like the README's cwd."""
    component = tmp_path / "Gym" / Path(script).parent
    component.mkdir(parents=True)
    shutil.copy2(REPO_ROOT / script, component / Path(script).name)
    return component


def _run(component: Path, script: str, version: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, Path(script).name, "--version", str(version)],
        cwd=component,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("version", [1, 2])
def test_builds_every_split_from_clone_in_gym_root(tmp_path: Path, script: str, version: int) -> None:
    component = _gym_tree(tmp_path, script)
    source = tmp_path / "Gym" / ("ARC-AGI" if version == 1 else "ARC-AGI-2") / "data"
    _write_task(source / "training", "t1")
    _write_task(source / "evaluation", "e1")
    _write_task(source / "evaluation", "e2")

    result = _run(component, script, version)

    assert result.returncode == 0, result.stderr
    data = component / "data"
    training = [json.loads(line) for line in (data / f"arc_agi_{version}_training.jsonl").read_text().splitlines()]
    evaluation = [json.loads(line) for line in (data / f"arc_agi_{version}_evaluation.jsonl").read_text().splitlines()]
    example = (data / f"example_{version}.jsonl").read_text().splitlines()
    assert [row["task_id"] for row in training] == ["t1"]
    assert [row["task_id"] for row in evaluation] == ["e1", "e2"]
    assert len(example) == 2
    assert "Created example dataset with 2 tasks" in result.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_clone_outside_gym_root_fails_without_writing(tmp_path: Path, script: str) -> None:
    # The old resources README cloned ARC-AGI-2 next to the script instead of
    # into the Gym root; the script then wrote empty datasets and exited 0.
    component = _gym_tree(tmp_path, script)
    _write_task(component / "ARC-AGI-2" / "data" / "training", "t1")
    _write_task(component / "ARC-AGI-2" / "data" / "evaluation", "e1")

    result = _run(component, script, 2)

    assert result.returncode != 0
    assert "No ARC-AGI-2 tasks" in result.stderr
    assert "https://github.com/arcprize/ARC-AGI-2" in result.stderr
    assert not (component / "data").exists()


@pytest.mark.parametrize("script", SCRIPTS)
@pytest.mark.parametrize("empty_split", ["training", "evaluation"])
def test_empty_split_fails_without_writing(tmp_path: Path, script: str, empty_split: str) -> None:
    component = _gym_tree(tmp_path, script)
    source = tmp_path / "Gym" / "ARC-AGI" / "data"
    for split in ("training", "evaluation"):
        if split == empty_split:
            (source / split).mkdir(parents=True)
        else:
            _write_task(source / split, "t1")

    result = _run(component, script, 1)

    assert result.returncode != 0
    assert f"No ARC-AGI-1 tasks in {(source / empty_split).resolve()}" in result.stderr
    assert not (component / "data").exists()
