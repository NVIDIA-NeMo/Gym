# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from runpy import run_path


verify = run_path(Path(__file__).parents[1] / "verifier.py")["verify"]
TaskData = run_path(Path(__file__).parents[1] / "task.py")["TaskData"]


@dataclass
class Workspace:
    root: Path

    async def read_text(self, path: str) -> str:
        relative = PurePosixPath(path).relative_to("/workspace")
        return self.root.joinpath(*relative.parts).read_text(encoding="utf-8")


@dataclass
class Attempt:
    workspace: Workspace


@dataclass
class VerifierInput:
    path: str
    content: str


def score(workspace: Path, path: str, content: str) -> float:
    return asyncio.run(
        verify(
            Attempt(workspace=Workspace(root=workspace)),
            VerifierInput(path=path, content=content),
        )
    )


def test_taskset_rows_match_task_model() -> None:
    taskset = Path(__file__).parents[1] / "tasksets/example.jsonl"

    for line in taskset.read_text(encoding="utf-8").splitlines():
        TaskData.model_validate(json.loads(line)["task_data"])


def test_matching_file_passes(tmp_path: Path) -> None:
    (tmp_path / "hello-gym.txt").write_text("Hello from NeMo Gym!", encoding="utf-8")

    assert score(tmp_path, "/workspace/hello-gym.txt", "Hello from NeMo Gym!") == 1.0


def test_one_trailing_newline_is_accepted(tmp_path: Path) -> None:
    (tmp_path / "hello-gym.txt").write_text("Hello from NeMo Gym!\n", encoding="utf-8")

    assert score(tmp_path, "/workspace/hello-gym.txt", "Hello from NeMo Gym!") == 1.0


def test_missing_file_fails(tmp_path: Path) -> None:
    assert score(tmp_path, "/workspace/hello-gym.txt", "Hello from NeMo Gym!") == 0.0


def test_wrong_content_fails(tmp_path: Path) -> None:
    (tmp_path / "hello-gym.txt").write_text("Wrong", encoding="utf-8")

    assert score(tmp_path, "/workspace/hello-gym.txt", "Hello from NeMo Gym!") == 0.0
