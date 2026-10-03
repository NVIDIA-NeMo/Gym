# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from runpy import run_path


verify = run_path(Path(__file__).parents[1] / "verifier.py")["verify"]


@dataclass
class Workspace:
    root: Path

    async def read_text(self, path: str) -> str:
        relative = PurePosixPath(path).relative_to("/workspace")
        return self.root.joinpath(*relative.parts).read_text(encoding="utf-8")


@dataclass
class Attempt:
    workspace: Workspace


def score(workspace: Path) -> float:
    return asyncio.run(verify(Attempt(workspace=Workspace(root=workspace)), None))


def test_matching_file_passes(tmp_path: Path) -> None:
    (tmp_path / "hello-gym.txt").write_text("Hello from NeMo Gym!", encoding="utf-8")

    assert score(tmp_path) == 1.0


def test_missing_file_fails(tmp_path: Path) -> None:
    assert score(tmp_path) == 0.0


def test_wrong_content_fails(tmp_path: Path) -> None:
    (tmp_path / "hello-gym.txt").write_text("Wrong", encoding="utf-8")

    assert score(tmp_path) == 0.0
