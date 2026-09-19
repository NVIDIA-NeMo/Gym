# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from runpy import run_path

from nemo_gym.verifiers.files import text_file_equals


uppercase_greeting_verify = run_path(Path(__file__).parents[1] / "tasks/uppercase-greeting/verifier.py")["verify"]


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


def test_reused_verifier_passes_exact_greeting(tmp_path: Path) -> None:
    (tmp_path / "hello-gym.txt").write_text("Hello from NeMo Gym!\n", encoding="utf-8")
    attempt = Attempt(workspace=Workspace(root=tmp_path))
    verifier_input = VerifierInput(
        path="/workspace/hello-gym.txt",
        content="Hello from NeMo Gym!",
    )

    assert asyncio.run(text_file_equals(attempt, verifier_input)) == 1.0


def test_reused_verifier_fails_when_exact_greeting_file_is_missing(tmp_path: Path) -> None:
    attempt = Attempt(workspace=Workspace(root=tmp_path))
    verifier_input = VerifierInput(
        path="/workspace/hello-gym.txt",
        content="Hello from NeMo Gym!",
    )

    assert asyncio.run(text_file_equals(attempt, verifier_input)) == 0.0


def test_task_local_verifier_passes_uppercase_greeting(tmp_path: Path) -> None:
    (tmp_path / "shout.txt").write_text("HELLO FROM NEMO GYM!", encoding="utf-8")
    attempt = Attempt(workspace=Workspace(root=tmp_path))

    assert asyncio.run(uppercase_greeting_verify(attempt, None)) == 1.0


def test_task_local_verifier_rejects_lowercase_greeting(tmp_path: Path) -> None:
    (tmp_path / "shout.txt").write_text("Hello from NeMo Gym!", encoding="utf-8")
    attempt = Attempt(workspace=Workspace(root=tmp_path))

    assert asyncio.run(uppercase_greeting_verify(attempt, None)) == 0.0
