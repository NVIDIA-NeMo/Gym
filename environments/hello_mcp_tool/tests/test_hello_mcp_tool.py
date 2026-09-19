# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from runpy import run_path
from typing import Any


root = Path(__file__).parents[1]
get_greeting = run_path(root / "tools/server.py")["get_greeting"]
verify = run_path(root / "verifier.py")["verify"]


@dataclass
class Workspace:
    root: Path

    async def read_text(self, path: str) -> str:
        relative = PurePosixPath(path).relative_to("/workspace")
        return self.root.joinpath(*relative.parts).read_text(encoding="utf-8")


@dataclass
class ToolCall:
    server: str
    name: str
    arguments: dict[str, Any]
    result: Any


@dataclass
class Attempt:
    workspace: Workspace
    tool_calls: list[ToolCall] = field(default_factory=list)


def score(workspace: Path, tool_calls: list[ToolCall]) -> float:
    attempt = Attempt(workspace=Workspace(root=workspace), tool_calls=tool_calls)
    return asyncio.run(verify(attempt, None))


def successful_call() -> ToolCall:
    return ToolCall(
        server="hello-tools",
        name="get_greeting",
        arguments={"name": "NeMo Gym"},
        result="Hello, NeMo Gym!",
    )


def test_get_greeting() -> None:
    assert get_greeting("NeMo Gym") == "Hello, NeMo Gym!"


def test_matching_call_and_file_pass(tmp_path: Path) -> None:
    (tmp_path / "tool-greeting.txt").write_text("Hello, NeMo Gym!\n", encoding="utf-8")

    assert score(tmp_path, [successful_call()]) == 1.0


def test_file_without_tool_call_fails(tmp_path: Path) -> None:
    (tmp_path / "tool-greeting.txt").write_text("Hello, NeMo Gym!", encoding="utf-8")

    assert score(tmp_path, []) == 0.0


def test_tool_call_without_file_fails(tmp_path: Path) -> None:
    assert score(tmp_path, [successful_call()]) == 0.0
