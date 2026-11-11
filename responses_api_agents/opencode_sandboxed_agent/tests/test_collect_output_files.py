# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest

from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.opencode_sandboxed_agent.app import collect_output_files


class FakeSandbox:
    """Holds files by absolute path; answers `test -f` and download like a sandbox provider."""

    def __init__(self, files: dict[str, str], *, broken_download: bool = False) -> None:
        self.files = files
        self.broken_download = broken_download
        self.commands: list[str] = []

    async def exec(self, command: str, **_: object) -> SandboxExecResult:
        self.commands.append(command)
        path = command.removeprefix("test -f ").strip("'")
        return SandboxExecResult(stdout="", stderr="", return_code=0 if path in self.files else 1)

    async def download(self, remote_path: str, local_path: Path) -> None:
        if self.broken_download:
            raise RuntimeError("provider error")
        Path(local_path).write_text(self.files[remote_path])


async def test_collects_the_named_files_from_the_workdir_and_skips_missing_ones() -> None:
    sandbox = FakeSandbox({"/workspace/results_a.jsonl": '{"a": 1}\nnot json\n', "/workspace/notes.md": "x"})
    files = await collect_output_files(sandbox, "/workspace", ["results_a.jsonl", "results_b.jsonl"])
    assert files == {"results_a.jsonl": '{"a": 1}\nnot json\n'}


@pytest.mark.parametrize("name", ["/etc/passwd", "../secret.jsonl", "a/../../b.jsonl"])
async def test_refuses_names_outside_the_workdir(name: str) -> None:
    sandbox = FakeSandbox({"/etc/passwd": "root"})
    assert await collect_output_files(sandbox, "/workspace", [name]) == {}
    assert sandbox.commands == []


async def test_a_failed_download_of_an_existing_file_is_raised_not_scored_as_missing() -> None:
    sandbox = FakeSandbox({"/workspace/results_a.jsonl": "{}\n"}, broken_download=True)
    with pytest.raises(RuntimeError, match="provider error"):
        await collect_output_files(sandbox, "/workspace", ["results_a.jsonl"])
