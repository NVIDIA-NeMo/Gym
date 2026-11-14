# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from pathlib import Path

import pytest

from nemo_gym.sandbox import SandboxExecResult
from resources_servers.swebench.sandbox_setup import read_sandbox_files, setup_command_runner


class TestReadSandboxFiles:
    def test_absolute_source(self, tmp_path: Path) -> None:
        source = tmp_path / "settings.xml"
        source.write_text("<settings/>")
        assert read_sandbox_files({"/root/.m2/settings.xml": str(source)}) == {"/root/.m2/settings.xml": "<settings/>"}

    def test_relative_source_resolves_against_the_gym_root(self) -> None:
        files = read_sandbox_files(
            {"/tmp/chrome-wrapper.sh": "resources_servers/swebench/toolchain/chrome-wrapper.sh"}
        )
        assert files["/tmp/chrome-wrapper.sh"].startswith("#!/bin/sh\n")

    def test_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            read_sandbox_files({"/tmp/x": str(tmp_path / "absent")})

    def test_empty(self) -> None:
        assert read_sandbox_files({}) == {}


class _ExecSandbox:
    def __init__(self, return_code: int, error_type: str | None = None) -> None:
        self.return_code = return_code
        self.error_type = error_type
        self.commands: list[str] = []

    async def exec(self, command: str) -> SandboxExecResult:
        self.commands.append(command)
        return SandboxExecResult(stdout="", stderr="boom", return_code=self.return_code, error_type=self.error_type)


class TestSetupCommandRunner:
    async def test_runs_the_command(self) -> None:
        sandbox = _ExecSandbox(0)
        await setup_command_runner("chmod +x /tmp/w.sh")(sandbox)
        assert sandbox.commands == ["chmod +x /tmp/w.sh"]

    async def test_failure_raises_with_stderr(self) -> None:
        with pytest.raises(RuntimeError, match="exited 2: boom"):
            await setup_command_runner("false")(_ExecSandbox(2))

    async def test_an_error_type_raises_even_with_exit_0(self) -> None:
        with pytest.raises(RuntimeError, match=r"exited 0 \(timeout\): boom"):
            await setup_command_runner("sleep 999")(_ExecSandbox(0, "timeout"))
