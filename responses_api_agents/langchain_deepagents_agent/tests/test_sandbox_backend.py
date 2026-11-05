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
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from deepagents.backends import DEFAULT_EXECUTE_TIMEOUT

from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.langchain_deepagents_agent.sandbox_backend import TaskSandboxBackend, TaskSandboxError


def _backend(exec_result: SandboxExecResult | None = None, **sandbox) -> tuple[TaskSandboxBackend, SimpleNamespace]:
    sandbox = SimpleNamespace(sandbox_id="sb-1", exec=AsyncMock(return_value=exec_result), **sandbox)
    return TaskSandboxBackend(sandbox, workdir="/app", env={"TASK": "1"}), sandbox


async def test_execute_runs_in_the_workdir_with_the_owner_env_and_combines_output():
    backend, sandbox = _backend(SandboxExecResult(stdout="out\n", stderr="warn", return_code=3))

    result = await backend.aexecute("make")

    sandbox.exec.assert_awaited_once_with("make", cwd="/app", env={"TASK": "1"}, timeout_s=DEFAULT_EXECUTE_TIMEOUT)
    assert (result.output, result.exit_code) == ("out\n\nwarn", 3)


async def test_a_timed_out_command_returns_its_output_and_a_timeout_note():
    backend, sandbox = _backend(
        SandboxExecResult(stdout="partial", stderr=None, return_code=124, error_type="timeout")
    )

    result = await backend.aexecute("sleep 99", timeout=5)

    assert sandbox.exec.await_args.kwargs["timeout_s"] == 5
    assert result.exit_code == 124
    assert result.output.startswith("partial\nError: Command timed out after 5 seconds.")


async def test_a_sandbox_runtime_failure_raises_instead_of_reaching_the_model():
    backend, _ = _backend(
        SandboxExecResult(stdout=None, stderr="container gone", return_code=-1, error_type="sandbox")
    )

    with pytest.raises(TaskSandboxError, match="container gone"):
        await backend.aexecute("ls")


async def test_file_transfers_report_each_failure_without_dropping_the_others():
    uploaded: dict[str, bytes] = {}

    async def upload(local: Path, remote: str) -> None:
        if remote == "/readonly/x":
            raise PermissionError("read-only file system")
        uploaded[remote] = local.read_bytes()

    async def download(remote: str, local: Path) -> None:
        if remote == "/missing":
            raise RuntimeError("no such file")
        local.write_bytes(b"content of " + remote.encode())

    backend, _ = _backend(upload=upload, download=download)

    uploads = await backend.aupload_files([("/app/a.txt", b"a"), ("/readonly/x", b"x"), ("/app/b.txt", b"b")])
    downloads = await backend.adownload_files(["/app/a.txt", "/missing"])

    assert [response.error for response in uploads] == [None, "PermissionError: read-only file system", None]
    assert uploaded == {"/app/a.txt": b"a", "/app/b.txt": b"b"}
    assert [(response.content, response.error) for response in downloads] == [
        (b"content of /app/a.txt", None),
        (None, "RuntimeError: no such file"),
    ]
