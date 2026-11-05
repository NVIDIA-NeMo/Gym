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
"""deepagents backend that runs a graph's file and shell tools in a Gym task sandbox.

`deepagents.backends.sandbox.BaseSandbox` builds every file tool (`ls`, `read_file`, `write_file`, `edit_file`,
`glob`, `grep`) on three primitives: run a command, upload files and download files. `TaskSandboxBackend` provides
them through `nemo_gym.sandbox.AsyncSandbox`, and `deepagents` offers the graph its `execute` tool because the
backend is a `SandboxBackendProtocol`. `BaseSandbox` runs its read, write, edit, `ls` and `glob` helpers as `python3`
scripts, so those tools need `python3` in the task image; `execute` and `grep` do not.

`task_sandbox_backend()` routes `deepagents`' own files (offloaded conversation history and evicted large tool
results) to graph state instead, so the harness writes nothing of its own into the task sandbox.
"""

import tempfile
from pathlib import Path

from deepagents.backends import DEFAULT_EXECUTE_TIMEOUT, CompositeBackend, StateBackend
from deepagents.backends.protocol import ExecuteResponse, FileDownloadResponse, FileUploadResponse
from deepagents.backends.sandbox import BaseSandbox

from nemo_gym.sandbox import AsyncSandbox


# Where deepagents keeps the files it writes for itself. `artifacts_root` moves both of its prefixes
# (`conversation_history/` and `large_tool_results/`) under this one route.
HARNESS_FILES_ROOT = "/.deepagents/"


class TaskSandboxError(RuntimeError):
    """The sandbox runtime could not run a command, so the graph cannot continue in that sandbox."""


class TaskSandboxBackend(BaseSandbox):
    """Run `deepagents`' `execute` and file tools in a task sandbox through `AsyncSandbox`.

    Only the async methods are implemented: the graph runs with `ainvoke()`, and `AsyncSandbox` belongs to the
    server's event loop. Commands run in `workdir`, with `env` over the sandbox's own environment. A command that
    times out returns its output and a timeout note, as `deepagents`' `LocalShellBackend` does. A runtime failure
    without an exit code raises `TaskSandboxError`, which ends the activation instead of letting the model continue
    in a sandbox that no longer runs commands.
    """

    def __init__(self, sandbox: AsyncSandbox, *, workdir: str, env: dict[str, str]) -> None:
        self._sandbox = sandbox
        self._workdir = workdir
        self._env = env

    @property
    def id(self) -> str:
        return self._sandbox.sandbox_id or ""

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        raise NotImplementedError("TaskSandboxBackend runs commands only through aexecute()")

    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        raise NotImplementedError("TaskSandboxBackend uploads files only through aupload_files()")

    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        raise NotImplementedError("TaskSandboxBackend downloads files only through adownload_files()")

    async def aexecute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:  # noqa: ASYNC109
        # deepagents documents timeout=0 as "no timeout where supported"; a sandbox command always gets a bound.
        timeout_s = timeout or DEFAULT_EXECUTE_TIMEOUT
        result = await self._sandbox.exec(command, cwd=self._workdir, env=self._env or None, timeout_s=timeout_s)
        # Combined as deepagents' own LangSmithSandbox does: stdout, then stderr.
        output = result.stdout or ""
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr
        if result.error_type == "timeout":
            note = f"Error: Command timed out after {timeout_s} seconds. For long-running commands, re-run using the timeout parameter."
            return ExecuteResponse(output=f"{output}\n{note}" if output else note, exit_code=124)
        if result.error_type is not None:
            raise TaskSandboxError(
                f"sandbox {self.id} could not run a command ({result.error_type}): {(result.stderr or '')[:500]}"
            )
        return ExecuteResponse(output=output, exit_code=result.return_code)

    async def aupload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        responses = []
        with tempfile.TemporaryDirectory(prefix="nemo-gym-deepagents-") as directory:
            for index, (path, content) in enumerate(files):
                local = Path(directory) / str(index)
                local.write_bytes(content)
                try:
                    await self._sandbox.upload(local, path)
                except Exception as error:
                    responses.append(FileUploadResponse(path=path, error=f"{type(error).__name__}: {error}"))
                else:
                    responses.append(FileUploadResponse(path=path))
        return responses

    async def adownload_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        responses = []
        with tempfile.TemporaryDirectory(prefix="nemo-gym-deepagents-") as directory:
            for index, path in enumerate(paths):
                local = Path(directory) / str(index)
                try:
                    await self._sandbox.download(path, local)
                    content = local.read_bytes()
                except Exception as error:
                    responses.append(FileDownloadResponse(path=path, error=f"{type(error).__name__}: {error}"))
                else:
                    responses.append(FileDownloadResponse(path=path, content=content))
        return responses


def task_sandbox_backend(sandbox: AsyncSandbox, *, workdir: str, env: dict[str, str]) -> CompositeBackend:
    """The backend for a graph whose tools run in `sandbox`, with `deepagents`' own files kept in graph state."""
    return CompositeBackend(
        default=TaskSandboxBackend(sandbox, workdir=workdir, env=env),
        routes={HARNESS_FILES_ROOT: StateBackend()},
        artifacts_root=HARNESS_FILES_ROOT,
    )
