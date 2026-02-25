# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import anyio
import re
import shutil
import subprocess
import tempfile
from typing import Dict, List

from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel, Field

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
)

SHELL_TIMEOUT = 30

class Session(BaseModel):
    # All code execution and file access happens in the temp directory
    temp_dir: Path

    def __init__(self, temp_dir_base: Path):
        temp_dir_base.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(tempfile.mkdtemp(prefix="local_sandbox_", dir=temp_dir_base))

class SessionManager:
    id_to_session: Dict[str, Session]
    temp_dir_base: Path

    def __init__(self, temp_dir_base: Path):
        self.id_to_session = {}
        self.temp_dir_base = temp_dir_base

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.id_to_session

    def get_session(self, session_id: str) -> Session:
        return self.id_to_session[session_id]

    def maybe_start_session(self, session_id: str) -> Session:
        if not self.session_exists(session_id):
            self.id_to_session[session_id] = Session(temp_dir_base=self.temp_dir_base)
        return self.get_session(session_id)

    def end_session(self, session_id: str) -> None:
        session = self.id_to_session.pop(session_id, None)
        if session:
            shutil.rmtree(session.temp_dir_path)

@dataclass
class UploadedFile:
    """Information about a file uploaded to the execution environment."""

    source_path: Path  # Original path on local filesystem
    dest_path: str  # Path in the execution environment
    size: int

@dataclass
class SavedFile:
    """Information about a file saved from the execution environment."""

    source_path: str  # Original path in execution environment
    output_path: Path  # Path where file was saved
    size: int

class BashSandboxResourcesServerConfig(BaseResourcesServerConfig):
    temp_dir_base: Path = Field(default_factory=lambda: Path("/tmp/nemo_gym_bash_sandboxes"))
    allowlist: List[str] = Field(default_factory=list)

class RunCommandRequest(BaseModel):
    command: str
    session_id: str
    timeout: int = SHELL_TIMEOUT

class RunCommandResponse(BaseModel):
    exit_code: int
    stdout: str
    stderr: str
    error_kind: str | None = None
    advice: str | None = None
    
class UploadFilesRequest(BaseModel):
    """
    paths: File or directory paths to upload.
    session_id: ID of the session to upload files to.
    """
   
    paths: List[str]
    session_id: str
    dest_dir: str | None = None

class UploadFilesResponse(BaseModel):
    """
    uploaded: List[UploadedFile]
    failed: List[str]
    """
    uploaded: List[UploadedFile]
    failed: Dict[str, str]

class SaveOutputFilesRequest(BaseModel):
    """
    paths: List[str]
    session_id: str
    output_dir: str
    """
    paths: List[str]
    session_id: str
    output_dir: str

class SaveOutputFilesResponse(BaseModel):
    """
    saved: List[SavedFile]
    failed: List[str]
    """
    saved: List[SavedFile]
    failed: List[str, str]
    error_message: str | None = None

class EndSessionRequest(BaseModel):
    """
    session_id: str
    """
    session_id: str
    paths: List[str] | None = None
    output_dir: str | None = None

class EndSessionResponse(BaseModel):
    """
    success: bool
    failed: List[str]
    """
    session_deleted: bool
    saved: List[SavedFile] = Field(default_factory=list)
    failed: List[str, str] = Field(default_factory=dict)
    error_message: str | None = None

class BashSandboxResourcesServer(SimpleResourcesServer):
    config: BashSandboxResourcesServerConfig
    session_manager: SessionManager

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_manager = SessionManager(Path(self.config.temp_dir_base))

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Register tool endpoints
        # These will be called via /{tool_name} pattern from GDPValAgent
        app.post("/run_command")(self.run_command)
        app.post("/upload_files")(self.upload_files)
        app.post("/save_files")(self.save_output_files)
        app.post("/end_session")(self.end_session)  # Finish tool for task completion
        # app.post("/web_search")(self.web_search) # TODO: Implement web search

        return app

    def _check_allowed(self, cmd: str) -> bool:
        """Check if command is allowed based on the allowlist.

        Returns:
            True if the command is allowed, False otherwise.

        """
        if self.config.allowlist is None:
            return True  # No allowlist = allow all
        return any(p.search(cmd) for p in self.config.allowlist)

    def _resolve_and_validate_path(self, path: str, session: Session) -> Path:
        """Resolve a path and validate it's within the temp directory.

        Args:
            path: File path (relative or absolute within the temp dir).

        Returns:
            Resolved absolute Path.

        Raises:
            RuntimeError: If environment not started.
            ValueError: If path is outside temp directory.
            FileNotFoundError: If path does not exist (for reads).

        """
        if session.temp_dir is None:
            raise RuntimeError("ExecutionEnvironment not started.")

        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = session.temp_dir / resolved

        # Security: ensure path is within temp directory
        try:
            resolved.resolve().relative_to(session.temp_dir.resolve())
        except ValueError as e:
            raise ValueError(f"Path is outside execution environment: {path}") from e

        return resolved

    def _check_absolute_paths(self, cmd: str) -> RunCommandResponse | None:
        """Check if command contains absolute paths that could escape the temp directory.

        Returns:
            CommandResult with error if absolute paths detected, None otherwise.

        Note:
            This check is specific to LocalCodeExecToolProvider since Docker and E2B
            providers are already sandboxed and absolute paths are safe within them.
        """
        absolute_patterns = [
            r"~/",  # ~/path - home directory shortcut
            r"/(?:home|Users|tmp|var|etc)/",  # /home/, /Users/, /tmp/, etc.
            r"\$HOME/",  # $HOME/path
            r"\$\{HOME\}/",  # ${HOME}/path
        ]
        for pattern in absolute_patterns:
            if re.search(pattern, cmd):
                return RunCommandResponse(
                    exit_code=1,
                    stdout="",
                    stderr=(
                        "Command appears to use absolute paths which could write outside "
                        "the execution environment. Use relative paths instead."
                    ),
                    error_kind="absolute_path_detected",
                    advice=(
                        "Use relative paths (e.g., './output.txt' instead of '~/output.txt'). "
                        "For full filesystem access, use DockerCodeExecToolProvider or E2BCodeExecToolProvider."
                    ),
                )
        return None

    async def run_command(self, body: RunCommandRequest) -> RunCommandResponse:
        """Execute command in the temp directory for the session specified by the session ID.

        Args:
            body: RunCommandRequest containing the command and session ID.

        Returns:
            RunCommandResponse with exit_code, stdout, stderr, and optional error info.

        """
        session = self.session_manager.get_session(body.session_id)

        if session.temp_dir is None:
            raise RuntimeError(
                "ExecutionEnvironment not started. Ensure current Agent is equipped with a CodeExecToolProvider."
            )

        # Check allowlist
        if not self._check_allowed(body.command):
            return RunCommandResponse(
                exit_code=1,
                stdout="",
                stderr=f"Command not allowed: '{body.command}' does not match any allowed patterns",
                error_kind="command_not_allowed",
                advice="Only commands matching the allowlist patterns are permitted.",
            )

        # Check for absolute paths (local environment is not sandboxed)
        absolute_path_error = self._check_absolute_paths(body.command)
        if absolute_path_error:
            return absolute_path_error

        process = None
        try:
            with anyio.fail_after(body.timeout):
                # Use shell=True by wrapping in a shell command
                process = await anyio.open_process(
                    ["bash", "-c", body.command],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=session.temp_dir,
                )

                # Read all output from streams concurrently
                stdout_chunks: list[bytes] = []
                stderr_chunks: list[bytes] = []

                async def read_stdout() -> None:
                    if process.stdout:
                        stdout_chunks.extend([chunk async for chunk in process.stdout])

                async def read_stderr() -> None:
                    if process.stderr:
                        stderr_chunks.extend([chunk async for chunk in process.stderr])

                async with anyio.create_task_group() as tg:
                    tg.start_soon(read_stdout)
                    tg.start_soon(read_stderr)

                await process.wait()

                return RunCommandResponse(
                    exit_code=process.returncode or 0,
                    stdout=b"".join(stdout_chunks).decode("utf-8", errors="replace"),
                    stderr=b"".join(stderr_chunks).decode("utf-8", errors="replace"),
                )

        except TimeoutError:
            if process:
                process.kill()
            return RunCommandResponse(
                exit_code=1,
                stdout="",
                stderr=f"Command timed out after {body.timeout} seconds",
                error_kind="timeout",
            )
        except Exception as exc:
            return RunCommandResponse(
                exit_code=1,
                stdout="",
                stderr=str(exc),
                error_kind="execution_error",
            )

    async def upload_files(self, body: UploadFilesRequest) -> UploadFilesResponse:
        """Upload files to the execution environment.

        Files are COPIED (not moved) - originals remain on the local filesystem.
        Directories are uploaded recursively, preserving their structure.

        Args:
            body: UploadFilesRequest containing the paths and session ID.

        Returns:
            UploadFilesResult containing lists of uploaded files and any failures.

        """
        session = self.session_manager.get_session(body.session_id)
        # Local filesystem - use optimized copy operation
        dest_base = session.temp_dir / body.dest_dir if body.dest_dir else session.temp_dir
        dest_base.mkdir(parents=True, exist_ok=True)

        result = UploadFilesResponse(uploaded=[], failed={})

        for source in body.paths:
            source = Path(source).resolve()

            if not source.exists():
                result.failed[str(source)] = "File or directory does not exist"
                continue

            try:
                if source.is_file():
                    dest = dest_base / source.name
                    shutil.copy2(source, dest)
                    result.uploaded.append(
                        UploadedFile(
                            source_path=source,
                            dest_path=str(dest.relative_to(session.temp_dir)),
                            size=source.stat().st_size,
                        ),
                    )

                elif source.is_dir():
                    # If dest_dir was explicitly provided, copy contents directly to dest_base
                    # Otherwise, create a subdirectory with the source's name
                    if body.dest_dir:
                        dest = dest_base
                        # Copy contents of source directory into dest_base
                        for item in source.iterdir():
                            item_dest = dest / item.name
                            if item.is_file():
                                shutil.copy2(item, item_dest)
                            else:
                                shutil.copytree(item, item_dest, dirs_exist_ok=True)
                    else:
                        dest = dest_base / source.name
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                    # Track all individual files uploaded
                    for file_path in source.rglob("*"):
                        if file_path.is_file():
                            relative = file_path.relative_to(source)
                            dest_file = dest / relative
                            result.uploaded.append(
                                UploadedFile(
                                    source_path=file_path,
                                    dest_path=str(dest_file.relative_to(self._temp_dir)),
                                    size=file_path.stat().st_size,
                                ),
                            )

            except Exception as exc:
                result.failed[str(source)] = str(exc)

        return result

    async def save_output_files(self, body: SaveOutputFilesRequest) -> SaveOutputFilesResponse:
        """Move files from the temp directory to a destination.

        Files are MOVED (not copied) - originals are deleted from the execution environment.
        Existing files in output_dir are silently overwritten.

        Args:
            body: SaveOutputFilesRequest containing the paths and session ID.

        Returns:
            SaveOutputFilesResponse containing lists of saved files and any failures.

        """
        try:
            session = self.session_manager.get_session(body.session_id)
        except Exception as e:
            return SaveOutputFilesResponse(
                saved=[], failed={}, error_message=str("Session not found; error: " + str(e))
            )

        # Local filesystem - use optimized move operation
        output_dir_path = Path(body.output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        result = SaveOutputFilesResponse(saved=[], failed={})

        for source_path_str in body.paths:
            try:
                source_path = Path(source_path_str)
                if not source_path.is_absolute():
                    source_path = session.temp_dir / source_path

                # Security: ensure path is within temp directory
                try:
                    source_path.resolve().relative_to(self._temp_dir.resolve())
                except ValueError:
                    result.failed[source_path_str] = "Path is outside execution environment directory"
                    continue

                if not source_path.exists():
                    result.failed[source_path_str] = "File does not exist"
                    continue

                if not source_path.is_file():
                    result.failed[source_path_str] = "Path is not a file"
                    continue

                file_size = source_path.stat().st_size
                dest_path = output_dir_path / source_path.name

                # Move file (overwrites if exists)
                shutil.move(str(source_path), str(dest_path))

                result.saved.append(
                    SavedFile(
                        source_path=source_path_str,
                        output_path=dest_path,
                        size=file_size,
                    ),
                )

            except Exception as exc:
                result.failed[source_path_str] = str(exc)

        return result

    async def end_session(self, body: EndSessionRequest) -> EndSessionResponse:
        """End the session.

        Args:
            body: EndSessionRequest containing the session ID.

        Returns:
            EndSessionResponse containing the session ID.
        """
        if body.paths is not None and body.output_dir is not None:
            result = await self.save_output_files(
                SaveOutputFilesRequest(
                    paths=body.paths,
                    session_id=body.session_id,
                    output_dir=body.output_dir
                )
            )
        else:
            result = SaveOutputFilesResponse(saved=[], failed={})

        try:
            self.session_manager.end_session(body.session_id)
        except Exception as e:
            return EndSessionResponse(
                session_deleted=False,
                saved=result.saved,
                failed=result.failed,
                error_message=f"Error ending session: {str(e)}\nFile Save Errors: {result.error_message}",
            )
        
        return EndSessionResponse(
            session_deleted=True,
            saved=result.saved,
            failed=result.failed,
            error_message=f"File Save Errors: {result.error_message}",
        )

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)

if __name__ == "__main__":
    BashSandboxResourcesServer.run_webserver()
