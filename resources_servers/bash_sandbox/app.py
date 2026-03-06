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
import asyncio
import logging
import os
from html import escape

import anyio
import httpx
import re
import shutil
import subprocess
import tempfile
import uuid

from typing import Dict, List
from fastapi import FastAPI
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field

from nemo_gym.base_resources_server import (
    SimpleResourcesServer,
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
)

logger = logging.getLogger(__name__)

SHELL_TIMEOUT = 30
MAX_LENGTH_WEB_FETCH = 40000
WEB_REQUEST_TIMEOUT = 60 * 3
DEFAULT_WEBFETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive",
}

class Session(BaseModel):
    """All code execution and file access happens in the temp directory."""
    temp_dir: Path

    @classmethod
    def create(cls, temp_dir_base: Path) -> "Session":
        temp_dir_base.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="local_sandbox_", dir=temp_dir_base))
        return cls(temp_dir=temp_dir)

class SessionManager:
    """
    In-memory session store. For multiple workers, the agent must use session affinity
    (worker_urls in config + affinity_key=session_id) so all requests for a session
    hit the same worker.
    """
    id_to_session: Dict[str, Session]
    temp_dir_base: Path

    def __init__(self, temp_dir_base: Path):
        self.id_to_session = {}
        self.temp_dir_base = temp_dir_base

    def session_exists(self, session_id: str) -> bool:
        return session_id in self.id_to_session

    def get_session(self, session_id: str) -> Session:
        return self.id_to_session[session_id]

    def start_session(self, session_id: str) -> Session:
        if self.session_exists(session_id):
            raise ValueError(f"Session {session_id} already exists")
        self.id_to_session[session_id] = Session.create(temp_dir_base=self.temp_dir_base)
        return self.get_session(session_id)

    def end_session(self, session_id: str) -> None:
        session = self.id_to_session.pop(session_id, None)
        if session:
            shutil.rmtree(session.temp_dir)

class UploadedFile(BaseModel):
    """Information about a file uploaded to the execution environment."""
    source_path: Path  # Original path on local filesystem
    dest_path: str  # Path in the execution environment
    size: int

class SavedFile(BaseModel):
    """Information about a file saved from the execution environment."""
    source_path: str  # Original path in execution environment
    output_path: Path  # Path where file was saved
    size: int

class CommitteeModelConfig(BaseModel):
    name: str
    output_dir: str


class JudgeConfig(BaseModel):
    enabled: bool = False
    judge_model_name: str = "gemini-3-pro-preview"
    gcp_project_id: str = ""
    gcp_location: str = "global"
    thinking_budget: int = 5000
    max_output_tokens: int = 65535
    num_trials: int = 4
    max_concurrent_judgements: int = 10
    committee_models: List[CommitteeModelConfig] = Field(default_factory=list)


class BashSandboxResourcesServerConfig(BaseResourcesServerConfig):
    temp_dir_base: Path = Field(default_factory=lambda: Path("/tmp/nemo_gym_bash_sandboxes"))
    allowlist: List[str] = Field(default_factory=list)
    judge: JudgeConfig = Field(default_factory=JudgeConfig)

class SeedSessionRequest(BaseSeedSessionRequest):
    session_id: str | None = None

class SeedSessionResponse(BaseSeedSessionResponse):
    session_id: str
    success: bool
    error_message: str | None = None

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
    paths: List[str]
    session_id: str
    dest_dir: str | None = None

class UploadFilesResponse(BaseModel):
    uploaded: List[UploadedFile]
    failed: Dict[str, str]

class SaveOutputFilesRequest(BaseModel):
    paths: List[str]
    session_id: str
    output_dir: str

class SaveOutputFilesResponse(BaseModel):
    saved: List[SavedFile]
    failed: Dict[str, str]
    error_message: str | None = None

class FinishRequest(BaseModel):
    session_id: str
    paths: List[str] | None = None
    output_dir: str | None = None

class FinishResponse(BaseModel):
    session_deleted: bool
    saved: List[SavedFile] = Field(default_factory=list)
    failed: Dict[str, str] = Field(default_factory=dict)
    error_message: str | None = None

class WebSearchRequest(BaseModel):
    query: str
    session_id: str

class WebSearchResponse(BaseModel):
    results_xml: str
    error: str | None = None

class WebFetchRequest(BaseModel):
    url: str
    session_id: str

class WebFetchResponse(BaseModel):
    content: str
    error: str | None = None

class VerifyRequest(BaseVerifyRequest):
    session_id: str
    paths: List[str]
    task_id: str = ""
    task_prompt: str = ""


class CommitteeModelVerdict(BaseModel):
    committee_model_name: str
    win_count_evaluated: int
    win_count_committee: int
    tie_count: int
    num_trials: int
    reward: float
    success: bool
    error_message: str | None = None


class GDPValVerifyResponse(BaseVerifyResponse):
    committee_verdicts: List[CommitteeModelVerdict] = Field(default_factory=list)

class BashSandboxResourcesServer(SimpleResourcesServer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: BashSandboxResourcesServerConfig
    session_manager: SessionManager = None  # type: ignore[assignment]
    _judge: object = None  # Lazily initialized GDPValJudge

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session_manager = SessionManager(Path(self.config.temp_dir_base))

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()

        # Register tool endpoints
        # Tool endpoints called via /{tool_name} pattern from GDPValAgent
        app.post("/run_command")(self.run_command)
        app.post("/upload_files")(self.upload_files)
        app.post("/save_files")(self.save_output_files)
        app.post("/finish")(self.finish)  # Finish tool for task completion
        app.post("/web_search")(self.web_search)
        app.post("/web_fetch")(self.web_fetch)

        return app

    def _check_allowed(self, cmd: str) -> bool:
        """Check if command is allowed based on the allowlist.

        Returns:
            True if the command is allowed, False otherwise.

        """
        # No allowlist configured means allow all commands.
        if not self.config.allowlist:
            return True

        for pattern in self.config.allowlist:
            try:
                if re.search(pattern, cmd):
                    return True
            except re.error:
                # Ignore invalid regex entries instead of crashing command execution.
                continue
        return False

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

    def _check_absolute_paths(self, cmd: str, session: Session) -> RunCommandResponse | None:
        """Check if command contains absolute paths outside the session directory.

        Absolute paths that resolve to the session directory are allowed.
        All other absolute paths, home-dir shortcuts, and env-var paths are rejected.

        Returns:
            RunCommandResponse with error if outside paths detected, None otherwise.
        """
        session_dir = str(session.temp_dir)
        session_msg = (
            f"You can only access and execute commands inside your session directory: {session_dir}. "
            "Use relative paths only (e.g. '.', './reference_files/file.xlsx')."
        )

        home_patterns = [
            r"~/",
            r"\$HOME\b",
            r"\$\{HOME\}",
        ]
        for pattern in home_patterns:
            if re.search(pattern, cmd):
                return RunCommandResponse(
                    exit_code=1,
                    stdout="",
                    stderr=(
                        "Command may access paths outside the session directory. This is not allowed. "
                        + session_msg
                    ),
                    error_kind="absolute_path_detected",
                    advice=session_msg,
                )

        abs_path_re = re.compile(r"(?:^|[\s;&|\"'=])(/[^\s;&|\"']*)")
        for match in abs_path_re.finditer(cmd):
            path = match.group(1)
            if path.startswith(session_dir):
                continue
            if "://" in cmd[max(0, match.start(1) - 8):match.start(1)]:
                continue
            return RunCommandResponse(
                exit_code=1,
                stdout="",
                stderr=(
                    f"Command references path '{path}' outside the session directory. This is not allowed. "
                    + session_msg
                ),
                error_kind="absolute_path_detected",
                advice=session_msg,
            )
        return None

    async def seed_session(self, body: SeedSessionRequest) -> SeedSessionResponse:
        if body.session_id is None:
            session_id = str(uuid.uuid4())
        else:
            session_id = body.session_id

        try:
            self.session_manager.start_session(session_id)
            print(f"seed_session: session_id: {session_id}, session_manager: {self.session_manager.id_to_session}")
        except Exception as e:
            return SeedSessionResponse(session_id=session_id, success=False, error_message=str(e))

        return SeedSessionResponse(session_id=session_id, success=True)

    async def run_command(self, body: RunCommandRequest) -> RunCommandResponse:
        """Execute command in the temp directory for the session specified by the session ID.

        Args:
            body: RunCommandRequest containing the command and session ID.

        Returns:
            RunCommandResponse with exit_code, stdout, stderr, and optional error info.

        """
        print(f"run_command: session_id: {body.session_id}, session_manager: {self.session_manager.id_to_session}")
        try:
            session = self.session_manager.get_session(body.session_id)
        except KeyError:
            return RunCommandResponse(
                exit_code=1,
                stdout="",
                stderr=f"Session not found: {body.session_id}",
                error_kind="session_not_found",
                advice="Create a session first via the seed_session endpoint.",
            )

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

        # Check for absolute paths — restrict all access to session directory only
        absolute_path_error = self._check_absolute_paths(body.command, session)
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
        print(f"upload_files: session_id: {body.session_id}, session_manager: {self.session_manager.id_to_session}")
        try:
            session = self.session_manager.get_session(body.session_id)
        except KeyError:
            return UploadFilesResponse(
                uploaded=[],
                failed={f"session:{body.session_id}": "Session not found"},
            )
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
                                    dest_path=str(dest_file.relative_to(session.temp_dir)),
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
        print(f"save_output_files: session_id: {body.session_id}, session_manager: {self.session_manager.id_to_session}")
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
                    source_path.resolve().relative_to(session.temp_dir.resolve())
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

    async def finish(self, body: FinishRequest) -> FinishResponse:
        """Finish the task: optionally save output files, then end the session.

        Args:
            body: FinishRequest containing the session ID and optional file save info.

        Returns:
            FinishResponse with session deletion status and saved file details.
        """
        print(f"finish: session_id: {body.session_id}, session_manager: {self.session_manager.id_to_session}")
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
            errors = [f"Error ending session: {str(e)}"]
            if result.error_message:
                errors.append(result.error_message)
            return FinishResponse(
                session_deleted=False,
                saved=result.saved,
                failed=result.failed,
                error_message="; ".join(errors),
            )
        
        return FinishResponse(
            session_deleted=True,
            saved=result.saved,
            failed=result.failed,
            error_message=result.error_message,
        )

    async def _http_request_with_retry(
        self,
        method: str,
        url: str,
        *,
        headers: dict | None = None,
        params: dict | None = None,
        max_attempts: int = 3,
    ) -> httpx.Response:
        """Execute an HTTP request with exponential backoff retry on network errors."""
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                async with httpx.AsyncClient(
                    timeout=WEB_REQUEST_TIMEOUT, follow_redirects=True
                ) as client:
                    response = await client.request(method, url, headers=headers, params=params)
                    response.raise_for_status()
                    return response
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2 ** attempt)
        raise last_exc  # type: ignore[misc]

    async def web_search(self, body: WebSearchRequest) -> WebSearchResponse:
        """Search the web using Brave Search API. Returns top 5 results as XML."""
        brave_api_key = os.getenv("BRAVE_API_KEY")
        if not brave_api_key:
            return WebSearchResponse(
                results_xml="", error="BRAVE_API_KEY environment variable not set"
            )

        try:
            response = await self._http_request_with_retry(
                "GET",
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "X-Subscription-Token": brave_api_key,
                    "Accept": "application/json",
                },
                params={"q": body.query, "count": 5},
            )
            data = response.json()
            results = data.get("web", {}).get("results", [])
            results_xml = (
                "<results>\n"
                + "\n".join(
                    (
                        "<result>"
                        f"\n<title>{escape(result.get('title', '') or '')}</title>"
                        f"\n<url>{escape(result.get('url', '') or '')}</url>"
                        f"\n<description>{escape(result.get('description', '') or '')}</description>"
                        "\n</result>"
                    )
                    for result in results
                )
                + "\n</results>"
            )
            return WebSearchResponse(results_xml=results_xml[:MAX_LENGTH_WEB_FETCH])
        except Exception as exc:
            return WebSearchResponse(results_xml="", error=str(exc))

    async def web_fetch(self, body: WebFetchRequest) -> WebFetchResponse:
        """Fetch a web page and extract main content as markdown."""
        import trafilatura

        try:
            response = await self._http_request_with_retry(
                "GET",
                body.url,
                headers=DEFAULT_WEBFETCH_HEADERS,
            )
            body_md = trafilatura.extract(response.text, output_format="markdown") or ""
            content = (
                f"<web_fetch><url>{escape(body.url)}</url>"
                f"<body>{body_md[:MAX_LENGTH_WEB_FETCH]}</body></web_fetch>"
            )
            return WebFetchResponse(content=content)
        except Exception as exc:
            content = (
                f"<web_fetch><url>{escape(body.url)}</url>"
                f"<error>{escape(str(exc))}</error></web_fetch>"
            )
            return WebFetchResponse(content=content, error=str(exc))

    def _get_or_create_judge(self):
        """Lazily initialize the GDPValJudge on first verify call."""
        if self._judge is None:
            from resources_servers.bash_sandbox.judge import GDPValJudge

            judge_config = self.config.judge
            self._judge = GDPValJudge(
                gcp_project_id=judge_config.gcp_project_id,
                gcp_location=judge_config.gcp_location,
                judge_model_name=judge_config.judge_model_name,
                thinking_budget=judge_config.thinking_budget,
                max_output_tokens=judge_config.max_output_tokens,
                num_trials=judge_config.num_trials,
                max_concurrent_judgements=judge_config.max_concurrent_judgements,
            )
        return self._judge

    async def verify(self, body: VerifyRequest) -> BaseVerifyResponse:
        judge_config = self.config.judge

        # Backward compatible: if judge not enabled or no committee models, return reward=1.0
        if not judge_config.enabled or not judge_config.committee_models:
            return GDPValVerifyResponse(**body.model_dump(), reward=1.0)

        judge = self._get_or_create_judge()

        # Determine evaluated output dir from the first saved file path
        if body.paths:
            evaluated_output_dir = str(Path(body.paths[0]).parent)
        else:
            logger.warning("No output paths in verify request, returning reward=1.0")
            return GDPValVerifyResponse(**body.model_dump(), reward=1.0)

        # Find reference files directory: check if reference_files/ exists in evaluated output
        refs_dir = None
        evaluated_refs = Path(evaluated_output_dir) / "reference_files"
        if evaluated_refs.exists() and any(evaluated_refs.iterdir()):
            refs_dir = str(evaluated_refs)

        # Build judge tasks for each committee model
        judge_tasks = []
        committee_configs = []
        for cm in judge_config.committee_models:
            cm_task_dir = Path(cm.output_dir) / f"task_{body.task_id}"
            finish_params = cm_task_dir / "finish_params.json"

            # H7: Skip committee models that didn't attempt this task
            if not cm_task_dir.exists() or not finish_params.exists():
                logger.warning(
                    "Committee model %s has no output for task %s, skipping",
                    cm.name, body.task_id,
                )
                continue

            # Check for reference files in committee dir too (H1)
            if refs_dir is None:
                cm_refs = cm_task_dir / "reference_files"
                if cm_refs.exists() and any(cm_refs.iterdir()):
                    refs_dir = str(cm_refs)

            judge_tasks.append(
                judge.judge_task(
                    task_prompt=body.task_prompt,
                    evaluated_output_dir=evaluated_output_dir,
                    committee_output_dir=str(cm_task_dir),
                    refs_dir=refs_dir,
                    committee_model_name=cm.name,
                )
            )
            committee_configs.append(cm)

        # H7: If no committee models have output for this task, fall back to reward=1.0
        if not judge_tasks:
            logger.warning("No committee models have output for task %s, returning reward=1.0", body.task_id)
            return GDPValVerifyResponse(**body.model_dump(), reward=1.0)

        # Run all judge tasks concurrently
        results = await asyncio.gather(*judge_tasks, return_exceptions=True)

        # Build verdicts and compute mean reward
        committee_verdicts = []
        successful_rewards = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Judge task for %s raised exception: %s", committee_configs[i].name, result)
                committee_verdicts.append(CommitteeModelVerdict(
                    committee_model_name=committee_configs[i].name,
                    win_count_evaluated=0,
                    win_count_committee=0,
                    tie_count=0,
                    num_trials=0,
                    reward=0.0,
                    success=False,
                    error_message=str(result),
                ))
                continue

            verdict = CommitteeModelVerdict(
                committee_model_name=result.committee_model_name,
                win_count_evaluated=result.win_count_evaluated,
                win_count_committee=result.win_count_committee,
                tie_count=result.tie_count,
                num_trials=result.num_trials,
                reward=result.reward,
                success=result.success,
                error_message=result.error_message,
            )
            committee_verdicts.append(verdict)

            # H6: Only include successful verdicts in the mean
            if result.success:
                successful_rewards.append(result.reward)

        # H6: If ALL verdicts fail, fall back to reward=1.0
        if not successful_rewards:
            logger.warning("All committee verdicts failed for task %s, returning reward=1.0", body.task_id)
            mean_reward = 1.0
        else:
            mean_reward = sum(successful_rewards) / len(successful_rewards)

        return GDPValVerifyResponse(
            **body.model_dump(),
            reward=mean_reward,
            committee_verdicts=committee_verdicts,
        )

if __name__ == "__main__":
    BashSandboxResourcesServer.run_webserver()
