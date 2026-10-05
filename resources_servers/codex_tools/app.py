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
"""Codex-style coding tools (exec_command, write_stdin, apply_patch, update_plan) on a git working tree.

Each session gets a workspace: by default a detached ``git worktree`` of ``repo_path`` at
``base_ref``, or the repository itself with ``isolation: in_place``. ``verify`` reports the
workspace diff against ``base_ref`` and, when a ``check_command`` is configured, rewards its success.

A ``repo_path`` outside any git repository is used in place, without a diff (``verify`` still runs
``check_command``).

Commands run unsandboxed with this server's permissions. Use a trusted model and repository.
"""

from __future__ import annotations

import asyncio
import fnmatch
import logging
import os
import shlex
import shutil
import signal
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Literal, Optional

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import ConfigDict, Field

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY, is_nemo_gym_fastapi_entrypoint
from resources_servers.codex_tools.apply_patch import run_apply_patch_tool
from resources_servers.codex_tools.exec_runtime import UNIFIED_EXEC_ENV, ExecError, ExecRuntime
from resources_servers.codex_tools.task_data import TaskData


LOG = logging.getLogger(__name__)
_APPLY_PATCH_SCRIPT = Path(__file__).with_name("apply_patch.py")
_CHECK_OUTPUT_TAIL_CHARS = 20_000
# Build artifacts left by running tests are never part of the change, even in repos that don't ignore them.
_DIFF_EXCLUDES = (":(exclude,glob)**/__pycache__/**", ":(exclude,glob)**/*.py[cod]")


class CodexToolsResourcesServerConfig(BaseResourcesServerConfig):
    repo_path: Optional[str] = Field(
        default=None,
        description="Default git repository (or plain directory, used in place without a diff); rows may override it.",
    )
    base_ref: str = "HEAD"
    isolation: Literal["worktree", "in_place"] = Field(
        default="worktree",
        description="worktree: a detached git worktree per session; in_place: edit repo_path directly.",
    )
    workspaces_dir: Optional[str] = Field(default=None, description="Parent directory for worktrees (default: temp).")
    keep_workspace: bool = Field(default=False, description="Keep worktrees after verify for inspection.")
    check_command: Optional[str] = Field(default=None, description="Shell command whose success earns reward 1.")
    check_timeout_s: float = 1800.0
    shell: str = "/bin/bash"
    login_shell: bool = True
    env_exclude_patterns: list[str] = Field(
        default_factory=lambda: ["*KEY*", "*SECRET*", "*TOKEN*", "NEMO_GYM_*"],
        description="Case-insensitive globs of server environment variables hidden from commands.",
    )
    session_idle_timeout_s: float = Field(default=3600.0, description="Idle sessions are closed after this long.")
    max_diff_bytes: int = 1_000_000


class CodexToolsSeedSessionRequest(BaseSeedSessionRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: TaskData = Field(default_factory=TaskData)


class CodexToolsVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: TaskData = Field(default_factory=TaskData)


class CodexToolsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")
    resolved: Optional[bool] = None
    diff: str = ""
    diff_truncated: bool = False
    check_exit_code: Optional[int] = None
    check_output: str = ""
    plan: Optional[dict[str, Any]] = None
    workspace: Optional[str] = None
    error: Optional[str] = None


@dataclass
class Workspace:
    path: str
    repo: str
    base_commit: Optional[str]  # None: not a git repository, so no diff
    isolation: str
    check_command: Optional[str]
    runtime: ExecRuntime
    plan: Optional[dict[str, Any]] = None
    last_used: float = field(default_factory=time.monotonic)


async def _run(
    argv: list[str], *, cwd: str, env: Optional[dict[str, str]] = None, timeout: Optional[float] = None
) -> tuple[int, str]:
    """Run a command to completion; returns (exit code, combined output). Timeouts kill its process group."""
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=cwd,
        env=env,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        start_new_session=True,
    )
    try:
        output, _ = await asyncio.wait_for(process.communicate(), timeout)
    except asyncio.TimeoutError:
        os.killpg(process.pid, signal.SIGKILL)
        output, _ = await process.communicate()
        return -1, output.decode("utf-8", "replace") + f"\n[timed out after {timeout} seconds]"
    return process.returncode, output.decode("utf-8", "replace")


async def _git(repo: str, *args: str, env: Optional[dict[str, str]] = None) -> str:
    code, output = await _run(["git", "-C", repo, *args], cwd=repo, env=env)
    if code != 0:
        raise RuntimeError(f"git {' '.join(args)} failed ({code}): {output.strip()}")
    return output


class CodexToolsResourcesServer(SimpleResourcesServer):
    config: CodexToolsResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        self._workspaces: dict[str, Workspace] = {}
        self._reaper: Optional[asyncio.Task] = None
        self._bin_dir: Optional[str] = None
        return super().model_post_init(context)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/exec_command")(self.exec_command)
        app.post("/write_stdin")(self.write_stdin)
        app.post("/apply_patch")(self.apply_patch)
        app.post("/update_plan")(self.update_plan)

        # Release workspaces (processes and worktrees) when the server shuts down.
        main_lifespan = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(app: FastAPI) -> AsyncIterator[Any]:
            async with main_lifespan(app) as state:
                try:
                    yield state
                finally:
                    await self._close_all()

        app.router.lifespan_context = lifespan
        return app

    # ---- Workspaces ----

    def _ensure_bin_dir(self) -> str:
        """A directory holding an `apply_patch` command, prepended to PATH for every command."""
        if self._bin_dir is None:
            self._bin_dir = tempfile.mkdtemp(prefix="codex_tools_bin_")
            launcher = Path(self._bin_dir) / "apply_patch"
            launcher.write_text(
                f'#!/bin/sh\nexec {shlex.quote(sys.executable)} {shlex.quote(str(_APPLY_PATCH_SCRIPT))} "$@"\n'
            )
            launcher.chmod(0o755)
        return self._bin_dir

    def _command_env(self, workspace_path: str) -> dict[str, str]:
        patterns = [pattern.upper() for pattern in self.config.env_exclude_patterns]
        env = {
            key: value
            for key, value in os.environ.items()
            if not any(fnmatch.fnmatchcase(key.upper(), pattern) for pattern in patterns)
        }
        env.update(UNIFIED_EXEC_ENV)
        env["PATH"] = f"{self._ensure_bin_dir()}{os.pathsep}{env.get('PATH', os.defpath)}"
        env["CODEX_TOOLS_WORKSPACE_ROOT"] = workspace_path
        return env

    async def _create_workspace(self, task: TaskData) -> Workspace:
        repo_path = task.repo_path or self.config.repo_path
        if not repo_path:
            raise HTTPException(400, "codex_tools: no repo_path in the server config or verifier_metadata")
        directory = os.path.abspath(os.path.expanduser(repo_path))
        if not os.path.isdir(directory):
            raise HTTPException(400, f"codex_tools: repo_path {repo_path!r} is not a directory")
        try:
            repo: Optional[str] = (await _git(directory, "rev-parse", "--show-toplevel")).strip()
        except (OSError, RuntimeError):
            repo = None
        if repo is None:
            # Not a git repository: edit the directory in place and report no diff.
            # TODO: isolate plain directories by copying them to a temp dir, `git init`ing and committing
            #   the copy as the baseline, so worktree-style isolation and diffs work unchanged (needs a
            #   size cap and default excludes such as node_modules/.venv, as there is no .gitignore).
            # TODO: alternatively, keep in-place edits but record a baseline in a git dir outside the
            #   directory (`git --git-dir=<tmp> --work-tree=<dir> add -A && commit`) and diff against it.
            LOG.warning("codex_tools: %s is not in a git repository; editing it in place, without a diff", directory)
            repo, base_commit, isolation = directory, None, "in_place"
        else:
            base_ref = task.base_ref or self.config.base_ref
            base_commit = (await _git(repo, "rev-parse", "--verify", f"{base_ref}^{{commit}}")).strip()
            isolation = self.config.isolation
        if isolation == "in_place":
            if any(workspace.path == repo for workspace in self._workspaces.values()):
                raise HTTPException(409, f"codex_tools: {repo} is already in use by an in_place session")
            path = repo
        else:
            parent = tempfile.mkdtemp(prefix="codex_tools_", dir=self.config.workspaces_dir)
            path = os.path.join(parent, os.path.basename(repo))
            await _git(repo, "worktree", "add", "--detach", path, base_commit)
        runtime = ExecRuntime(
            cwd=path, shell=self.config.shell, login=self.config.login_shell, env=self._command_env(path)
        )
        check_command = task.check_command or self.config.check_command
        return Workspace(path, repo, base_commit, isolation, check_command, runtime)

    async def _release(self, workspace: Workspace) -> None:
        workspace.runtime.close()
        if workspace.isolation != "worktree" or self.config.keep_workspace:
            return
        try:
            await _git(workspace.repo, "worktree", "remove", "--force", workspace.path)
        except RuntimeError as error:
            LOG.warning("Could not remove worktree %s: %s", workspace.path, error)
        shutil.rmtree(os.path.dirname(workspace.path), ignore_errors=True)

    async def _reap_idle(self) -> None:
        while True:
            await asyncio.sleep(min(60.0, self.config.session_idle_timeout_s))
            cutoff = time.monotonic() - self.config.session_idle_timeout_s
            for session_id, workspace in list(self._workspaces.items()):
                if workspace.last_used < cutoff and self._workspaces.get(session_id) is workspace:
                    LOG.info("Closing idle codex_tools session %s (%s)", session_id, workspace.path)
                    del self._workspaces[session_id]
                    await self._release(workspace)

    async def _close_all(self) -> None:
        workspaces, self._workspaces = list(self._workspaces.values()), {}
        for workspace in workspaces:
            await self._release(workspace)
        if self._bin_dir is not None:
            shutil.rmtree(self._bin_dir, ignore_errors=True)
            self._bin_dir = None

    async def _workspace(self, request: Request) -> Workspace:
        session_id = request.session[SESSION_ID_KEY]
        workspace = self._workspaces.get(session_id)
        if workspace is None:
            # Direct /v1/responses calls skip seed_session; fall back to the configured defaults.
            workspace = await self._open(session_id, TaskData())
        workspace.last_used = time.monotonic()
        return workspace

    async def _open(self, session_id: str, task: TaskData) -> Workspace:
        if self._reaper is None:
            self._reaper = asyncio.get_running_loop().create_task(self._reap_idle())
        previous = self._workspaces.pop(session_id, None)
        if previous is not None:
            await self._release(previous)
        workspace = await self._create_workspace(task)
        self._workspaces[session_id] = workspace
        return workspace

    async def seed_session(self, request: Request, body: CodexToolsSeedSessionRequest) -> BaseSeedSessionResponse:
        workspace = await self._open(request.session[SESSION_ID_KEY], body.verifier_metadata)
        LOG.info("codex_tools workspace %s at %s", workspace.path, workspace.base_commit or "(not a git repository)")
        return BaseSeedSessionResponse()

    # ---- Tools ----

    async def exec_command(
        self, request: Request, body: dict[str, Any] = Body(default_factory=dict)
    ) -> PlainTextResponse:
        workspace = await self._workspace(request)
        try:
            return PlainTextResponse(await workspace.runtime.exec_command(body))
        except ExecError as error:
            return PlainTextResponse(str(error))

    async def write_stdin(
        self, request: Request, body: dict[str, Any] = Body(default_factory=dict)
    ) -> PlainTextResponse:
        workspace = await self._workspace(request)
        try:
            return PlainTextResponse(await workspace.runtime.write_stdin(body))
        except ExecError as error:
            return PlainTextResponse(str(error))

    async def apply_patch(
        self, request: Request, body: dict[str, Any] = Body(default_factory=dict)
    ) -> PlainTextResponse:
        workspace = await self._workspace(request)
        patch = body.get("input")
        if not isinstance(patch, str):
            return PlainTextResponse("failed to parse function arguments: missing field `input`")
        output, _ = await asyncio.to_thread(run_apply_patch_tool, patch, workspace.path, workspace.path)
        return PlainTextResponse(output)

    async def update_plan(
        self, request: Request, body: dict[str, Any] = Body(default_factory=dict)
    ) -> PlainTextResponse:
        workspace = await self._workspace(request)
        plan = body.get("plan")
        explanation = body.get("explanation")
        valid = isinstance(plan, list) and all(
            isinstance(item, dict)
            and isinstance(item.get("step"), str)
            and item.get("status") in ("pending", "in_progress", "completed")
            for item in plan
        )
        if not valid or (explanation is not None and not isinstance(explanation, str)):
            return PlainTextResponse(
                "failed to parse function arguments: expected {explanation?: string, "
                "plan: [{step: string, status: pending|in_progress|completed}]}"
            )
        workspace.plan = {"explanation": explanation, "plan": plan}
        return PlainTextResponse("Plan updated")

    # ---- Verification ----

    async def _diff(self, workspace: Workspace) -> str:
        """Diff of the workspace (tracked and untracked, honouring .gitignore) against the base commit,
        staged into a throwaway index so the repository's own index is untouched."""
        if workspace.base_commit is None:
            return ""
        with tempfile.TemporaryDirectory() as index_dir:
            env = {**os.environ, "GIT_INDEX_FILE": os.path.join(index_dir, "index")}
            await _git(workspace.path, "read-tree", workspace.base_commit, env=env)
            await _git(workspace.path, "add", "-A", "--", ".", *_DIFF_EXCLUDES, env=env)
            return await _git(workspace.path, "diff", "--cached", "--binary", workspace.base_commit, env=env)

    async def verify(self, request: Request, body: CodexToolsVerifyRequest) -> CodexToolsVerifyResponse:
        workspace = self._workspaces.pop(request.session[SESSION_ID_KEY], None)
        base = body.model_dump()
        if workspace is None:
            return CodexToolsVerifyResponse(
                **base, reward=0.0, mask_sample=True, error="no workspace for this session"
            )
        workspace.runtime.close()
        result: dict[str, Any] = {"plan": workspace.plan}
        try:
            diff = await self._diff(workspace)
            result["diff_truncated"] = len(diff.encode("utf-8")) > self.config.max_diff_bytes
            result["diff"] = diff.encode("utf-8")[: self.config.max_diff_bytes].decode("utf-8", "ignore")
            if workspace.check_command:
                code, output = await _run(
                    [self.config.shell, "-lc" if self.config.login_shell else "-c", workspace.check_command],
                    cwd=workspace.path,
                    env=self._command_env(workspace.path),
                    timeout=self.config.check_timeout_s,
                )
                result |= {
                    "resolved": code == 0,
                    "check_exit_code": code,
                    "check_output": output[-_CHECK_OUTPUT_TAIL_CHARS:],
                }
        except (OSError, RuntimeError) as error:
            result |= {"error": str(error), "mask_sample": True}
        finally:
            await self._release(workspace)
        if workspace.isolation == "in_place" or self.config.keep_workspace:
            result["workspace"] = workspace.path
        return CodexToolsVerifyResponse(**base, reward=1.0 if result.get("resolved") else 0.0, **result)


if __name__ == "__main__":
    CodexToolsResourcesServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = CodexToolsResourcesServer.run_webserver()  # noqa: F401
