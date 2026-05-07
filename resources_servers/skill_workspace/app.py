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
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.skill_workspace.schemas import (
    CloseRequest,
    CloseResponse,
    ReadFileRequest,
    ReadFileResponse,
    RunBashRequest,
    RunBashResponse,
    SkillWorkspaceResourcesServerConfig,
    SkillWorkspaceSeedSessionRequest,
    SkillWorkspaceSeedSessionResponse,
)


logger = logging.getLogger(__name__)


_SANDBOX_BIN = ".sandbox_bin"
_SANDBOX_PATH = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"


def _seed_sandbox_bin(workspace: Path) -> None:
    """Create a workspace-local bin dir with `python -> python3` so rollouts relying on
    the `python` command (common in docs/examples) don't thrash on `command not found`.
    Idempotent; safe to call even if python3 isn't resolvable."""
    python3 = shutil.which("python3") or "/usr/bin/python3"
    bin_dir = workspace / _SANDBOX_BIN
    bin_dir.mkdir(exist_ok=True)
    link = bin_dir / "python"
    if not link.exists() and Path(python3).exists():
        link.symlink_to(python3)


def _build_sandbox_env(workspace: Path) -> dict[str, str]:
    """Minimal env for rollout subprocesses. Strips the host's NeMo Gym venv from PATH
    so rollouts can't see host `ng_*` binaries, Ray state, or HF/MLflow credentials.
    Prepends a workspace-local bin dir that provides `python -> python3`."""
    return {
        "PATH": f"{workspace / _SANDBOX_BIN}:{_SANDBOX_PATH}",
        "HOME": os.environ.get("HOME", "/tmp"),
        "USER": os.environ.get("USER", "nobody"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "LC_ALL": os.environ.get("LC_ALL", "C.UTF-8"),
        "SHELL": "/bin/bash",
        "TERM": "dumb",
        "PWD": str(workspace),
        "TMPDIR": os.environ.get("TMPDIR", "/tmp"),
    }


class SkillWorkspaceResourcesServer(SimpleResourcesServer):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: SkillWorkspaceResourcesServerConfig
    env_id_to_workspace: dict[str, Path] = Field(default_factory=dict)

    _bash_semaphore: Optional[asyncio.Semaphore] = PrivateAttr(default=None)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/run_bash")(self.run_bash)
        app.post("/read_file")(self.read_file)
        app.post("/close")(self.close)
        return app

    def _get_bash_semaphore(self) -> asyncio.Semaphore:
        if self._bash_semaphore is None:
            self._bash_semaphore = asyncio.Semaphore(self.config.max_concurrent_bash)
        return self._bash_semaphore

    def _resolve_in_workspace(self, env_id: str, rel_path: str) -> Path:
        workspace = self.env_id_to_workspace.get(env_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail=f"Unknown env_id: {env_id}")

        p = Path(rel_path)
        if p.is_absolute():
            raise HTTPException(status_code=400, detail="Absolute paths are not allowed")

        resolved = (workspace / p).resolve()
        try:
            resolved.relative_to(workspace.resolve())
        except ValueError:
            raise HTTPException(status_code=400, detail="Path escapes the workspace")
        return resolved

    async def seed_session(self, body: SkillWorkspaceSeedSessionRequest) -> SkillWorkspaceSeedSessionResponse:
        skill_src = Path(body.skill_path).resolve()
        if not skill_src.is_dir():
            raise HTTPException(status_code=400, detail=f"skill_path is not a directory: {body.skill_path}")

        workspace = Path(
            tempfile.mkdtemp(
                prefix=f"skill_ws_{body.scenario_id}_",
                dir=self.config.workspace_root,
            )
        )
        env_id = str(uuid.uuid4())

        try:
            _seed_sandbox_bin(workspace)

            # Do NOT copy SKILL.md into the workspace. The skill body belongs in
            # the system prompt (when with_skill=True) or nowhere (when
            # with_skill=False). Putting it on disk lets the model `cat SKILL.md`
            # during the without_skill arm — we measured 100% peek rate in the
            # shape probe run, which contaminates every with-vs-without delta.
            # scripts/ and references/ are still copied because they are
            # supplementary artifacts the skill's scenarios may legitimately
            # reference by path.

            seed_subdirs: list[str] = []
            if body.with_scripts:
                seed_subdirs.append("scripts")
            if body.with_references:
                seed_subdirs.append("references")
            for subdir in seed_subdirs:
                src = skill_src / subdir
                if src.is_dir():
                    shutil.copytree(src, workspace / subdir)

            for rel in body.files:
                rel_path = Path(rel)
                if rel_path.is_absolute():
                    raise HTTPException(status_code=400, detail=f"Fixture path must be relative: {rel}")
                fixture_src = (skill_src / rel_path).resolve()
                try:
                    fixture_src.relative_to(skill_src)
                except ValueError:
                    raise HTTPException(status_code=400, detail=f"Fixture path escapes skill dir: {rel}")
                if not fixture_src.is_file():
                    raise HTTPException(status_code=400, detail=f"Fixture file not found: {rel}")
                dest = workspace / rel_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(fixture_src, dest)
        except Exception:
            shutil.rmtree(workspace, ignore_errors=True)
            raise

        self.env_id_to_workspace[env_id] = workspace
        return SkillWorkspaceSeedSessionResponse(env_id=env_id)

    async def run_bash(self, body: RunBashRequest) -> RunBashResponse:
        workspace = self.env_id_to_workspace.get(body.env_id)
        if workspace is None:
            raise HTTPException(status_code=404, detail=f"Unknown env_id: {body.env_id}")

        cfg = self.config
        requested = body.timeout_seconds if body.timeout_seconds is not None else cfg.bash_timeout_default_seconds
        timeout = max(1, min(requested, cfg.bash_timeout_hard_cap_seconds))
        cap = cfg.output_cap_bytes

        async with self._get_bash_semaphore():
            proc = await asyncio.create_subprocess_shell(
                body.cmd,
                cwd=str(workspace),
                env=_build_sandbox_env(workspace),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            timed_out = False
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                timed_out = True
                proc.kill()
                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=5)
                except asyncio.TimeoutError:  # pragma: no cover - defensive for SIGKILL-resistant procs
                    stdout_bytes, stderr_bytes = b"", b""

        truncated = False
        if len(stdout_bytes) + len(stderr_bytes) > cap:
            truncated = True
            stdout_budget = min(len(stdout_bytes), cap)
            stdout_bytes = stdout_bytes[:stdout_budget]
            stderr_bytes = stderr_bytes[: max(0, cap - stdout_budget)]

        return RunBashResponse(
            stdout=stdout_bytes.decode("utf-8", errors="replace"),
            stderr=stderr_bytes.decode("utf-8", errors="replace"),
            exit_code=proc.returncode if proc.returncode is not None else -1,
            truncated=truncated,
            timed_out=timed_out,
        )

    async def read_file(self, body: ReadFileRequest) -> ReadFileResponse:
        resolved = self._resolve_in_workspace(body.env_id, body.path)
        if not resolved.is_file():
            raise HTTPException(status_code=404, detail=f"File not found: {body.path}")

        cap = self.config.output_cap_bytes
        with resolved.open("rb") as f:
            data = f.read(cap + 1)
        truncated = len(data) > cap
        if truncated:
            data = data[:cap]
        return ReadFileResponse(content=data.decode("utf-8", errors="replace"), truncated=truncated)

    async def close(self, body: CloseRequest) -> CloseResponse:
        workspace = self.env_id_to_workspace.pop(body.env_id, None)
        if workspace is None:
            return CloseResponse(message=f"Unknown env_id: {body.env_id}", success=False)
        try:
            shutil.rmtree(workspace, ignore_errors=False)
        except Exception as e:
            logger.exception("Error removing workspace")
            return CloseResponse(message=repr(e), success=False)
        return CloseResponse(message="Success", success=True)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=0.0)


if __name__ == "__main__":
    SkillWorkspaceResourcesServer.run_webserver()
