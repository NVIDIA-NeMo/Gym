# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import os
import shlex
import shutil
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_utils import SESSION_ID_KEY


# ----------------------------
# Config
# ----------------------------


class CVDPAgenticHeavyConfig(BaseResourcesServerConfig):
    oss_sim_image: str = "ghcr.io/hdl/sim/osvb"
    oss_pnr_image: str = "ghcr.io/hdl/impl/pnr"
    execution_backend: str = "apptainer"  # apptainer | docker
    container_timeout: int = 0
    docker_timeout: int = 600  # Backward-compatible alias for older configs.
    tool_timeout: int = 120
    num_processes: int = 4
    sif_cache_dir: str = ""
    harness_workspace_dir: str = ""
    container_tmp_bind_path: str = ""
    disable_network: bool = False


# ----------------------------
# Schemas
# ----------------------------


class CVDPAgenticHeavyVerifierMetadata(BaseModel):
    task_id: str
    categories: List[str] = []
    difficulty: str = ""
    context_files: Dict[str, str]
    harness_files: Dict[str, Optional[str]]
    patch: Optional[Dict[str, str]] = None
    origin: Optional[Dict[str, str]] = None


class AgenticHeavySeedRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    verifier_metadata: Optional[Dict[str, Any]] = None


class CVDPAgenticHeavyVerifyRequest(BaseVerifyRequest):
    verifier_metadata: Dict[str, Any]


class CVDPAgenticHeavyVerifyResponse(BaseVerifyResponse):
    task_id: Optional[str] = None
    category: Optional[str] = None
    difficulty: Optional[str] = None
    container_exit_code: Optional[int] = None
    container_stderr: Optional[str] = None
    container_services: Optional[List[Dict]] = None
    execution_time: Optional[float] = None
    tests_passed: Optional[int] = None
    tests_total: Optional[int] = None
    # Deprecated Docker-named aliases kept for older analysis scripts.
    docker_exit_code: Optional[int] = None
    docker_stderr: Optional[str] = None
    docker_services: Optional[List[Dict]] = None


# ----------------------------
# Compose / container helpers
# ----------------------------


def _apply_substitutions(content: str, config: CVDPAgenticHeavyConfig) -> str:
    substitutions = {
        "__OSS_SIM_IMAGE__": config.oss_sim_image,
        "__OSS_PNR_IMAGE__": config.oss_pnr_image,
    }
    for placeholder, value in substitutions.items():
        if value and placeholder in content:
            content = content.replace(placeholder, value)
    return content


def _filter_code_volumes(compose_content: str) -> str:
    data = yaml.safe_load(compose_content)
    if not data or "services" not in data:
        return compose_content
    for service_config in data["services"].values():
        if "volumes" in service_config:
            service_config["volumes"] = [vol for vol in service_config["volumes"] if "/code" not in vol]
    return yaml.dump(data, default_flow_style=False)


def _parse_compose_service(compose_content: str, service_name: str) -> Dict[str, Any]:
    data = yaml.safe_load(compose_content) or {}
    service = (data.get("services") or {}).get(service_name, {})
    return {
        "image": service.get("image", ""),
        "build": service.get("build", {}),
        "command": service.get("command", ""),
        "entrypoint": service.get("entrypoint"),
        "volumes": service.get("volumes", []),
        "working_dir": service.get("working_dir", "/code/rundir"),
        "environment": service.get("environment", {}),
    }


def _resolve_image_for_service(
    compose_data: dict,
    service_name: str,
    harness_files: Dict[str, Optional[str]],
    config: CVDPAgenticHeavyConfig,
) -> Tuple[str, List[str]]:
    svc = (compose_data.get("services") or {}).get(service_name, {})
    image = svc.get("image", "")
    if image:
        return image, []

    build_cfg = svc.get("build", {})
    if isinstance(build_cfg, str):
        dockerfile_path = os.path.join(build_cfg, "Dockerfile")
    elif isinstance(build_cfg, dict):
        dockerfile_path = build_cfg.get("dockerfile", "Dockerfile")
    else:
        return "", []

    dockerfile_content = None
    candidates = [
        dockerfile_path,
        f"src/{dockerfile_path}",
        dockerfile_path.replace("src/", ""),
    ]
    for candidate in candidates:
        for hf_path, hf_content in harness_files.items():
            if hf_content and (hf_path == candidate or hf_path.endswith(os.path.basename(candidate))):
                dockerfile_content = _apply_substitutions(hf_content, config)
                break
        if dockerfile_content:
            break
    if not dockerfile_content:
        return "", []

    base_image = ""
    post_commands: List[str] = []
    for line in dockerfile_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        upper = line.upper()
        if upper.startswith("FROM "):
            parts = line.split()
            base_image = parts[1] if len(parts) > 1 else ""
        elif upper.startswith("RUN "):
            post_commands.append(line[4:].strip())
        elif upper.startswith("ADD ") and "http" in line.lower():
            parts = line.split()
            if len(parts) >= 3:
                url, dest = parts[1], parts[2]
                post_commands.append(f"wget -q -O {dest} {url} || curl -sL -o {dest} {url}")
    return base_image, post_commands


def _load_dot_env(workdir: str) -> Dict[str, str]:
    env_path = os.path.join(workdir, "src", ".env")
    env_vars: Dict[str, str] = {}
    if not os.path.isfile(env_path):
        return env_vars
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                env_vars[key.strip()] = val.strip()
    return env_vars


def _build_env_args(environment: Any, dot_env: Optional[Dict[str, str]] = None) -> List[str]:
    env_args: List[str] = []
    if dot_env:
        for key, val in dot_env.items():
            env_args += ["--env", f"{key}={val}"]
    if isinstance(environment, dict):
        for key, val in environment.items():
            env_args += ["--env", f"{key}={val}"]
    elif isinstance(environment, list):
        for item in environment:
            env_args += ["--env", str(item)]
    return env_args


def _build_runtime_tmp_env_args(container_tmp_path: str) -> List[str]:
    runtime_env = {
        "TMPDIR": container_tmp_path,
        "TMP": container_tmp_path,
        "TEMP": container_tmp_path,
        "TEMPDIR": container_tmp_path,
        "JAVA_TOOL_OPTIONS": f"-Djava.io.tmpdir={container_tmp_path}",
    }
    env_args: List[str] = []
    for key, value in runtime_env.items():
        env_args += ["--env", f"{key}={value}"]
    return env_args


def _build_command(entrypoint: Any, command: Any) -> List[str]:
    cmd_parts: List[str] = []
    if entrypoint:
        cmd_parts = shlex.split(entrypoint) if isinstance(entrypoint, str) else list(entrypoint)
    if command:
        cmd_parts += shlex.split(command) if isinstance(command, str) else list(command)
    return cmd_parts


def _build_non_code_bind_args(workdir: str, compose_volumes: List[str]) -> List[str]:
    bind_args: List[str] = []
    for vol_str in compose_volumes:
        parts = str(vol_str).split(":")
        host_path = parts[0]
        container_path = parts[1] if len(parts) > 1 else host_path
        opts = parts[2] if len(parts) > 2 else ""
        if "/code" in container_path:
            continue
        if host_path.startswith("./") or host_path.startswith("../") or not os.path.isabs(host_path):
            host_path = os.path.normpath(os.path.join(workdir, host_path))
        bind_spec = f"{host_path}:{container_path}"
        if opts:
            bind_spec += f":{opts}"
        bind_args += ["--bind", bind_spec]
    return bind_args


# ----------------------------
# Per-session state
# ----------------------------


class SessionState:
    """Tracks the sandbox directory and optional Docker container for one rollout."""

    def __init__(self, sandbox: Path, container_id: str = ""):
        self.sandbox = sandbox
        self.container_id = container_id
        self.created_at = time.time()

    async def cleanup(self) -> None:
        if self.container_id:
            for cmd in [
                ["docker", "kill", self.container_id],
                ["docker", "rm", "-f", self.container_id],
            ]:
                try:
                    proc = await asyncio.create_subprocess_exec(
                        *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    )
                    await asyncio.wait_for(proc.wait(), timeout=15)
                except Exception:
                    pass
        if self.sandbox.exists():
            shutil.rmtree(self.sandbox, ignore_errors=True)


# ----------------------------
# Server
# ----------------------------


class ToolRequest(BaseModel):
    model_config = ConfigDict(extra="allow")


class ToolResponse(BaseModel):
    output: str


MAX_OUTPUT_CHARS = 50_000


def _sanitize_docker_name(name: str) -> str:
    import re
    name = re.sub(r"[^a-z0-9_-]", "_", name.lower())
    name = re.sub(r"^[^a-z0-9]+", "", name)
    return name or "cvdp"


class CVDPAgenticHeavyResourcesServer(SimpleResourcesServer):
    config: CVDPAgenticHeavyConfig
    _sessions: Dict[str, SessionState] = {}

    def model_post_init(self, context: Any) -> None:
        self._semaphore = asyncio.Semaphore(value=self.config.num_processes)
        self._sessions = {}
        self._sif_locks: Dict[str, asyncio.Lock] = {}
        self._sif_lock_guard = asyncio.Lock()
        cache = self.config.sif_cache_dir or os.path.join(Path.home(), ".cache", "nemo-gym", "sif")
        self._sif_cache_dir = cache
        os.makedirs(self._sif_cache_dir, exist_ok=True)

    def _backend(self) -> str:
        backend = (self.config.execution_backend or "apptainer").lower()
        if backend not in {"apptainer", "docker"}:
            raise ValueError(f"Unsupported execution_backend={self.config.execution_backend!r}")
        return backend

    def _container_timeout(self) -> int:
        return self.config.container_timeout or self.config.docker_timeout

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        app.post("/ls")(self.tool_ls)
        app.post("/cat")(self.tool_cat)
        app.post("/echo")(self.tool_echo)
        app.post("/edit")(self.tool_edit)
        app.post("/iverilog")(self.tool_iverilog)
        app.post("/vvp")(self.tool_vvp)
        app.post("/pwd")(self.tool_pwd)
        app.post("/cleanup_session")(self.cleanup_session)

        @app.on_event("startup")
        async def start_reaper():
            asyncio.create_task(self._reap_stale_sessions())

        return app

    async def _reap_stale_sessions(self, max_age: int = 3600) -> None:
        while True:
            await asyncio.sleep(120)
            now = time.time()
            stale = [sid for sid, s in self._sessions.items() if now - s.created_at > max_age]
            for sid in stale:
                session = self._sessions.pop(sid, None)
                if session:
                    print(f"[reaper] Cleaning stale session {sid[:8]} (age {now - session.created_at:.0f}s)")
                    await session.cleanup()

    async def cleanup_session(self, request: Request) -> dict:
        session_id = request.session.get(SESSION_ID_KEY, "")
        session = self._sessions.pop(session_id, None)
        if session:
            await session.cleanup()
            return {"status": "cleaned", "session_id": session_id[:8]}
        return {"status": "no_session"}

    def _get_session(self, request: Request) -> SessionState:
        session_id = request.session.get(SESSION_ID_KEY, "")
        session = self._sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=400, detail="No session found. Call seed_session first.")
        return session

    async def _docker_exec(self, session: SessionState, argv: List[str], timeout: int) -> str:
        if not session.container_id:
            return "Error: no Docker container for this session"
        command = f"cd /code && {shlex.join(argv)}"
        exec_cmd = ["docker", "exec", session.container_id, "sh", "-c", command]
        return await self._run_process(exec_cmd, timeout)

    async def _apptainer_exec(self, sandbox: Path, argv: List[str], timeout: int, *, workdir: str = "/code") -> str:
        sif_path = await self._ensure_sif(self.config.oss_sim_image)
        bind_args = ["--bind", f"{sandbox.resolve()}:/code"]
        env_args: List[str] = []
        if self.config.container_tmp_bind_path:
            (sandbox / "rundir" / "tmp").mkdir(parents=True, exist_ok=True)
            bind_args += ["--bind", f"{sandbox.resolve()}/rundir/tmp:{self.config.container_tmp_bind_path}"]
            env_args += _build_runtime_tmp_env_args(self.config.container_tmp_bind_path)
        network_args = ["--net", "--network", "none"] if self.config.disable_network else []
        cmd = [
            "apptainer", "exec",
            "--writable-tmpfs",
            "--home", "/code/rundir",
            *network_args,
            *bind_args,
            *env_args,
            "--pwd", workdir,
            sif_path,
            *argv,
        ]
        return await self._run_process(cmd, timeout, cwd=str(sandbox), kill_process_group=True)

    async def _tool_exec(self, session: SessionState, argv: List[str], timeout: int) -> str:
        if self._backend() == "docker":
            return await self._docker_exec(session, argv, timeout)
        return await self._apptainer_exec(session.sandbox, argv, timeout)

    async def _run_process(
        self,
        cmd: List[str],
        timeout: int,
        *,
        cwd: Optional[str] = None,
        kill_process_group: bool = False,
    ) -> str:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=kill_process_group,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            output = (stdout + stderr).decode("utf-8", errors="replace")
            if len(output) > MAX_OUTPUT_CHARS:
                output = output[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
            exit_info = f"[exit code: {proc.returncode}]"
            return f"{output}\n{exit_info}" if output.strip() else exit_info
        except asyncio.TimeoutError:
            if kill_process_group:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
            partial_out = ""
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
                partial_out = (stdout + stderr).decode("utf-8", errors="replace")
            except Exception:
                pass
            note = f"[command timed out after {timeout}s; process killed]"
            if partial_out.strip():
                if len(partial_out) > MAX_OUTPUT_CHARS:
                    partial_out = partial_out[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
                return f"{partial_out}\n{note}"
            return f"Error: command timed out after {timeout}s (no output produced)"

    # ------ Session management ------

    async def seed_session(self, request: Request, body: AgenticHeavySeedRequest) -> BaseSeedSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        meta = body.verifier_metadata or {}
        context_files = meta.get("context_files", {})
        task_id = meta.get("task_id", "unknown")

        if session_id in self._sessions:
            await self._sessions[session_id].cleanup()

        tmp_root = self.config.harness_workspace_dir.strip()
        if tmp_root:
            os.makedirs(tmp_root, exist_ok=True)
        sandbox = Path(tempfile.mkdtemp(prefix=f"cvdp_agentic_{task_id}_", dir=tmp_root or None))

        for filepath, content in context_files.items():
            dest = sandbox / filepath
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        for d in ["rundir", "verif", "docs", "src"]:
            (sandbox / d).mkdir(exist_ok=True)

        container_id = ""
        if self._backend() == "docker":
            container_id = await self._start_docker_tool_container(sandbox, task_id, session_id)

        self._sessions[session_id] = SessionState(sandbox=sandbox, container_id=container_id)
        return BaseSeedSessionResponse(**body.model_dump())

    async def _start_docker_tool_container(self, sandbox: Path, task_id: str, session_id: str) -> str:
        container_name = _sanitize_docker_name(f"cvdp_ah_{task_id}_{session_id[:8]}_{int(time.time())}")
        uid = os.getuid()
        gid = os.getgid()
        start_cmd = [
            "docker", "run", "-d",
            "--name", container_name,
            "--user", f"{uid}:{gid}",
            "-v", f"{sandbox}:/code",
            "-w", "/code",
            "-e", "HOME=/code/rundir",
            self.config.oss_sim_image,
            "sleep", "infinity",
        ]
        proc = await asyncio.create_subprocess_exec(
            *start_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=60)
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Failed to start Docker container for {task_id}: {err[:500]}")
        return stdout.decode().strip()

    # ------ Filesystem tools (direct host operations on sandbox) ------

    @staticmethod
    def _normalize_path(user_path: str) -> str:
        if user_path.startswith("/code/"):
            return user_path[len("/code/"):]
        if user_path == "/code":
            return "."
        if user_path.startswith("/"):
            return user_path.lstrip("/")
        return user_path

    def _resolve_sandbox_path(self, session: SessionState, user_path: str) -> Path:
        root = session.sandbox.resolve()
        target = (root / user_path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            raise ValueError(f"path escapes sandbox: {user_path}")
        return target

    async def tool_ls(self, request: Request, body: ToolRequest) -> ToolResponse:
        session = self._get_session(request)
        user_path = self._normalize_path(body.model_dump().get("path", "."))
        try:
            target = self._resolve_sandbox_path(session, user_path)
            if not target.exists():
                return ToolResponse(output=f"Error: no such directory: {user_path}")
            if not target.is_dir():
                return ToolResponse(output=f"Error: not a directory: {user_path}")
            return ToolResponse(output="\n".join(sorted(os.listdir(target))))
        except Exception as e:
            return ToolResponse(output=f"Error: {e}")

    async def tool_cat(self, request: Request, body: ToolRequest) -> ToolResponse:
        session = self._get_session(request)
        filename = self._normalize_path(body.model_dump().get("filename", ""))
        if not filename:
            return ToolResponse(output="Error: filename is required")
        try:
            target = self._resolve_sandbox_path(session, filename)
            if not target.exists():
                return ToolResponse(output=f"Error: file not found: {filename}")
            if not target.is_file():
                return ToolResponse(output=f"Error: not a file: {filename}")
            content = target.read_text(encoding="utf-8", errors="replace")
            if len(content) > MAX_OUTPUT_CHARS:
                content = content[:MAX_OUTPUT_CHARS] + f"\n... (truncated, {len(content)} chars total)"
            return ToolResponse(output=content)
        except Exception as e:
            return ToolResponse(output=f"Error: {e}")

    async def tool_echo(self, request: Request, body: ToolRequest) -> ToolResponse:
        session = self._get_session(request)
        data = body.model_dump()
        content = data.get("content", "")
        filename = self._normalize_path(data.get("filename", ""))
        if not filename:
            return ToolResponse(output="Error: filename is required")
        try:
            target = self._resolve_sandbox_path(session, filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
            return ToolResponse(output=f"Written {len(content)} bytes to {filename}")
        except Exception as e:
            return ToolResponse(output=f"Error: {e}")

    async def tool_edit(self, request: Request, body: ToolRequest) -> ToolResponse:
        session = self._get_session(request)
        data = body.model_dump()
        filename = self._normalize_path(data.get("filename", ""))
        old_text = data.get("old_text", "")
        new_text = data.get("new_text", "")
        if not filename:
            return ToolResponse(output="Error: filename is required")
        if old_text == "":
            return ToolResponse(output="Error: old_text must be non-empty. To create a brand-new file, use echo instead.")
        try:
            target = self._resolve_sandbox_path(session, filename)
            if not target.exists():
                return ToolResponse(output=f"Error: file not found: {filename}. edit modifies existing files only; use echo to create a new one.")
            if not target.is_file():
                return ToolResponse(output=f"Error: not a file: {filename}")
            original = target.read_text(encoding="utf-8", errors="replace")
            count = original.count(old_text)
            if count == 0:
                return ToolResponse(output=f"Error: old_text not found in {filename}. Re-read the file and copy an exact span.")
            if count > 1:
                return ToolResponse(output=f"Error: old_text matches {count} locations in {filename}. Include more context.")
            updated = original.replace(old_text, new_text, 1)
            target.write_text(updated, encoding="utf-8")
            delta = len(updated) - len(original)
            sign = "+" if delta >= 0 else ""
            return ToolResponse(output=f"Edited {filename}: replaced {len(old_text)} chars with {len(new_text)} chars ({sign}{delta} net). File is now {len(updated)} bytes.")
        except Exception as e:
            return ToolResponse(output=f"Error: {e}")

    async def tool_pwd(self, request: Request, body: ToolRequest) -> ToolResponse:
        return ToolResponse(output="/code")

    # ------ Simulation tools ------

    async def tool_iverilog(self, request: Request, body: ToolRequest) -> ToolResponse:
        session = self._get_session(request)
        args = body.model_dump().get("args", "")
        if not args:
            return ToolResponse(output="Error: args is required")
        try:
            argv = ["iverilog", *shlex.split(args)]
        except ValueError as exc:
            return ToolResponse(output=f"Error: could not parse args: {exc}")
        result = await self._tool_exec(session, argv, timeout=self.config.tool_timeout)
        return ToolResponse(output=result)

    async def tool_vvp(self, request: Request, body: ToolRequest) -> ToolResponse:
        session = self._get_session(request)
        filename = body.model_dump().get("filename", "")
        if not filename:
            return ToolResponse(output="Error: filename is required")
        if len(shlex.split(filename)) != 1:
            return ToolResponse(output="Error: filename must be a single simulation file path; shell redirection is not supported")
        result = await self._tool_exec(session, ["vvp", filename], timeout=self.config.tool_timeout)
        return ToolResponse(output=result)

    # ------ Verification ------

    async def verify(self, request: Request, body: CVDPAgenticHeavyVerifyRequest) -> CVDPAgenticHeavyVerifyResponse:
        meta = CVDPAgenticHeavyVerifierMetadata.model_validate(body.verifier_metadata)
        category = meta.categories[0] if meta.categories else ""
        difficulty = meta.categories[1] if len(meta.categories) > 1 else ""

        session_id = request.session.get(SESSION_ID_KEY, "")
        session = self._sessions.get(session_id)
        sandbox = session.sandbox if session else None

        async with self._semaphore:
            t0 = time.time()
            exit_code, stderr, service_results = await self._run_container_harness(
                sandbox=sandbox,
                harness_files=meta.harness_files,
                task_id=meta.task_id,
            )
            execution_time = time.time() - t0

        if session:
            await session.cleanup()
        self._sessions.pop(session_id, None)

        return CVDPAgenticHeavyVerifyResponse(
            **body.model_dump(),
            reward=1.0 if exit_code == 0 else 0.0,
            task_id=meta.task_id,
            category=category,
            difficulty=difficulty,
            container_exit_code=exit_code,
            container_stderr=stderr,
            container_services=service_results,
            docker_exit_code=exit_code,
            docker_stderr=stderr,
            docker_services=service_results,
            execution_time=execution_time,
        )

    async def _run_container_harness(
        self,
        sandbox: Optional[Path],
        harness_files: Dict[str, Optional[str]],
        task_id: str,
    ) -> Tuple[int, str, List[Dict]]:
        if sandbox is None:
            return 1, "No sandbox found for this session", []

        compose_content: Optional[str] = None
        for filepath, content in harness_files.items():
            if content is None:
                continue
            content = _apply_substitutions(content, self.config)
            if filepath.endswith("docker-compose.yml"):
                content = _filter_code_volumes(content)
                compose_content = content
            dest = sandbox / filepath
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        if compose_content is None:
            return 1, "No docker-compose.yml found in harness_files", []
        if "__VERIF_EDA_IMAGE__" in compose_content:
            return 1, (
                "Commercial EDA/Xcelium harness service is not supported by this resource server. "
                "Regenerate the dataset without --include-commercial-eda, or provide a separate EDA-enabled server."
            ), []

        compose_data = yaml.safe_load(compose_content) or {}
        services = list((compose_data.get("services") or {}).keys())
        service_results: List[Dict] = []
        for service in services:
            if self._backend() == "docker":
                exit_code, output = await self._run_docker_harness_service(str(sandbox), service, task_id)
            else:
                exit_code, output = await self._run_apptainer_harness_service(
                    str(sandbox), service, task_id, compose_content, harness_files,
                )
            service_results.append({"service": service, "exit_code": exit_code, "stderr": output})

        final_exit_code = 0 if all(r["exit_code"] == 0 for r in service_results) else 1
        combined_stderr = "\n".join(f"[{r['service']}] {r['stderr']}" for r in service_results if r["stderr"])
        return final_exit_code, combined_stderr, service_results

    # Backward-compatible name used by older tests/scripts.
    async def _run_docker_harness(
        self,
        sandbox: Optional[Path],
        harness_files: Dict[str, Optional[str]],
        task_id: str,
    ) -> Tuple[int, str, List[Dict]]:
        return await self._run_container_harness(sandbox, harness_files, task_id)

    async def _run_docker_harness_service(self, workdir: str, service: str, task_id: str) -> Tuple[int, str]:
        path = os.path.abspath(workdir)
        docker_file = os.path.join(path, "docker-compose.yml")
        project_name = _sanitize_docker_name(f"cvdp_ah_{task_id}_{int(time.time())}")
        uid = os.getuid()
        gid = os.getgid()
        cmd = [
            "docker", "compose", "-f", docker_file, "-p", project_name,
            "run", "--rm",
            "--user", f"{uid}:{gid}",
            "-e", "HOME=/code/rundir",
            "-w", "/code/rundir",
            "-v", f"{path}:/code",
            service,
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=workdir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=self._container_timeout())
            exit_code = proc.returncode
        except asyncio.TimeoutError:
            proc.kill()
            stdout_bytes, stderr_bytes = await proc.communicate()
            exit_code = -1
            stderr_bytes = f"docker compose run timed out after {self._container_timeout()}s".encode()
        await self._cleanup_docker_harness(docker_file, project_name, service)
        return exit_code, (stderr_bytes + stdout_bytes).decode("utf-8", errors="replace")

    async def _run_apptainer_harness_service(
        self,
        workdir: str,
        service: str,
        task_id: str,
        compose_content: str,
        harness_files: Dict[str, Optional[str]],
    ) -> Tuple[int, str]:
        path = os.path.abspath(workdir)
        svc = _parse_compose_service(compose_content, service)
        image = svc["image"]
        post_commands: List[str] = []
        if not image:
            compose_data = yaml.safe_load(compose_content) or {}
            image, post_commands = _resolve_image_for_service(compose_data, service, harness_files, self.config)
        if not image:
            return 1, f"No image defined for service '{service}'"

        try:
            sif_path = await self._ensure_built_sif(image, post_commands) if post_commands else await self._ensure_sif(image)
        except RuntimeError as exc:
            return 1, str(exc)

        bind_args = ["--bind", f"{path}:/code", *_build_non_code_bind_args(path, svc["volumes"])]
        dot_env = _load_dot_env(path)
        env_args = _build_env_args(svc["environment"], dot_env)
        if self.config.container_tmp_bind_path:
            os.makedirs(os.path.join(path, "rundir", "tmp"), exist_ok=True)
            bind_args += ["--bind", f"{path}/rundir/tmp:{self.config.container_tmp_bind_path}"]
            env_args += _build_runtime_tmp_env_args(self.config.container_tmp_bind_path)
        network_args = ["--net", "--network", "none"] if self.config.disable_network else []
        cmd_parts = _build_command(svc["entrypoint"], svc["command"])
        working_dir = svc["working_dir"] or "/code/rundir"
        if not working_dir.startswith("/code"):
            working_dir = "/code/rundir"

        if cmd_parts:
            cmd = [
                "apptainer", "exec",
                "--writable-tmpfs",
                "--home", "/code/rundir",
                *network_args,
                *bind_args,
                *env_args,
                "--pwd", working_dir,
                sif_path,
                *cmd_parts,
            ]
        else:
            cmd = [
                "apptainer", "run",
                "--writable-tmpfs",
                "--home", "/code/rundir",
                *network_args,
                *bind_args,
                *env_args,
                "--pwd", working_dir,
                sif_path,
            ]

        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=workdir, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=self._container_timeout())
            exit_code = proc.returncode
        except asyncio.TimeoutError:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout_bytes, stderr_bytes = await proc.communicate()
            exit_code = -1
            stderr_bytes = f"apptainer exec timed out after {self._container_timeout()}s".encode()
        return exit_code, (stderr_bytes + stdout_bytes).decode("utf-8", errors="replace")

    async def _cleanup_docker_harness(self, docker_file: str, project_name: str, service: str) -> None:
        for cmd in [
            ["docker", "compose", "-f", docker_file, "-p", project_name, "kill", service],
            ["docker", "rmi", f"{project_name}-{service}"],
            ["docker", "network", "rm", f"{project_name}_default"],
        ]:
            try:
                proc = await asyncio.create_subprocess_exec(
                    *cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                )
                await asyncio.wait_for(proc.wait(), timeout=30)
            except Exception:
                pass

    async def _ensure_built_sif(self, base_image: str, post_commands: List[str]) -> str:
        if not post_commands:
            return await self._ensure_sif(base_image)
        cmd_hash = hashlib.md5("\n".join(post_commands).encode()).hexdigest()[:12]
        safe_name = base_image.replace("/", "_").replace(":", "_") + f"__built_{cmd_hash}.sif"
        sif_path = os.path.join(self._sif_cache_dir, safe_name)
        if os.path.exists(sif_path):
            return sif_path
        async with self._sif_lock_guard:
            if safe_name not in self._sif_locks:
                self._sif_locks[safe_name] = asyncio.Lock()
            lock = self._sif_locks[safe_name]
        async with lock:
            if os.path.exists(sif_path):
                return sif_path
            base_sif = await self._ensure_sif(base_image)
            post_section = "\n    ".join(post_commands)
            def_content = f"Bootstrap: localimage\nFrom: {base_sif}\n\n%post\n    {post_section}\n"
            tmp_def = sif_path + ".def"
            tmp_sif = sif_path + ".building"
            with open(tmp_def, "w") as f:
                f.write(def_content)
            proc = await asyncio.create_subprocess_exec(
                "apptainer", "build", "--force", tmp_sif, tmp_def,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            os.unlink(tmp_def)
            if proc.returncode != 0:
                if os.path.exists(tmp_sif):
                    os.unlink(tmp_sif)
                raise RuntimeError(f"apptainer build failed: {stderr.decode(errors='replace')}")
            os.rename(tmp_sif, sif_path)
            return sif_path

    async def _ensure_sif(self, image: str) -> str:
        safe_name = image.replace("/", "_").replace(":", "_") + ".sif"
        sif_path = os.path.join(self._sif_cache_dir, safe_name)
        if os.path.exists(sif_path):
            return sif_path
        async with self._sif_lock_guard:
            if image not in self._sif_locks:
                self._sif_locks[image] = asyncio.Lock()
            lock = self._sif_locks[image]
        async with lock:
            if os.path.exists(sif_path):
                return sif_path
            tmp_path = sif_path + ".pulling"
            proc = await asyncio.create_subprocess_exec(
                "apptainer", "pull", "--force", tmp_path, f"docker://{image}",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                raise RuntimeError(
                    f"apptainer pull failed for {image} (exit {proc.returncode}): {stderr.decode(errors='replace')}"
                )
            os.rename(tmp_path, sif_path)
            return sif_path


if __name__ == "__main__":
    CVDPAgenticHeavyResourcesServer.run_webserver()
