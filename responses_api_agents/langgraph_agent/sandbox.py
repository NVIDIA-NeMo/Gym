# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install and supervise the exact LangGraph runtime in a borrowed or owned sandbox."""

import asyncio
import hashlib
import importlib.metadata
import io
import json
import tarfile
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from shlex import quote

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from pydantic import JsonValue

import nemo_gym
from nemo_gym.agents import DEFAULT_UV_CACHE_DIR, AgentSessionState, SandboxCommand, SandboxSession, uv_for_sandbox
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text


@dataclass(frozen=True)
class RuntimeBundle:
    """Source snapshot, dependency manifest and version constraints installed by every graph runner."""

    digest: str
    archive: bytes
    constraints: str


@lru_cache(maxsize=1)
def build_runtime_bundle() -> RuntimeBundle:
    """Package this server's Gym source, not an unrelated PyPI or main-branch revision.

    Install only Gym's base dependencies plus the pinned graph runtime. The host's
    dev/telemetry extras and provider credentials never enter this source archive.
    """
    distribution = importlib.metadata.distribution("nemo-gym")
    dependencies = [
        item
        for item in distribution.requires or []
        if Requirement(item).marker is None
        or "extra" not in str(Requirement(item).marker)
        or Requirement(item).marker.evaluate({"extra": ""})
    ]
    agent_dir = Path(__file__).parent
    runtime_requirements = [
        line.strip()
        for line in (agent_dir / "runtime-requirements.txt").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    dependencies.extend(runtime_requirements)
    # Hold every other dependency at the version this server resolved, so sandboxes install what it was tested
    # with rather than whatever the index serves that day. Packages the host lacks still resolve freely, and a
    # local build ("+" version) cannot come from an index, so it stays unconstrained.
    pinned: dict[str, str] = {}
    for installed in importlib.metadata.distributions():
        name = installed.metadata["Name"]
        if name and "+" not in installed.version:
            pinned.setdefault(canonicalize_name(name), f"{name}=={installed.version}")
    for name in ["nemo-gym", *(Requirement(item).name for item in runtime_requirements)]:
        pinned.pop(canonicalize_name(name), None)
    constraints = "".join(f"{line}\n" for line in sorted(pinned.values()))
    files: dict[str, bytes] = {}
    gym_dir = Path(nemo_gym.__file__).parent
    for path in sorted(gym_dir.rglob("*")):
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml", ".json", ".sh", ".toml", ".j2"}:
            files[f"nemo_gym/{path.relative_to(gym_dir)}"] = path.read_bytes()
    for filename in ("workflows.py", "sandbox_runner.py"):
        files[f"responses_api_agents/langgraph_agent/{filename}"] = (agent_dir / filename).read_bytes()
    project = (
        '[build-system]\nrequires = ["setuptools>=75"]\nbuild-backend = "setuptools.build_meta"\n'
        '[project]\nname = "nemo-gym"\n'
        f'version = {json.dumps(distribution.version)}\nrequires-python = ">=3.13.14"\n'
        f"dependencies = {json.dumps(dependencies)}\n"
        '[tool.setuptools.packages.find]\ninclude = ["nemo_gym*", "responses_api_agents*"]\n'
        '[tool.setuptools.package-data]\n"*" = ["**/*.yaml", "**/*.yml", "**/*.json", "**/*.sh", "**/*.toml", "**/*.j2"]\n'
    )
    files["pyproject.toml"] = project.encode()
    digest = hashlib.sha256(constraints.encode() + b"\0")
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, content in sorted(files.items()):
            digest.update(name.encode() + b"\0" + content)
            info = tarfile.TarInfo(f"runtime/{name}")
            info.size = len(content)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(content))
    return RuntimeBundle(digest=digest.hexdigest(), archive=stream.getvalue(), constraints=constraints)


@dataclass
class LangGraphSandboxSession(AgentSessionState):
    """One graph activation and the connection retained until its cleanup succeeds."""

    session: SandboxSession[dict[str, JsonValue]]
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    activation_path: str | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_digest: str | None = None

    @property
    def python(self) -> str:
        """Interpreter isolated to this session's exact source and dependencies."""
        return f"{self.session.session_dir}/venv/bin/python"

    async def install_runtime(self, *, install_timeout: float, uv_cache_dir: str = DEFAULT_UV_CACHE_DIR) -> None:
        """Install the whole graph runtime before any model call or activation.

        Each session owns its installation directory, avoiding shared-sandbox install
        races and leaving all task files outside the directory intact on close.
        """
        sandbox = self.session.sandbox
        directory = self.session.session_dir
        prepared = await sandbox.exec(
            f"mkdir -p {quote(directory)}",
            cwd=self.session.workdir,
            timeout_s=30,
            user=self.session.user,
        )
        if prepared.error_type or prepared.return_code != 0:
            raise RuntimeError(f"Could not prepare LangGraph runtime: {prepared.stderr or prepared.stdout}")
        bundle = await asyncio.to_thread(build_runtime_bundle)
        uv_path = await uv_for_sandbox(sandbox, self.session.workdir, uv_cache_dir)
        remote_uv = f"{directory}/uv"
        await sandbox.upload(uv_path, remote_uv)
        with tempfile.TemporaryDirectory(prefix="langgraph-runtime-") as temporary:
            archive = Path(temporary) / "runtime.tar.gz"
            archive.write_bytes(bundle.archive)
            await sandbox.upload(archive, f"{directory}/runtime.tar.gz")
        await upload_text(sandbox, path=f"{directory}/constraints.txt", text=bundle.constraints)
        # The executable, archive and constraints arrive through file transfer, never shell interpolation.
        executable = await sandbox.exec(f"chmod 755 {quote(remote_uv)}", timeout_s=30)
        if executable.error_type or executable.return_code != 0:
            raise RuntimeError("Could not make the LangGraph runtime installer executable")
        install_env = {
            **(self.session.env or {}),
            "UV_CACHE_DIR": f"{directory}/uv-cache",
            "UV_PYTHON_INSTALL_DIR": f"{directory}/python",
        }
        installed = await sandbox.exec(
            f"{quote(remote_uv)} venv {quote(directory + '/venv')} --python 3.13.14 && "
            f"{quote(remote_uv)} pip install --python {quote(self.python)} "
            f"--constraint {quote(directory + '/constraints.txt')} {quote(directory + '/runtime.tar.gz')} && "
            f"{quote(self.python)} -I -c 'from responses_api_agents.langgraph_agent.workflows import run_workflow'",
            cwd=directory,
            env=install_env,
            user=self.session.user,
            timeout_s=install_timeout,
        )
        if installed.error_type or installed.return_code != 0:
            raise RuntimeError(f"LangGraph runtime installation failed: {installed.stderr or installed.stdout}")
        self.runtime_digest = bundle.digest

    async def stage_activation(self, payload: dict[str, JsonValue]) -> SandboxCommand:
        """Transfer a request containing graph settings and the Gym model URL only."""
        directory = self.session.session_dir
        await upload_text(self.session.sandbox, path=f"{directory}/input.json", text=json.dumps(payload))
        return SandboxCommand(
            argv=[
                self.python,
                "-I",
                "-m",
                "responses_api_agents.langgraph_agent.sandbox_runner",
                f"{directory}/input.json",
                f"{directory}/output.json",
            ]
        )

    async def collect_artifacts(self) -> dict[str, JsonValue]:
        """Read the runner output only after the shared supervisor confirmed termination."""
        try:
            output = json.loads(await read_text(self.session.sandbox, path=f"{self.session.session_dir}/output.json"))
        except (OSError, ValueError, RuntimeError) as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(f"LangGraph runner exited without a valid result: {logs}") from error
        if not isinstance(output, dict):
            raise TypeError("LangGraph runner output must be a JSON object")
        return output

    async def execute(
        self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float
    ) -> dict[str, JsonValue]:
        """Run and drain the graph before returning its result or propagating its failure."""
        output = await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
        )
        if output.get("error") is not None:
            raise RuntimeError(f"LangGraph runner failed ({output.get('error_type', 'unknown')}): {output['error']}")
        if not isinstance(output.get("response"), dict) or not isinstance(output.get("runtime"), dict):
            raise TypeError("LangGraph runner result must include response and runtime objects")
        return output

    async def close(self, timeout: float) -> None:
        """Confirm remote termination and release before draining the activation waiter."""
        await self.session.close(timeout=timeout)
        if self.task is not None:
            if not self.task.done() and not self.task.cancelling():
                self.task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
            except asyncio.CancelledError:
                if not self.task.cancelled():
                    raise
            except Exception:
                if not self.task.done():
                    raise
