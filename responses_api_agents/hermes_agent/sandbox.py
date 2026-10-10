# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-server side of Hermes execution in borrowed or owned task sandboxes."""

import asyncio
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agent_utils.sandbox_session import SandboxCommand, SandboxSession
from nemo_gym.base_responses_api_agent import AgentSessionState
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.rollout_observability import AgentObservationBundle
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.utils import read_text, upload_text


LOG = logging.getLogger(__name__)


def _sandbox_hermes_install() -> tuple[str, str]:
    """Return the requirement the sandbox installs and the key that names its runtime directory.

    Both come from the Hermes installed with this server, so ``requirements.txt`` is the only version pin and
    the sandbox runs the same Hermes as the host. A git install is fetched as a GitHub archive, so the sandbox
    does not need git. The ``mcp`` extra carries Hermes' MCP client, which episode tool grants use.
    """
    distribution = importlib.metadata.distribution("hermes-agent")
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")
    commit = (direct_url.get("vcs_info") or {}).get("commit_id")
    if commit is None:
        return f"hermes-agent[mcp]=={distribution.version}", distribution.version
    url = str(direct_url.get("url") or "").removesuffix(".git")
    if not url.startswith("https://github.com/"):
        raise RuntimeError(f"Cannot build a sandbox install URL for hermes-agent installed from {url!r}")
    return f"hermes-agent[mcp] @ {url}/archive/{commit}.tar.gz", commit[:12]


_HERMES_REQUIREMENT, _HERMES_RUNTIME_KEY = _sandbox_hermes_install()
_SANDBOX_RUNTIME_DIR = f"/tmp/nemo-gym-hermes-runtime-{_HERMES_RUNTIME_KEY}"
_SANDBOX_UV = f"{_SANDBOX_RUNTIME_DIR}/uv"
_SANDBOX_PYTHON = f"{_SANDBOX_RUNTIME_DIR}/venv/bin/python"
_SANDBOX_RUNNER = f"{_SANDBOX_RUNTIME_DIR}/sandbox_runner.py"
_SANDBOX_OBSERVER = f"{_SANDBOX_RUNTIME_DIR}/sandbox_observer.py"
_SANDBOX_MODEL_KWARGS = f"{_SANDBOX_RUNTIME_DIR}/model_kwargs.py"

# uv release targets by the sandbox's `uname -m`; the host binary is reused when the
# architectures match, otherwise a matching build of the host's uv version is fetched once.
_UV_RELEASE_TARGETS = {
    "x86_64": "x86_64-unknown-linux-gnu",
    "amd64": "x86_64-unknown-linux-gnu",
    "aarch64": "aarch64-unknown-linux-gnu",
    "arm64": "aarch64-unknown-linux-gnu",
}
_UV_RELEASE_URL = "https://github.com/astral-sh/uv/releases/download/{version}/uv-{target}.tar.gz"
# Where uv builds for sandbox architectures other than the host's are cached.
DEFAULT_UV_CACHE_DIR = "~/.cache/nemo_gym/uv"


def _host_uv_version(uv_path: str) -> str:
    """``uv --version`` -> ``0.12.3``."""
    output = subprocess.run([uv_path, "--version"], check=True, capture_output=True, text=True).stdout
    parts = output.split()
    if len(parts) < 2 or parts[0] != "uv":
        raise RuntimeError(f"Unexpected uv version output: {output!r}")
    return parts[1]


def _download_uv(version: str, target: str, destination: Path) -> None:
    """Fetch the uv release for ``target`` into ``destination`` (atomic, idempotent)."""
    url = _UV_RELEASE_URL.format(version=version, target=target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nemo-gym-uv-", dir=destination.parent) as tmp:
        archive = Path(tmp) / "uv.tar.gz"
        try:
            with urllib.request.urlopen(url, timeout=120) as response, archive.open("wb") as stream:
                shutil.copyfileobj(response, stream)
        except OSError as exc:
            raise RuntimeError(f"Could not download uv {version} for {target} from {url}: {exc}") from exc
        with tarfile.open(archive, "r:gz") as tar:
            member = next((m for m in tar.getmembers() if m.isfile() and m.name.endswith("/uv")), None)
            if member is None:
                raise RuntimeError(f"uv release archive {url} has no uv binary")
            tar.extract(member, tmp, filter="data")
            extracted = Path(tmp) / member.name
        extracted.chmod(0o755)
        os.replace(extracted, destination)


async def uv_for_sandbox(sandbox: AsyncSandbox, workdir: str | None, cache_dir: str) -> Path:
    """The uv binary to upload: the host's when architectures match, else a matching release.

    An aarch64 agent host driving x86_64 sandboxes cannot run its own uv inside them, so the sandbox's
    ``uname -m`` decides which build is uploaded; builds for other architectures are fetched once into
    ``cache_dir``.
    """
    host_uv = shutil.which("uv")
    if host_uv is None:
        raise RuntimeError("Hermes agent server requires uv to install the sandbox runtime")
    probe = await sandbox.exec("uname -m", cwd=workdir, timeout_s=30)
    arch = (probe.stdout or "").strip()
    if not arch or arch == platform.machine():
        return Path(host_uv)
    target = _UV_RELEASE_TARGETS.get(arch)
    if target is None:
        raise RuntimeError(f"No uv build is known for sandbox architecture {arch!r}")
    version = await asyncio.to_thread(_host_uv_version, host_uv)
    cached = Path(cache_dir).expanduser() / version / target / "uv"
    if not cached.is_file():
        await asyncio.to_thread(_download_uv, version, target, cached)
    return cached


class HarnessProcessInfo(BaseModel):
    """Optional identity of the Hermes harness process, separate from cleanup evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int
    python: str | None = None


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read Hermes diagnostics without failing an otherwise valid episode."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("Hermes runtime metadata is missing or malformed")
        return None


@dataclass
class HermesSandboxSession(AgentSessionState):
    """Hermes transport and cleanup; HTTP retry bookkeeping stays in the agent base."""

    session: SandboxSession[dict[str, JsonValue]]
    observations: AgentObservationBundle | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None
    task: asyncio.Task[NeMoGymResponse] | None = None
    runtime_info: HarnessProcessInfo | None = None

    async def install_runtime(self, *, install_timeout: float, uv_cache_dir: str = DEFAULT_UV_CACHE_DIR) -> None:
        """Reuse or install the pinned runtime and stage the Hermes harness files."""
        prepared = await self.session.sandbox.exec(
            f"mkdir -p {quote(_SANDBOX_RUNTIME_DIR)} {quote(self.session.session_dir)}",
            cwd=self.session.workdir,
            timeout_s=30,
        )
        if prepared.return_code != 0:
            raise RuntimeError(prepared.stderr or prepared.stdout or "Failed to prepare Hermes sandbox paths")
        if not await self._runtime_installed():
            await self._install_runtime(install_timeout, uv_cache_dir)
        await self.session.sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), _SANDBOX_RUNNER)
        await self.session.sandbox.upload(Path(__file__).with_name("sandbox_observer.py"), _SANDBOX_OBSERVER)
        await self.session.sandbox.upload(Path(__file__).with_name("model_kwargs.py"), _SANDBOX_MODEL_KWARGS)

    async def _runtime_installed(self) -> bool:
        """Whether the pinned Hermes and its MCP client import from its runtime path.

        The path is keyed by the pinned commit, so a runtime baked into the image or left by an earlier
        session in this sandbox is reused.
        """
        check = await self.session.sandbox.exec(
            f"{quote(_SANDBOX_PYTHON)} -c 'import run_agent, mcp'",
            cwd=self.session.workdir,
            timeout_s=120,
        )
        return check.return_code == 0

    async def _install_runtime(self, timeout: float, uv_cache_dir: str) -> None:
        uv_path = await uv_for_sandbox(self.session.sandbox, self.session.workdir, uv_cache_dir)
        await self.session.sandbox.upload(uv_path, _SANDBOX_UV)
        venv = quote(_SANDBOX_RUNTIME_DIR + "/venv")
        # A runtime that failed the import check is incomplete, so rebuild it rather than reuse it.
        install = await self.session.sandbox.exec(
            f"chmod 755 {quote(_SANDBOX_UV)} && rm -rf {venv} && "
            f"{quote(_SANDBOX_UV)} venv {venv} --python 3.13 && "
            f"{quote(_SANDBOX_UV)} pip install --python {quote(_SANDBOX_PYTHON)} {quote(_HERMES_REQUIREMENT)}",
            cwd=self.session.workdir,
            timeout_s=timeout,
        )
        if install.return_code != 0 or not await self._runtime_installed():
            raise RuntimeError(install.stderr or install.stdout or "Hermes sandbox installation failed")

    async def upload_json(self, name: str, payload: dict[str, JsonValue]) -> None:
        """Write a Hermes input under the session directory using file transfer."""
        await upload_text(self.session.sandbox, path=f"{self.session.session_dir}/{name}", text=json.dumps(payload))

    async def read_json(self, name: str) -> dict[str, JsonValue]:
        """Read a Hermes output object without mixing it with the cleanup contract."""
        path = f"{self.session.session_dir}/{name}"
        payload = json.loads(await read_text(self.session.sandbox, path=path))
        if not isinstance(payload, dict):
            raise TypeError(f"Hermes sandbox payload at {path} is not an object")
        return payload

    async def close(self, timeout: float) -> None:
        """Finish sandbox capture/release before cancelling the HTTP activation."""
        await self.session.close(timeout=timeout)
        if self.task is not None and not self.task.done() and not self.task.cancelling():
            self.task.cancel()
        if self.task is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=timeout)
            except asyncio.CancelledError:
                if not self.task.cancelled():
                    raise
            except Exception:
                if not self.task.done():
                    raise
                # The agent base replays the activation error independently of close.

    async def execute(
        self, payload: dict[str, JsonValue], *, timeout: float, close_timeout: float
    ) -> dict[str, JsonValue]:
        """Use the common lifecycle, then check the Hermes-specific result."""
        output = await self.session.execute(
            stage_activation=lambda: self.stage_activation(payload),
            collect=self.collect_artifacts,
            timeout=timeout,
            close_timeout=close_timeout,
        )
        if output.get("error") is not None:
            raise RuntimeError(f"Hermes sandbox runner failed: {output['error']}\n{output.get('traceback', '')}")
        return output

    async def stage_activation(self, payload: dict[str, JsonValue]) -> SandboxCommand:
        """Stage this activation's input and describe its harness process."""
        await self.upload_json("input.json", {**payload, "stop_request_path": self.session.stop_request_path})
        return SandboxCommand(
            argv=[
                _SANDBOX_PYTHON,
                _SANDBOX_RUNNER,
                f"{self.session.session_dir}/input.json",
                f"{self.session.session_dir}/output.json",
            ],
            python=_SANDBOX_PYTHON,
        )

    async def collect_artifacts(self) -> dict[str, JsonValue]:
        """Copy Hermes output before close removes files, including interrupted output."""
        try:
            output = await self.read_json("output.json")
        except Exception as error:
            logs = await self.session.read_output_log()
            raise RuntimeError(f"Hermes sandbox runner exited without output: {logs}") from error
        self.runtime_info = parse_runtime_info(output.get("runtime"))
        return output
