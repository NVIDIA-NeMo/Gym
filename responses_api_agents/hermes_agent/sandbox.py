# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-server side of Hermes execution in borrowed or owned task sandboxes."""

import asyncio
import json
import logging
import os
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

from pydantic import BaseModel, ConfigDict, JsonValue, ValidationError

from nemo_gym.agents import AgentSessionState, SandboxCommand, SandboxPlatform, SandboxPythonRuntime, SandboxSession
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.trajectory import AgentObservationBundle
from responses_api_agents.hermes_agent.runtime import (
    HERMES_MIN_GLIBC_MINOR,
    HERMES_PYTHON_VERSION,
    HERMES_SOURCE,
    source_checkout,
    stage_source,
)


LOG = logging.getLogger(__name__)


class HermesPlatformUnsupportedError(RuntimeError):
    """The task sandbox's platform cannot run the locked Hermes wheels, so every attempt on its image fails."""


class HermesPythonRuntime(SandboxPythonRuntime):
    """Bundle the exact upstream lock and source tree for an offline task sandbox."""

    async def _build_site_packages(self, platform: SandboxPlatform, destination: Path) -> None:
        if platform.libc == "gnu" and (platform.glibc_minor or 0) < HERMES_MIN_GLIBC_MINOR:
            version = f"2.{platform.glibc_minor}" if platform.glibc_minor is not None else "unknown"
            raise HermesPlatformUnsupportedError(
                f"Hermes requires glibc 2.{HERMES_MIN_GLIBC_MINOR} or newer for its locked wheels. "
                f"The sandbox reported glibc {version} on {platform.arch}."
            )
        await asyncio.to_thread(self.build_packages, platform.wheel_platform, destination)

    def build_packages(self, wheel_platform: str, destination: Path) -> None:
        """Create the relocatable package archive for one Python 3.14 platform."""
        source = source_checkout(self.requirements[0])
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
            scratch = Path(temporary)
            requirements = scratch / "requirements.txt"
            with requirements.open("w") as output:
                subprocess.run(
                    ["uv", "export", "--frozen", "--no-dev", "--extra", "mcp", "--no-emit-project"],
                    cwd=source,
                    stdout=output,
                    check=True,
                )
            minor = ".".join(self.python_version.split(".")[:2])
            site = scratch / f"python/lib/python{minor}/site-packages"
            # Export already applied the upstream overrides. Install exactly those hashes.
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--no-config",
                    "--no-deps",
                    "--require-hashes",
                    "--only-binary",
                    ":all:",
                    "--python-platform",
                    wheel_platform,
                    "--python-version",
                    self.python_version,
                    "--target",
                    str(site),
                    "--link-mode",
                    "copy",
                    "-r",
                    str(requirements),
                ],
                cwd=scratch,
                check=True,
            )
            stage_source(source, site)
            archive = scratch / "packages.tar.gz"
            with tarfile.open(archive, "w:gz", compresslevel=1) as bundle:
                bundle.add(scratch / "python", arcname="python")
            os.replace(archive, destination)


_HERMES_RUNTIME = HermesPythonRuntime(
    harness="hermes",
    python_version=HERMES_PYTHON_VERSION,
    requirements=(HERMES_SOURCE,),
    constraints=(),
    import_check=(
        "import os; os.environ['HERMES_HOME'] = os.path.join(os.environ['HOME'], '.hermes'); "
        "from run_agent import AIAgent; from tools.mcp_tool_discovery import discover_mcp_tools; import mcp"
    ),
)
# These paths are inside the task sandbox. The runner uses the absolute interpreter path regardless of the task's PATH.
_SANDBOX_RUNTIME_DIR = _HERMES_RUNTIME.directory
_SANDBOX_PYTHON = _HERMES_RUNTIME.python
_SANDBOX_RUNNER = f"{_SANDBOX_RUNTIME_DIR}/sandbox_runner.py"
_SANDBOX_OBSERVER = f"{_SANDBOX_RUNTIME_DIR}/sandbox_observer.py"
_SANDBOX_MODEL_KWARGS = f"{_SANDBOX_RUNTIME_DIR}/model_kwargs.py"


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
    # Multi-turn continuation: Hermes keeps no session of its own between processes, so the session holds its
    # conversation (as Hermes' model context had it after the last activation) and the first activation's system
    # message, and gives both to the next activation's process.
    conversation: list[dict[str, JsonValue]] | None = None
    system_message: str | None = None
    activation_id: int | None = None
    # Whether the current activation's observations are already in ``observations``.
    activation_recorded: bool = False

    async def prepare_activation(self, activation_id: int, *, timeout: float) -> None:
        """Rearm the supervised launch for a later activation in a fresh directory of the same sandbox.

        The first activation uses the directory the seed prepared. Files Hermes changed in the sandbox stay; the
        shell state of the previous activation's process (working directory, environment, background jobs) does not.
        """
        self.activation_id = activation_id
        self.activation_recorded = False
        if activation_id == 1:
            return
        self.runtime_info = None
        directory = f"{self.session.root_dir}/activation-{activation_id}"
        await self.session.begin_activation(session_dir=directory, timeout=timeout)
        result = await self.session.sandbox.exec(f"mkdir -p -- {quote(directory)}", cwd="/", timeout_s=timeout)
        if result.error_type == "timeout":
            raise TimeoutError("Hermes activation directory preparation timed out")
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(f"Cannot prepare Hermes activation directory: {result.stderr or result.stdout}")

    async def install_runtime(self, *, install_timeout: float) -> None:
        """Reuse or install the pinned runtime and stage the Hermes harness files.

        The runtime directory is keyed by its inputs, so a runtime an earlier session left in this sandbox is
        reused when it passes the import check.
        """
        prepared = await self.session.sandbox.exec(
            f"mkdir -p {quote(_SANDBOX_RUNTIME_DIR)} {quote(self.session.session_dir)}", cwd="/", timeout_s=30
        )
        if prepared.return_code != 0:
            raise RuntimeError(prepared.stderr or prepared.stdout or "Failed to prepare Hermes sandbox paths")
        if not await _HERMES_RUNTIME.is_installed(self.session.sandbox):
            await _HERMES_RUNTIME.install(self.session.sandbox, timeout_s=install_timeout)
        await self.session.sandbox.upload(Path(__file__).with_name("sandbox_runner.py"), _SANDBOX_RUNNER)
        await self.session.sandbox.upload(Path(__file__).with_name("sandbox_observer.py"), _SANDBOX_OBSERVER)
        await self.session.sandbox.upload(Path(__file__).with_name("model_kwargs.py"), _SANDBOX_MODEL_KWARGS)

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
                # Ignore task Python settings without changing the environment inherited by tool subprocesses.
                "-E",
                "-s",
                _SANDBOX_RUNNER,
                f"{self.session.session_dir}/input.json",
                f"{self.session.session_dir}/output.json",
            ],
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
