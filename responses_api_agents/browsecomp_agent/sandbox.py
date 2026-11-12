# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run the repository's BrowseComp loop in a supervised task sandbox."""

import asyncio
import functools
import importlib.metadata
import json
import logging
import os
import re
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from shlex import quote

from packaging.requirements import Requirement
from pydantic import JsonValue

import nemo_gym
from nemo_gym.agents import AgentSessionState, SandboxCommand, SandboxPythonRuntime, SandboxSession
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.server_runtime import DirectHTTPToolAccess


LOG = logging.getLogger(__name__)


# Most characters of upstream error text that an error message keeps; failure rows carry these messages.
_ERROR_DETAIL_CHARS = 4000


@functools.cache
def _runtime() -> SandboxPythonRuntime:
    # Upload the exact installed Gym sources separately. Resolve their external dependencies using the
    # same pinned host environment as every other SandboxPythonRuntime, without dev/provider extras.
    requirements = tuple(
        raw
        for raw in importlib.metadata.requires("nemo-gym") or ()
        if (requirement := Requirement(raw)).marker is None or requirement.marker.evaluate({"extra": ""})
    )
    # Hydra/OmegaConf need this pure-Python dependency, whose pinned release has no wheel.
    # The direct, hash-checked sdist opts only this package into the shared runtime's host build path.
    # Version and digest match this adapter's uv.lock; all other packages still require target wheels.
    antlr = (
        "antlr4-python3-runtime @ https://files.pythonhosted.org/packages/3e/38/"
        "7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/"
        "antlr4-python3-runtime-4.9.3.tar.gz"
        "#sha256=f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b"
    )
    return SandboxPythonRuntime(
        harness="browsecomp",
        requirements=(*requirements, antlr),
        import_check="import aiohttp, openai, pydantic, fastapi, omegaconf",
    )


def _write_sources(destination: Path) -> None:
    """Package code only, never the host configuration, datasets, credentials or caches."""
    library = Path(nemo_gym.__file__).parent
    adapter = Path(__file__).parent
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for root, prefix in ((library, "nemo_gym"), (adapter, "responses_api_agents/browsecomp_agent")):
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if any(part.startswith(".") or part in {"tests", "__pycache__"} for part in relative.parts):
                    continue
                if path.is_file() and path.suffix in {".py", ".sh"}:
                    archive.write(path, f"{prefix}/{relative.as_posix()}")


class SandboxIsolationError(RuntimeError):
    """The provider's sandbox does not isolate the harness's processes; no retry can fix it."""


@dataclass
class BrowsecompSandboxSession(AgentSessionState):
    """Sandbox ownership and retained activation; resource state stays behind its HTTP grant."""

    session: SandboxSession[dict[str, JsonValue]]
    tool_access: DirectHTTPToolAccess | None
    resources_cookies: dict[str, str]
    started_at: float | None = None
    _stop_logged: bool = False
    task: asyncio.Task[NeMoGymResponse] | None = None
    activation_request: NeMoGymResponseCreateParamsNonStreaming | None = None

    async def install_runtime(self, *, timeout: float) -> None:
        """Install a private Python, then upload this revision of Gym and the harness."""
        runtime = _runtime()
        await runtime.install(self.session.sandbox, timeout_s=timeout)
        await self._check_process_isolation(python=runtime.python)
        directory = self.session.session_dir
        prepared = await self.session.sandbox.exec(
            f"mkdir -p {quote(directory)} {quote(self.session.workdir)}", cwd="/", timeout_s=30
        )
        if prepared.return_code != 0 or prepared.error_type is not None:
            raise RuntimeError(prepared.stderr or "Cannot prepare BrowseComp sandbox directory")
        with tempfile.TemporaryDirectory(prefix="browsecomp-source-") as temporary:
            archive = Path(temporary) / "source.zip"
            await asyncio.to_thread(_write_sources, archive)
            await self.session.sandbox.upload(archive, f"{directory}/source.zip")
        result = await self.session.sandbox.exec(
            f"{quote(runtime.python)} -I -m zipfile -e {quote(directory + '/source.zip')} "
            f"{quote(directory + '/source')}",
            cwd="/",
            timeout_s=60,
        )
        if result.return_code != 0 or result.error_type is not None:
            raise RuntimeError(result.stderr or "Cannot unpack BrowseComp sandbox sources")

    async def _check_process_isolation(self, *, python: str) -> None:
        # Older Enroot silently ignores ENROOT_UNSHARE_PID. Check the actual exec
        # namespace and its proc mount before delivering any rollout input. The resources server's shell sandbox
        # (resources_servers/browsecomp_advanced_harness/sandbox.py) applies the same rule: only a finished probe
        # that names the host's namespace, or a /proc mount of another one, is final.
        sandbox = self.session.sandbox
        if sandbox.provider_name != "enroot":
            return
        host_namespace = os.readlink("/proc/self/ns/pid")
        probe = "import os; print(os.readlink('/proc/self/ns/pid')); print(os.readlink('/proc/1/ns/pid'))"
        result = await sandbox.exec(f"{quote(python)} -I -c {quote(probe)}", cwd="/", timeout_s=30)
        if result.return_code != 0 or result.error_type is not None:
            # The probe did not run to an answer, which says nothing about isolation; a retry may get one.
            outcome = result.error_type or f"exit code {result.return_code}"
            raise RuntimeError(f"BrowseComp could not finish its process isolation probe: {outcome}")
        namespaces = (result.stdout or "").splitlines()
        if (
            re.fullmatch(r"pid:\[\d+\]", host_namespace) is None
            or len(namespaces) != 2
            or any(re.fullmatch(r"pid:\[\d+\]", namespace) is None for namespace in namespaces)
        ):
            raise RuntimeError("BrowseComp's process isolation probe did not name a PID namespace")
        if namespaces[0] != namespaces[1] or namespaces[0] == host_namespace:
            raise SandboxIsolationError(
                "BrowseComp requires an isolated PID namespace and matching /proc mount. "
                "Install a complete Enroot distribution with PID isolation support (4.2 or newer)."
            )

    async def stage_activation(self, payload: dict[str, JsonValue]) -> SandboxCommand:
        """Write input as data; model-controlled strings never become a launch command."""
        directory = self.session.session_dir
        await upload_text(self.session.sandbox, path=f"{directory}/input.json", text=json.dumps(payload))
        return SandboxCommand(
            argv=[
                _runtime().python,
                "-I",
                f"{directory}/source/responses_api_agents/browsecomp_agent/sandbox_runner.py",
                f"{directory}/input.json",
                f"{directory}/output.json",
            ]
        )

    async def collect(self) -> dict[str, JsonValue]:
        """Read the runner receipt only after the shared supervisor confirms termination."""
        try:
            output = json.loads(await read_text(self.session.sandbox, path=f"{self.session.session_dir}/output.json"))
        except Exception as error:
            # The tail holds the runner's last words; the whole log of a long run would bloat failure rows.
            log = (await self.session.read_output_log())[-_ERROR_DETAIL_CHARS:]
            raise RuntimeError(f"BrowseComp runner did not produce a receipt: {log}") from error
        if not isinstance(output, dict):
            raise TypeError("BrowseComp runner receipt must be an object")
        return output

    async def close(self, *, timeout: float) -> None:
        """Capture and release the sandbox before cancelling an HTTP activation waiter."""
        sandbox = self.session.sandbox
        provider, sandbox_id = sandbox.provider_name, sandbox.sandbox_id
        status = "ok"
        try:
            await self.session.close(timeout=timeout)
        except BaseException as error:
            status = f"failed:{type(error).__name__}"
            raise
        finally:
            if not self._stop_logged:
                lifetime = time.monotonic() - self.started_at if self.started_at is not None else -1.0
                event = "sandbox_stop" if self.session.owns_sandbox else "sandbox_disconnect"
                LOG.info(
                    "[browsecomp][%s] role=agent session=%s provider=%s id=%s status=%s lifetime_s=%.3f",
                    event,
                    self.request.agent_session_id,
                    provider,
                    sandbox_id,
                    status,
                    lifetime,
                )
                self._stop_logged = status == "ok"
        if self.task is not None and not self.task.done():
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
