# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources-owned, supervised execution of the existing read-only shell."""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from time import monotonic
from uuid import uuid4

from nemo_gym.agents import SandboxCommand, SandboxPythonRuntime, SandboxSession
from nemo_gym.sandbox import read_text, upload_text


LOG = logging.getLogger(__name__)

_RUNTIME = SandboxPythonRuntime(harness="browsecomp-shell", requirements=("regex",), import_check="import regex")
_RUNNER = f"{_RUNTIME.directory}/shell_runner.py"
_ENROOT_PID_ERROR = (
    "BrowseComp requires an Enroot PID namespace separate from the host and a matching /proc mount. "
    "Upgrade or configure Enroot with PID isolation; older Enroot builds can ignore ENROOT_UNSHARE_PID."
)


class ShellCommandTimedOut(TimeoutError):
    """The supervisor ended the model's command at its deadline; any other TimeoutError is a sandbox failure."""


class SandboxIsolationError(RuntimeError):
    """A completed probe showed the host's PID namespace or a mismatched /proc mount; no retry can change that."""


def shell_source_files() -> dict[str, str]:
    """Deliver the checked-out shell implementation through SandboxSpec's cleanup-aware startup."""
    source = Path(__file__).parent
    files = {
        f"{_RUNTIME.directory}/pocketshell/{path.name}": path.read_text(encoding="utf-8")
        for path in sorted((source / "pocketshell/src/pocketshell").glob("*.py"))
    }
    for name in ("shell_runner.py", "shell_output.py"):
        files[f"{_RUNTIME.directory}/{name}"] = (source / name).read_text(encoding="utf-8")
    return files


@dataclass
class BrowsecompShellSandbox:
    """One Resources session owns its shell sandbox and immutable page copies."""

    session: SandboxSession[str]
    resources_session_id: str = ""
    started_at: float | None = None
    bash_calls: int = 0
    _stop_logged: bool = False
    ready: bool = False
    uploaded_pages: set[str] = field(default_factory=set)

    async def install(self, *, timeout: float) -> None:
        """Install the pinned Python/regex runtime and the checked-out pocketshell source."""
        sandbox = self.session.sandbox
        if not await _RUNTIME.is_installed(sandbox):
            await _RUNTIME.install(sandbox, timeout_s=timeout)
        await self._require_enroot_pid_namespace()
        prepared = await sandbox.exec(
            f"mkdir -p {quote(_RUNTIME.directory + '/pocketshell')} "
            f"{quote(self.session.session_dir)} {quote(self.session.workdir + '/pages')}",
            cwd="/",
            timeout_s=30,
        )
        if prepared.return_code != 0 or prepared.error_type is not None:
            raise RuntimeError(f"Could not prepare BrowseComp shell paths: {prepared.stderr or prepared.error_type}")
        self.ready = True

    async def _require_enroot_pid_namespace(self) -> None:
        """Check actual process isolation before any model-directed command can run."""
        sandbox = self.session.sandbox
        if sandbox.provider_name != "enroot":
            return
        try:
            host_namespace = os.readlink("/proc/self/ns/pid")
        except OSError as error:
            raise RuntimeError(f"{_ENROOT_PID_ERROR} Cannot read the host PID namespace.") from error
        probe = "import os; print(os.readlink('/proc/self/ns/pid')); print(os.readlink('/proc/1/ns/pid'))"
        result = await sandbox.exec(f"{quote(_RUNTIME.python)} -I -c {quote(probe)}", cwd="/", timeout_s=30)
        if result.return_code != 0 or result.error_type is not None:
            raise RuntimeError(f"{_ENROOT_PID_ERROR} The namespace probe did not complete successfully.")
        namespaces = (result.stdout or "").splitlines()
        # Output that names no namespace says nothing about isolation, so it stays retryable like an unfinished probe.
        if (
            not re.fullmatch(r"pid:\[[0-9]+\]", host_namespace)
            or len(namespaces) != 2
            or any(not re.fullmatch(r"pid:\[[0-9]+\]", value) for value in namespaces)
        ):
            raise RuntimeError(f"{_ENROOT_PID_ERROR} The namespace probe did not confirm isolation.")
        if namespaces[0] != namespaces[1] or namespaces[0] == host_namespace:
            raise SandboxIsolationError(f"{_ENROOT_PID_ERROR} The namespace probe did not confirm isolation.")

    async def execute(self, *, command: str, duration: float, workspace: Path, close_timeout: float) -> str:
        """Copy new retrieved pages, then interpret the command only inside the sandbox."""
        if not self.ready:
            raise RuntimeError("BrowseComp shell setup has not completed")
        self.bash_calls += 1
        directory = f"{self.session.session_dir.rsplit('/calls/', 1)[0]}/calls/{uuid4().hex}"
        await self.session.begin_activation(session_dir=directory, timeout=close_timeout)

        async def stage() -> SandboxCommand:
            prepared = await self.session.sandbox.exec(f"mkdir -p {quote(directory)}", cwd="/", timeout_s=30)
            if prepared.return_code != 0 or prepared.error_type is not None:
                raise RuntimeError("Could not prepare BrowseComp shell activation")
            # Pages are immutable after _PageWriter publishes them; only the manifest changes.
            # Never copy arbitrary files, symlinks, credentials, or verifier inputs from the host.
            for path in sorted((workspace / "pages").glob("*.txt")):
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"BrowseComp page is not a regular file: {path.name}")
                if path.name not in self.uploaded_pages:
                    await self.session.sandbox.upload(path, f"{self.session.workdir}/pages/{path.name}")
                    self.uploaded_pages.add(path.name)
            manifest = workspace / "manifest.tsv"
            if manifest.is_symlink() or not manifest.is_file():
                raise ValueError("BrowseComp manifest is not a regular file")
            await self.session.sandbox.upload(manifest, f"{self.session.workdir}/manifest.tsv")
            await upload_text(
                self.session.sandbox,
                path=f"{directory}/input.json",
                text=json.dumps({"command": command, "duration": duration, "workspace": self.session.workdir}),
            )
            return SandboxCommand(
                argv=[_RUNTIME.python, "-E", "-s", _RUNNER, f"{directory}/input.json", f"{directory}/output.txt"]
            )

        async def collect() -> str:
            return await read_text(self.session.sandbox, path=f"{directory}/output.txt")

        try:
            return await self.session.execute(
                stage_activation=stage,
                collect=collect,
                timeout=duration + 30,
                close_timeout=close_timeout,
            )
        except Exception as error:
            # The supervisor ends a command at its deadline and records timed_out; the command wrote no output.
            if self.session.cleanup is not None and self.session.cleanup["timed_out"]:
                raise ShellCommandTimedOut("BrowseComp shell command passed its supervisor deadline") from error
            raise

    async def close(self, *, timeout: float) -> None:
        """Confirm termination and stop the owned sandbox; a failed stop remains retryable."""
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
                lifetime = monotonic() - self.started_at if self.started_at is not None else -1.0
                LOG.info(
                    "[browsecomp][sandbox_stop] role=resources session=%s provider=%s id=%s "
                    "status=%s lifetime_s=%.3f bash_calls=%d pages=%d",
                    self.resources_session_id,
                    provider,
                    sandbox_id,
                    status,
                    lifetime,
                    self.bash_calls,
                    len(self.uploaded_pages),
                )
                self._stop_logged = status == "ok"
