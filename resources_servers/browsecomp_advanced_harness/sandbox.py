# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources-owned, supervised execution of the existing read-only shell."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from uuid import uuid4

from nemo_gym.agents import SandboxCommand, SandboxPythonRuntime, SandboxSession
from nemo_gym.sandbox import read_text, upload_text


_RUNTIME = SandboxPythonRuntime(harness="browsecomp-shell", requirements=("regex",), import_check="import regex")
_RUNNER = f"{_RUNTIME.directory}/shell_runner.py"


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
    ready: bool = False
    uploaded_pages: set[str] = field(default_factory=set)

    async def install(self, *, timeout: float) -> None:
        """Install the pinned Python/regex runtime and the checked-out pocketshell source."""
        sandbox = self.session.sandbox
        if not await _RUNTIME.is_installed(sandbox):
            await _RUNTIME.install(sandbox, timeout_s=timeout)
        prepared = await sandbox.exec(
            f"mkdir -p {quote(_RUNTIME.directory + '/pocketshell')} "
            f"{quote(self.session.session_dir)} {quote(self.session.workdir + '/pages')}",
            cwd="/",
            timeout_s=30,
        )
        if prepared.return_code or prepared.error_type:
            raise RuntimeError(f"Could not prepare BrowseComp shell paths: {prepared.stderr or prepared.error_type}")
        self.ready = True

    async def execute(self, *, command: str, duration: float, workspace: Path, close_timeout: float) -> str:
        """Copy new retrieved pages, then interpret the command only inside the sandbox."""
        if not self.ready:
            raise RuntimeError("BrowseComp shell setup has not completed")
        directory = f"{self.session.session_dir.rsplit('/calls/', 1)[0]}/calls/{uuid4().hex}"
        await self.session.begin_activation(session_dir=directory, timeout=close_timeout)

        async def stage() -> SandboxCommand:
            prepared = await self.session.sandbox.exec(f"mkdir -p {quote(directory)}", cwd="/", timeout_s=30)
            if prepared.return_code or prepared.error_type:
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

        return await self.session.execute(
            stage_activation=stage,
            collect=collect,
            timeout=duration + 30,
            close_timeout=close_timeout,
        )

    async def close(self, *, timeout: float) -> None:
        """Confirm termination and stop the owned sandbox; a failed stop remains retryable."""
        await self.session.close(timeout=timeout)
