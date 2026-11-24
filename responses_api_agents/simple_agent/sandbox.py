# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-server side of a Simple Agent whose turn loop runs in the sandbox its environment lends.

At seed the agent borrows the sandbox, installs a private Python runtime in it and unpacks this server's own
``nemo_gym`` sources and Simple Agent module there. Each activation stages the loop's arguments, runs
``sandbox_runner.py`` under Gym's process supervisor, and reads back what the loop returned. The runner builds the same
``SimpleAgent`` class and calls the same ``_create_episode`` loop the agent server runs in its own process, so a
sandboxed loop cannot drift from the in-process one.
"""

import functools
import gzip
import hashlib
import io
import json
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from shlex import quote
from typing import Any
from urllib.parse import urlsplit
from uuid import uuid4

import nemo_gym
from nemo_gym.agents import SandboxCommand, SandboxPythonRuntime, SandboxSession
from nemo_gym.agents.utils import BorrowedSandbox
from nemo_gym.sandbox import read_text, upload_text


# The loop's import closure: these distributions and their dependencies, at this server's versions. A unit test
# checks that importing the runner needs nothing else.
RUNTIME_REQUIREMENTS = (
    "aiohttp",
    "fastapi",
    "httptools",
    "hydra-core",
    "itsdangerous",
    "openai",
    "opentelemetry-sdk",
    "orjson",
    "pydot",
    "python-multipart",
    "requests",
    "rich",
    "uvicorn",
    "yappi",
    # OmegaConf pins antlr4-python3-runtime 4.9.*, which PyPI ships only as source. A direct URL is the one kind of
    # requirement the runtime builds from source on this host; it is pure Python, so the build fits every sandbox.
    "antlr4-python3-runtime @ https://files.pythonhosted.org/packages/3e/38/7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/"
    "antlr4-python3-runtime-4.9.3.tar.gz#sha256=f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b",
)
RUNTIME = SandboxPythonRuntime(
    harness="simple_agent",
    requirements=RUNTIME_REQUIREMENTS,
    import_check="import aiohttp, fastapi, hydra, omegaconf, openai, pydantic, uvicorn",
)
# The harness sources live beside the runtime, outside the task's working directory.
HARNESS_DIR = f"{RUNTIME.directory}/harness"
RUNNER = f"{HARNESS_DIR}/responses_api_agents/simple_agent/sandbox_runner.py"
# The Simple Agent files the runner imports; the rest of nemo_gym comes from this server's own package.
_AGENT_FILES = ("__init__.py", "app.py", "sandbox.py", "sandbox_runner.py")


@functools.cache
def harness_archive() -> Path:
    """This server's ``nemo_gym`` Python sources and the Simple Agent files, as a gzipped tar to unpack in a sandbox.

    It is built once per process from the code this server runs, into the system's temporary directory, under a name
    that hashes its contents.
    """
    package = Path(nemo_gym.__file__).parent
    agent = Path(__file__).parent
    members = [(path, f"nemo_gym/{path.relative_to(package)}") for path in sorted(package.rglob("*.py"))]
    members += [(agent / name, f"responses_api_agents/simple_agent/{name}") for name in _AGENT_FILES]
    buffer = io.BytesIO()
    # Fixed member times and order, and a gzip header without a time, keep the archive's bytes, and so its name,
    # the same for the same sources.
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path, name in members:
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    content = gzip.compress(buffer.getvalue(), compresslevel=6, mtime=0)
    target = Path(tempfile.gettempdir()) / f"nemo-gym-simple-agent-{hashlib.sha256(content).hexdigest()[:16]}.tar.gz"
    if not target.is_file():
        partial = target.with_name(f".{target.name}.{uuid4().hex}")
        partial.write_bytes(content)
        partial.replace(target)
    return target


def model_server_endpoint(url: str) -> tuple[str, int]:
    """The host and port of a model server URL that a sandbox reaches, such as ``sandbox_server_url`` returns."""
    parts = urlsplit(url)
    if parts.hostname is None or parts.port is None:
        raise ValueError(f"model server URL {url!r} names no host and port")
    return parts.hostname, parts.port


@dataclass
class SandboxLoop:
    """One agent session's borrowed sandbox and the supervised runs of the turn loop in it."""

    session: SandboxSession[dict[str, Any]]
    # Each activation gets its own staged input, output and supervisor files.
    activations: int = field(default=0)

    @classmethod
    def borrowed(cls, sandbox: BorrowedSandbox, *, workdir: str) -> "SandboxLoop":
        return cls(
            session=SandboxSession(
                sandbox=sandbox,
                workdir=workdir,
                session_dir=f"/tmp/nemo-gym-simple-agent-sessions/{uuid4().hex}",
                owns_sandbox=False,
                harness="Simple Agent",
            )
        )

    async def install(self, *, timeout: float) -> None:
        """Install the runtime unless the sandbox has it, unpack the harness sources, and check the runner imports."""
        sandbox = self.session.sandbox
        if not await RUNTIME.is_installed(sandbox):
            await RUNTIME.install(sandbox, timeout_s=timeout)
        archive = f"{self.session.session_dir}/harness.tar.gz"
        prepared = await sandbox.exec(
            f"mkdir -p {quote(self.session.session_dir)} {quote(HARNESS_DIR)}", cwd="/", timeout_s=30
        )
        _check(prepared, "prepare the harness directories")
        await sandbox.upload(harness_archive(), archive)
        unpacked = await sandbox.exec(
            f"tar -xzf {quote(archive)} -C {quote(HARNESS_DIR)} && rm -f {quote(archive)}", cwd="/", timeout_s=120
        )
        _check(unpacked, "unpack the harness sources")
        # Fail the seed, not the first activation, when the runtime cannot import the loop.
        checked = await sandbox.exec(f"{quote(RUNTIME.python)} -E -s {quote(RUNNER)} --check", cwd="/", timeout_s=120)
        _check(checked, "import the turn loop")

    async def run(self, payload: dict[str, Any], *, timeout: float, close_timeout: float) -> dict[str, Any]:
        """Run the turn loop once with ``payload`` and return what it wrote; raise what the loop raised."""
        self.activations += 1
        if self.activations > 1:
            # A later activation of a continued session runs a fresh supervisor in a fresh directory.
            await self.session.begin_activation(
                session_dir=f"{self.session.root_dir}/activation-{self.activations}", timeout=close_timeout
            )
        input_path = f"{self.session.session_dir}/input.json"
        output_path = f"{self.session.session_dir}/output.json"

        async def stage() -> SandboxCommand:
            await upload_text(self.session.sandbox, path=input_path, text=json.dumps(payload))
            return SandboxCommand(argv=[RUNTIME.python, "-E", "-s", RUNNER, input_path, output_path])

        async def collect() -> dict[str, Any]:
            try:
                return json.loads(await read_text(self.session.sandbox, path=output_path))
            except Exception as error:
                logs = await self.session.read_output_log()
                raise RuntimeError(f"Simple Agent turn loop exited without output: {logs}") from error

        output = await self.session.execute(
            stage_activation=stage, collect=collect, timeout=timeout, close_timeout=close_timeout
        )
        if output.get("error") is not None:
            raise RuntimeError(
                f"Simple Agent turn loop failed in the sandbox: {output['error']}\n{output.get('traceback', '')}"
            )
        return output

    async def close(self, *, timeout: float) -> None:
        """Stop the loop if it still runs and release the borrowed sandbox; its owner stops it."""
        await self.session.close(timeout=timeout)


def _check(result: Any, action: str) -> None:
    if result.return_code != 0 or result.error_type is not None:
        detail = result.error_type or (result.stderr or result.stdout or "").strip()[-2000:]
        raise RuntimeError(f"Simple Agent could not {action} in the sandbox: {detail}")
