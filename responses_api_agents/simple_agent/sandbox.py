# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Agent-server side of a Simple Agent whose agent loop runs in the sandbox its environment lends.

At seed the agent borrows the sandbox, installs a private Python runtime in it and unpacks this server's own
``nemo_gym`` sources and Simple Agent module there. Each activation stages the loop's arguments, runs
``sandbox_runner.py`` under Gym's process supervisor, and reads back what the loop returned. The runner builds the same
``SimpleAgent`` class and calls the same ``_create_episode`` loop the agent server runs in its own process, so a
sandboxed loop cannot drift from the in-process one.
"""

import asyncio
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

from aiohttp import ClientResponseError, RequestInfo
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

import nemo_gym
from nemo_gym.agents import AgentSeedRejectedError, SandboxCommand, SandboxPythonRuntime, SandboxSession
from nemo_gym.core import FailureHeader, RequestFailureLog, rebuild_failure
from nemo_gym.sandbox import read_text, upload_text
from nemo_gym.server_runtime import BorrowedSandbox
from responses_api_agents.simple_agent.python_tool import PythonToolConfig, PythonToolSpec


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
# The Simple Agent files the runner imports; the rest of nemo_gym comes from this server's own package.
_AGENT_FILES = (
    "__init__.py",
    "app.py",
    "python_tool.py",
    "python_tool_worker.py",
    "sandbox.py",
    "sandbox_runner.py",
)


def harness_dir() -> str:
    """Where the harness sources live in the sandbox: beside the runtime, outside the task's working directory.

    A function, so importing this module does not key the runtime on this server's installed packages.
    """
    return f"{RUNTIME.directory}/harness"


def runner_path() -> str:
    return f"{harness_dir()}/responses_api_agents/simple_agent/sandbox_runner.py"


def python_tool_worker_path() -> str:
    return f"{harness_dir()}/responses_api_agents/simple_agent/python_tool_worker.py"


def python_tool_runtime(config: PythonToolConfig) -> SandboxPythonRuntime | None:
    """A Python tool's own runtime in the sandbox, or None for a tool without requirements, which runs on the loop's.

    The requirements carry their own pins; the tool does not take this server's versions, which may differ.
    """
    if not config.requirements:
        return None
    return SandboxPythonRuntime(
        harness="simple_agent_python_tool",
        requirements=tuple(config.requirements),
        import_check=_import_check(config),
        constraints=(),
    )


def _import_check(config: PythonToolConfig) -> str:
    return f"import {', '.join(sorted(set(config.preload.values())))}" if config.preload else "import sys"


@functools.cache
def _harness_bytes() -> bytes:
    """This server's ``nemo_gym`` Python sources and the Simple Agent files, as a gzipped tar, built once a process.

    Fixed member times and order, and a gzip header without a time, keep the bytes the same for the same sources.
    """
    package = Path(nemo_gym.__file__).parent
    agent = Path(__file__).parent
    members = [(path, f"nemo_gym/{path.relative_to(package)}") for path in sorted(package.rglob("*.py"))]
    members += [(agent / name, f"responses_api_agents/simple_agent/{name}") for name in _AGENT_FILES]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path, name in members:
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return gzip.compress(buffer.getvalue(), compresslevel=6, mtime=0)


@functools.cache
def _private_dir() -> Path:
    """A directory only this process's user can write, so no one else can stage what the sandbox runs."""
    return Path(tempfile.mkdtemp(prefix="nemo-gym-simple-agent-"))


def harness_archive() -> Path:
    """The harness archive as a file to upload, written again if it is gone or was changed since.

    Blocking file IO: call it off the event loop.
    """
    content = _harness_bytes()
    directory = _private_dir()
    target = directory / f"harness-{hashlib.sha256(content).hexdigest()[:16]}.tar.gz"
    if not target.is_file() or target.read_bytes() != content:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
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


def loop_error(output: dict[str, Any]) -> Exception:
    """The error the sandboxed loop raised, as the agent server raises it.

    A model server's HTTP error comes back as one, with its status, headers and body, so the agent boundary reports it
    as it reports one the in-process loop raises. Any other error is a ``RuntimeError`` naming the loop's error.
    """
    http = output.get("http_error")
    if http is not None:
        url = URL(http["url"])
        error = ClientResponseError(
            RequestInfo(url, http["method"], CIMultiDictProxy(CIMultiDict()), url),
            (),
            status=http["status"],
            message=http["message"],
            headers=CIMultiDictProxy(CIMultiDict(http["headers"])),
        )
        if http.get("content") is not None:
            error.response_content = http["content"].encode()
        report = FailureHeader.decode(error.headers)
        if report is not None:
            error.__cause__ = rebuild_failure(report)
            if (log := RequestFailureLog.current()) is not None:
                log.add(report)
        return error
    return RuntimeError(
        f"Simple Agent agent loop failed in the sandbox: {output['error']}\n{output.get('traceback', '')}"
    )


@dataclass
class SandboxLoop:
    """One agent session's borrowed sandbox and the supervised runs of the agent loop in it."""

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
            f"mkdir -p {quote(self.session.session_dir)} {quote(harness_dir())}", cwd="/", timeout_s=30
        )
        _check(prepared, "prepare the harness directories")
        await sandbox.upload(await asyncio.to_thread(harness_archive), archive)
        unpacked = await sandbox.exec(
            f"tar -xzf {quote(archive)} -C {quote(harness_dir())} && rm -f {quote(archive)}", cwd="/", timeout_s=120
        )
        _check(unpacked, "unpack the harness sources")
        # Fail the seed, not the first activation, when the runtime cannot import the loop.
        checked = await sandbox.exec(
            f"{quote(RUNTIME.python)} -E -s {quote(runner_path())} --check", cwd="/", timeout_s=120
        )
        _check(checked, "import the agent loop")

    async def install_python_tools(
        self, tools: dict[str, PythonToolConfig], *, timeout: float
    ) -> dict[str, PythonToolSpec]:
        """Install each Python tool's runtime unless the sandbox has it, check that it imports the tool's preloaded
        modules, and return where each tool's files are. Call it after ``install``, which unpacks the worker."""
        specs: dict[str, PythonToolSpec] = {}
        sandbox = self.session.sandbox
        for name, config in tools.items():
            runtime = python_tool_runtime(config)
            if runtime is not None and not await runtime.is_installed(sandbox):
                await runtime.install(sandbox, timeout_s=timeout)
            python = runtime.python if runtime is not None else RUNTIME.python
            checked = await sandbox.exec(
                f"{quote(python)} -I -c {quote(_import_check(config))}", cwd="/", timeout_s=min(timeout, 300)
            )
            if checked.error_type is None and checked.return_code != 0:
                # The runtime runs but lacks a module: the agent's config is wrong, and no retry changes that.
                detail = (checked.stderr or checked.stdout or "").strip()[-2000:]
                raise AgentSeedRejectedError(
                    f"Python tool {name!r} cannot import its preloaded modules in the sandbox: {detail}"
                )
            _check(checked, f"import the preloaded modules of Python tool {name!r}")
            specs[name] = PythonToolSpec(
                config=config, python=python, worker=python_tool_worker_path(), workdir=self.session.workdir
            )
        return specs

    async def run(self, payload: dict[str, Any], *, timeout: float, close_timeout: float) -> dict[str, Any]:
        """Run the agent loop once with ``payload`` and return what it wrote; raise what the loop raised."""
        self.activations += 1
        if self.activations > 1:
            # A later activation of a continued session runs a fresh supervisor in a fresh directory.
            await self.session.begin_activation(
                session_dir=f"{self.session.root_dir}/activation-{self.activations}", timeout=close_timeout
            )
        session_dir = self.session.session_dir
        input_path = f"{session_dir}/input.json"
        output_path = f"{session_dir}/output.json"

        async def stage() -> SandboxCommand:
            # A provider's upload need not create the directory a later activation's files go to.
            prepared = await self.session.sandbox.exec(f"mkdir -p {quote(session_dir)}", cwd="/", timeout_s=30)
            _check(prepared, "prepare the activation directory")
            await upload_text(self.session.sandbox, path=input_path, text=json.dumps(payload))
            return SandboxCommand(argv=[RUNTIME.python, "-E", "-s", runner_path(), input_path, output_path])

        async def collect() -> dict[str, Any]:
            try:
                return json.loads(await read_text(self.session.sandbox, path=output_path))
            except Exception as error:
                logs = await self.session.read_output_log()
                raise RuntimeError(f"Simple Agent agent loop exited without output: {logs}") from error

        output = await self.session.execute(
            stage_activation=stage, collect=collect, timeout=timeout, close_timeout=close_timeout
        )
        if output.get("error") is not None:
            raise loop_error(output)
        return output

    async def close(self, *, timeout: float) -> None:
        """Stop the loop if it still runs and release the borrowed sandbox; its owner stops it."""
        await self.session.close(timeout=timeout)


def _check(result: Any, action: str) -> None:
    if result.return_code != 0 or result.error_type is not None:
        detail = result.error_type or (result.stderr or result.stdout or "").strip()[-2000:]
        raise RuntimeError(f"Simple Agent could not {action} in the sandbox: {detail}")
