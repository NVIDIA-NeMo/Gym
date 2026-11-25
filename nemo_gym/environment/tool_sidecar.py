# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Resources Server's side of a tool sidecar: start its tool host in the group, and read its tool state back.

A Resources Server opts in by listing :class:`ToolSidecarResources` first among its bases and using a config that
derives from :class:`ToolSidecarResourcesConfig`. With ``tool_sidecar`` set:

- its seed declares the sidecar in ``sandbox_group``, and the Environment Server starts it with the attempt's group;
- ``/prepare_sandbox`` installs a private Python runtime in the sidecar if the image lacks it, unpacks this server's
  ``nemo_gym`` sources and its own package there, starts the tool host (``tool_sidecar_host.py``), and seeds the
  episode's session in it with the seed's identity and ``task_data``, which the prepare request carries. The tool host
  signs sessions with this server's key, so the Resources session's cookies name the same session there;
- ``/tool_sidecar/collect`` reads the session's tool state from the tool host (:meth:`export_tool_state` there) and
  hands it to :meth:`import_tool_state` here, so ``/verify`` grades the state the agent's tool calls left.

A server whose tools keep no state the grader reads, such as a lookup, overrides neither hook.
"""

import asyncio
import functools
import gzip
import hashlib
import inspect
import io
import json
import logging
import re
import shlex
import tarfile
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, ClassVar
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from pydantic import JsonValue, model_validator

import nemo_gym
from nemo_gym.agents.utils.sandbox_python import SandboxPythonRuntime
from nemo_gym.client.server_client import SESSION_ID_KEY
from nemo_gym.environment.tool_sidecar_host import STATE_PATH
from nemo_gym.sandbox.api import AsyncSandboxSidecar
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_runtime.attempt_sandbox import ResourcesPrepareSandboxRequest, SandboxGroupSpec
from nemo_gym.server_runtime.base_resources_server import BaseResourcesServerConfig, ResourcesSeedSessionRequest
from nemo_gym.server_runtime.sandbox_borrow import borrow
from nemo_gym.server_runtime.tool_sidecar import (
    TOOL_SIDECAR_COLLECT_PATH,
    ToolSidecarCollectRequest,
    ToolSidecarCollectResponse,
    ToolSidecarConfig,
)


LOG = logging.getLogger(__name__)

# The tool host's import closure: a Resources Server built from nemo_gym, at this server's versions. A server whose
# own package needs more lists it in ``tool_sidecar_requirements``.
TOOL_HOST_REQUIREMENTS = (
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
    # OmegaConf pins antlr4-python3-runtime 4.9.*, which PyPI ships only as source; a direct URL is built on the host.
    "antlr4-python3-runtime @ https://files.pythonhosted.org/packages/3e/38/7859ff46355f76f8d19459005ca000b6e7012f2f1ca597746cbcd1fbfe5e/"
    "antlr4-python3-runtime-4.9.3.tar.gz#sha256=f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b",
)
_IMPORT_CHECK = "import aiohttp, fastapi, hydra, omegaconf, pydantic, uvicorn"
# Where the tool host keeps its sources, spec, log and call files in the sidecar, outside any working directory.
_ROOT = "/tmp/nemo-gym-tool-sidecar"
_HOST = f"{_ROOT}/harness/nemo_gym/environment/tool_sidecar_host.py"
# A config key that names a credential. Code in the agent container can read a tool sidecar's files and environment
# (the provider's exec daemon is reachable from it), so no such value may go there.
_SECRET_KEY = re.compile(r"(api_?key|token|secret|password|credential)s?$", re.IGNORECASE)


def secret_paths(value: Any, path: str = "") -> list[str]:
    """The dotted paths of ``value``'s keys that name a credential and hold a string, such as an API key."""
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}" if path else str(key)
            if isinstance(key, str) and _SECRET_KEY.search(key) and isinstance(item, str) and item:
                found.append(child)
            else:
                found.extend(secret_paths(item, child))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(secret_paths(item, f"{path}[{index}]"))
    return found


class ToolSidecarResourcesConfig(BaseResourcesServerConfig):
    """A Resources Server config that may serve its tools from a sidecar (``tool_sidecar``)."""

    tool_sidecar: ToolSidecarConfig | None = None

    @model_validator(mode="after")
    def _no_secret_in_the_sidecar(self) -> "ToolSidecarResourcesConfig":
        if self.tool_sidecar is not None:
            secrets = secret_paths(self.model_dump(mode="json", exclude={"tool_sidecar"}))
            if secrets:
                raise ValueError(
                    f"tool_sidecar would copy {secrets} into a sidecar that code in the agent container can read; "
                    "keep credentials out of a config whose tools run in a sidecar"
                )
            if self.mcp_bearer_token:
                raise ValueError(
                    "tool_sidecar needs sessions signed with the server's public key, which the tool host shares; "
                    "mcp_bearer_token signs them with a secret made for the run"
                )
        return self


class _SidecarShell:
    """A sidecar as :class:`SandboxPythonRuntime` installs into one: shell commands through ``sh``, files by upload."""

    def __init__(self, sidecar: AsyncSandboxSidecar, provider_name: str) -> None:
        self._sidecar = sidecar
        self.provider_name = provider_name

    async def exec(
        self, command: str, *, cwd: str | None = None, timeout_s: float, env: Mapping[str, str] | None = None
    ) -> SandboxExecResult:
        prefix = "".join(f"{name}={shlex.quote(value)} " for name, value in (env or {}).items())
        script = f"cd {shlex.quote(cwd)} && {prefix}{command}" if cwd else f"{prefix}{command}"
        return await self._sidecar.exec("sh", "-c", script, timeout_s=timeout_s)

    async def upload(self, local_path: Path | str, remote_path: str) -> None:
        await self._sidecar.upload(local_path, remote_path)


def _check(result: SandboxExecResult, action: str) -> None:
    if result.return_code != 0 or result.error_type is not None:
        detail = result.error_type or (result.stderr or result.stdout or "").strip()[-2000:]
        raise RuntimeError(f"Could not {action} in the tool sidecar: {detail}")


@functools.cache
def _private_dir() -> Path:
    """A directory only this process's user can write, so no one else can stage what the sidecar runs."""
    return Path(tempfile.mkdtemp(prefix="nemo-gym-tool-sidecar-"))


@functools.cache
def _sources_bytes(sources: tuple[tuple[str, str], ...]) -> bytes:
    """``nemo_gym``'s Python sources and each ``(directory, name in the archive)`` pair's, as a reproducible tarball.

    Fixed member times and order, and a gzip header without a time, keep the bytes the same for the same sources.
    """
    package = Path(nemo_gym.__file__).parent
    members = [(path, f"nemo_gym/{path.relative_to(package)}") for path in sorted(package.rglob("*.py"))]
    for directory, name in sources:
        root = Path(directory)
        members += [
            (path, f"{name}/{path.relative_to(root)}")
            for path in sorted(root.rglob("*.py"))
            if not {"tests", ".venv", "__pycache__"} & set(path.relative_to(root).parts)
        ]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for path, name in members:
            data = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
    return gzip.compress(buffer.getvalue(), compresslevel=6, mtime=0)


def _write_private(content: bytes, suffix: str) -> Path:
    """``content`` in a file only this process's user can write, named by its hash. Blocking: run it off the loop."""
    directory = _private_dir()
    target = directory / f"{hashlib.sha256(content).hexdigest()[:16]}{suffix}"
    if not target.is_file() or target.read_bytes() != content:
        partial = target.with_name(f".{target.name}.{uuid4().hex}")
        partial.write_bytes(content)
        partial.replace(target)
    return target


class ToolSidecarResources:
    """Serve this Resources Server's tools from a sidecar of the attempt's group, when its config sets
    ``tool_sidecar``. List it before the Resources Server class among the bases.

    Attributes:
        tool_sidecar_requirements: Packages the server's own module needs beyond Gym's, as requirement strings.
        tool_sidecar_sources: Packages the server imports that are not installed from an index, such as a benchmark's
            own package: each directory, relative to the repository root, and the path it is imported from in the
            sidecar. The server's own directory is always included, at its own path.
    """

    tool_sidecar_requirements: ClassVar[tuple[str, ...]] = ()
    tool_sidecar_sources: ClassVar[Mapping[str, str]] = {}

    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)  # type: ignore[misc]
        if self.config.tool_sidecar is not None and self.session_secret != self.get_session_middleware_key():  # type: ignore[attr-defined]
            raise ValueError(
                f"{type(self).__name__} signs sessions with a secret the tool host does not share, so its tools "
                "cannot run in a sidecar"
            )

    def export_tool_state(self, session_id: str) -> JsonValue:
        """In the sidecar: the session's tool state that grading reads, as JSON. None when the tools keep none."""
        return None

    async def import_tool_state(self, session_id: str, state: JsonValue) -> None:
        """On the Resources host: put ``state``, what :meth:`export_tool_state` returned, into the session."""
        return None

    def register_protocol_routes(self, app: FastAPI) -> None:
        super().register_protocol_routes(app)  # type: ignore[misc]
        if self.config.tool_sidecar is not None:  # type: ignore[attr-defined]
            app.post(TOOL_SIDECAR_COLLECT_PATH)(self.collect_tool_sidecar)

    def sandbox_group(self, body: ResourcesSeedSessionRequest) -> SandboxGroupSpec | None:
        """The attempt's group holds the tool sidecar, which serves this server's tools."""
        config = self.config.tool_sidecar  # type: ignore[attr-defined]
        if config is None:
            return super().sandbox_group(body)  # type: ignore[misc]
        return SandboxGroupSpec(sidecars=[config.group_sidecar()])

    def _prepares_sandbox(self) -> bool:
        # Only a server whose tools run in a sidecar prepares the group: with them in process, its seed declares none.
        overridden = type(self).prepare_sandbox is not ToolSidecarResources.prepare_sandbox
        return self.config.tool_sidecar is not None or overridden  # type: ignore[attr-defined]

    async def prepare_sandbox(self, body: ResourcesPrepareSandboxRequest) -> None:
        """Start the tool host in the group's sidecar unless it runs, and seed the episode's session in it."""
        config = self.config.tool_sidecar  # type: ignore[attr-defined]
        if config is None:
            return
        seed = ResourcesSeedSessionRequest(
            resources_session_id=body.resources_session_id,
            episode_id=body.episode_id,
            task_id=body.task_id,
            task_data=body.task_data,
        ).model_dump(mode="json")
        python = self.tool_host_runtime().python
        sandbox = await borrow(body.sandbox_access)
        try:
            sidecar = sandbox.sidecar(config.name)
            await self._start_tool_host(sidecar, sandbox.provider_name, config)
            reply = await _call(sidecar, python, config.port, "POST", "/seed_session", seed, cookies={})
        finally:
            await sandbox.disconnect()
        if reply["status"] != 200:
            raise RuntimeError(f"The tool host refused the seed with HTTP {reply['status']}: {reply['body'][:2000]}")

    async def collect_tool_sidecar(
        self, request: Request, body: ToolSidecarCollectRequest
    ) -> ToolSidecarCollectResponse:
        """Copy the session's tool state from the sidecar into this server's session of the same episode."""
        config = self._tool_sidecar_config()
        python = self.tool_host_runtime().python
        sandbox = await borrow(body.sandbox_access)
        try:
            reply = await _call(
                sandbox.sidecar(config.name), python, config.port, "POST", STATE_PATH, {}, cookies=request.cookies
            )
        finally:
            await sandbox.disconnect()
        if reply["status"] != 200:
            raise RuntimeError(
                f"The tool host did not export its state: HTTP {reply['status']}: {reply['body'][:2000]}"
            )
        state = json.loads(reply["body"])["state"]
        if state is None:
            return ToolSidecarCollectResponse(imported=False)
        await self.import_tool_state(request.session[SESSION_ID_KEY], state)
        return ToolSidecarCollectResponse(imported=True)

    def _tool_sidecar_config(self) -> ToolSidecarConfig:
        config = self.config.tool_sidecar  # type: ignore[attr-defined]
        if config is None:
            raise HTTPException(status_code=404, detail="this Resources Server serves its tools in process")
        return config

    def tool_host_spec(self) -> dict[str, Any]:
        """What the tool host builds: this server's class, from its module, with this config, on the sidecar's port."""
        config = self._tool_sidecar_config()
        server_dir = Path(inspect.getfile(type(self))).resolve().parent
        root = server_dir.parents[1]
        module = PurePosixPath(Path(inspect.getfile(type(self))).resolve().relative_to(root).with_suffix("")).parts
        return {
            "module": ".".join(module),
            "class": type(self).__qualname__,
            "config": self.config.model_dump(mode="json", exclude={"tool_sidecar"}),  # type: ignore[attr-defined]
            "port": config.port,
        }

    def _tool_host_sources(self) -> tuple[tuple[str, str], ...]:
        server_dir = Path(inspect.getfile(type(self))).resolve().parent
        root = server_dir.parents[1]
        own = str(server_dir.relative_to(root))
        return tuple(
            (str(root / source), target) for source, target in {own: own, **self.tool_sidecar_sources}.items()
        )

    def tool_host_runtime(self) -> SandboxPythonRuntime:
        return SandboxPythonRuntime(
            harness="tool-sidecar",
            requirements=(*TOOL_HOST_REQUIREMENTS, *self.tool_sidecar_requirements),
            import_check=_IMPORT_CHECK,
        )

    async def _start_tool_host(
        self, sidecar: AsyncSandboxSidecar, provider_name: str, config: ToolSidecarConfig
    ) -> None:
        """Install the runtime and sources unless the sidecar has them, start the tool host, and wait until it answers.

        A tool host that already answers, from an earlier prepare of this group, is left as it is.
        """
        runtime = self.tool_host_runtime()
        if await _ready(sidecar, runtime.python, config.port):
            return
        shell = _SidecarShell(sidecar, provider_name)
        async with asyncio.timeout(config.install_timeout_seconds):
            if not await runtime.is_installed(shell):  # type: ignore[arg-type]
                await runtime.install(shell, timeout_s=config.install_timeout_seconds)  # type: ignore[arg-type]
            _check(
                await shell.exec(f"mkdir -p {_ROOT}/harness {_ROOT}/calls", timeout_s=30), "prepare its directories"
            )
            archive = await asyncio.to_thread(
                lambda: _write_private(_sources_bytes(self._tool_host_sources()), ".tar.gz")
            )
            await sidecar.upload(archive, f"{_ROOT}/sources.tar.gz")
            _check(
                await shell.exec(f"tar -xzf {_ROOT}/sources.tar.gz -C {_ROOT}/harness", timeout_s=120),
                "unpack the tool host's sources",
            )
            spec = await asyncio.to_thread(lambda: _write_private(json.dumps(self.tool_host_spec()).encode(), ".json"))
            await sidecar.upload(spec, f"{_ROOT}/spec.json")
            command = shlex.join([runtime.python, "-E", "-s", _HOST, f"{_ROOT}/spec.json"])
            # Detached, so the tool host outlives this exec; it stops with the group.
            started = await shell.exec(
                f"setsid nohup {command} >{_ROOT}/host.log 2>&1 </dev/null & echo $! >{_ROOT}/host.pid",
                cwd="/",
                timeout_s=30,
            )
            _check(started, "start the tool host")
        deadline = time.monotonic() + config.ready_timeout_seconds
        while not await _ready(sidecar, runtime.python, config.port):
            alive = await shell.exec(f"kill -0 $(cat {_ROOT}/host.pid)", timeout_s=30)
            if alive.return_code != 0 or time.monotonic() > deadline:
                log = await shell.exec(f"tail -c 4000 {_ROOT}/host.log", timeout_s=30)
                state = (
                    "exited" if alive.return_code != 0 else f"did not answer within {config.ready_timeout_seconds}s"
                )
                raise RuntimeError(f"The tool host {state}: {(log.stdout or '').strip()}")
            await asyncio.sleep(0.5)


async def _ready(sidecar: AsyncSandboxSidecar, python: str, port: int) -> bool:
    """Whether a tool host answers in the sidecar.

    Through ``sh``, so a sidecar without the runtime yet answers with exit 127 instead of a provider error.
    """
    command = shlex.join([python, "-E", "-s", _HOST, "--ready", str(port)])
    result = await sidecar.exec("sh", "-c", f"test -x {shlex.quote(python)} && exec {command}", timeout_s=30)
    return result.return_code == 0 and result.error_type is None


async def _call(
    sidecar: AsyncSandboxSidecar,
    python: str,
    port: int,
    method: str,
    path: str,
    body: Mapping[str, Any],
    *,
    cookies: Mapping[str, str],
) -> dict[str, Any]:
    """Send one HTTP request to the tool host from inside the sidecar; return its status, body and cookies.

    The request and the reply travel as files, so a large seed or state passes no argument or output limit.
    """
    call = f"{_ROOT}/calls/{uuid4().hex}"
    request = await asyncio.to_thread(
        lambda: _write_private(
            json.dumps({"method": method, "path": path, "body": body, "cookies": dict(cookies)}).encode(), ".json"
        )
    )
    await sidecar.upload(request, f"{call}.request.json")
    result = await sidecar.exec(
        python, "-E", "-s", _HOST, "--call", str(port), f"{call}.request.json", f"{call}.reply.json", timeout_s=300
    )
    _check(result, f"call {path} on the tool host")
    with tempfile.TemporaryDirectory(dir=_private_dir()) as scratch:
        local = Path(scratch) / "reply.json"
        await sidecar.download(f"{call}.reply.json", local)
        return json.loads(await asyncio.to_thread(local.read_text))
