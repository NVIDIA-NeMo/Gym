# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The execution settings and content-addressed packages used by pinned TB4."""

import asyncio
import hashlib
import os
import re
import tarfile
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal
from urllib.parse import quote

import aiohttp
import pathspec
from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym.server_utils import request


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Healthcheck(Settings):
    command: str
    interval_sec: float = Field(default=5, ge=0)
    timeout_sec: float = Field(default=30, gt=0)
    start_period_sec: float = Field(default=0, ge=0)
    start_interval_sec: float = Field(default=5, ge=0)
    retries: int = Field(default=3, gt=0)


class MCPServer(Settings):
    name: str
    transport: Literal["stdio", "sse", "streamable-http"] = "sse"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data):
        if isinstance(data, dict) and data.get("transport") == "http":
            data = {**data, "transport": "streamable-http"}
        return data

    @model_validator(mode="after")
    def validate_transport(self):
        if not (self.command if self.transport == "stdio" else self.url):
            raise ValueError("MCP transport requires command or URL")
        return self


class PhaseSettings(Settings):
    network_mode: Literal["public", "no-network"] | None = None
    allowed_hosts: list[str] | None = None

    @model_validator(mode="after")
    def no_allowlist(self):
        if self.allowed_hosts is not None:
            raise ValueError("TB4 does not support network allowlists")
        return self


class EnvironmentSettings(PhaseSettings):
    docker_image: str
    build_timeout_sec: float = Field(default=600, gt=0)
    os: Literal["linux"] = "linux"
    cpus: int | None = Field(default=None, gt=0)
    memory_mb: int | None = Field(default=None, gt=0)
    storage_mb: int | None = Field(default=None, gt=0)
    gpus: int | None = Field(default=None, ge=0)
    gpu_types: list[str] | None = Field(default=None, max_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    mcp_servers: list[MCPServer] = Field(default_factory=list)
    skills_dir: str | None = None
    healthcheck: Healthcheck | None = None
    workdir: str | None = None
    network_mode: Literal["public", "no-network"] = "public"

    @model_validator(mode="before")
    @classmethod
    def legacy_fields(cls, data):
        data = dict(data)
        allow = data.pop("allow_internet", None)
        if allow is not None:
            data.setdefault("network_mode", "public" if allow else "no-network")
        for old, new in (("memory", "memory_mb"), ("storage", "storage_mb")):
            if old in data:
                raw = data.pop(old).strip().upper()
                size = int(float(raw[:-1]) * {"G": 1024, "M": 1, "K": 1 / 1024}[raw[-1]])
                if new in data and data[new] != size:
                    raise ValueError(f"Conflicting {old} and {new}")
                data[new] = size
        return data


class AgentSettings(PhaseSettings):
    timeout_sec: float = Field(default=28800, gt=0)
    user: str | int | None = None


class CollectHook(Settings):
    command: str
    service: str = "main"
    timeout_sec: float = Field(default=60, gt=0)
    user: str | int | None = None


class VerifierSettings(PhaseSettings):
    timeout_sec: float = Field(default=600, gt=0)
    user: str | int | None = None
    env: dict[str, str] = Field(default_factory=dict)
    environment_mode: Literal["separate", "shared"] | None = None
    environment: EnvironmentSettings | None = None
    collect: list[CollectHook] = Field(default_factory=list)


class Artifact(Settings):
    source: str
    destination: str | None = None
    service: str | None = None
    exclude: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_paths(self):
        for path in (self.source, self.destination):
            if path and (".." in PurePosixPath(path).parts or "\\" in path):
                raise ValueError("Artifact paths must be contained POSIX paths")
        if self.destination and (
            PurePosixPath(self.destination).is_absolute()
            or PurePosixPath(self.destination) in (PurePosixPath("."), PurePosixPath("manifest.json"))
        ):
            raise ValueError("Artifact destination must be relative and cannot shadow the manifest")
        if self.service not in (None, "main") and not self.source.startswith("/"):
            raise ValueError("Sidecar artifacts require absolute source paths")
        return self

    @property
    def host_path(self):
        return Path(self.destination or self.source.lstrip("/"))


class TaskSettings(Settings):
    schema_version: str = "1.4"
    task: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None
    solution: dict[str, Any] = Field(default_factory=dict)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    environment: EnvironmentSettings
    verifier: VerifierSettings = Field(default_factory=VerifierSettings)
    artifacts: list[Artifact] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def normalize(cls, data):
        data = dict(data)
        if data.pop("steps", None) or data.pop("multi_step_reward_strategy", None):
            raise ValueError("TB4 requires single-step task packages")
        if "version" in data:
            data.setdefault("schema_version", data.pop("version"))
        data["artifacts"] = [{"source": a} if isinstance(a, str) else a for a in data.get("artifacts", [])]
        return data

    @model_validator(mode="after")
    def validate_execution(self):
        if self.verifier.environment_mode == "shared" and self.verifier.environment is not None:
            raise ValueError("Shared verification cannot define a separate environment")
        # All 66 pinned packages use separate verification. Reject unobserved
        # shared mode rather than silently changing collection or test upload.
        if self.verifier.environment_mode != "separate" and self.verifier.environment is None:
            raise ValueError("The pinned TB4 profile requires separate verification")
        for phase, env in ((self.agent, self.environment), (self.verifier, self.verifier_environment)):
            if phase.network_mode is not None and phase.network_mode != env.network_mode:
                raise ValueError("Dynamic network policy transitions are unsupported")
        return self

    @property
    def verifier_environment(self):
        return self.verifier.environment or self.environment.model_copy(deep=True)

    @property
    def collected_artifacts(self):
        entries = list(self.artifacts)
        if not any(a.source.rstrip("/") == "/logs/artifacts" and a.service in (None, "main") for a in entries):
            entries.insert(0, Artifact(source="/logs/artifacts"))
        return entries


@dataclass
class Task:
    path: Path
    name: str
    ref: str
    config: TaskSettings
    instruction: str

    @classmethod
    def read(cls, path, name, ref):
        path = Path(path).resolve()
        for entry in path.rglob("*"):
            if entry.is_symlink() and not entry.resolve().is_relative_to(path):
                raise ValueError(f"Package link escapes task: {entry.relative_to(path)}")
        config = TaskSettings.model_validate(tomllib.loads((path / "task.toml").read_text()))
        if config.task.get("name", name) != name or not (path / "environment").is_dir():
            raise ValueError("Package identity or environment layout is invalid")
        lines = (path / "instruction.md").read_text().split("\n")
        while lines and re.match(r"^(<!--.*canary.*-->|#.*canary.*)$", lines[0].strip(), re.IGNORECASE):
            lines.pop(0)
        while lines and not lines[0].strip():
            lines.pop(0)
        return cls(path, name, ref, config, "\n".join(lines))


def content_hash(path: Path) -> str:
    """SHA256 of sorted `relative_path + NUL + file_sha256_hex + LF` records.

    Matches the package publisher's file set, not dirhash or archive bytes.
    A package .gitignore replaces the publisher's default ignore patterns.
    """
    singles = [path / name for name in ("task.toml", "instruction.md", "README.md", "trajectory.json")]
    files = [p for p in singles if p.is_file()]
    for name in ("environment", "tests", "solution", "steps"):
        files.extend(p for p in (path / name).rglob("*") if p.is_file())
    ignores = ["__pycache__/", "*.pyc", ".DS_Store", "*.swp", "*.swo", "*~"]
    if (path / ".gitignore").exists():
        ignores = (path / ".gitignore").read_text().splitlines()
    spec = pathspec.PathSpec.from_lines("gitignore", ignores)
    digest = hashlib.sha256()
    for file in sorted(files, key=lambda p: p.relative_to(path).as_posix()):
        relative = file.relative_to(path).as_posix()
        if spec.match_file(relative):
            continue
        if not file.resolve().is_relative_to(path.resolve()):
            raise ValueError(f"Package file escapes task: {relative}")
        with file.open("rb") as stream:
            file_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        digest.update(f"{relative}\0{file_hash}\n".encode())
    return digest.hexdigest()


def resolve_env(values):
    def substitute(match):
        name, default = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"Missing task environment variable: {name}")

    pattern = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
    return {k: substitute(match) if (match := pattern.fullmatch(v)) else v for k, v in values.items()}


class PackageLoader:
    def __init__(self, download_dir=None):
        self.root = Path(download_dir or Path.home() / ".cache/harbor/tasks/packages")
        self._locks = {}

    async def load(self, name, ref):
        if not re.fullmatch(r"terminal-bench/[a-z0-9][a-z0-9-]*", name) or not re.fullmatch(
            r"sha256:[a-f0-9]{64}", ref
        ):
            raise ValueError("A trusted task name and SHA256 pin are required")
        target = self.root / name / ref[7:]
        async with self._locks.setdefault((name, ref), asyncio.Lock()):
            if not target.exists():
                await self._download(name, ref, target)
            actual = await asyncio.to_thread(content_hash, target)
            if actual != ref[7:]:
                raise ValueError(f"Package content hash mismatch for {name}: expected {ref}, got sha256:{actual}")
            return await asyncio.to_thread(Task.read, target, name, ref)

    async def _download(self, name, ref, target):
        # Public registry credentials identify the public project, not a user.
        base = "https://ofhuhcpkvzjlejydnvyd.supabase.co"
        headers = {"apikey": "sb_publishable_Z-vuQbpvpG-PStjbh4yE0Q_e-d3MTIH"}
        org, short_name = name.split("/")
        async with await request(
            "POST",
            base + "/rest/v1/rpc/resolve_task_version",
            headers=headers,
            json={"p_org": org, "p_name": short_name, "p_ref": ref},
            timeout=aiohttp.ClientTimeout(total=120),
        ) as response:
            response.raise_for_status()
            resolved = await response.json()
        if not resolved or resolved["content_hash"].removeprefix("sha256:") != ref[7:]:
            raise ValueError("Registry returned a different package identity")
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".tb4-download-", dir=target.parent) as tmp:
            archive = Path(tmp) / "task.tar.gz"
            url = base + "/storage/v1/object/packages/" + quote(resolved["archive_path"], safe="/")
            async with await request(
                "GET", url, headers=headers, timeout=aiohttp.ClientTimeout(total=120)
            ) as response:
                response.raise_for_status()
                with archive.open("wb") as stream:
                    async for chunk in response.content.iter_chunked(1024 * 1024):
                        stream.write(chunk)
            staged = Path(tmp) / "package"
            staged.mkdir()
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(staged, filter="data")
            Task.read(staged, name, ref)
            if await asyncio.to_thread(content_hash, staged) != ref[7:]:
                raise ValueError("Downloaded package content hash mismatch")
            # A second process may have promoted the same immutable pin.
            try:
                staged.rename(target)
            except OSError:
                if not target.is_dir():
                    raise
                # load() validates the winning process's package before use.
