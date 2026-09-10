# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read a Terminal-Bench 4.0 task directory into the pieces the separate verifier needs.

Field names follow Harbor's ``task.toml`` schema (``harbor.models.task.config``): the top-level
``artifacts`` list (string ``"/path"`` or ``"/path@service"`` entries, or tables with ``source``,
``destination``, ``exclude`` and ``service``), ``[verifier]`` (``timeout_sec``, ``env``, ``user``,
``environment_mode``, ``environment``, ``[[verifier.collect]]`` hooks), ``[agent]`` (``user``) and
``[environment]`` (``cpus``, ``memory_mb``, ``storage_mb``, ``gpus``). Nothing here talks to a
sandbox; the server decides what to do with sidecars and compose tasks.
"""

from __future__ import annotations

import tomllib
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Optional, Sequence, Tuple, Union


MAIN_SERVICE = "main"
# Harbor's conventional agent publish directory; always collected in addition to the declared artifacts.
CONVENTION_ARTIFACTS_DIR = "/logs/artifacts"
COMPOSE_FILENAMES = ("docker-compose.yaml", "docker-compose.yml")


@dataclass(frozen=True)
class ArtifactEntry:
    source: str
    service: str = MAIN_SERVICE
    destination: Optional[str] = None
    exclude: Tuple[str, ...] = ()

    @property
    def is_main(self) -> bool:
        return self.service == MAIN_SERVICE

    @property
    def normalized_source(self) -> str:
        """Absolute container path without a trailing slash (``/app/`` -> ``/app``)."""
        stripped = self.source.rstrip("/")
        return stripped or "/"

    @property
    def relative_to_root(self) -> str:
        """Path relative to ``/`` for ``tar -C /`` (``/app/out.step`` -> ``app/out.step``)."""
        parts = [p for p in PurePosixPath(self.normalized_source).parts if p not in ("", "/")]
        return "/".join(parts)


@dataclass(frozen=True)
class CollectHook:
    command: str
    service: str = MAIN_SERVICE
    timeout_sec: float = 300.0
    user: Optional[Union[str, int]] = None

    @property
    def is_main(self) -> bool:
        return self.service == MAIN_SERVICE


@dataclass(frozen=True)
class TB4Task:
    task_name: str
    task_dir: Path
    artifacts: Tuple[ArtifactEntry, ...]
    verifier_timeout_sec: float
    verifier_env: Dict[str, str]
    verifier_user: Optional[Union[str, int]]
    verifier_environment_mode: str
    verifier_environment: Optional[Dict[str, Any]]
    environment: Dict[str, Any]
    collect_hooks: Tuple[CollectHook, ...]
    agent_user: Optional[Union[str, int]]
    agent_timeout_sec: Optional[float]
    compose_services: Tuple[str, ...]
    solution_env: Dict[str, str] = field(default_factory=dict)

    @property
    def is_compose(self) -> bool:
        return bool(self.compose_services)

    @property
    def main_artifacts(self) -> Tuple[ArtifactEntry, ...]:
        return tuple(a for a in self.artifacts if a.is_main)

    @property
    def sidecar_artifacts(self) -> Tuple[ArtifactEntry, ...]:
        return tuple(a for a in self.artifacts if not a.is_main)

    @property
    def main_collect_hooks(self) -> Tuple[CollectHook, ...]:
        return tuple(h for h in self.collect_hooks if h.is_main)

    @property
    def sidecar_collect_hooks(self) -> Tuple[CollectHook, ...]:
        return tuple(h for h in self.collect_hooks if not h.is_main)

    def effective_verifier_environment(self) -> Dict[str, Any]:
        """Harbor's resolution: ``[verifier.environment]`` when present, else a copy of ``[environment]``."""
        if self.verifier_environment is not None:
            return deepcopy(self.verifier_environment)
        return deepcopy(self.environment)

    @property
    def requires_gpu(self) -> bool:
        return (
            int(self.environment.get("gpus") or 0) > 0
            or int(self.effective_verifier_environment().get("gpus") or 0) > 0
        )

    @property
    def tests_dir(self) -> Path:
        return self.task_dir / "tests"

    @property
    def solution_dir(self) -> Path:
        return self.task_dir / "solution"


def parse_artifact(entry: Any) -> ArtifactEntry:
    """Accept Harbor's string form (``/path`` or ``/path@service``) or table form."""
    if isinstance(entry, str):
        source, _, service = entry.partition("@")
        if not source:
            raise ValueError(f"Artifact entry has an empty source: {entry!r}")
        return _checked(ArtifactEntry(source=source, service=service or MAIN_SERVICE))
    if isinstance(entry, dict):
        source = entry.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError(f"Artifact table needs a string `source`: {entry!r}")
        exclude = entry.get("exclude") or ()
        return _checked(
            ArtifactEntry(
                source=source,
                service=entry.get("service") or MAIN_SERVICE,
                destination=entry.get("destination"),
                exclude=tuple(str(x) for x in exclude),
            )
        )
    raise ValueError(f"Unsupported artifact entry: {entry!r}")


def _checked(entry: ArtifactEntry) -> ArtifactEntry:
    if ".." in PurePosixPath(entry.source).parts:
        raise ValueError(f"Artifact source {entry.source!r} must not contain '..'")
    if entry.is_main and not entry.relative_to_root:
        raise ValueError(
            f"Artifact source {entry.source!r} is the filesystem root; refusing (it would empty the verifier)"
        )
    return entry


def with_convention_entry(entries: Sequence[ArtifactEntry]) -> Tuple[ArtifactEntry, ...]:
    """Prepend the implicit ``/logs/artifacts`` main entry unless the task declared it."""
    for entry in entries:
        if entry.is_main and entry.normalized_source == CONVENTION_ARTIFACTS_DIR:
            return tuple(entries)
    return (ArtifactEntry(source=CONVENTION_ARTIFACTS_DIR), *entries)


def _parse_hook(entry: Dict[str, Any]) -> CollectHook:
    command = entry.get("command")
    if not isinstance(command, str) or not command:
        raise ValueError(f"[[verifier.collect]] entry needs a string `command`: {entry!r}")
    return CollectHook(
        command=command,
        service=entry.get("service") or MAIN_SERVICE,
        timeout_sec=float(entry.get("timeout_sec", 300.0)),
        user=entry.get("user"),
    )


def compose_service_names(task_dir: Path) -> Tuple[str, ...]:
    """Service names of ``environment/docker-compose.y(a)ml`` when the task ships one.

    Parsed with PyYAML when available; when it is not, the presence of the file alone marks
    the task as a compose task (the caller only needs the boolean to refuse the task).
    """
    for name in COMPOSE_FILENAMES:
        compose_path = task_dir / "environment" / name
        if not compose_path.is_file():
            continue
        try:
            import yaml  # type: ignore[import-not-found]
        except ModuleNotFoundError:
            return ("<unparsed compose file>",)
        with compose_path.open("rb") as handle:
            document = yaml.safe_load(handle) or {}
        services = document.get("services") if isinstance(document, dict) else None
        if isinstance(services, dict) and services:
            return tuple(str(k) for k in services)
        return ("<compose file without services>",)
    return ()


def load_task(task_folder: Union[str, Path], repo_root: Optional[Path] = None) -> TB4Task:
    """Load ``<task_folder>/task.toml``; relative folders resolve under ``repo_root`` (default: cwd)."""
    task_dir = Path(task_folder).expanduser()
    if not task_dir.is_absolute():
        task_dir = (repo_root or Path.cwd()) / task_dir
    task_dir = task_dir.resolve()
    toml_path = task_dir / "task.toml"
    if not toml_path.is_file():
        raise FileNotFoundError(f"task.toml not found under {task_dir}")
    with toml_path.open("rb") as handle:
        document = tomllib.load(handle)

    task_table = document.get("task") or {}
    task_name = task_table.get("name") or task_dir.name
    verifier = document.get("verifier") or {}
    agent = document.get("agent") or {}
    environment = deepcopy(document.get("environment") or {})

    artifacts = with_convention_entry([parse_artifact(a) for a in document.get("artifacts") or []])
    hooks = tuple(_parse_hook(h) for h in verifier.get("collect") or [])

    return TB4Task(
        task_name=str(task_name),
        task_dir=task_dir,
        artifacts=artifacts,
        verifier_timeout_sec=float(verifier.get("timeout_sec", 600.0)),
        verifier_env={str(k): str(v) for k, v in (verifier.get("env") or {}).items()},
        verifier_user=verifier.get("user"),
        verifier_environment_mode=str(verifier.get("environment_mode") or "shared"),
        verifier_environment=deepcopy(verifier.get("environment"))
        if verifier.get("environment") is not None
        else None,
        environment=environment,
        collect_hooks=hooks,
        agent_user=agent.get("user"),
        agent_timeout_sec=float(agent["timeout_sec"]) if agent.get("timeout_sec") is not None else None,
        compose_services=compose_service_names(task_dir),
        solution_env={str(k): str(v) for k, v in ((document.get("solution") or {}).get("env") or {}).items()},
    )
