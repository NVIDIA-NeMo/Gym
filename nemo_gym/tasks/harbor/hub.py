# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve ``harbor:<dataset>[@<version>]`` through the Harbor registry.

The registry is one JSON file listing datasets; each task entry names a git
repository, a commit and a path. Fetching copies the task folders into Gym's shared
datasets folder so that every server reads the same bytes and nothing downloads at
run time. A ``HEAD`` pin is resolved to a commit before anything is copied.
"""

import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path


REGISTRY_URL = "https://raw.githubusercontent.com/harbor-framework/harbor/main/registry.json"
REF_PREFIX = "harbor:"
DATASETS_DIR_ENV = "NEMO_GYM_DATASETS_DIR"
MANIFEST_FILE = "manifest.toml"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


class HubError(ValueError):
    """The reference cannot be resolved or fetched."""


def is_hub_ref(target: str) -> bool:
    return target.startswith(REF_PREFIX)


def datasets_dir() -> Path:
    """Gym's shared datasets folder: ``$NEMO_GYM_DATASETS_DIR`` or ``./datasets``."""
    return Path(os.environ.get(DATASETS_DIR_ENV) or Path.cwd() / "datasets").expanduser().resolve()


@dataclass(frozen=True)
class HubRef:
    name: str
    version: str | None

    @classmethod
    def parse(cls, target: str) -> "HubRef":
        if not is_hub_ref(target):
            raise HubError(f"Not a Harbor hub reference: {target!r} (expected `harbor:<dataset>[@<version>]`)")
        body = target[len(REF_PREFIX) :]
        name, _, version = body.partition("@")
        if not _NAME.fullmatch(name) or (version and not _NAME.fullmatch(version)):
            raise HubError(f"Malformed Harbor hub reference: {target!r}")
        return cls(name=name, version=version or None)


@dataclass(frozen=True)
class RegistryTask:
    name: str
    git_url: str
    git_commit_id: str
    path: str


@dataclass(frozen=True)
class RegistryDataset:
    name: str
    version: str
    description: str
    tasks: tuple[RegistryTask, ...]

    @property
    def folder_name(self) -> str:
        return self.name


def _version_key(version: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"[.-]", version))


def load_registry(cache_dir: Path, *, refresh: bool = False) -> list[RegistryDataset]:
    """Read the registry, downloading it into ``cache_dir`` when missing or ``refresh`` is set."""
    cache = Path(cache_dir) / "registry.json"
    if refresh or not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(REGISTRY_URL, timeout=120) as response:
                payload = response.read()
        except OSError as exc:
            raise HubError(f"Could not download the Harbor registry from {REGISTRY_URL}: {exc}") from exc
        cache.write_bytes(payload)
    datasets: list[RegistryDataset] = []
    for entry in json.loads(cache.read_text()):
        tasks = tuple(
            RegistryTask(
                name=task["name"], git_url=task["git_url"], git_commit_id=task["git_commit_id"], path=task["path"]
            )
            for task in entry.get("tasks", [])
        )
        datasets.append(
            RegistryDataset(
                name=entry["name"],
                version=str(entry.get("version", "")),
                description=entry.get("description", ""),
                tasks=tasks,
            )
        )
    return datasets


def resolve_dataset(ref: HubRef, registry: list[RegistryDataset]) -> RegistryDataset:
    matches = [dataset for dataset in registry if dataset.name == ref.name]
    if not matches:
        raise HubError(f"Dataset {ref.name!r} is not in the Harbor registry")
    if ref.version is not None:
        for dataset in matches:
            if dataset.version == ref.version:
                return dataset
        available = ", ".join(sorted(dataset.version for dataset in matches))
        raise HubError(f"Dataset {ref.name!r} has no version {ref.version!r}; available: {available}")
    return max(matches, key=lambda dataset: _version_key(dataset.version))


def _git(*args: str, cwd: Path | None = None) -> str:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            errors="replace",
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except FileNotFoundError as exc:
        raise HubError("git is required to fetch Harbor hub datasets") from exc
    except subprocess.CalledProcessError as exc:
        raise HubError(f"git {' '.join(args)} failed: {exc.stderr.strip() or exc.stdout.strip()}") from exc
    return completed.stdout


def resolve_commit(git_url: str, git_commit_id: str) -> str:
    """Pin ``HEAD`` (or a branch name) to the commit it points at right now."""
    if re.fullmatch(r"[0-9a-f]{40}", git_commit_id):
        return git_commit_id
    ref = "HEAD" if git_commit_id == "HEAD" else git_commit_id
    output = _git("ls-remote", git_url, ref)
    for line in output.splitlines():
        sha, _, name = line.partition("\t")
        if name.strip() == ref or (ref != "HEAD" and name.strip() == f"refs/heads/{ref}"):
            return sha.strip()
    raise HubError(f"{git_url} has no ref {git_commit_id!r}")


def _write_manifest(folder: Path, dataset: RegistryDataset, pins: dict[str, tuple[str, str, str]]) -> None:
    def quote(value: str) -> str:
        return json.dumps(value)

    lines = [
        "[dataset]",
        f"name = {quote(dataset.name)}",
        f"version = {quote(dataset.version)}",
        'source = "harbor-registry"',
        "",
    ]
    for task_name in sorted(pins):
        git_url, commit, path = pins[task_name]
        lines += [
            f"[tasks.{quote(task_name)}]",
            f"git_url = {quote(git_url)}",
            f"git_commit_id = {quote(commit)}",
            f"path = {quote(path)}",
            "",
        ]
    (folder / MANIFEST_FILE).write_text("\n".join(lines))


def fetch_dataset(dataset: RegistryDataset, root: Path) -> Path:
    """Copy every task of ``dataset`` into ``root/<dataset name>/<task name>/``.

    Task folders that already exist are kept; only missing ones are fetched, so a
    rerun after a partial fetch completes it. Returns the dataset folder.
    """
    folder = Path(root) / dataset.folder_name
    folder.mkdir(parents=True, exist_ok=True)
    pins: dict[str, tuple[str, str, str]] = {}
    by_repo: dict[tuple[str, str], list[RegistryTask]] = {}
    for task in dataset.tasks:
        commit = resolve_commit(task.git_url, task.git_commit_id)
        pins[task.name] = (task.git_url, commit, task.path)
        if not (folder / task.name / "task.toml").is_file():
            by_repo.setdefault((task.git_url, commit), []).append(task)

    for (git_url, commit), tasks in by_repo.items():
        with tempfile.TemporaryDirectory(prefix="nemo-gym-harbor-", dir=folder) as tmp:
            clone = Path(tmp) / "repo"
            _git("clone", "--quiet", "--filter=blob:none", "--no-checkout", git_url, str(clone))
            _git("sparse-checkout", "set", "--no-cone", *(task.path for task in tasks), cwd=clone)
            _git("checkout", "--quiet", commit, cwd=clone)
            for task in tasks:
                source = clone / task.path
                if not (source / "task.toml").is_file():
                    raise HubError(f"{git_url}@{commit[:12]}:{task.path} has no task.toml")
                staged = Path(tmp) / task.name
                shutil.copytree(source, staged, symlinks=False)
                shutil.move(str(staged), str(folder / task.name))
    _write_manifest(folder, dataset, pins)
    return folder


def fetch_ref(target: str, root: Path | None = None, *, refresh_registry: bool = False) -> Path:
    """Resolve and fetch a ``harbor:`` reference; return the dataset folder."""
    root = Path(root) if root is not None else datasets_dir()
    ref = HubRef.parse(target)
    registry = load_registry(root / ".harbor", refresh=refresh_registry)
    return fetch_dataset(resolve_dataset(ref, registry), root)
