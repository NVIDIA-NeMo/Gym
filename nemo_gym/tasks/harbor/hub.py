# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve ``harbor:<dataset>[@<version>]`` through the Harbor registry.

The registry is one JSON file listing datasets; each task entry names a git
repository, a commit and a path. Fetching copies the task folders into Gym's shared
datasets folder so that every server reads the same bytes and nothing downloads at
run time. A ``HEAD`` pin is resolved to a commit before anything is copied.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
import urllib.request
from dataclasses import dataclass
from pathlib import Path


logger = logging.getLogger(__name__)

REGISTRY_URL = "https://raw.githubusercontent.com/harbor-framework/harbor/main/registry.json"
REF_PREFIX = "harbor:"
DATASETS_DIR_ENV = "NEMO_GYM_DATASETS_DIR"
MANIFEST_FILE = "manifest.toml"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_SHA = re.compile(r"[0-9a-f]{40}")
# Registry entries may only point at network git remotes: a URL scheme git fetches over,
# or the scp-like `user@host:path` form. Anything else (file paths, `ext::`, option-looking
# strings) is refused before it reaches a git command line.
_GIT_URL_SCHEME = re.compile(r"(https|ssh|git)://[A-Za-z0-9]", re.IGNORECASE)
_GIT_URL_SCP = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*@[A-Za-z0-9][A-Za-z0-9.-]*:(?!-)[^:]*")
# Package-store datasets are `org/name`; registry datasets are a bare name.
_PACKAGE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*")
# Task names become one path component under the dataset folder; these can never be.
_NAME_SEPARATORS = ("/", "\\", "\0")


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
        if not (_NAME.fullmatch(name) or _PACKAGE_NAME.fullmatch(name)) or (
            version and not (_NAME.fullmatch(version) or version.startswith("sha256:"))
        ):
            raise HubError(f"Malformed Harbor hub reference: {target!r}")
        return cls(name=name, version=version or None)

    @property
    def is_package(self) -> bool:
        """``org/name`` references live in Harbor's package store, not in ``registry.json``."""
        return "/" in self.name


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
        """``<name>-<version>``, so two versions of one dataset never share a folder."""
        if not self.version:
            return self.name
        return f"{self.name}-{_UNSAFE.sub('-', self.version)}"


def _numeric_version(version: str) -> tuple[int, ...] | None:
    parts = version.split(".")
    return tuple(int(part) for part in parts) if all(part.isdigit() for part in parts) else None


def _version_key(version: str) -> tuple:
    """Total order over versions: numeric versions first (by component), then the rest by text."""
    numeric = _numeric_version(version)
    return (0, numeric, "") if numeric is not None else (1, (), version)


def validate_task_name(name: str, *, error: type[ValueError] = HubError) -> str:
    """Return ``name`` if it is safe as a single folder name under the dataset folder; else raise ``error``.

    Task names arrive from ``registry.json`` and from package-store replies, and the fetchers
    use them as path components (and remove folders by them), so empty names, ``.``, ``..``,
    anything containing ``..``, a path separator or NUL is refused. Other characters, such as
    ``$``, which real Harbor task names contain, are accepted.
    """
    if not name or name in (".", "..") or ".." in name or any(separator in name for separator in _NAME_SEPARATORS):
        raise error(f"Unsafe Harbor task name {name!r}: a task name must be a single folder name")
    return name


def contained_path(target: Path, root: Path, *, error: type[ValueError] = HubError) -> Path:
    """Return ``target`` after checking that it resolves to a path strictly inside ``root``; else raise ``error``.

    Called before anything under the dataset folder is removed or replaced, so a spoofed name
    or a symlink pointing elsewhere can never make a fetch touch files outside the dataset.
    """
    resolved_root = Path(root).resolve()
    resolved = Path(target).resolve()
    if resolved == resolved_root or not resolved.is_relative_to(resolved_root):
        raise error(
            f"Refusing to replace {target}: it resolves to {resolved}, outside the dataset folder {resolved_root}"
        )
    return target


def validate_git_url(git_url: str) -> str:
    """Return ``git_url`` if it is an https, ssh or git URL (or ``user@host:path``); else raise."""
    if _GIT_URL_SCHEME.match(git_url) or _GIT_URL_SCP.fullmatch(git_url):
        return git_url
    raise HubError(
        f"Unsupported git_url {git_url!r} in the Harbor registry (expected https://, ssh://, git:// or git@host:path)"
    )


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
                name=validate_task_name(task["name"]),
                git_url=validate_git_url(task["git_url"]),
                git_commit_id=task["git_commit_id"],
                path=task["path"],
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
    if len(matches) == 1:
        return matches[0]
    if any(_numeric_version(dataset.version) is None for dataset in matches):
        versions = ", ".join(sorted((dataset.version for dataset in matches), key=_version_key))
        raise HubError(
            f"Dataset {ref.name!r} has versions that do not order numerically ({versions}); "
            f"pin one with `harbor:{ref.name}@<version>`"
        )
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


def resolve_commit(git_url: str, git_commit_id: str, cache: dict[tuple[str, str], str] | None = None) -> str:
    """Pin ``HEAD`` (or a branch name) to the commit it points at right now.

    ``cache`` memoizes the answer per ``(git_url, ref)`` so one fetch asks each remote once.
    """
    if _SHA.fullmatch(git_commit_id):
        return git_commit_id
    ref = "HEAD" if git_commit_id == "HEAD" else git_commit_id
    if cache is not None and (git_url, ref) in cache:
        return cache[(git_url, ref)]
    output = _git("ls-remote", "--", git_url, ref)
    for line in output.splitlines():
        sha, _, name = line.partition("\t")
        if name.strip() == ref or (ref != "HEAD" and name.strip() == f"refs/heads/{ref}"):
            if cache is not None:
                cache[(git_url, ref)] = sha.strip()
            return sha.strip()
    raise HubError(f"{git_url} has no ref {git_commit_id!r}")


def _read_manifest_pins(folder: Path) -> dict[str, tuple[str, str, str]]:
    """The ``(git_url, commit, path)`` pins a previous fetch recorded, or ``{}``."""
    manifest = folder / MANIFEST_FILE
    if not manifest.is_file():
        return {}
    try:
        tasks = tomllib.loads(manifest.read_text()).get("tasks", {})
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        logger.warning("%s is unreadable and will be rewritten: %s", manifest, exc)
        return {}
    pins = {}
    for name, pin in tasks.items():
        if isinstance(pin, dict) and all(
            isinstance(pin.get(key), str) for key in ("git_url", "git_commit_id", "path")
        ):
            pins[name] = (pin["git_url"], pin["git_commit_id"], pin["path"])
    return pins


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
    """Copy every task of ``dataset`` into ``root/<dataset>-<version>/<task name>/``.

    Task folders that already exist are kept with the pins the manifest recorded for
    them; only missing ones are resolved and fetched, so a rerun after a partial fetch
    completes it without moving a ``HEAD`` pin. Returns the dataset folder.
    """
    folder = Path(root) / dataset.folder_name
    folder.mkdir(parents=True, exist_ok=True)
    previous = _read_manifest_pins(folder)
    resolved: dict[tuple[str, str], str] = {}
    pins: dict[str, tuple[str, str, str]] = {}
    by_repo: dict[tuple[str, str], list[RegistryTask]] = {}
    for task in dataset.tasks:
        present = (folder / task.name / "task.toml").is_file()
        if present and task.name in previous:
            pins[task.name] = previous[task.name]
            continue
        commit = resolve_commit(task.git_url, task.git_commit_id, resolved)
        pins[task.name] = (task.git_url, commit, task.path)
        if not present:
            by_repo.setdefault((task.git_url, commit), []).append(task)

    for (git_url, commit), tasks in by_repo.items():
        with tempfile.TemporaryDirectory(prefix="nemo-gym-harbor-", dir=folder) as tmp:
            clone = Path(tmp) / "repo"
            _git("clone", "--quiet", "--filter=blob:none", "--no-checkout", "--", git_url, str(clone))
            _git("sparse-checkout", "set", "--no-cone", *(task.path for task in tasks), cwd=clone)
            _git("checkout", "--quiet", commit, cwd=clone)
            for task in tasks:
                source = clone / task.path
                if not (source / "task.toml").is_file():
                    raise HubError(f"{git_url}@{commit[:12]}:{task.path} has no task.toml")
                staged = Path(tmp) / task.name
                shutil.copytree(source, staged, symlinks=False)
                shutil.move(str(staged), str(contained_path(folder / task.name, folder)))
    _write_manifest(folder, dataset, pins)
    return folder


def fetch_ref(target: str, root: Path | None = None, *, refresh_registry: bool = False, force: bool = False) -> Path:
    """Resolve and fetch a ``harbor:`` reference; return the dataset folder.

    ``force`` lets a package-store fetch replace a task folder whose content no longer matches
    the store (see :func:`nemo_gym.tasks.harbor.package_store.fetch_package_dataset`).
    """
    root = Path(root) if root is not None else datasets_dir()
    ref = HubRef.parse(target)
    if ref.is_package:
        from nemo_gym.tasks.harbor.package_store import PackageRef, fetch_package_dataset

        org, _, name = ref.name.partition("/")
        return fetch_package_dataset(PackageRef(org=org, name=name, ref=ref.version or "latest"), root, force=force)
    registry = load_registry(root / ".harbor", refresh=refresh_registry)
    return fetch_dataset(resolve_dataset(ref, registry), root)
