# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve ``harbor:<dataset>[@<version>]`` through the Harbor registry.

The registry is one JSON file listing datasets; each task entry names a git
repository, a commit and a path. Fetching copies the task folders into Gym's pulled
datasets folder, ``environments/harbor/`` under the working root, so that every server reads the
same bytes and nothing downloads at run time. A ``HEAD`` pin is resolved to a commit before anything is copied.
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from nemo_gym import WORKING_DIR
from nemo_gym.data.tasks.harbor.digest import content_hash
from nemo_gym.data.tasks.harbor.manifest import (
    MANIFEST_FILE,
    Manifest,
    ManifestError,
    UnsupportedManifestError,
    task_folders,
)


logger = logging.getLogger(__name__)

REGISTRY_URL = "https://raw.githubusercontent.com/harbor-framework/harbor/main/registry.json"
REF_PREFIX = "harbor:"
DATASETS_DIR_ENV = "NEMO_GYM_DATASETS_DIR"
REGISTRY_SOURCE = "harbor-registry"

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_SHA = re.compile(r"[0-9a-f]{40}")
# The fetch stages its clone inside the dataset folder (the same filesystem, for the atomic rename into
# place). Hidden, so recursive task discovery, which skips dot-folders, never enters a staging directory an
# interrupted fetch left behind: its clone holds another copy of each task.
_STAGING_PREFIX = ".nemo-gym-harbor-"
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
    """Where pulled datasets live: ``$NEMO_GYM_DATASETS_DIR``, else ``environments/harbor/`` under the working root.

    That root is :data:`nemo_gym.WORKING_DIR`: the Gym checkout of an editable install, the working directory of
    a package install, whose ``PARENT_DIR`` is ``site-packages`` (shared, often read-only) and could hold neither
    a dataset nor the registry cache.

    Each fetched dataset is one folder under it (``environments/harbor/<dataset>/``), gitignored there
    next to the committed per-dataset entries, which keep only their README, YAML files and ``dataset.toml``.
    An entry that declares a ``[gym] source`` is pulled into its own folder when that folder is under the working
    root and not installed, else (an entry a package install ships under ``site-packages``, a project-local ``.venv``
    included, or one an extra search root commits, or any entry when ``NEMO_GYM_DATASETS_DIR`` is set) into
    ``<datasets dir>/<entry>/`` (:mod:`nemo_gym.data.tasks.harbor.entries`).
    """
    return Path(os.environ.get(DATASETS_DIR_ENV) or WORKING_DIR / "environments" / "harbor").expanduser().resolve()


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

    @property
    def reference(self) -> str:
        """The reference a fetch of this dataset records in its manifest: ``harbor:<name>@<version>``, the version
        resolved, so a fetch of ``harbor:<name>`` records which version the registry picked."""
        return f"{REF_PREFIX}{self.name}@{self.version}" if self.version else f"{REF_PREFIX}{self.name}"


def _numeric_version(version: str) -> tuple[int, ...] | None:
    parts = version.split(".")
    return tuple(int(part) for part in parts) if all(part.isdigit() for part in parts) else None


def _version_key(version: str) -> tuple:
    """Total order over versions: numeric versions first (by component), then the rest by text."""
    numeric = _numeric_version(version)
    return (0, numeric, "") if numeric is not None else (1, (), version)


def validate_task_name(name: str, *, error: type[ValueError] = HubError) -> str:
    """Require one folder component, reserving the dataset manifest's name; raise ``error`` otherwise.

    Task names arrive from ``registry.json`` and from package-store replies, and the fetchers
    use them as path components, so empty names, ``.``, anything containing ``..``, a path
    separator or NUL is refused. Other characters, such as ``$``, which real Harbor task names
    contain, are accepted.
    """
    if not name or ".." in name or name in (".", MANIFEST_FILE) or any(char in name for char in _NAME_SEPARATORS):
        raise error(f"Unsafe Harbor task name {name!r}: expected a single folder name")
    return name


def _validate_task_path(path: str) -> None:
    source = PurePosixPath(path)
    if (
        not path
        or source.is_absolute()
        or ".." in source.parts
        or path.startswith("-")
        or "\\" in path
        or "\0" in path
    ):
        raise HubError(f"Unsafe Harbor task path {path!r}: expected a relative path inside the repository")


def _validate_task_tree(source: Path) -> None:
    """Refuse links that could import host files, before any task bytes are copied."""
    for link in source.rglob("*"):
        if not link.is_symlink():
            continue
        target = link.readlink()
        try:
            resolved = link.resolve(strict=True)
        except (OSError, ValueError) as exc:
            raise HubError(f"Invalid task symlink {link}: {exc}") from exc
        if target.is_absolute() or not resolved.is_relative_to(source.resolve()):
            raise HubError(f"Task symlink {link} leaves the task folder")


def contained_path(target: Path, root: Path, *, error: type[ValueError] = HubError) -> Path:
    """Return ``target`` after checking that it resolves to a path strictly inside ``root``; else raise ``error``.

    Called before anything under a dataset folder is removed or replaced, so a spoofed name
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


def _parse_registry(payload: bytes) -> list[RegistryDataset]:
    entries = json.loads(payload)
    if not isinstance(entries, list):
        raise ValueError("expected a list of datasets")
    datasets: list[RegistryDataset] = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry["name"], str):
            raise ValueError("each dataset must be an object with a string name")
        task_entries = entry.get("tasks", [])
        if not isinstance(task_entries, list):
            raise ValueError(f"dataset {entry['name']!r}: tasks must be a list")
        tasks: list[RegistryTask] = []
        for task in task_entries:
            if not isinstance(task, dict):
                raise ValueError(f"dataset {entry['name']!r}: each task must be an object")
            for key in ("name", "git_url", "git_commit_id", "path"):
                if not isinstance(task[key], str):
                    raise ValueError(f"dataset {entry['name']!r}: task {key} must be a string")
            tasks.append(
                RegistryTask(
                    name=validate_task_name(task["name"]),
                    git_url=validate_git_url(task["git_url"]),
                    git_commit_id=task["git_commit_id"],
                    path=task["path"],
                )
            )
        datasets.append(
            RegistryDataset(
                name=entry["name"],
                version=str(entry.get("version", "")),
                description=entry.get("description", ""),
                tasks=tuple(tasks),
            )
        )
    return datasets


def load_registry(cache_dir: Path, *, refresh: bool = False) -> list[RegistryDataset]:
    """Read the registry, validating downloads before atomically replacing the cache.

    A damaged cache raises ``HubError`` with its path and recovery instructions.
    Failed downloads or updates leave any previous cache intact.
    """
    cache = Path(cache_dir).resolve() / "registry.json"
    download = refresh or not cache.is_file()
    if download:
        try:
            with urllib.request.urlopen(REGISTRY_URL, timeout=120) as response:
                payload = response.read()
        except OSError as exc:
            raise HubError(f"Could not download the Harbor registry from {REGISTRY_URL}: {exc}") from exc
    else:
        try:
            payload = cache.read_bytes()
        except OSError as exc:
            raise HubError(
                f"Could not read Harbor registry cache {cache}: {exc}. Check file permissions and retry."
            ) from exc
    try:
        datasets = _parse_registry(payload)
    except (ValueError, KeyError, TypeError) as exc:
        if download:
            hint = "The downloaded registry is invalid; retry later. Any existing cache was left unchanged."
        else:
            hint = "Remove this file and retry to download a fresh registry."
        raise HubError(f"Invalid Harbor registry at {cache}: {exc}. {hint}") from exc
    if download:
        try:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=cache.parent, prefix=".registry-", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    stream.write(payload)
                    stream.close()
                    temporary.replace(cache)
                finally:
                    temporary.unlink(missing_ok=True)
        except OSError as exc:
            raise HubError(
                f"Could not update Harbor registry cache {cache}: {exc}. Check write permissions and retry."
            ) from exc
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


def _read_manifest(folder: Path) -> Manifest | None:
    """The manifest a previous fetch wrote, or ``None`` (a malformed one is rewritten).

    One a newer Gym wrote (a schema this Gym does not read) raises instead, so the fetch stops
    with the folder untouched rather than replacing a manifest this Gym cannot read in full.
    """
    try:
        return Manifest.read(folder)
    except UnsupportedManifestError:
        raise
    except ManifestError as exc:
        logger.warning("%s is unreadable and will be rewritten: %s", folder / MANIFEST_FILE, exc)
        return None


def _read_manifest_pins(manifest: Manifest | None) -> dict[str, tuple[str, str, str]]:
    """The ``(git_url, commit, path)`` pins a previous fetch recorded, or ``{}``."""
    if manifest is None:
        return {}
    pins = {}
    for name, pin in manifest.pins.items():
        if all(isinstance(pin.get(key), str) for key in ("git_url", "git_commit_id", "path")):
            pins[name] = (pin["git_url"], pin["git_commit_id"], pin["path"])
    return pins


def _write_manifest(
    folder: Path, pins: dict[str, tuple[str, str, str]], digests: dict[str, str], *, reference: str
) -> None:
    """Publish the pins with the digests of their folders, and the ``reference`` fetched; only pinned tasks are recorded.

    ``digests`` caches the hashing across the per-task checkpoints, and naming the pinned tasks keeps each
    checkpoint from scanning the dataset folder again: a folder the fetch did not pin stays unrecorded,
    so prepare refuses it instead of running it as part of the dataset.
    """
    Manifest.build(
        folder,
        source=REGISTRY_SOURCE,
        pins={
            name: {"git_url": url, "git_commit_id": commit, "path": path} for name, (url, commit, path) in pins.items()
        },
        digests=digests,
        names=pins,
        ref=reference,
    ).write(folder)


def refuse_unrecorded_folders(folder: Path, known: set[str], *, what: str, error: type[ValueError] = HubError) -> None:
    """Refuse task folders under ``folder`` that neither the source names nor a previous fetch recorded.

    The manifest a fetch writes is the record prepare trusts, so a folder nobody fetched must not sit
    beside the fetched ones when it is written: it would run and score as part of the dataset.
    """
    extra = sorted(path.name for path in task_folders(folder) if path.name not in known)
    if extra:
        raise error(
            f"Task folder(s) {', '.join(extra)} in {folder} are not in the {what} and not recorded by a previous "
            "fetch; move them aside and run again"
        )


def _refuse_dropped_folders(
    folder: Path, previous: dict[str, tuple[str, str, str]], names: set[str], dataset: RegistryDataset
) -> None:
    """Refuse the task folders a previous fetch pinned for tasks that ``dataset`` no longer lists.

    The manifest a fetch writes pins the dataset's tasks alone, so such a folder would stay on disk unrecorded and
    the next prepare would refuse the whole folder as edited by hand, after a pull that reported success. The folder
    holds another version's task (a committed entry's ``source`` moved to a version with fewer tasks, the registry
    revised this one, or ``harbor:<dataset>`` now resolves to such a version), so it is refused here, named, with
    what to move aside, before anything is cloned or rewritten: the registry's counterpart of the store's refusal of
    a manifest written for another version (``tasks removed``, :mod:`nemo_gym.data.tasks.harbor.package_store`).
    """
    dropped = sorted(name for name in previous if name not in names and (folder / name / "task.toml").is_file())
    if dropped:
        raise HubError(
            f"Task folder(s) {', '.join(dropped)} in {folder} were fetched for another version of {dataset.name}: a "
            f"previous fetch pinned them and {dataset.reference} does not list them; move them aside and run again "
            f"to fetch {dataset.reference}"
        )


def _refuse_repinned_folders(folder: Path, repinned: dict[str, tuple[str, str]], dataset: RegistryDataset) -> None:
    """Refuse the task folders a previous fetch pinned at a commit other than the one ``dataset`` pins them at.

    A recorded pin is kept when the registry still names the same repository and path, so a version that pins a
    task at another commit (a committed entry's ``source`` moved to it, ``harbor:<dataset>`` now resolves to it, or
    the registry revised this version in place) used to keep the folder of the commit fetched before and record it
    under the new reference: the other version's content, run by name as this one's. Harbor caches a git task by
    ``(git_url, git_commit_id, path)`` and downloads a task pinned at another commit afresh
    (``harbor.models.task.id.GitTaskId``), so the folder is refused here, named with both commits, before anything
    is cloned or rewritten; moved aside, the fetch clones the commit the registry pins. A registry pin that is not
    a commit (``HEAD``, a branch) keeps the recorded one while the reference stays the same: the fetch pinned what
    it pointed at then. A fetch of another reference into the folder (the version changed) resolves such a pin
    afresh, as the first fetch of that version into a folder of its own did before entries shared one, and Harbor
    never serves a task without a resolved commit from its cache (``harbor.tasks.client``); a commit that moved
    since is refused here the same way, named with both commits.
    """
    if repinned:
        was = ", ".join(f"{name} at {old[:12]}" for name, (old, _) in sorted(repinned.items()))
        now = ", ".join(f"{name} at {new[:12]}" for name, (_, new) in sorted(repinned.items()))
        raise HubError(
            f"Task folder(s) {', '.join(sorted(repinned))} in {folder} were fetched for another version of "
            f"{dataset.name}: a previous fetch pinned {was} and {dataset.reference} pins {now}; move them aside and "
            f"run again to fetch {dataset.reference}"
        )


def _check_kept_folders(folder: Path, kept: list[str], previous: Manifest | None, digests: dict[str, str]) -> None:
    """Hash the task folders a fetch keeps and refuse one whose content differs from the manifest.

    A rewritten manifest records what is on disk, so without this check a fetch that adds a task would
    bless a hand-edited folder next to it. A folder recorded without a digest (a manifest written
    before digests existed) is accepted and recorded now.
    """
    recorded = previous.task_digests if previous is not None else {}
    for name in kept:
        digests[name] = content_hash(folder / name)
        expected = recorded.get(name)
        if expected is not None and expected != digests[name]:
            raise HubError(
                f"{folder / name} differs from its fetch record in {folder / MANIFEST_FILE}: its content hash is "
                f"{digests[name]} but the manifest records {expected}. The folder was edited by hand. Delete the "
                "dataset folder, or move it aside, and run again to fetch the registry's copy"
            )


def fetch_dataset(dataset: RegistryDataset, root: Path, *, folder: Path | None = None) -> Path:
    """Fetch tasks, retaining the recorded source pins of existing folders.

    Publish a pin after each task, so a later failure leaves completed tasks reusable.
    A folder without a trustworthy pin (including interruption before pin publication)
    is refused; the user can move it aside and retry without losing local edits. The
    manifest records each folder's digest beside its pin; when a fetch adds tasks to an
    existing folder, the kept folders must still match their recorded digests.

    The tasks land in ``root/<name>-<version>/`` (:attr:`RegistryDataset.folder_name`) unless ``folder``
    names the dataset folder itself: a committed entry under ``environments/harbor/`` whose ``dataset.toml``
    declares this dataset as its ``[gym] source`` (:mod:`nemo_gym.data.tasks.harbor.entries`). The folder's other
    files (the entry's README, YAML overlays and ``dataset.toml``) are not task folders and stay as they are.
    A folder holding a task of another version (the entry's source moved to a version that no longer lists it, or
    that pins it at another commit, a ``HEAD`` pin that moved since included) is refused with the folder named, as
    the store's fetch refuses a manifest written for another version (:func:`_refuse_dropped_folders`,
    :func:`_refuse_repinned_folders`), never left unrecorded beside the fetched tasks or recorded as this version's.
    """
    if not _NAME.fullmatch(dataset.name):
        raise HubError(f"Unsafe Harbor dataset name {dataset.name!r}")
    names: set[str] = set()
    for task in dataset.tasks:
        validate_task_name(task.name)
        _validate_task_path(task.path)
        if task.name in names:
            raise HubError(f"Duplicate Harbor task name {task.name!r}")
        names.add(task.name)
    folder = Path(folder).resolve() if folder is not None else Path(root).resolve() / dataset.folder_name
    if folder.is_symlink():
        raise HubError(f"Dataset folder must not be a symlink: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / MANIFEST_FILE).is_symlink():
        raise HubError(f"Dataset manifest must not be a symlink: {folder / MANIFEST_FILE}")
    manifest = _read_manifest(folder)
    previous = _read_manifest_pins(manifest)
    # The record names another reference than this fetch's (another version of the dataset: a committed entry's
    # source moved, or the registry's pick did), so the folder holds that version's fetch; one that records no
    # reference (written before it was recorded) is this fetch's as far as it says.
    moved = manifest is not None and manifest.ref is not None and manifest.ref != dataset.reference
    resolved: dict[tuple[str, str], str] = {}
    pins: dict[str, tuple[str, str, str]] = {}
    digests: dict[str, str] = {}
    kept: list[str] = []
    repinned: dict[str, tuple[str, str]] = {}
    by_repo: dict[tuple[str, str], list[RegistryTask]] = {}
    for task in dataset.tasks:
        destination = folder / task.name
        if destination.is_symlink():
            raise HubError(f"Task destination must not be a symlink: {destination}")
        pin = previous.get(task.name)
        if pin is not None and (pin[0] != task.git_url or pin[2] != task.path or not _SHA.fullmatch(pin[1])):
            raise HubError(
                f"Recorded source pin for {destination} is invalid or differs from the registry; move the folder aside and retry"
            )
        pinned = task.git_commit_id if _SHA.fullmatch(task.git_commit_id) else None
        if pin is not None and pinned is None and moved:
            # A pin that is not a commit (`HEAD`, a branch) stays as the recorded fetch resolved it while the
            # reference stays the same, but this fetch is another version's, into the folder the previous version's
            # fetch filled: its pin is resolved afresh, as a fetch into a folder of this version's own resolved it,
            # and a commit that moved since is handled below like one the registry changed.
            pinned = resolve_commit(task.git_url, task.git_commit_id, resolved)
        if pin is not None and pinned is not None and pinned != pin[1]:
            # The registry pins this task at another commit than the recorded fetch. An existing folder holds that
            # other commit's content and is refused below (_refuse_repinned_folders); a missing one is no repair of
            # the recorded bytes but a fetch of the commit the registry pins.
            if destination.exists():
                repinned[task.name] = (pin[1], pinned)
                continue
            pin = None
        if pin is not None:
            # A missing folder is a repair of these pinned bytes. Keep its validated pin in every
            # publication, even before copying it, so an interrupted repair cannot follow new HEAD.
            pins[task.name] = pin
        if destination.exists():
            if not (destination / "task.toml").is_file() or pin is None:
                raise HubError(
                    f"Existing task folder {destination} has no trustworthy source pin; move it aside and retry"
                )
            _validate_task_tree(destination)
            kept.append(task.name)
            continue
        commit = pin[1] if pin else resolve_commit(task.git_url, task.git_commit_id, resolved)
        by_repo.setdefault((task.git_url, commit), []).append(task)

    _refuse_repinned_folders(folder, repinned, dataset)
    # A manifest that records another reference for these pins (one written before the reference was recorded, or a
    # version the registry lists with the same tasks) is rewritten with this fetch's, so a run by the entry's name
    # finds the fetch it declares on record (nemo_gym.data.tasks.harbor.cli._check_fetch_record).
    rerecord = manifest is not None and manifest.ref != dataset.reference
    if by_repo or not dataset.tasks or rerecord:
        # The manifest is about to be rewritten: a folder it would not pin must not be there (one a previous fetch
        # pinned for a task this version does not list is named as such, with the version), and the kept folders
        # must still be what it recorded.
        _refuse_dropped_folders(folder, previous, names, dataset)
        refuse_unrecorded_folders(folder, names, what="registry")
        _check_kept_folders(folder, kept, manifest, digests)
    for (git_url, commit), tasks in by_repo.items():
        with tempfile.TemporaryDirectory(prefix=_STAGING_PREFIX, dir=folder) as tmp:
            clone = Path(tmp) / "repo"
            _git("clone", "--quiet", "--filter=blob:none", "--no-checkout", "--", git_url, str(clone))
            _git("sparse-checkout", "set", "--no-cone", "--", *(task.path for task in tasks), cwd=clone)
            _git("checkout", "--quiet", commit, cwd=clone)
            for task in tasks:
                source = clone
                for part in PurePosixPath(task.path).parts:
                    source /= part
                    if source.is_symlink():
                        raise HubError(f"Task source path contains a symlink: {source}")
                _validate_task_tree(source)
                if not (source / "task.toml").is_file():
                    raise HubError(f"{git_url}@{commit[:12]}:{task.path} has no task.toml")
                staged = Path(tmp) / "staged"
                shutil.copytree(source, staged, symlinks=True)
                destination = folder / task.name
                if destination.exists() or destination.is_symlink():
                    raise HubError(f"Task destination appeared during fetch: {destination}; retry preparation")
                staged.rename(destination)
                pins[task.name] = (git_url, commit, task.path)
                _write_manifest(folder, pins, digests, reference=dataset.reference)
    if not dataset.tasks or (rerecord and not by_repo):
        _write_manifest(folder, pins, digests, reference=dataset.reference)
    return folder


def expected_folder(target: str, root: Path | None = None) -> Path | None:
    """Where a fetch of ``target`` lands when that follows from the reference alone, without resolving it.

    A package-store dataset's folder is named from its reference (``<name>`` for ``latest``, ``<name>-<tag>``,
    ``<name>-<12 digest chars>`` for a digest pin, :meth:`PackageRef.folder_name`), and a registry dataset's from
    its name and version; ``None`` for a registry reference whose version the registry picks. ``gym dataset pull``
    reads the folder's manifest here before the fetch to report an unchanged dataset.
    """
    root = Path(root) if root is not None else datasets_dir()
    ref = HubRef.parse(target)
    if ref.is_package:
        from nemo_gym.data.tasks.harbor.package_store import PackageRef

        package = PackageRef.of(ref)
        return root / package.folder_name(package.digest or "")
    if ref.version is None:
        return None
    return root / RegistryDataset(name=ref.name, version=ref.version, description="", tasks=()).folder_name


def resolved_folder(target: str, root: Path | None = None, *, refresh_registry: bool = False) -> Path:
    """Where a fetch of ``target`` lands, resolving a registry reference whose version the registry picks.

    :func:`expected_folder` names the folder when the reference alone does; for ``harbor:<dataset>`` without a
    version the registry's cache under ``root/.harbor/`` (downloaded, or refreshed when ``refresh_registry``) is
    read and the version it picks names the folder, as :func:`fetch_ref` will. ``gym dataset pull`` resolves such a
    reference here, before the fetch, so it can read the folder's manifest and report an unchanged dataset.
    """
    root = Path(root) if root is not None else datasets_dir()
    folder = expected_folder(target, root)
    if folder is not None:
        return folder
    registry = load_registry(root / ".harbor", refresh=refresh_registry)
    return root / resolve_dataset(HubRef.parse(target), registry).folder_name


def canonical_reference(target: str) -> str:
    """The ``harbor:`` reference ``target`` as the fetchers record it, so two spellings of one reference compare equal.

    A package-store reference is read as the store reads it (:attr:`~nemo_gym.data.tasks.harbor.package_store.PackageRef.reference`:
    ``harbor:org/name`` is ``@latest``, a digest is its ``sha256:`` form). A registry reference is itself: a version it
    names is the version, and ``harbor:<dataset>`` without one is the registry's pick, another reference than any
    version it may resolve to. A malformed reference raises :class:`HubError`, as :meth:`HubRef.parse` does.
    """
    ref = HubRef.parse(target)
    if ref.is_package:
        from nemo_gym.data.tasks.harbor.package_store import PackageRef

        return PackageRef.of(ref).reference
    return f"{REF_PREFIX}{ref.name}@{ref.version}" if ref.version else f"{REF_PREFIX}{ref.name}"


def declares_recorded_fetch(declared: str, recorded: str) -> bool:
    """Whether the ``harbor:`` reference ``declared`` (a committed entry's ``[gym] source``) names the fetch a manifest
    records as ``recorded`` (:attr:`~nemo_gym.data.tasks.harbor.manifest.Manifest.ref`), without asking the hub.

    A package-store reference matches by package and by tag, ``latest`` or digest, read as the store reads them
    (:attr:`~nemo_gym.data.tasks.harbor.package_store.PackageRef.reference`: ``harbor:org/name`` is ``@latest``, a digest is
    its ``sha256:`` form). A registry reference matches by dataset and, when it pins one, by version: the version an
    unpinned ``harbor:<dataset>`` gets is the registry's pick, which the record holds and only a pull can revise.
    A record that does not parse as a reference matches nothing.
    """
    try:
        wanted, found = HubRef.parse(declared), HubRef.parse(recorded)
    except HubError:
        return False
    if wanted.is_package != found.is_package or wanted.name != found.name:
        return False
    if wanted.is_package:
        from nemo_gym.data.tasks.harbor.package_store import PackageRef

        return PackageRef.of(wanted).reference == PackageRef.of(found).reference
    return wanted.version is None or wanted.version == found.version


def fetch_ref(
    target: str, root: Path | None = None, *, refresh_registry: bool = False, folder: Path | None = None
) -> Path:
    """Resolve and fetch a ``harbor:`` reference; return the dataset folder.

    The dataset lands in its own folder under ``root`` (:func:`datasets_dir` unless given), named after the
    dataset and its version, or in ``folder`` when a caller names the destination: the folder of a committed
    entry that declares the reference as its ``[gym] source`` (:func:`nemo_gym.data.tasks.harbor.entries.plan_pull`
    decides that; this function does not read entries). The registry cache stays under ``root/.harbor/``.

    A package-store fetch refuses a task folder whose content no longer matches the store
    (see :func:`nemo_gym.data.tasks.harbor.package_store.fetch_package_dataset`). Whatever the store fetch
    refuses or cannot read (a ``PackageStoreError``) is raised as a :class:`HubError`, so a caller handles
    one error type for a ``harbor:`` reference of either source.
    """
    root = Path(root) if root is not None else datasets_dir()
    ref = HubRef.parse(target)
    if ref.is_package:
        from nemo_gym.data.tasks.harbor.package_store import PackageRef, PackageStoreError, fetch_package_dataset

        try:
            return fetch_package_dataset(PackageRef.of(ref), root, folder=folder)
        except PackageStoreError as exc:
            raise HubError(str(exc)) from exc
    registry = load_registry(root / ".harbor", refresh=refresh_registry)
    return fetch_dataset(resolve_dataset(ref, registry), root, folder=folder)
