# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Committed dataset entries under ``environments/harbor/<entry>/`` and where a pull lands.

An entry is the folder a Gym checkout commits for one dataset: its README, deployment YAML overlays and a
``dataset.toml`` whose ``[gym]`` table may declare ``source``, the hub reference the task folders come from
(``harbor:<org>/<name>@<tag | sha256:digest>`` for the package store, ``harbor:<dataset>@<version>`` for the
registry). ``gym dataset pull <entry>`` fetches that reference into the entry's own folder, where the pulled task
folders and ``manifest.toml`` are gitignored next to the committed files, and a pull of the reference itself
lands there too, so one dataset never lives in two sibling folders. A reference no entry declares lands in
``<datasets dir>/<dataset>-<version>/`` as before. ``gym dataset validate <entry>`` and ``gym eval run <entry>``
resolve the name to that folder without fetching (:func:`entry_target`, where an existing path of that name keeps
meaning the path, since those commands take paths; the pull takes none, so a bare name is always the entry there).

An entry is pulled into its own folder wherever the search root that commits it lives (the Gym checkout, a
``NEMO_GYM_EXTRA_ROOTS`` root or the working directory). ``NEMO_GYM_DATASETS_DIR`` still moves every pulled
dataset: an entry's tasks then live in ``<datasets dir>/<entry>/`` with a copy of the entry's Gym layer beside them,
its ``dataset.toml`` and whatever dataset-level ``task_data.py`` or ``verifier.py`` it commits
(:func:`gym_layer`), kept current with the entry's files (:func:`prepare_destination`), so the files that shape the
tasks, and the dataset digest that covers them, are the same wherever the pull landed.
"""

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from nemo_gym import component_search_roots
from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE, read_dataset_config
from nemo_gym.tasks.harbor.hub import DATASETS_DIR_ENV, HubError, HubRef, datasets_dir, is_hub_ref
from nemo_gym.tasks.harbor.manifest import (
    DATASET_FILES,
    Manifest,
    ManifestError,
    dataset_files_digest,
    dataset_toml_digest,
    refresh_gym_layer,
    task_folders,
)


ENTRIES_SUBDIR = Path("environments") / "harbor"
# An entry is named by its folder, one path component; a hub folder name (`terminal-bench-4.0.0`) fits too, which
# is fine: such a folder commits no dataset.toml with a source and is never an entry a pull resolves.
_ENTRY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# The files of an entry's Gym layer beside the task folders: what enters the
# dataset digest besides the tasks, and so what a destination elsewhere needs a copy of.
LAYER_FILES = (DATASET_FILE, *DATASET_FILES)


@dataclass(frozen=True)
class DatasetEntry:
    """A committed entry: its name (the folder's), its folder and the ``[gym] source`` it declares, if any."""

    name: str
    folder: Path
    source: str | None

    @property
    def destination(self) -> Path:
        """Where the entry's tasks are pulled to and read from: the entry's own folder, in whichever search root
        commits it, unless ``NEMO_GYM_DATASETS_DIR`` moves every pull, then ``<datasets dir>/<name>/``
        (:func:`nemo_gym.tasks.harbor.hub.datasets_dir`)."""
        return datasets_dir() / self.name if os.environ.get(DATASETS_DIR_ENV) else self.folder


@dataclass(frozen=True)
class PullPlan:
    """What a pull of a target fetches and where: the hub reference, and the entry it lands in, if one."""

    ref: str
    entry: DatasetEntry | None

    @property
    def folder(self) -> Path | None:
        """The dataset folder to fetch into, or ``None`` for the hub's own folder name under the datasets dir."""
        return self.entry.destination if self.entry is not None else None


def _roots() -> list[Path]:
    return component_search_roots()


def _entry(folder: Path) -> DatasetEntry:
    """The entry at ``folder``; a ``dataset.toml`` that does not read raises ``ValueError`` with the file named."""
    return DatasetEntry(name=folder.name, folder=folder.resolve(), source=read_dataset_config(folder).source)


def find_entry(name: str) -> DatasetEntry | None:
    """The entry called ``name`` in the first search root that commits one, or ``None``.

    An entry need not declare a source: one whose tasks come from elsewhere (a checkout the user points
    ``gym eval run`` at) still resolves by name to its folder. A ``dataset.toml`` that does not read raises
    ``ValueError`` naming the file, as prepare would for the folder.
    """
    if not _ENTRY_NAME.fullmatch(name):
        return None
    for root in _roots():
        folder = root / ENTRIES_SUBDIR / name
        if folder.is_dir():
            return _entry(folder)
    return None


def committed_entries() -> list[DatasetEntry]:
    """Every entry that declares a source, across the search roots; an earlier root wins a name.

    Only folders with a ``dataset.toml`` are read (a pulled hub folder has none unless its user wrote one) and
    one whose table does not read is skipped here: the mapping must not fail every pull over a stray file, and
    the entry's own pull reports it.
    """
    found: dict[str, DatasetEntry] = {}
    for root in _roots():
        parent = root / ENTRIES_SUBDIR
        if not parent.is_dir():
            continue
        for folder in sorted(parent.iterdir()):
            if folder.name in found or folder.name.startswith(".") or not folder.is_dir():
                continue
            if not (folder / DATASET_FILE).is_file():
                continue
            try:
                entry = _entry(folder)
            except ValueError:
                continue
            if entry.source is not None:
                found[folder.name] = entry
    return list(found.values())


def entry_for_ref(target: str) -> DatasetEntry | None:
    """The entry whose source names the same dataset and version as the ``harbor:`` reference ``target``, or ``None``.

    ``harbor:org/name`` (the store's ``latest``) and ``harbor:org/name@4.0.0`` are different references, so a
    plain reference never lands in an entry pinned to a version. Two entries declaring one source is a mistake in
    the checkout and is refused with both named.
    """
    ref = HubRef.parse(target)
    matches = [entry for entry in committed_entries() if HubRef.parse(entry.source) == ref]
    if len(matches) > 1:
        names = ", ".join(entry.name for entry in matches)
        raise HubError(
            f"{target} is the source of {len(matches)} entries under environments/harbor/ ({names}); keep one"
        )
    return matches[0] if matches else None


def entry_target(target: str) -> DatasetEntry | None:
    """The entry ``target`` names on a command line, or ``None`` when it is a path or a hub reference.

    A bare name (one path component, no ``./`` or ``/``) that is not an existing path and that a search root
    commits under ``environments/harbor/`` is the entry. An existing folder of that name, or an explicit path,
    keeps meaning the path, as it does for Hydra overrides.
    """
    if is_hub_ref(target) or target in (".", "..") or Path(target).name != target:
        return None
    if Path(target).exists():
        return None
    return find_entry(target)


def plan_pull(target: str) -> PullPlan | None:
    """How ``gym dataset pull TARGET`` lands, or ``None`` when ``target`` is neither a hub reference nor an entry.

    A hub reference is fetched into the entry that declares it as its source, else into the hub's own folder.
    An entry name is fetched from the source it declares, into its folder; an entry without a source has
    nothing to pull and says so. The pull takes no path, so a bare name is looked up as an entry outright: a
    folder of the same name in the working directory does not hide it, as it would from validate and run
    (:func:`entry_target`).
    """
    if is_hub_ref(target):
        return PullPlan(ref=target, entry=entry_for_ref(target))
    entry = find_entry(target)
    if entry is None:
        return None
    if entry.source is None:
        raise HubError(
            f"{entry.folder / DATASET_FILE} declares no [gym] source, so there is nothing to pull for entry "
            f'{entry.name}; add source = "harbor:<org>/<name>@<tag>" there, or pull a harbor: reference'
        )
    return PullPlan(ref=entry.source, entry=entry)


def gym_layer(folder: Path) -> dict[Path, bytes]:
    """The Gym-layer files under ``folder`` by relative path, with their content: ``dataset.toml``, a dataset-level
    ``task_data.py`` or ``verifier.py`` (:data:`LAYER_FILES`), each when present.
    These are the files the dataset digest covers besides the task folders, so they are what an entry's destination
    elsewhere must hold a copy of for a run there to be the entry's."""
    folder = Path(folder)
    paths = [Path(name) for name in LAYER_FILES if (folder / name).is_file()]
    return {path: (folder / path).read_bytes() for path in paths}


def prepare_destination(entry: DatasetEntry, *, create: bool = True) -> Path:
    """``entry``'s destination with the entry's Gym layer current there, ready for a fetch or a prepare.

    The entry's own folder needs nothing. A destination elsewhere (``NEMO_GYM_DATASETS_DIR``) holds a copy of the
    entry's Gym layer (:func:`gym_layer`: ``dataset.toml``, a dataset-level ``task_data.py`` or ``verifier.py``),
    so the files that shape the tasks and the digest that covers them are the entry's wherever the
    pull landed: the copies are placed before the first task lands (``create`` makes the folder; a prepare, which
    fetches nothing, passes ``False`` and leaves a missing folder missing) and are replaced, a file the entry no
    longer commits removed, when the entry's files changed since, as long as the copies are still the ones on record:
    the ones the folder's ``manifest.toml`` hashed as its ``dataset_toml_digest`` and
    ``dataset_files_digest``, or, with no manifest and no task folder (nothing was pulled yet), the ones this function
    placed. Whenever the copies are replaced the manifest is brought up to date with them
    (:func:`~nemo_gym.tasks.harbor.manifest.refresh_gym_layer`), so the record follows the copies even when nothing
    rewrites the manifest afterwards (a registry fetch with every folder in place, a fetch that fails, a prepare
    refused before its manifest check), and the next edit of the entry's files is recognized as such. A copy that
    matches neither the entry's file nor that record was edited in place and is refused, never overwritten: the
    entry's files are where the layer is edited. ``gym dataset pull <entry>``, ``gym dataset validate <entry>`` and
    ``gym eval run <entry>`` all come through here, so they agree on the folder.
    """
    destination = entry.destination
    if destination == entry.folder:
        return destination
    layer = gym_layer(entry.folder)
    if not layer or (not create and not destination.is_dir()):
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    copies = gym_layer(destination)
    if copies == layer:
        return destination
    if copies:
        _refuse_copies_edited_in_place(entry, destination, layer, copies)
    for path in sorted(set(copies) - set(layer)):
        _remove_copy(destination, path)
    for path, content in layer.items():
        if copies.get(path) != content:
            (destination / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry.folder / path, destination / path)
    try:
        refresh_gym_layer(destination)
    except ManifestError:
        # A manifest this Gym cannot read records nothing to keep current; the fetch or the prepare that follows
        # reports it, or rewrites it, in its own words.
        pass
    return destination


def _remove_copy(destination: Path, path: Path) -> None:
    """Remove the copy of a file the entry no longer commits, and the folders left empty by it."""
    (destination / path).unlink()
    for parent in (destination / path).parents:
        if parent == destination:
            break
        try:
            parent.rmdir()
        except OSError:
            break


def _refuse_copies_edited_in_place(
    entry: DatasetEntry, destination: Path, layer: dict[Path, bytes], copies: dict[Path, bytes]
) -> None:
    """Refuse copies of the entry's Gym layer that are neither the entry's files nor the ones the last pull recorded."""
    recorded = Manifest.read(destination)
    if recorded is None:
        # Nothing pulled yet: a first fetch that failed before claiming the folder left only the copies behind, which
        # follow the entry's files. Task folders without a manifest were placed by hand, and so may the files beside
        # them have been, so those copies are kept.
        if not task_folders(destination):
            return
    elif (dataset_toml_digest(destination), dataset_files_digest(destination)) == (
        recorded.dataset_toml_digest,
        recorded.dataset_files_digest,
    ):
        return
    differing = sorted(path for path in set(layer) | set(copies) if layer.get(path) != copies.get(path))
    if len(differing) == 1:
        raise HubError(
            f"{destination / differing[0]} was edited in place and differs from the entry's "
            f"{entry.folder / differing[0]}; edit the entry's file instead, or move the copy aside and run again"
        )
    names = ", ".join(str(destination / path) for path in differing)
    raise HubError(
        f"{names} were edited in place and differ from the entry's files under {entry.folder}; edit the entry's files "
        "instead, or move the copies aside and run again"
    )
