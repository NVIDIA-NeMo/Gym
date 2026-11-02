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

An entry is pulled into its own folder when that folder is under the working root (:data:`nemo_gym.WORKING_DIR`:
the Gym checkout of an editable install, the working directory of a package install) and is not installed. An
installed entry, one a package install ships under ``site-packages`` or another Python installation prefix
(:func:`_installed_entry`; shared, often read-only, and under the working root too when the project keeps its venv
there), and an entry committed outside the working root (one a ``NEMO_GYM_EXTRA_ROOTS`` root commits) are pulled
into ``<datasets dir>/<entry>/`` (:func:`~nemo_gym.tasks.harbor.hub.datasets_dir`, the writable default every other
pull uses), and ``NEMO_GYM_DATASETS_DIR`` moves every pulled dataset there (an entry without a source is never
pulled, so it stays at its folder). An entry's tasks then live in ``<datasets dir>/<entry>/`` with a copy of the
entry's Gym layer beside them, its ``dataset.toml`` and whatever dataset-level ``task_data.py`` or ``verifier.py`` it
commits (:func:`gym_layer`), kept current with the entry's files (:func:`prepare_destination`), so the files that
shape the tasks, and the dataset digest that covers them, are the same wherever the pull landed. That copy folder,
told by the :data:`LAYER_RECORD_FILE` the placing writes beside the copies, is never itself an entry: lookup by name
and by reference (:func:`find_entry`, :func:`committed_entries`) skips it and goes on to the root that commits the
entry, so the pull keeps following the entry's files, while a folder of the entry's name a user committed in an
earlier root (no record file) keeps precedence, as any shadowing component does.
"""

import hashlib
import json
import os
import re
import shutil
import site
import sysconfig
from dataclasses import dataclass
from pathlib import Path

from nemo_gym import WORKING_DIR, component_search_roots
from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE, read_dataset_config
from nemo_gym.tasks.harbor.hub import (
    DATASETS_DIR_ENV,
    REGISTRY_SOURCE,
    HubError,
    canonical_reference,
    datasets_dir,
    is_hub_ref,
)
from nemo_gym.tasks.harbor.manifest import DATASET_FILES, MANIFEST_FILE, Manifest, ManifestError, refresh_gym_layer
from nemo_gym.tasks.harbor.package_store import PACKAGE_STORE_SOURCE


ENTRIES_SUBDIR = Path("environments") / "harbor"
# An entry is named by its folder, one path component; a hub folder name (`terminal-bench-4.0.0`) fits too, which
# is fine: such a folder commits no dataset.toml with a source and is never an entry a pull resolves.
_ENTRY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# The files of an entry's Gym layer beside the task folders: what enters the
# dataset digest besides the tasks, and so what a destination elsewhere needs a copy of.
LAYER_FILES = (DATASET_FILE, *DATASET_FILES)
# The record of the copies :func:`prepare_destination` placed at a destination elsewhere: the digest of each copy as
# placed, in a hidden file beside them (hidden, so task discovery never looks at it) that nothing else writes. The
# folder's manifest.toml is no such record: every command that prepares the folder as a plain path rewrites its
# Gym-layer digests over whatever the copies hold.
LAYER_RECORD_FILE = ".gym-layer.json"
# The manifests a hub fetch writes; any other one beside differing copies says the folder is not the entry's pull.
_FETCH_SOURCES = (PACKAGE_STORE_SOURCE, REGISTRY_SOURCE)


@dataclass(frozen=True)
class DatasetEntry:
    """A committed entry: its name (the folder's), its folder and the ``[gym] source`` it declares, if any."""

    name: str
    folder: Path
    source: str | None

    @property
    def destination(self) -> Path:
        """Where the entry's tasks are pulled to and read from: the entry's own folder when it is under the working
        root (:data:`nemo_gym.WORKING_DIR`) and not installed, else ``<datasets dir>/<name>/``
        (:func:`nemo_gym.tasks.harbor.hub.datasets_dir`). An installed entry, one a package install ships under
        ``site-packages`` or another installation prefix (:func:`_installed_entry`), falls back to the datasets dir
        wherever the working root is, a project-local ``.venv`` under it included, since the installation may be
        shared or read-only and is in any case not where the user's pulls go; so does an entry committed outside the
        working root (one a ``NEMO_GYM_EXTRA_ROOTS`` root commits), and ``NEMO_GYM_DATASETS_DIR`` moves every pull
        there, the working root's entries included. The name is taken
        by the first root that commits it (:func:`find_entry` order), so the fallback folder is another committed
        entry's own folder (the shadowed one's) only when the same name is committed twice, and that case is the
        user's to resolve. An entry without a source has no pull to move: its tasks are wherever its folder holds
        them (a checkout the user placed there), so it stays at its folder and resolves by name as it does by path,
        wherever it is and whether or not the variable is set."""
        if self.source is None:
            return self.folder
        if os.environ.get(DATASETS_DIR_ENV) or _installed_entry(self.folder) or not _under_working_root(self.folder):
            return datasets_dir() / self.name
        return self.folder


def _under_working_root(folder: Path) -> bool:
    """Whether ``folder`` lies under :data:`nemo_gym.WORKING_DIR`, the root whose ``environments/harbor/`` is the
    default datasets dir (the checkout of an editable install, the working directory of a package install). A
    project-local ``.venv`` lies under it too, which is why :func:`_installed_entry` is asked first."""
    return _under(folder, Path(WORKING_DIR))


def _installed_entry(folder: Path) -> bool:
    """Whether ``folder`` is installed: it lies under a Python installation prefix, so it is a package's, shared and
    often read-only, and never where a user's pulls go, whatever the working directory. Installed is a path with a
    ``site-packages`` or ``dist-packages`` component (as given or resolved), or one under the running interpreter's
    ``purelib``/``platlib`` (:func:`sysconfig.get_paths`), its site-packages folders (:func:`site.getsitepackages`) or
    its user site-packages (:func:`site.getusersitepackages`)."""
    given = Path(folder)
    resolved = given.resolve()
    if any(part in ("site-packages", "dist-packages") for path in (given, resolved) for part in path.parts):
        return True
    return any(_under(resolved, prefix) for prefix in _install_prefixes())


def _install_prefixes() -> list[Path]:
    """The running interpreter's installation prefixes: ``purelib`` and ``platlib``, its site-packages folders and
    the user site-packages (a venv's ``site`` module may lack ``getsitepackages``: then the first two stand)."""
    paths = sysconfig.get_paths()
    candidates = [paths.get("purelib"), paths.get("platlib")]
    getsitepackages = getattr(site, "getsitepackages", None)
    if getsitepackages is not None:
        candidates.extend(getsitepackages())
    candidates.append(site.getusersitepackages())
    return [Path(candidate) for candidate in candidates if candidate]


def _under(folder: Path, root: Path) -> bool:
    """Whether ``folder`` resolves to ``root`` or a path under it (``root`` resolved too)."""
    try:
        Path(folder).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True


def _pull_copy(folder: Path) -> bool:
    """Whether ``folder`` is the copy a pull placed at ``<datasets dir>/<entry>/`` for an entry elsewhere, told by
    the :data:`LAYER_RECORD_FILE` only :func:`prepare_destination` writes, beside the copies of the entry's Gym layer.
    Such a folder is never itself an entry: the entry is in a later root, and lookup goes on to it."""
    return (Path(folder) / LAYER_RECORD_FILE).is_file()


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


def _committed_folder(name: str) -> Path | None:
    """The folder ``environments/harbor/<name>`` of the first search root that has one, or ``None``.

    The copy a pull placed there for an entry committed in a later root (:func:`_pull_copy`: the working root's
    ``<datasets dir>/<name>/`` of an installed entry, after its first pull) is not that folder: it is skipped and the
    search goes on, so the name keeps resolving to the entry whose files the copy follows.
    """
    if not _ENTRY_NAME.fullmatch(name):
        return None
    for root in _roots():
        folder = root / ENTRIES_SUBDIR / name
        if folder.is_dir() and not _pull_copy(folder):
            return folder
    return None


def find_entry(name: str) -> DatasetEntry | None:
    """The entry called ``name`` in the first search root that commits one, or ``None``.

    An entry need not declare a source: one whose tasks come from elsewhere (a checkout the user points
    ``gym eval run`` at) still resolves by name to its folder. A ``dataset.toml`` that does not read raises
    ``ValueError`` naming the file, as prepare would for the folder.
    """
    folder = _committed_folder(name)
    return _entry(folder) if folder is not None else None


def committed_entries() -> list[DatasetEntry]:
    """Every entry that declares a source, across the search roots; an earlier root wins a name.

    A name is taken by the first root that has a folder of it, as :func:`find_entry` resolves it, whether or
    not that folder declares a source: a source-less ``tb4`` in an earlier root shadows a sourced one in a later
    root for the name, so it shadows it for the reference too, and the pull of that reference lands in the hub's
    own folder rather than in an entry the name no longer reaches. The copy a pull placed for an entry of a later
    root (:func:`_pull_copy`) takes no name: it is skipped as :func:`find_entry` skips it, so the installed entry it
    follows keeps the name and the reference. Only folders with a ``dataset.toml`` are read
    (a pulled hub folder has none unless its user wrote one) and one whose table does not read is skipped here:
    the mapping must not fail every pull over a stray file, and the entry's own pull reports it.
    """
    found: dict[str, DatasetEntry] = {}
    taken: set[str] = set()
    for root in _roots():
        parent = root / ENTRIES_SUBDIR
        if not parent.is_dir():
            continue
        for folder in sorted(parent.iterdir()):
            if folder.name in taken or folder.name.startswith(".") or not folder.is_dir() or _pull_copy(folder):
                continue
            taken.add(folder.name)
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

    The two are compared as the fetchers record them (:func:`~nemo_gym.tasks.harbor.hub.canonical_reference`), so a
    digest with or without its ``sha256:`` prefix, or ``harbor:org/name`` with or without ``@latest``, finds the entry
    that declares the other spelling instead of landing beside it in the hub's own folder. ``harbor:org/name`` (the
    store's ``latest``) and ``harbor:org/name@4.0.0`` are different references, so a plain reference never lands in
    an entry pinned to a version. Two entries declaring one source is a mistake in the checkout and is refused with
    both named.
    """
    ref = canonical_reference(target)
    matches = [entry for entry in committed_entries() if canonical_reference(entry.source) == ref]
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


def entry_at(folder: Path) -> DatasetEntry | None:
    """The committed entry whose pull ``folder`` holds when that is not the entry's own folder:
    ``<datasets dir>/<name>/`` for the entry called ``name`` under a moved ``NEMO_GYM_DATASETS_DIR``, for an installed
    entry (one a package install ships) or for an entry committed outside the working root, or ``None`` for every
    other folder (an entry's own folder included: nothing is copied there). The folder is looked up as the name is
    (:func:`_committed_folder`), so the copy itself, once placed, is passed over for the entry it follows and the path
    form agrees with the name form. A path target naming such a folder is prepared with the entry's
    Gym layer kept current beside the tasks, or refused, exactly as the pull and the entry's name are
    (:func:`prepare_destination`), so the three commands agree on the folder and an edited copy is refused in the
    same words by each. An unreadable ``dataset.toml`` of the entry raises ``ValueError`` naming the file, as
    :func:`find_entry` does: it is the table the folder's copy follows. The folder's own table is never read here:
    the folder is compared with the first root's ``environments/harbor/<name>`` before any entry is read, so a plain
    folder under the datasets dir (every pulled hub folder of an editable checkout, whose datasets dir is that root's
    ``environments/harbor/``) is nobody's pull elsewhere, and a ``dataset.toml`` written into it that does not read is
    reported by the prepare that follows, as it was before.
    """
    folder = Path(folder).resolve()
    if folder.parent != datasets_dir():
        return None
    committed = _committed_folder(folder.name)
    if committed is None or committed.resolve() == folder:
        return None
    entry = _entry(committed)
    if entry.destination != folder:
        return None
    return entry


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

    The entry's own folder needs nothing. A destination elsewhere (``NEMO_GYM_DATASETS_DIR``, an installed entry or
    one committed outside the working root, :attr:`DatasetEntry.destination`) holds a copy of the entry's Gym layer
    (:func:`gym_layer`: ``dataset.toml``, a dataset-level ``task_data.py`` or ``verifier.py``), so the files that
    shape the tasks and the digest that covers them are the entry's wherever the pull landed: the copies are placed
    before the first task lands (``create`` makes the folder; a prepare, which fetches nothing, passes ``False`` and
    leaves a missing folder missing) and are replaced, a file the entry no longer commits removed, when the entry's
    files changed since, as long as the copies are still the ones on record:
    the ones this function placed, whose digests it writes to :data:`LAYER_RECORD_FILE` beside them as it places them
    (:func:`_record_copies`; the record is brought up to date whenever the copies are found to be the entry's files).
    Only this function writes that record, so no other command can vouch for an edited copy: the manifest's own
    Gym-layer digests are rewritten by a prepare of the folder by its path in a shell without
    ``NEMO_GYM_DATASETS_DIR``, which cannot know the folder is an entry's pull, and were once what the copies were
    checked against, so such a prepare absorbed an edit of the copy and the next pull overwrote it.
    Whenever the copies are replaced the manifest is brought up to date with them
    (:func:`~nemo_gym.tasks.harbor.manifest.refresh_gym_layer`), so its digests follow the copies even when nothing
    rewrites the manifest afterwards (a registry fetch with every folder in place, a fetch that fails, a prepare
    refused before its manifest check). A copy that matches neither the entry's file nor the record was edited in
    place and is refused, never overwritten, whatever the folder's manifest says: the entry's files are where the
    layer is edited. ``gym dataset pull <entry>``, ``gym dataset validate <entry>`` and ``gym eval run <entry>`` all
    come through here, so they agree on the folder.
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
        _record_copies(destination, layer)
        return destination
    if copies:
        _refuse_copies_edited_in_place(entry, destination, layer, copies)
    for path in sorted(set(copies) - set(layer)):
        _remove_copy(destination, path)
    for path, content in layer.items():
        if copies.get(path) != content:
            (destination / path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(entry.folder / path, destination / path)
    _record_copies(destination, layer)
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


def _layer_digests(files: dict[Path, bytes]) -> dict[str, str]:
    """The digest of each Gym-layer file by its relative path, as the record of the copies holds them."""
    return {path.as_posix(): hashlib.sha256(content).hexdigest() for path, content in files.items()}


def _record_copies(destination: Path, layer: dict[Path, bytes]) -> None:
    """Record the copies of ``layer`` now at ``destination`` in :data:`LAYER_RECORD_FILE`, when the record differs.

    Written whole and renamed into place, so an interrupted write leaves the previous record, not a torn one.
    """
    text = json.dumps(_layer_digests(layer), indent=2, sort_keys=True) + "\n"
    record = destination / LAYER_RECORD_FILE
    if _recorded_copies(destination) == _layer_digests(layer):
        return
    staged = record.with_name(record.name + ".tmp")
    staged.write_text(text, encoding="utf-8")
    staged.replace(record)


def _recorded_copies(destination: Path) -> dict[str, str] | None:
    """The digests :data:`LAYER_RECORD_FILE` at ``destination`` holds, or ``None`` without a readable record."""
    record = destination / LAYER_RECORD_FILE
    if not record.is_file():
        return None
    try:
        data = json.loads(record.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    return {str(path): digest for path, digest in data.items() if isinstance(digest, str)}


def _refuse_copies_edited_in_place(
    entry: DatasetEntry, destination: Path, layer: dict[Path, bytes], copies: dict[Path, bytes]
) -> None:
    """Refuse copies of the entry's Gym layer that are neither the entry's files nor the ones this module placed.

    The copies are at a destination elsewhere than the entry's folder (``NEMO_GYM_DATASETS_DIR``, or an entry
    committed outside the working root, :attr:`DatasetEntry.destination`). The record of the placed copies is
    :data:`LAYER_RECORD_FILE` beside them (:func:`_record_copies`), which only :func:`prepare_destination` writes:
    copies whose digests it holds, every one of them, were placed there and may be replaced. The folder's
    ``manifest.toml`` is no such record, not even one a hub fetch wrote: a prepare of the folder by its path in a
    shell without ``NEMO_GYM_DATASETS_DIR`` rewrites its Gym-layer digests over an edited copy, as it does for any
    dataset whose own ``dataset.toml`` changed, so a copy checked against those digests read as the one on record and
    the next pull overwrote the edit. A ``local`` manifest beside the differing copies is named as such: it is
    prepare's record of a dataset the user put at ``<datasets dir>/<entry>/`` by hand and ran by its path, before
    ``NEMO_GYM_DATASETS_DIR`` or an entry committed outside the working root moved the entry's pull there, or in a
    shell without the variable, so the folder is not the entry's pull at all; the refusal says which case it is and
    what to do (move the folder aside; with the variable, running it by its path without it is the other way), and
    either way nothing is overwritten.
    """
    recorded = _recorded_copies(destination)
    if recorded is not None and all(recorded.get(path) == digest for path, digest in _layer_digests(copies).items()):
        return
    differing = sorted(path for path in set(layer) | set(copies) if layer.get(path) != copies.get(path))
    names = ", ".join(str(destination / path) for path in differing)
    manifest = Manifest.read(destination)
    if manifest is not None and manifest.source not in _FETCH_SOURCES:
        if os.environ.get(DATASETS_DIR_ENV):
            where = f"under {DATASETS_DIR_ENV}"
            aside = f"move the folder aside, or run it by its path with {DATASETS_DIR_ENV} unset"
        elif _installed_entry(entry.folder):
            where = f"(the entry is installed, under {entry.folder.parent})"
            aside = "move the folder aside"
        else:
            where = f"(the entry is committed outside the working root, under {entry.folder.parent})"
            aside = "move the folder aside"
        raise HubError(
            f"{destination} is where entry {entry.name}'s pull lands {where}, but its {MANIFEST_FILE} "
            f"records a {manifest.source} dataset, not a fetch, and {names} differ{'s' if len(differing) == 1 else ''} "
            f"from the entry's files under {entry.folder}; it is not the entry's pull: {aside}"
        )
    if len(differing) == 1:
        raise HubError(
            f"{destination / differing[0]} was edited in place and differs from the entry's "
            f"{entry.folder / differing[0]}; edit the entry's file instead, or move the copy aside and run again"
        )
    raise HubError(
        f"{names} were edited in place and differ from the entry's files under {entry.folder}; edit the entry's files "
        "instead, or move the copies aside and run again"
    )
