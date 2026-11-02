# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``manifest.toml``: the derived identity of a dataset folder.

The file sits beside the task folders and holds only facts computed from them: where the folders
came from (``source``), one dataset digest, the digest of ``dataset.toml`` alone, and per task the
folder digest (the value rows carry as
``_ng_digest``) with its upstream pin. The dataset's name and version live only in ``dataset.toml``, so nothing
appears in both files; what a fetch records besides is the hub reference it was made from (``ref``), a fact of the
fetch like the pins, which is what a committed entry's declared ``source`` is checked against without the hub
(:func:`nemo_gym.tasks.harbor.cli._check_fetch_record`). Nobody edits the manifest by hand.

The output is deterministic: a fixed key order, sorted tasks and no timestamps, so equal manifests
mean identical inputs and two runs compare by ``[dataset].digest``. The dataset digest is a hash over
the sorted per-task digests, then the ``dataset.toml`` digest, then, when the
dataset ships them, the digest of the dataset-level Gym files that shape every task (``task_data.py``, the
row model, and ``verifier.py``, the shared grader; ``dataset_files_digest``, recorded only when present, so a
dataset without them hashes and reads as before). Two datasets that differ only in a timeout override in
``dataset.toml``, or only in the shared grader, are therefore different datasets, on purpose: either can
change a score. Like ``dataset.toml``, those files are Gym's layer, not fetched content: an edit rewrites the
manifest and is reported, while an edited task folder of a fetched dataset is refused.

Fetchers write the manifest (:mod:`nemo_gym.tasks.harbor.hub`, :mod:`nemo_gym.tasks.harbor.package_store`)
and record only the tasks they fetched, so a folder nobody fetched is never blessed; prepare writes it for a
local folder, unless git tracks the folder (the repository's own example datasets), whose identity is
computed in memory. Prepare recomputes the per-task digests from the folders and refuses to run a fetched
dataset whose folders no longer match (:func:`check_manifest`). A task folder that cannot be hashed (a file
under it escapes it through a link, or cannot be read) is recorded by name without a digest and does not
enter the dataset digest: the loader refuses the same folder, so no row runs it, and the rest of a local
dataset still prepares; a fetched manifest that recorded its digest names it when it refuses.
"""

import errno
import hashlib
import json
import logging
import os
import re
import secrets
import subprocess
import threading
import tomllib
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from nemo_gym.tasks.harbor import task as harbor_task
from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE
from nemo_gym.tasks.harbor.digest import content_hash
from nemo_gym.tasks.harbor.python_verifier import VERIFIER_FILE
from nemo_gym.tasks.harbor.task import iter_task_folders
from nemo_gym.tasks.harbor.task_data import TASK_DATA_MODULE


logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.toml"
MANIFEST_SCHEMA_VERSION = 1
# The Gym files beside dataset.toml that shape every task: the row model and the dataset-level grader.
DATASET_FILES = (TASK_DATA_MODULE, VERIFIER_FILE)
TASK_FILE = "task.toml"
# Where the task folders came from. `local` is a folder prepare met on disk with no fetch record.
SOURCES = ("harbor-registry", "harbor-package-store", "git", "local")
LOCAL_SOURCE = "local"
# The per-task keys, in the order they are written: the digest, then the upstream pin of a registry or
# git source (`git_url`, `git_commit_id`, `path`) or of a package-store source (`package`, `content_hash`).
TASK_KEYS = ("digest", "git_url", "git_commit_id", "path", "package", "content_hash")
# The sources whose fetcher recorded a digest per landed task from its first version on. Before the schema was
# numbered, the registry fetcher wrote pins only, while the package-store fetcher wrote `[dataset]` with no digest
# and per task `digest = "sha256:<hex>"` once the folder landed: a manifest in that format records content, and an
# entry of it without a digest is a download the fetch never finished, not a pin from before digests existed.
DIGEST_RECORDING_SOURCES = ("harbor-package-store",)
_EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()
# What os.link raises on a filesystem without hard links (vfat, some FUSE and SMB mounts).
_NO_HARD_LINKS = {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EACCES}
# The UTF-16 surrogate pair json.dumps writes for a character above U+FFFF.
_SURROGATE_PAIR = re.compile(r"\\u(d[89ab][0-9a-f]{2})\\u(d[c-f][0-9a-f]{2})")


class ManifestError(ValueError):
    """The manifest cannot be read, or the task folders no longer match it."""


class UnsupportedManifestError(ManifestError):
    """The manifest was written by a newer Gym (a ``schema_version`` this Gym does not read).

    Distinct from a malformed manifest: the file is intact and says more than this Gym understands,
    so a fetcher must leave it alone rather than rewrite it with what it knows.
    """


def _quote(value: str) -> str:
    """``value`` as a TOML basic string, ASCII only.

    JSON string escapes are valid TOML basic-string escapes (the quote, the backslash, the controls and
    U+007F as ``\\u007f``, other non-ASCII as ``\\uXXXX``) except that JSON writes a character above U+FFFF
    as a UTF-16 surrogate pair, which TOML forbids; that pair becomes TOML's ``\\UXXXXXXXX``.
    """

    def scalar(match: re.Match) -> str:
        high, low = int(match[1], 16), int(match[2], 16)
        return "\\U%08x" % (0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))

    return _SURROGATE_PAIR.sub(scalar, json.dumps(value))


def _file_digest(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def task_folders(folder: Path) -> list[Path]:
    """The task folders under ``folder``, as the loader discovers them (:func:`iter_task_folders`: recursively,
    hidden folders skipped, links not followed), sorted by path. A task is named by its folder's name there too."""
    return iter_task_folders(folder)


def dataset_toml_digest(folder: Path) -> str:
    """Digest of ``dataset.toml`` alone, or of nothing when the file is absent."""
    path = Path(folder) / DATASET_FILE
    return _file_digest(path) if path.is_file() else _EMPTY_DIGEST


def dataset_files_digest(folder: Path) -> str:
    """Digest of the dataset-level Gym files that shape tasks (:data:`DATASET_FILES`): name and content of each one
    present, in order; the digest of nothing when the dataset ships none (every hub dataset)."""
    root = Path(folder)
    digest = hashlib.sha256()
    for name in DATASET_FILES:
        if (root / name).is_file():
            digest.update(f"{name}\0{_file_digest(root / name)}\n".encode())
    return digest.hexdigest()


def dataset_digest(task_digests: dict[str, str], dataset_toml: str, dataset_files: str = _EMPTY_DIGEST) -> str:
    """The dataset's identity: sorted per-task digests, then ``dataset.toml``, then the dataset-level files when
    there are any (``dataset_files`` other than the digest of nothing)."""
    digest = hashlib.sha256()
    for name in sorted(task_digests):
        digest.update(f"{name}\0{task_digests[name]}\n".encode())
    digest.update(f"{DATASET_FILE}\0{dataset_toml}\n".encode())
    if dataset_files != _EMPTY_DIGEST:
        digest.update(f"dataset-files\0{dataset_files}\n".encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class Manifest:
    """``manifest.toml`` in memory. ``tasks`` maps a task name to its ``digest`` and pin keys."""

    source: str
    digest: str
    dataset_toml_digest: str
    tasks: dict[str, dict[str, str]] = field(default_factory=dict)
    # The dataset-level Gym files (:data:`DATASET_FILES`); the digest of nothing, and not written, without them.
    dataset_files_digest: str = _EMPTY_DIGEST
    # The hub reference the fetch was made from: ``harbor:<org>/<name>@<tag | latest | sha256:digest>`` for the
    # package store, ``harbor:<dataset>@<version>`` with the version it resolved for the registry; ``None``, and not
    # written, for a local folder.
    ref: str | None = None
    # The task folders :meth:`build` could not hash (recorded by name, no digest): computed, never written or read.
    unhashable: frozenset[str] = field(default_factory=frozenset)

    @property
    def task_digests(self) -> dict[str, str | None]:
        """Per-task folder digests; ``None`` for a task pinned before its folder was written or one that cannot be hashed."""
        return {name: entry.get("digest") for name, entry in self.tasks.items()}

    @property
    def pins(self) -> dict[str, dict[str, str]]:
        """The per-task entries without their digests: the upstream pins alone."""
        return {name: {k: v for k, v in entry.items() if k != "digest"} for name, entry in self.tasks.items()}

    @classmethod
    def build(
        cls,
        folder: Path,
        *,
        source: str,
        pins: dict[str, dict[str, str]] | None = None,
        digests: dict[str, str] | None = None,
        names: Iterable[str] | None = None,
        ref: str | None = None,
    ) -> "Manifest":
        """Compute the manifest of ``folder`` from its task folders and ``dataset.toml``.

        ``pins`` are the upstream pins to record per task; a pinned task whose folder is absent
        (an interrupted fetch) keeps its pin and gets no digest. ``digests`` are folder digests
        already computed (by a loader or a fetch); any task folder not in it is hashed here, and the
        mapping is extended in place so a caller can reuse the work. ``ref`` is the hub reference a fetch
        was made from, recorded as the pins are and kept by every rewrite of the manifest.

        ``names`` are the tasks to record. A fetcher passes the tasks it fetched, so a folder it did not
        fetch stays unrecorded (and prepare refuses it as not in the manifest) and no folder is scanned
        or stat-ed again for a name whose digest is known. Without it, every task folder on disk is
        recorded: a local dataset, or prepare recomputing a fetched one to compare. A folder found there
        that cannot be hashed is recorded by name with no digest (``unhashable``) and left out of the dataset
        digest, with a warning; the loader refuses it per task, so the rest of the dataset still prepares.
        """
        if source not in SOURCES:
            raise ValueError(f"Unknown manifest source {source!r}; expected one of {', '.join(SOURCES)}")
        folder = Path(folder)
        digests = digests if digests is not None else {}
        present: dict[str, str] = {}
        unhashable: set[str] = set()
        if names is None:
            for task in task_folders(folder):
                if task.name not in digests:
                    try:
                        digests[task.name] = content_hash(task)
                    except (OSError, ValueError) as exc:
                        logger.warning(
                            "Task %s cannot be hashed (%s); recorded by name without a digest", task.name, exc
                        )
                        unhashable.add(task.name)
                        continue
                present[task.name] = digests[task.name]
        else:
            for name in names:
                if name not in digests:
                    if not (folder / name / TASK_FILE).is_file():
                        continue
                    digests[name] = content_hash(folder / name)
                present[name] = digests[name]
        tasks: dict[str, dict[str, str]] = {}
        for name in sorted(set(present) | set(pins or {}) | unhashable):
            entry: dict[str, str] = {}
            if name in present:
                entry["digest"] = present[name]
            entry.update({k: v for k, v in (pins or {}).get(name, {}).items() if k in TASK_KEYS and k != "digest"})
            tasks[name] = entry
        toml_digest = dataset_toml_digest(folder)
        files_digest = dataset_files_digest(folder)
        return cls(
            source=source,
            digest=dataset_digest(present, toml_digest, files_digest),
            dataset_toml_digest=toml_digest,
            tasks=tasks,
            dataset_files_digest=files_digest,
            ref=ref,
            unhashable=frozenset(unhashable),
        )

    @classmethod
    def read(cls, folder: Path) -> "Manifest | None":
        """The manifest beside the task folders, ``None`` when there is none; malformed raises."""
        path = Path(folder) / MANIFEST_FILE
        if not path.is_file():
            return None
        try:
            data = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ManifestError(f"{path} is not a valid manifest: {exc}") from exc
        # A manifest written before the schema was numbered (pins only) has no version line and reads as 1.
        schema_version = data.get("schema_version", MANIFEST_SCHEMA_VERSION)
        # An int, exactly: a bool or the float 1.0 is not a version this Gym wrote.
        if type(schema_version) is not int or schema_version != MANIFEST_SCHEMA_VERSION:
            raise UnsupportedManifestError(
                f"{path} has schema_version {schema_version!r}; this Gym reads {MANIFEST_SCHEMA_VERSION}. It was "
                "written by a newer Gym: upgrade Gym, or delete the folder, or move it aside, and run again"
            )
        dataset = data.get("dataset", {})
        tasks = data.get("tasks", {})
        if not isinstance(dataset, dict) or not isinstance(tasks, dict):
            raise ManifestError(f"{path} is not a valid manifest: [dataset] and [tasks] must be tables")
        ref = dataset.get("ref")
        return cls(
            source=str(dataset.get("source", "")),
            digest=str(dataset.get("digest", "")),
            dataset_toml_digest=str(dataset.get("dataset_toml_digest", "")),
            dataset_files_digest=str(dataset.get("dataset_files_digest", _EMPTY_DIGEST)),
            ref=ref if isinstance(ref, str) and ref else None,
            tasks={
                str(name): {
                    # The package-store fetcher wrote `digest = "sha256:<hex>"` before the schema was numbered; the
                    # digest is bare everywhere else (`_ng_digest`, `content_hash`), so it reads bare. Pins keep theirs.
                    k: (v.removeprefix("sha256:") if k == "digest" else str(v))
                    for k, v in entry.items()
                    if k in TASK_KEYS and isinstance(v, str)
                }
                for name, entry in tasks.items()
                if isinstance(entry, dict)
            },
        )

    @property
    def records_content(self) -> bool:
        """Whether the manifest records what the task folders held, so a pin without a digest is unfinished business.

        One with a dataset digest does, and so does one written before the dataset digest existed by a fetcher
        that recorded a digest per landed task from the start (:data:`DIGEST_RECORDING_SOURCES`), whether or not
        a task had landed when it was written; a registry manifest from before digests has pins only, and
        :func:`check_manifest` upgrades it by hashing the folders it finds.
        """
        return (
            bool(self.digest)
            or self.source in DIGEST_RECORDING_SOURCES
            or any("digest" in entry for entry in self.tasks.values())
        )

    def text(self) -> str:
        """The file's bytes: fixed key order, sorted tasks, no timestamps."""
        lines = [
            f"schema_version = {MANIFEST_SCHEMA_VERSION}",
            "",
            "[dataset]",
            f"source = {_quote(self.source)}",
        ]
        if self.ref is not None:
            lines.append(f"ref = {_quote(self.ref)}")
        lines.append(f"digest = {_quote(self.digest)}")
        lines.append(f"dataset_toml_digest = {_quote(self.dataset_toml_digest)}")
        if self.dataset_files_digest != _EMPTY_DIGEST:
            lines.append(f"dataset_files_digest = {_quote(self.dataset_files_digest)}")
        lines.append("")
        for name in sorted(self.tasks):
            lines.append(f"[tasks.{_quote(name)}]")
            entry = self.tasks[name]
            lines.extend(f"{key} = {_quote(entry[key])}" for key in TASK_KEYS if key in entry)
            lines.append("")
        return "\n".join(lines)

    def write(self, folder: Path, *, exclusive: bool = False) -> Path:
        """Write ``manifest.toml`` atomically: a failed write leaves the previous, truthful file intact.

        The file keeps the permissions it had, so a shared dataset another user could read stays
        readable after a rewrite; a new manifest gets the usual umask-aware mode (``0o666`` before
        the umask, like ``Path.write_text``), not the owner-only mode a ``mkstemp`` file would have.

        With ``exclusive``, the manifest is created only where none is and :class:`FileExistsError` is
        raised otherwise, the target untouched (the temporary is written and removed either way): the
        complete file is linked into place, so of two writers racing for an empty folder exactly one
        succeeds and a reader never sees a partial file. A filesystem without hard links falls back to
        an exclusive create of the target itself, still one winner, though a reader may then glimpse
        the file before its last byte. A fetcher claims a folder this way before its first task lands.
        """
        folder = Path(folder)
        target = folder / MANIFEST_FILE
        try:
            mode: int | None = target.stat().st_mode & 0o777
        except FileNotFoundError:
            mode = None
        # Unique per writer, so concurrent fetches (processes or threads) of one folder never share it.
        temporary = folder / f".manifest-{os.getpid()}-{threading.get_ident()}-{secrets.token_hex(4)}.tmp"
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        try:
            self._write_text(fd)
            if mode is not None:
                os.chmod(temporary, mode)
            if not exclusive:
                os.replace(temporary, target)
                return target
            try:
                # A hard link lands only where nothing is yet: the atomic create-unless-present os.replace lacks.
                os.link(temporary, target)
            except OSError as exc:
                if exc.errno not in _NO_HARD_LINKS:
                    raise
                self._write_text(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
        finally:
            try:
                os.unlink(temporary)
            except OSError:
                pass
        return target

    def _write_text(self, fd: int) -> None:
        """Write the file's bytes to ``fd`` and flush them to disk; the descriptor is closed."""
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(self.text())
            handle.flush()
            os.fsync(handle.fileno())


def task_differences(
    recorded: dict[str, str | None], on_disk: dict[str, str | None], *, unhashable: Iterable[str] = ()
) -> str | None:
    """One line naming the tasks whose digest differs between a manifest and the folders, or ``None``.

    A task is ``missing`` when the manifest recorded its digest and the folder is gone (``on_disk`` lacks
    it, or maps it to ``None`` because its pin outlived the folder), ``changed`` when both digests exist
    and differ, and ``not in the manifest`` when only the folder exists. One of ``unhashable`` (its folder
    is there but has no digest now, see :meth:`Manifest.build`) whose digest the manifest recorded is
    ``cannot be hashed``, not missing.
    """
    unhashed = sorted(name for name in set(unhashable) & set(recorded) if recorded[name] is not None)
    missing = sorted(
        name for name in recorded if recorded[name] is not None and on_disk.get(name) is None and name not in unhashed
    )
    changed = sorted(
        name
        for name in set(recorded) & set(on_disk)
        if recorded[name] != on_disk[name] and name not in missing and name not in unhashed
    )
    unrecorded = sorted(set(on_disk) - set(recorded))
    if not (changed or unhashed or missing or unrecorded):
        return None

    parts = []
    if changed:
        parts.append(f"changed: {_some(changed)}")
    if unhashed:
        parts.append(f"cannot be hashed: {_some(unhashed)}")
    if missing:
        parts.append(f"missing: {_some(missing)}")
    if unrecorded:
        parts.append(f"not in the manifest: {_some(unrecorded)}")
    return "; ".join(parts)


def _some(names: list[str]) -> str:
    """The first five of ``names`` and how many there are in all."""
    return ", ".join(names[:5]) + (f", ... ({len(names)} in all)" if len(names) > 5 else "")


def tracked_by_git(folder: Path) -> bool:
    """Whether git tracks files under ``folder``: the repository's own example datasets, or a checkout.

    Such a folder is under version control already, so prepare computes its identity in memory
    instead of leaving a derived ``manifest.toml`` in the working tree. No git, no repository or any
    other failure reads as not tracked.
    """
    try:
        completed = subprocess.run(
            ["git", "-C", str(folder), "ls-files", "-z", "--", "."], capture_output=True, timeout=30, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and bool(completed.stdout.strip(b"\0"))


def _write_or_warn(manifest: "Manifest", folder: Path) -> bool:
    """Write the manifest; a folder that cannot be written (read-only shared storage) keeps its in-memory identity."""
    try:
        manifest.write(folder)
    except OSError as exc:
        logger.warning(
            "Could not write %s (%s); dataset digest %s is recorded in the run config and the rows only",
            folder / MANIFEST_FILE,
            exc,
            manifest.digest[:12],
        )
        return False
    return True


def enclosing_manifest(path: Path, *, levels: int) -> tuple[Path, Manifest] | None:
    """The nearest fetched manifest above ``path`` whose dataset folder's check reaches ``path``, with its folder.

    A dataset folder's check (:func:`check_manifest`) sees the task folders :func:`iter_task_folders` discovers: at
    most :data:`MAX_DISCOVERY_DEPTH` levels down, never inside a hidden folder, never through a symbolic link, never
    inside a task folder it found (a task's own files are its content, not more tasks). So
    the folders whose manifest can hold ``path`` are its nearest ``levels`` parents, reached from ``path`` through
    visible folders that are not links or task folders; a manifest further up, or above a hidden, linked or task
    folder, belongs to a
    dataset whose check never sees ``path`` and is not read. Only a manifest this Gym reads as a fetch's record
    (a source it knows other than ``local``) claims the folders below it: a ``local`` one is passed over (a local
    folder inside a fetched dataset's range is no dataset of its own, ``check_manifest`` refuses to prepare it as
    one, so the fetched manifest above holds its task folders), and so is a ``manifest.toml`` that is no record
    this Gym can read, such as another tool's in a project root, or a newer Gym's, which claims nothing here and is
    refused where it is the folder's own, by that folder's check. ``None`` when no such manifest is in reach.
    """
    path = Path(path).absolute()
    for depth, folder in enumerate(path.parents[:levels]):
        below = path if depth == 0 else path.parents[depth - 1]
        if below.name.startswith(".") or below.is_symlink() or (depth and harbor_task.is_task_folder(below)):
            return None
        try:
            manifest = Manifest.read(folder)
        except ManifestError:
            continue
        if manifest is not None and manifest.source in SOURCES and manifest.source != LOCAL_SOURCE:
            return folder, manifest
    return None


def check_task_manifest(task: Path, *, digests: dict[str, str] | None = None) -> Manifest | None:
    """Hold one task folder to the manifest of the dataset folder whose check reaches it, when that is a fetch's.

    A single task folder target (``<dataset>/<task>``, or a task deeper under the dataset folder) is prepared with
    the ``dataset.toml`` and image records beside it but is no folder of task folders, so :func:`check_manifest`
    never runs for it; without this check an edited task of a fetched dataset prepared alone, as ``local``, while
    the dataset folder was refused. The manifest that holds the task is the one the dataset folder's check holds it
    to, not only its parent's: that check discovers tasks under subfolders too (:func:`iter_task_folders`, at most
    :data:`MAX_DISCOVERY_DEPTH` levels down, never inside a hidden folder or through a link) and names each by its
    folder name, so a task moved into a subfolder of a fetched dataset is still the task its manifest records and
    is held to it here, while a task that check never sees (hidden, deeper, under a hidden or linked folder, or inside
    another task folder, as a fixture a task ships) is held to nothing, and a manifest above that range is not read
    (:func:`enclosing_manifest`; a folder in between
    that carries a fetched manifest of its own, such as a dataset fetched inside another folder, holds its tasks
    itself, and a ``local`` one in between is passed over, as is a ``manifest.toml`` this Gym cannot read as a
    fetch's record, another tool's for one). The task's own parent is read as it was: it is the dataset root the
    task is prepared with, so its manifest is the task's dataset's whatever it says, and one naming a source this
    Gym does not know, or written by a newer Gym, refuses the task as it refuses the dataset folder. The task's
    folder digest is compared with the one that manifest
    records for it, with :func:`task_differences`' words: a folder that differs was edited by hand or its source
    moved, one the manifest does not record was never fetched, one that cannot be hashed is refused as the dataset
    folder's check refuses it, and one recorded without a digest is a fetch that never finished. Nothing is
    written: the manifest stays the dataset folder's record, and it is not upgraded or rewritten here. Returns that
    manifest, for the task's identity to carry its source (the task is still prepared with the ``dataset.toml`` and
    image records beside it, as any single task folder target is: the manifest holds its content, not the settings
    the preparation applies), or ``None`` when no fetched manifest in that range holds the task (a standalone task
    folder, whose parent is whatever directory holds it, or one whose nearest manifest is ``local``: the folder is
    its own source of truth, and the dataset folder's next prepare records the edit) or that manifest is from
    before digests were recorded (pins only; the dataset folder's prepare upgrades it).

    ``digests`` are folder digests already computed, keyed by task name as :func:`check_manifest` keys them: the
    task's is used when present (a loader's, to check the folder as it loaded) and hashed and stored otherwise.
    """
    task = Path(task).absolute()
    if task.name.startswith("."):
        return None
    folder, previous = task.parent, Manifest.read(task.parent)
    if previous is not None and previous.source not in SOURCES:
        raise ManifestError(
            f"{folder / MANIFEST_FILE} names source {previous.source!r}, which this Gym does not know "
            f"({', '.join(SOURCES)}); it was written by a newer Gym, or edited. Upgrade Gym, or delete the folder, "
            "or move it aside, and run again to fetch the source's copy"
        )
    if previous is None or previous.source == LOCAL_SOURCE:
        enclosing = enclosing_manifest(task, levels=harbor_task.MAX_DISCOVERY_DEPTH)
        if enclosing is None:
            return None
        folder, previous = enclosing
    if not previous.records_content:
        return None
    recorded = previous.task_digests
    if task.name in recorded and recorded[task.name] is None:
        raise ManifestError(
            f"the fetch of {folder} was interrupted (tasks not yet recorded: {task.name}); run the fetch again"
        )
    digests = digests if digests is not None else {}
    unhashable: tuple[str, ...] = ()
    on_disk: dict[str, str | None] = {task.name: digests.get(task.name)}
    if on_disk[task.name] is None:
        try:
            on_disk[task.name] = digests[task.name] = content_hash(task)
        except (OSError, ValueError):
            unhashable = (task.name,)
    differences = task_differences(
        {task.name: recorded[task.name]} if task.name in recorded else {}, on_disk, unhashable=unhashable
    )
    if differences is not None:
        raise ManifestError(
            f"{task} does not match {folder / MANIFEST_FILE} ({differences}). The task folder was edited by hand, "
            f"or its source moved since it was fetched. Delete {folder}, or move it aside, and run again to fetch "
            f"the source's copy"
        )
    return previous


def check_manifest(folder: Path, *, digests: dict[str, str] | None = None) -> Manifest:
    """Recompute the manifest of ``folder`` from disk and reconcile it with the one on disk.

    - No manifest: the folder is a local dataset; one is written with source ``local``, unless git
      tracks the folder, whose identity is then computed in memory and nothing written.
    - A manifest naming a source this Gym does not know (written by a newer Gym, or edited): refused,
      never read as local, because this Gym cannot tell whether the folders still match it.
    - A manifest without digests (written before they existed): upgraded in place, pins kept.
    - A manifest with per-task digests but no dataset digest (the package-store format before the schema
      was numbered): the folders are held to the recorded digests, and an unchanged folder is upgraded in
      place, the recorded digests kept, so a cache written by an earlier Gym keeps its fetch record.
    - A fetched manifest pinning a task with no digest: the fetch that wrote it was interrupted
      before that task landed (a fetcher claims the folder with every pin before its first download),
      so the folder is refused, named with the tasks still to come, until the fetch is run again.
    - Task folders differ from a ``local`` manifest: the folder is its own source of truth, so the
      manifest is rewritten and the new digest reported.
    - No manifest, or a ``local`` one, in a folder that a fetched dataset's check reaches (a subfolder of a
      fetched dataset, within its discovery range, see :func:`enclosing_manifest`): refused, and no local manifest
      written there. The task folders under it are that dataset's, recorded in its manifest and held to it by its
      check and by :func:`check_task_manifest`; a local manifest beside them would prepare an edited one as
      ``local`` and shadow the fetched manifest for the single-task check.
    - Task folders differ from a fetched manifest: the folder was edited by hand, or its source
      moved since the fetch. :class:`ManifestError` says which tasks and what to do.
    - Only ``dataset.toml`` or a dataset-level ``task_data.py`` / ``verifier.py`` moved: the Gym
      layer changed on purpose, the task folders are the same, so the derived facts are rewritten without a fetch,
      and the rewrite says which.

    ``digests`` are folder digests a loader already computed, extended in place with the rest.
    Returns the manifest now on disk.
    """
    folder = Path(folder)
    previous = Manifest.read(folder)
    if previous is not None and previous.source not in SOURCES:
        raise ManifestError(
            f"{folder / MANIFEST_FILE} names source {previous.source!r}, which this Gym does not know "
            f"({', '.join(SOURCES)}); it was written by a newer Gym, or edited. Upgrade Gym, or delete the folder, "
            "or move it aside, and run again to fetch the source's copy"
        )
    if previous is None or previous.source == LOCAL_SOURCE:
        enclosing = enclosing_manifest(folder, levels=harbor_task.MAX_DISCOVERY_DEPTH - 1)
        if enclosing is not None:
            above, _ = enclosing
            raise ManifestError(
                f"{folder} is no dataset of its own: it sits inside {above}, whose {MANIFEST_FILE} records the task "
                f"folders its check discovers under it, {folder}'s included when it has any. Prepare {above} "
                f"(selecting tasks with --only-tasks or --exclude-tasks) or one task folder under {folder}, or move "
                f"{folder} out of {above}"
            )
    source = previous.source if previous is not None else LOCAL_SOURCE
    pins = previous.pins if previous is not None else None
    ref = previous.ref if previous is not None else None
    if previous is not None and source != LOCAL_SOURCE and previous.records_content:
        # A pin without a digest in a manifest that records digests is a task the fetch never finished:
        # the claim is complete, the folder is not. Checked before the folders, since a claim-only folder
        # (interrupted during its first download) matches its manifest exactly.
        unfetched = sorted(name for name, digest in previous.task_digests.items() if digest is None)
        if unfetched:
            raise ManifestError(
                f"the fetch of {folder} was interrupted (tasks not yet recorded: {_some(unfetched)}); run the fetch again"
            )
    fresh = Manifest.build(folder, source=source, pins=pins, digests=digests, ref=ref)
    if previous is None:
        if tracked_by_git(folder):
            print(
                f"Dataset digest {fresh.digest[:12]} computed in memory: git tracks {folder}, no {MANIFEST_FILE} written"
            )
        elif _write_or_warn(fresh, folder):
            print(f"Wrote {folder / MANIFEST_FILE} (dataset digest {fresh.digest[:12]})")
        return fresh
    if previous.text() == fresh.text():
        return previous
    differences = (
        task_differences(previous.task_digests, fresh.task_digests, unhashable=fresh.unhashable)
        if previous.records_content
        else None
    )
    if differences is not None and source != LOCAL_SOURCE:
        raise ManifestError(
            f"{folder} does not match its {MANIFEST_FILE} ({differences}). The task folders were edited by hand "
            f"or deleted, or their source moved since they were fetched. Delete the folder, or move it aside, and "
            f"run again to fetch the source's copy"
        )
    if not previous.records_content:
        why = "it recorded pins only"
    elif differences is not None:
        why = f"task folders changed ({differences})"
    else:
        why = gym_layer_change(previous, fresh)
    if _write_or_warn(fresh, folder):
        report_rewrite(folder, previous, fresh, why)
    return fresh


def gym_layer_change(previous: "Manifest", fresh: "Manifest") -> str:
    """Which part of the Gym layer differs between two manifests of one task set, for a rewrite notice.

    A manifest from before the dataset digest existed (per-task digests only, see :attr:`Manifest.records_content`)
    is rewritten for that alone, whatever ``dataset.toml`` held when it was written.
    """
    if not previous.digest:
        return "it recorded no dataset digest"
    if previous.dataset_files_digest != fresh.dataset_files_digest:
        return f"a dataset-level {' or '.join(DATASET_FILES)} changed"
    return f"{DATASET_FILE} changed"


def report_rewrite(folder: Path, previous: "Manifest", fresh: "Manifest", why: str) -> None:
    """The one-line notice that ``folder``'s manifest was rewritten, and how the dataset digest moved."""
    print(
        f"Rewrote {Path(folder) / MANIFEST_FILE}: {why}; dataset digest {previous.digest[:12] or '-'} -> {fresh.digest[:12]}"
    )


def refresh_gym_layer(folder: Path) -> Manifest | None:
    """Bring a fetched manifest's Gym-layer digests up to date with ``folder`` without touching its task records.

    A fetch records ``dataset.toml`` and the dataset-level Gym files as it found them, and
    rewrites the manifest only when it fetches something: a registry fetch with every folder in place, or a
    pull that only replaced the copy of an entry's ``dataset.toml`` beside the tasks
    (:func:`nemo_gym.tasks.harbor.entries.prepare_destination`), leaves ``dataset_toml_digest`` and the dataset
    digest describing a file that is no longer there. This recomputes those digests from disk and keeps every
    task's recorded digest and pin, and the fetch's reference, as they are (nothing is hashed, so a hand-edited
    folder is still caught by :func:`check_manifest`, which rehashes), then writes the manifest when its text
    changed, with the notice ``check_manifest`` prints for the same rewrite (``Rewrote ...; dataset digest <old> ->
    <new>``), so a change of
    the dataset's identity is announced whichever function applies it. Returns the manifest now on disk: ``None``
    without one, and the old one untouched when it records pins only (``check_manifest`` upgrades those) or names a
    source this Gym does not know (``check_manifest`` refuses those).
    """
    folder = Path(folder)
    previous = Manifest.read(folder)
    if previous is None or not previous.records_content or previous.source not in SOURCES:
        return previous
    digests = {name: digest for name, digest in previous.task_digests.items() if digest is not None}
    fresh = Manifest.build(
        folder, source=previous.source, pins=previous.pins, digests=digests, names=list(digests), ref=previous.ref
    )
    if fresh.text() == previous.text():
        return previous
    if _write_or_warn(fresh, folder):
        report_rewrite(folder, previous, fresh, gym_layer_change(previous, fresh))
    return fresh
