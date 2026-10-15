# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``manifest.toml``: the derived identity of a dataset folder.

The file sits beside the task folders and holds only facts computed from them: where the folders
came from (``source``), one dataset digest, the digest of ``dataset.toml`` alone, and per task the
folder digest (the value rows carry as
``_ng_digest``) with its upstream pin. Name and version live only in ``dataset.toml``, so nothing
appears in both files. Nobody edits the manifest by hand.

The output is deterministic: a fixed key order, sorted tasks and no timestamps, so equal manifests
mean identical inputs and two runs compare by ``[dataset].digest``. The dataset digest is a hash over
the sorted per-task digests, then the ``dataset.toml`` digest. Two
datasets that differ only in a timeout override in ``dataset.toml`` are therefore different datasets,
on purpose: the override can change a score.

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

from nemo_gym.tasks.harbor.dataset_config import DATASET_FILE
from nemo_gym.tasks.harbor.digest import content_hash
from nemo_gym.tasks.harbor.task import iter_task_folders


logger = logging.getLogger(__name__)

MANIFEST_FILE = "manifest.toml"
MANIFEST_SCHEMA_VERSION = 1
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


def dataset_digest(task_digests: dict[str, str], dataset_toml: str) -> str:
    """The dataset's identity: sorted per-task digests, then ``dataset.toml``."""
    digest = hashlib.sha256()
    for name in sorted(task_digests):
        digest.update(f"{name}\0{task_digests[name]}\n".encode())
    digest.update(f"{DATASET_FILE}\0{dataset_toml}\n".encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class Manifest:
    """``manifest.toml`` in memory. ``tasks`` maps a task name to its ``digest`` and pin keys."""

    source: str
    digest: str
    dataset_toml_digest: str
    tasks: dict[str, dict[str, str]] = field(default_factory=dict)
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
    ) -> "Manifest":
        """Compute the manifest of ``folder`` from its task folders and ``dataset.toml``.

        ``pins`` are the upstream pins to record per task; a pinned task whose folder is absent
        (an interrupted fetch) keeps its pin and gets no digest. ``digests`` are folder digests
        already computed (by a loader or a fetch); any task folder not in it is hashed here, and the
        mapping is extended in place so a caller can reuse the work.

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
        return cls(
            source=source,
            digest=dataset_digest(present, toml_digest),
            dataset_toml_digest=toml_digest,
            tasks=tasks,
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
        return cls(
            source=str(dataset.get("source", "")),
            digest=str(dataset.get("digest", "")),
            dataset_toml_digest=str(dataset.get("dataset_toml_digest", "")),
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
            f"digest = {_quote(self.digest)}",
            f"dataset_toml_digest = {_quote(self.dataset_toml_digest)}",
            "",
        ]
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
    - Task folders differ from a fetched manifest: the folder was edited by hand, or its source
      moved since the fetch. :class:`ManifestError` says which tasks and what to do.
    - Only ``dataset.toml`` moved: the Gym layer changed on purpose, the task folders
      are the same, so the derived facts are rewritten without a fetch.

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
    source = previous.source if previous is not None else LOCAL_SOURCE
    pins = previous.pins if previous is not None else None
    if previous is not None and source != LOCAL_SOURCE and previous.records_content:
        # A pin without a digest in a manifest that records digests is a task the fetch never finished:
        # the claim is complete, the folder is not. Checked before the folders, since a claim-only folder
        # (interrupted during its first download) matches its manifest exactly.
        unfetched = sorted(name for name, digest in previous.task_digests.items() if digest is None)
        if unfetched:
            raise ManifestError(
                f"the fetch of {folder} was interrupted (tasks not yet recorded: {_some(unfetched)}); run the fetch again"
            )
    fresh = Manifest.build(folder, source=source, pins=pins, digests=digests)
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
    elif not previous.digest:
        # Written before the dataset digest existed (per-task digests only, see Manifest.records_content):
        # rewritten for that alone, whatever dataset.toml held when it was written.
        why = "it recorded no dataset digest"
    elif differences is not None:
        why = f"task folders changed ({differences})"
    else:
        why = f"{DATASET_FILE} changed"
    if _write_or_warn(fresh, folder):
        print(
            f"Rewrote {folder / MANIFEST_FILE}: {why}; dataset digest {previous.digest[:12] or '-'} -> {fresh.digest[:12]}"
        )
    return fresh
