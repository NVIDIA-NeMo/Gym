# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``manifest.toml``: the derived identity of a dataset folder.

The file sits beside the task folders and holds only facts computed from them: where the folders
came from (``source``), one dataset digest, the digest of ``dataset.toml`` alone, the digest of
the ``downstream/`` patch set alone, and per task the folder digest (the value rows carry as
``_ng_digest``) with its upstream pin. Name and version live only in ``dataset.toml``, so nothing
appears in both files. Nobody edits the manifest by hand.

The output is deterministic: a fixed key order, sorted tasks and no timestamps, so equal manifests
mean identical inputs and two runs compare by ``[dataset].digest``. The dataset digest is a hash over
the sorted per-task digests, then the ``dataset.toml`` digest, then the downstream digest. Two
datasets that differ only in a timeout override in ``dataset.toml`` are therefore different datasets,
on purpose: the override can change a score.

Fetchers write the manifest (:mod:`nemo_gym.tasks.harbor.hub`, :mod:`nemo_gym.tasks.harbor.package_store`)
and record only the tasks they fetched, so a folder nobody fetched is never blessed; prepare writes it for a
local folder, unless git tracks the folder (the repository's own example datasets), whose identity is
computed in memory. Prepare recomputes the per-task digests from the folders and refuses to run a fetched
dataset whose folders no longer match (:func:`check_manifest`).
"""

import hashlib
import json
import logging
import re
import subprocess
import tempfile
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
DOWNSTREAM_DIR = "downstream"
TASK_FILE = "task.toml"
# Where the task folders came from. `local` is a folder prepare met on disk with no fetch record.
SOURCES = ("harbor-registry", "harbor-package-store", "git", "local")
LOCAL_SOURCE = "local"
# The per-task keys, in the order they are written: the digest, then the upstream pin of a registry or
# git source (`git_url`, `git_commit_id`, `path`) or of a package-store source (`package`, `content_hash`).
TASK_KEYS = ("digest", "git_url", "git_commit_id", "path", "package", "content_hash")
_EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()
# The UTF-16 surrogate pair json.dumps writes for a character above U+FFFF.
_SURROGATE_PAIR = re.compile(r"\\u(d[89ab][0-9a-f]{2})\\u(d[c-f][0-9a-f]{2})")


class ManifestError(ValueError):
    """The manifest cannot be read, or the task folders no longer match it."""


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


def downstream_digest(folder: Path) -> str:
    """Digest of the ``downstream/`` patch set: relative path and content of every file, in order.

    A folder without ``downstream/`` (every fetched or local folder today) hashes to the digest of
    nothing, so the value is stable until patches exist.
    """
    root = Path(folder) / DOWNSTREAM_DIR
    if not root.is_dir():
        return _EMPTY_DIGEST
    digest = hashlib.sha256()
    for file in sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix()):
        digest.update(f"{file.relative_to(root).as_posix()}\0{_file_digest(file)}\n".encode())
    return digest.hexdigest()


def dataset_digest(task_digests: dict[str, str], dataset_toml: str, downstream: str) -> str:
    """The dataset's identity: sorted per-task digests, then ``dataset.toml``, then the downstream set."""
    digest = hashlib.sha256()
    for name in sorted(task_digests):
        digest.update(f"{name}\0{task_digests[name]}\n".encode())
    digest.update(f"{DATASET_FILE}\0{dataset_toml}\n".encode())
    digest.update(f"{DOWNSTREAM_DIR}\0{downstream}\n".encode())
    return digest.hexdigest()


@dataclass(frozen=True)
class Manifest:
    """``manifest.toml`` in memory. ``tasks`` maps a task name to its ``digest`` and pin keys."""

    source: str
    digest: str
    dataset_toml_digest: str
    downstream_digest: str
    tasks: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def task_digests(self) -> dict[str, str | None]:
        """Per-task folder digests; ``None`` for a task pinned before its folder was written."""
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
        """Compute the manifest of ``folder`` from its task folders, ``dataset.toml`` and ``downstream/``.

        ``pins`` are the upstream pins to record per task; a pinned task whose folder is absent
        (an interrupted fetch) keeps its pin and gets no digest. ``digests`` are folder digests
        already computed (by a loader or a fetch); any task folder not in it is hashed here, and the
        mapping is extended in place so a caller can reuse the work.

        ``names`` are the tasks to record. A fetcher passes the tasks it fetched, so a folder it did not
        fetch stays unrecorded (and prepare refuses it as not in the manifest) and no folder is scanned
        or stat-ed again for a name whose digest is known. Without it, every task folder on disk is
        recorded: a local dataset, or prepare recomputing a fetched one to compare.
        """
        if source not in SOURCES:
            raise ValueError(f"Unknown manifest source {source!r}; expected one of {', '.join(SOURCES)}")
        folder = Path(folder)
        digests = digests if digests is not None else {}
        present: dict[str, str] = {}
        if names is None:
            for task in task_folders(folder):
                if task.name not in digests:
                    digests[task.name] = content_hash(task)
                present[task.name] = digests[task.name]
        else:
            for name in names:
                if name not in digests:
                    if not (folder / name / TASK_FILE).is_file():
                        continue
                    digests[name] = content_hash(folder / name)
                present[name] = digests[name]
        tasks: dict[str, dict[str, str]] = {}
        for name in sorted(set(present) | set(pins or {})):
            entry: dict[str, str] = {}
            if name in present:
                entry["digest"] = present[name]
            entry.update({k: v for k, v in (pins or {}).get(name, {}).items() if k in TASK_KEYS and k != "digest"})
            tasks[name] = entry
        toml_digest = dataset_toml_digest(folder)
        patch_digest = downstream_digest(folder)
        return cls(
            source=source,
            digest=dataset_digest(present, toml_digest, patch_digest),
            dataset_toml_digest=toml_digest,
            downstream_digest=patch_digest,
            tasks=tasks,
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
            raise ManifestError(
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
            downstream_digest=str(dataset.get("downstream_digest", "")),
            tasks={
                str(name): {k: str(v) for k, v in entry.items() if k in TASK_KEYS and isinstance(v, str)}
                for name, entry in tasks.items()
                if isinstance(entry, dict)
            },
        )

    @property
    def records_content(self) -> bool:
        """Whether the manifest carries digests at all (one written before them has pins only)."""
        return bool(self.digest)

    def text(self) -> str:
        """The file's bytes: fixed key order, sorted tasks, no timestamps."""
        lines = [
            f"schema_version = {MANIFEST_SCHEMA_VERSION}",
            "",
            "[dataset]",
            f"source = {_quote(self.source)}",
            f"digest = {_quote(self.digest)}",
            f"dataset_toml_digest = {_quote(self.dataset_toml_digest)}",
            f"downstream_digest = {_quote(self.downstream_digest)}",
            "",
        ]
        for name in sorted(self.tasks):
            lines.append(f"[tasks.{_quote(name)}]")
            entry = self.tasks[name]
            lines.extend(f"{key} = {_quote(entry[key])}" for key in TASK_KEYS if key in entry)
            lines.append("")
        return "\n".join(lines)

    def write(self, folder: Path) -> Path:
        """Write ``manifest.toml`` atomically: a failed write leaves the previous, truthful file intact."""
        folder = Path(folder)
        target = folder / MANIFEST_FILE
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=folder, prefix=".manifest-", delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(self.text())
                stream.close()
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        return target


def task_differences(recorded: dict[str, str | None], on_disk: dict[str, str | None]) -> str | None:
    """One line naming the tasks whose digest differs between a manifest and the folders, or ``None``.

    A task is ``missing`` when the manifest recorded its digest and the folder is gone (``on_disk`` lacks
    it, or maps it to ``None`` because its pin outlived the folder), ``changed`` when both digests exist
    and differ, and ``not in the manifest`` when only the folder exists.
    """
    missing = sorted(name for name in recorded if recorded[name] is not None and on_disk.get(name) is None)
    changed = sorted(
        name for name in set(recorded) & set(on_disk) if recorded[name] != on_disk[name] and name not in missing
    )
    unrecorded = sorted(set(on_disk) - set(recorded))
    if not (changed or missing or unrecorded):
        return None

    def some(names: list[str]) -> str:
        return ", ".join(names[:5]) + (f", ... ({len(names)} in all)" if len(names) > 5 else "")

    parts = []
    if changed:
        parts.append(f"changed: {some(changed)}")
    if missing:
        parts.append(f"missing: {some(missing)}")
    if unrecorded:
        parts.append(f"not in the manifest: {some(unrecorded)}")
    return "; ".join(parts)


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
    - Task folders differ from a ``local`` manifest: the folder is its own source of truth, so the
      manifest is rewritten and the new digest reported.
    - Task folders differ from a fetched manifest: the folder was edited by hand, or its source
      moved since the fetch. :class:`ManifestError` says which tasks and what to do.
    - Only ``dataset.toml`` or ``downstream/`` moved: the Gym layer changed on purpose, the task folders
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
    differences = task_differences(previous.task_digests, fresh.task_digests) if previous.records_content else None
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
        why = f"{DATASET_FILE} or {DOWNSTREAM_DIR}/ changed"
    if _write_or_warn(fresh, folder):
        print(
            f"Rewrote {folder / MANIFEST_FILE}: {why}; dataset digest {previous.digest[:12] or '-'} -> {fresh.digest[:12]}"
        )
    return fresh
