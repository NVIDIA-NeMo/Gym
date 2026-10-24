# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor package-store datasets: ``harbor:<org>/<name>[@<tag | sha256:digest>]``.

Harbor publishes some datasets (Terminal Bench 4 among them) as packages rather than
git checkouts. A dataset version is a list of task packages, each pinned by the
content hash Gym also uses as its task digest. Fetching a dataset resolves the version,
downloads every task archive, checks each against Harbor's published hash and writes
``manifest.toml``, whose ``[dataset]`` table pins the package (``org/name``), the reference
and the content hash the fetch resolved, and whose ``[tasks]`` entries pin every task package and
record Gym's digest of each task folder as fetched. The manifest is also the folder's claim: it is
written with the identity and the task pins, and no digests, before the first task lands, so a
folder an interrupted fetch leaves behind still says which package and version it belongs to. A
later fetch into the same folder reads that back first, refuses a folder claimed for another
package or an earlier resolution of this one, and compares each task folder with the digest
recorded for it.

The store is Harbor's public PostgREST and storage API. Only the public
(publishable) key is needed to read published packages.
"""

import errno
import http.client
import io
import json
import logging
import os
import re
import secrets
import tarfile
import tempfile
import threading
import tomllib
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_gym.tasks.harbor.digest import archive_hash, content_hash
from nemo_gym.tasks.harbor.hub import contained_path, validate_task_name


logger = logging.getLogger(__name__)

PACKAGE_STORE_URL = "https://ofhuhcpkvzjlejydnvyd.supabase.co"
# Harbor's public read-only key, the same one its CLI ships with.
PACKAGE_STORE_PUBLIC_KEY = "sb_publishable_Z-vuQbpvpG-PStjbh4yE0Q_e-d3MTIH"  # pragma: allowlist secret
MANIFEST_FILE = "manifest.toml"
_PAGE_SIZE = 1000
_DIGEST = re.compile(r"(sha256:)?([a-f0-9]{64})")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")
# os.link errors that mean "no hard links here" (a filesystem or mount that lacks them), not "the target exists".
_NO_HARD_LINKS = {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.ENOSYS, errno.EACCES}


class PackageStoreError(ValueError):
    """The reference cannot be resolved or a package cannot be fetched."""


@dataclass(frozen=True)
class PackageRef:
    """``org/name@ref``; ``ref`` is a tag such as ``4.0.0``, ``latest`` or ``sha256:<digest>``."""

    org: str
    name: str
    ref: str = "latest"

    @property
    def package(self) -> str:
        return f"{self.org}/{self.name}"

    @property
    def digest(self) -> str | None:
        match = _DIGEST.fullmatch(self.ref)
        return match.group(2) if match else None

    def folder_name(self, resolved_digest: str) -> str:
        """``<name>`` for ``latest``, ``<name>-<tag>`` for a tag, ``<name>-<12 digest chars>`` for a digest pin.

        ``latest`` is what a plain ``harbor:org/name`` reference means, so its folder carries the plain
        name and ``manifest.toml`` records the content hash it resolved to. A pinned tag or digest keeps
        its suffix, so two pins of one dataset never share a folder.
        """
        if self.ref == "latest":
            return self.name
        if self.digest:
            return f"{self.name}-{resolved_digest[:12]}"
        return f"{self.name}-{_SAFE.sub('-', self.ref)}"


@dataclass(frozen=True)
class PackageTask:
    org: str
    name: str
    content_hash: str

    @property
    def package(self) -> str:
        return f"{self.org}/{self.name}"


@dataclass(frozen=True)
class ResolvedDataset:
    ref: PackageRef
    version_id: str
    content_hash: str
    tasks: tuple[PackageTask, ...]


def _holds_store_copy(target: Path, task: PackageTask) -> bool:
    """Whether the task folder at ``target`` is the store's ``task``, by either digest.

    A freshly unpacked archive matches Harbor's published hash by :func:`archive_hash` (its files are
    exactly the published set, names Gym's folder digest skips included), and a folder a run has
    since left byte-code in matches it by :func:`content_hash` unless the package itself ships such a
    name. Either match means the folder is the store's copy.
    """
    return archive_hash(target) == task.content_hash or content_hash(target) == task.content_hash


class PackageStore:
    """Read-only client for Harbor's package store."""

    def __init__(
        self, base_url: str = PACKAGE_STORE_URL, api_key: str = PACKAGE_STORE_PUBLIC_KEY, timeout_s: float = 120
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    # -- transport -------------------------------------------------------------------------

    def _request(self, method: str, path: str, *, params: dict[str, str] | None = None, body: Any = None) -> bytes:
        url = f"{self.base_url}{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params, safe=":(),*!.")
        headers = {"apikey": self.api_key}
        data = None
        if body is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(body).encode()
        request = urllib.request.Request(url, method=method, headers=headers, data=data)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode(errors="replace")[:300]
            except (OSError, http.client.HTTPException):
                detail = ""
            raise PackageStoreError(f"Harbor package store returned {exc.code} for {method} {path}: {detail}") from exc
        except http.client.HTTPException as exc:
            # The server closed the connection before the advertised body arrived (IncompleteRead), or did
            # not speak HTTP (BadStatusLine): a transport failure like an unreachable store, not a traceback.
            raise PackageStoreError(
                f"Harbor package store closed the connection before the reply to {method} {path} was complete: {exc}"
            ) from exc
        except OSError as exc:
            raise PackageStoreError(f"Harbor package store unreachable for {method} {path}: {exc}") from exc

    def _json(self, method: str, path: str, *, params: dict[str, str] | None = None, body: Any = None) -> Any:
        """The decoded JSON reply; a body that is not JSON (an HTML error page, a reply cut short) is a store error."""
        raw = self._request(method, path, params=params, body=body)
        try:
            return json.loads(raw)
        except ValueError as exc:  # JSONDecodeError, or bytes that are not UTF-8
            excerpt = raw.decode(errors="replace")[:200]
            raise PackageStoreError(
                f"Harbor package store reply to {method} {path} is not JSON ({exc}): {excerpt!r}"
            ) from exc

    def _rows(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        rows = self._json("GET", f"/rest/v1/{table}", params=params)
        if not isinstance(rows, list):
            raise PackageStoreError(f"Unexpected reply from {table}: {rows!r}")
        return rows

    # -- resolution ------------------------------------------------------------------------

    def resolve_dataset(self, ref: PackageRef) -> ResolvedDataset:
        """Resolve a dataset reference to its version row and task list."""
        package_filter = {
            "package.name": f"eq.{ref.name}",
            "package.type": "eq.dataset",
            "package.org.name": f"eq.{ref.org}",
            "limit": "1",
        }
        if ref.digest:
            rows = self._rows(
                "dataset_version",
                {"select": "*,package:package_id!inner(*,org:org_id!inner(name))", "content_hash": f"eq.{ref.digest}"}
                | package_filter,
            )
            version = rows[0] if rows else None
        else:
            rows = self._rows(
                "dataset_version_tag",
                {
                    "select": "dataset_version:dataset_version_id(*),package:package_id!inner(*,org:org_id!inner(name))",
                    "tag": f"eq.{ref.ref}",
                }
                | package_filter,
            )
            version = rows[0]["dataset_version"] if rows else None
        if not version:
            raise PackageStoreError(f"No dataset {ref.package}@{ref.ref} in the Harbor package store")
        if version.get("yanked_at"):
            logger.warning("Dataset %s@%s is yanked: %s", ref.package, ref.ref, version.get("yanked_reason") or "")
        tasks = self._dataset_tasks(str(version["id"]))
        return ResolvedDataset(
            ref=ref, version_id=str(version["id"]), content_hash=str(version["content_hash"]), tasks=tasks
        )

    def _dataset_tasks(self, version_id: str) -> tuple[PackageTask, ...]:
        tasks: list[PackageTask] = []
        offset = 0
        while True:
            rows = self._rows(
                "dataset_version_task",
                {
                    "select": "task_version_id,task_version:task_version_id(content_hash,package:package_id(name,org:org_id(name)))",
                    "dataset_version_id": f"eq.{version_id}",
                    "order": "task_version_id",
                    "limit": str(_PAGE_SIZE),
                    "offset": str(offset),
                },
            )
            for row in rows:
                task = row.get("task_version")
                if not task:
                    raise PackageStoreError("A task in this dataset version is not readable with the public key")
                tasks.append(
                    PackageTask(
                        org=task["package"]["org"]["name"],
                        name=validate_task_name(task["package"]["name"], error=PackageStoreError),
                        content_hash=task["content_hash"],
                    )
                )
            if len(rows) < _PAGE_SIZE:
                return tuple(tasks)
            offset += _PAGE_SIZE

    def archive_path(self, task: PackageTask) -> str:
        resolved = self._json(
            "POST",
            "/rest/v1/rpc/resolve_task_version",
            body={"p_org": task.org, "p_name": task.name, "p_ref": f"sha256:{task.content_hash}"},
        )
        if not resolved or resolved.get("content_hash", "").removeprefix("sha256:") != task.content_hash:
            raise PackageStoreError(f"The store returned a different identity for {task.package}")
        return str(resolved["archive_path"])

    # -- download --------------------------------------------------------------------------

    def download_task(self, task: PackageTask, target: Path) -> None:
        """Download and unpack one task package into ``target``, checking its digest.

        The unpacked archive is checked with :func:`~nemo_gym.tasks.harbor.digest.archive_hash`, the
        exclusion-free digest Harbor publishes: the archive holds exactly the files Harbor's packager
        kept, so a file Gym's folder digest would skip (a ``.pyc`` the task ships on purpose) still
        counts, and a file the packager never put there is a mismatch. The folder digest of what was
        unpacked is what :func:`fetch_package_dataset` records in the manifest and compares reruns with,
        so such a package is fetched once and kept thereafter. ``target`` is
        ``<dataset folder>/<task name>``, checked to really resolve inside the dataset folder; an empty
        folder there is replaced, a concurrent fetch's copy of the same package is kept, and anything
        else stops the fetch.

        The unpacked package is promoted with one ``os.rename`` of the staging folder, a sibling of
        ``target`` on the same filesystem, so the destination is only ever absent, a complete previous
        copy or the complete new one: no rename-away, no copy fallback and nothing to roll back.
        """
        archive = self._request(
            "GET", "/storage/v1/object/packages/" + urllib.parse.quote(self.archive_path(task), safe="/")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".nemo-gym-package-", dir=target.parent) as tmp:
            staged = Path(tmp) / "package"
            staged.mkdir()
            try:
                with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                    tar.extractall(staged, filter="data")
            except (tarfile.TarError, EOFError) as exc:
                # A body that is not a gzip tar (an HTML or JSON error page, another compression, a
                # truncated stream, which tarfile reports as EOFError) is a store failure, not a traceback.
                raise PackageStoreError(
                    f"{task.package} archive from the store is not a readable tar.gz package: {exc}"
                ) from exc
            actual = archive_hash(staged)
            if actual != task.content_hash:
                raise PackageStoreError(
                    f"{task.package} content hash mismatch: the store advertises sha256:{task.content_hash} but the "
                    f"archive's files hash to sha256:{actual}"
                )
            contained_path(target, target.parent, error=PackageStoreError)
            try:
                # Atomic: the staging folder is a sibling on the same filesystem, so this is a true rename
                # (never shutil.move's copy fallback, which can leave a partial destination). It replaces
                # an empty folder and refuses a non-empty one, a file or a link, so a peer's finished copy
                # is never moved away and no reader ever sees the folder without its task.toml.
                os.rename(staged, target)
            except OSError as exc:
                if (
                    not target.is_symlink()
                    and target.is_dir()
                    and (target / "task.toml").is_file()
                    and _holds_store_copy(target, task)
                ):
                    # A concurrent fetch promoted the same bytes first; its copy stays.
                    return
                if target.exists() or target.is_symlink():
                    raise PackageStoreError(
                        f"{target} appeared while fetching and is not the store's copy of {task.package}; move "
                        "it aside and run again"
                    ) from exc
                raise


_UNREADABLE_REMEDY = (
    "The manifest records which package and version the task folders belong to, so the folder cannot be "
    "completed or reused without it. Move the folder aside (or delete it) and run again"
)


def _read_manifest(folder: Path) -> dict[str, Any] | None:
    """The manifest a previous fetch left in ``folder``, parsed, or ``None`` when there is none.

    One this Gym cannot read stops the fetch with the folder untouched: not TOML, or without the
    ``[dataset]`` table, or with ``[dataset]`` or ``[tasks]`` not a table, is a :class:`PackageStoreError`.
    The manifest is the folder's claim, which package and version its task folders belong to, so a folder
    whose claim cannot be read is never completed, reused or rewritten; the remedy is the same as for an
    edited task folder.
    """
    manifest = folder / MANIFEST_FILE
    if not manifest.is_file():
        return None
    try:
        parsed = tomllib.loads(manifest.read_text())
    except (tomllib.TOMLDecodeError, UnicodeDecodeError) as exc:
        raise PackageStoreError(f"{manifest} is not a valid manifest: {exc}. {_UNREADABLE_REMEDY}") from exc
    if not isinstance(parsed.get("dataset"), dict):
        raise PackageStoreError(f"{manifest} is not a valid manifest: no [dataset] table. {_UNREADABLE_REMEDY}")
    if not isinstance(parsed.get("tasks", {}), dict):
        raise PackageStoreError(f"{manifest} is not a valid manifest: [tasks] is not a table. {_UNREADABLE_REMEDY}")
    return parsed


def _recorded_dataset(manifest: dict[str, Any]) -> dict[str, str]:
    """The ``[dataset]`` table of a parsed manifest (string fields only), or ``{}``."""
    recorded = manifest.get("dataset", {})
    if not isinstance(recorded, dict):
        return {}
    return {key: value for key, value in recorded.items() if isinstance(value, str)}


def _recorded_digests(previous: dict[str, Any] | None, dataset: ResolvedDataset) -> dict[str, str]:
    """Gym's digest of each task folder as the previous fetch wrote it, by task name.

    Only an entry that pins the task version ``dataset`` names counts, so a hand-edited or foreign
    manifest never vouches for a folder, and a task the claim pinned but that has not landed yet (its
    entry has no ``digest``) is not recorded.
    """
    if previous is None:
        return {}
    tasks = previous.get("tasks", {})
    if not isinstance(tasks, dict):
        return {}
    digests: dict[str, str] = {}
    for task in dataset.tasks:
        entry = tasks.get(task.name)
        if not isinstance(entry, dict) or entry.get("content_hash") != f"sha256:{task.content_hash}":
            continue
        digest = entry.get("digest")
        if isinstance(digest, str) and digest.startswith("sha256:"):
            digests[task.name] = digest.removeprefix("sha256:")
    return digests


def _check_recorded_identity(folder: Path, dataset: ResolvedDataset, previous: dict[str, Any] | None) -> None:
    """Refuse ``folder`` when its manifest's ``[dataset]`` table pins another dataset than ``dataset``.

    Folder names stay readable (``<name>``, ``<name>-<tag>``), so the manifest's ``[dataset]`` table is
    what tells ``alice/evals@1.0`` from ``bob/evals@1.0`` in the same ``evals-1.0`` folder, and a
    ``latest`` fetch from the version the tag has since moved to. Either mismatch names the recorded
    identity and the requested one, and a manifest whose table names no package-store dataset at all is
    refused the same way: a folder with a manifest is never treated as unclaimed. The remedy is the same
    as for an edited task folder.
    """
    if previous is None:
        return
    recorded = _recorded_dataset(previous)
    ref = dataset.ref
    recorded_name = recorded.get("name", "?")
    recorded_version = recorded.get("version", "?")
    recorded_hash = recorded.get("content_hash", "").removeprefix("sha256:")
    source = recorded.get("source")
    requested = f"{ref.package}@{ref.ref} (sha256:{dataset.content_hash})"
    if source != "harbor-package-store" or recorded_name != ref.package:
        found = f"{recorded_name}@{recorded_version}"
        if recorded_hash:
            found += f" (sha256:{recorded_hash})"
        if source != "harbor-package-store":
            found += f" from {source or 'an unknown source'}"
        raise PackageStoreError(
            f"{folder} was fetched for {found}, not {requested}. Move the folder aside (or delete it) and run again"
        )
    if recorded_hash != dataset.content_hash:
        raise PackageStoreError(
            f"{folder} is stale: it was fetched when {ref.package}@{recorded_version} resolved to "
            f"sha256:{recorded_hash}, and {ref.package}@{ref.ref} now resolves to sha256:{dataset.content_hash}. "
            "Move the folder aside (or delete it) and run again to fetch the current version"
        )


def _manifest_text(dataset: ResolvedDataset, digests: dict[str, str]) -> str:
    """The manifest: the dataset pin, then per task the store's hash and, once the task has landed, Gym's digest.

    Every task is pinned from the first write on; a task whose folder is not there yet has no ``digest``, so
    the manifest an interrupted fetch leaves names exactly the tasks still to come.
    """

    def quote(value: str) -> str:
        return json.dumps(value)

    lines = [
        "[dataset]",
        f"name = {quote(dataset.ref.package)}",
        f"version = {quote(dataset.ref.ref)}",
        'source = "harbor-package-store"',
        f"content_hash = {quote('sha256:' + dataset.content_hash)}",
        "",
    ]
    for task in sorted(dataset.tasks, key=lambda t: t.name):
        lines += [
            f"[tasks.{quote(task.name)}]",
            f"package = {quote(task.package)}",
            f"content_hash = {quote('sha256:' + task.content_hash)}",
        ]
        if task.name in digests:
            lines.append(f"digest = {quote('sha256:' + digests[task.name])}")
        lines.append("")
    return "\n".join(lines)


def _write_text(fd: int, text: str) -> None:
    """Write ``text`` to ``fd`` and flush it to disk; the descriptor is closed."""
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_manifest(
    folder: Path, dataset: ResolvedDataset, digests: dict[str, str], *, exclusive: bool = False
) -> None:
    """Write ``manifest.toml`` atomically: a failed write leaves the previous, truthful file intact.

    The complete file is written to a temporary sibling (unique per process and thread, so concurrent
    fetches of one folder never share it) and moved into place, so a reader never sees a partial manifest.
    A rewrite keeps the permissions the file had; a new manifest gets the usual umask-aware mode.

    With ``exclusive``, the manifest is created only where none is and :class:`FileExistsError` is raised
    otherwise, the target untouched (the temporary is written and removed either way): the complete file is
    hard-linked into place, the atomic create-unless-present that ``os.replace`` lacks, so of two writers
    racing for an empty folder exactly one succeeds. A filesystem without hard links falls back to an
    exclusive create of the target itself, still one winner, though a reader may then glimpse the file
    before its last byte. :func:`_claim_folder` claims a folder this way before its first task lands.
    """
    target = folder / MANIFEST_FILE
    text = _manifest_text(dataset, digests)
    try:
        mode: int | None = target.stat().st_mode & 0o777
    except FileNotFoundError:
        mode = None
    temporary = folder / f".manifest-{os.getpid()}-{threading.get_ident()}-{secrets.token_hex(4)}.tmp"
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        _write_text(fd, text)
        if mode is not None:
            os.chmod(temporary, mode)
        if not exclusive:
            os.replace(temporary, target)
            return
        try:
            # A hard link lands only where nothing is yet; an existing target is FileExistsError, re-raised.
            os.link(temporary, target)
        except OSError as exc:
            if exc.errno not in _NO_HARD_LINKS:
                raise
            _write_text(os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666), text)
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass


def _claim_folder(folder: Path, dataset: ResolvedDataset, digests: dict[str, str]) -> None:
    """Record which dataset ``folder`` holds before its first task lands.

    Called when the folder has no manifest, right before the first download. The claim is the manifest
    with the ``[dataset]`` identity, every task pinned and only the digests verified so far (none in a
    fresh folder), created exclusively; each task downloaded after it is recorded as it lands. Of two
    fetches racing for an unclaimed folder exactly one writes the claim, and the other reads it back and
    is refused unless it fetches the same dataset (then both complete the folder, and
    :meth:`PackageStore.download_task` keeps whichever copy of a task lands first). An interrupted fetch
    leaves the claim behind, so a later fetch of another package, or of another version of this one, is
    refused before it looks at any task, and a later fetch of the same dataset completes the folder.
    """
    try:
        _write_manifest(folder, dataset, digests, exclusive=True)
    except FileExistsError:
        previous = _read_manifest(folder)
        if previous is None:
            raise PackageStoreError(
                f"{folder / MANIFEST_FILE} was written and removed by another fetch while this one ran; run again"
            ) from None
        _check_recorded_identity(folder, dataset, previous)
    except OSError as exc:
        raise PackageStoreError(
            f"Could not record which dataset {folder} holds in {folder / MANIFEST_FILE} before fetching its tasks: "
            f"{exc}. Nothing was downloaded; fix the folder's permissions or filesystem and run again"
        ) from exc


def fetch_package_dataset(ref: PackageRef, root: Path, store: PackageStore | None = None) -> Path:
    """Fetch every task of a package-store dataset into ``root/<folder>/<task>/``.

    A task folder whose digest already matches is kept, so a rerun after a partial
    fetch completes it: a folder the manifest records is compared with the digest recorded
    for it (Gym's digest of the folder as fetched, which a package shipping a name that
    digest skips makes differ from the store's hash), and a folder no manifest records yet
    (the fetch was interrupted) with the store's hash by either digest. A task folder whose
    digest differs (it was edited locally, or the store republished it) stops the fetch with
    both digests named, so edits are never discarded silently; deleting or moving the folder
    aside and running again fetches the store's copy. A dataset folder whose ``manifest.toml``
    pins another package, or an earlier resolution of this one (``latest`` has moved), stops
    the fetch before any task is looked at, with the recorded and the requested identity named,
    so one folder never mixes two versions or two packages. That manifest exists from the first
    download on: an unclaimed folder is claimed (:func:`_claim_folder`, the identity and the
    pins without digests, created exclusively) before its first task lands, so an interrupted
    fetch leaves a folder that says whose it is, and two fetches racing for one folder never
    fill it with two datasets. A manifest this Gym cannot read stops the fetch with the folder
    untouched, since this Gym cannot tell what it claimed. Anything else in a task's place (a
    folder without ``task.toml``, a file, a symlink) also stops the fetch: only what this
    function fetched is ever replaced. Returns the dataset folder.
    """
    store = store or PackageStore()
    dataset = store.resolve_dataset(ref)
    if not dataset.tasks:
        raise PackageStoreError(f"{ref.package}@{ref.ref} has no tasks")
    counts = Counter(task.name for task in dataset.tasks)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise PackageStoreError(f"{ref.package}@{ref.ref} has tasks sharing a folder name: {', '.join(duplicates)}")
    folder = Path(root) / ref.folder_name(dataset.content_hash)
    if folder.is_symlink():
        raise PackageStoreError(f"Dataset folder must not be a symlink: {folder}")
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / MANIFEST_FILE).is_symlink():
        raise PackageStoreError(f"Dataset manifest must not be a symlink: {folder / MANIFEST_FILE}")
    previous = _read_manifest(folder)
    _check_recorded_identity(folder, dataset, previous)
    recorded = _recorded_digests(previous, dataset)
    claimed = previous is not None
    digests: dict[str, str] = {}
    for task in dataset.tasks:
        target = folder / task.name
        if target.is_symlink():
            raise PackageStoreError(f"Task destination must not be a symlink: {target}")
        if (target / "task.toml").is_file():
            actual = content_hash(target)
            if task.name in recorded:
                expected = recorded[task.name]
                matches = actual == expected
                whose = f"the copy the previous fetch wrote of {task.package} (in {ref.package}@{ref.ref}) hashed to"
            else:
                expected = task.content_hash
                matches = actual == expected or _holds_store_copy(target, task)
                whose = f"the store's {task.package} (in {ref.package}@{ref.ref}) is"
            if matches:
                digests[task.name] = actual
                continue
            raise PackageStoreError(
                f"{target} differs from the store: its content hash is sha256:{actual} but {whose} "
                f"sha256:{expected}. Keep your local copy, or delete the folder (or move it aside) and run again to "
                "fetch the store's copy"
            )
        if target.exists():
            raise PackageStoreError(
                f"{target} exists but is not a fetched task (no task.toml); move it aside and run again"
            )
        if not claimed:
            _claim_folder(folder, dataset, digests)
            claimed = True
        logger.info("Fetching %s (sha256:%s)", task.package, task.content_hash[:12])
        store.download_task(task, target)
        # download_task checked the unpacked archive against the store's hash; the manifest records Gym's
        # digest of the folder it left, which differs when the package ships a name that digest skips.
        digests[task.name] = content_hash(target)
        # Recorded as it lands, so an interrupted fetch names exactly the tasks still to come and a rerun
        # compares the landed folders with their recorded digests instead of hashing them against the store.
        _write_manifest(folder, dataset, digests)
    _write_manifest(folder, dataset, digests)
    return folder
