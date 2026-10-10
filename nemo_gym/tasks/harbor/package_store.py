# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harbor package-store datasets: ``harbor:<org>/<name>[@<tag | sha256:digest>]``.

Harbor publishes some datasets (Terminal Bench 4 among them) as packages rather than
git checkouts. A dataset version is a list of task packages, each pinned by the
content hash Gym also uses as its task digest. Fetching a dataset resolves the version,
downloads every task archive, checks each digest and writes ``manifest.toml``.

The store is Harbor's public PostgREST and storage API. Only the public
(publishable) key is needed to read published packages.
"""

import io
import json
import logging
import re
import shutil
import tarfile
import tempfile
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemo_gym.tasks.harbor.digest import content_hash
from nemo_gym.tasks.harbor.hub import contained_path, validate_task_name


logger = logging.getLogger(__name__)

PACKAGE_STORE_URL = "https://ofhuhcpkvzjlejydnvyd.supabase.co"
# Harbor's public read-only key, the same one its CLI ships with.
PACKAGE_STORE_PUBLIC_KEY = "sb_publishable_Z-vuQbpvpG-PStjbh4yE0Q_e-d3MTIH"
MANIFEST_FILE = "manifest.toml"
_PAGE_SIZE = 1000
_DIGEST = re.compile(r"(sha256:)?([a-f0-9]{64})")
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


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
        """``<name>-<tag>`` for a tag, ``<name>-<12 digest chars>`` for ``latest`` or a digest."""
        if self.ref == "latest" or self.digest:
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
            detail = exc.read().decode(errors="replace")[:300]
            raise PackageStoreError(f"Harbor package store returned {exc.code} for {method} {path}: {detail}") from exc
        except OSError as exc:
            raise PackageStoreError(f"Harbor package store unreachable for {method} {path}: {exc}") from exc

    def _rows(self, table: str, params: dict[str, str]) -> list[dict[str, Any]]:
        rows = json.loads(self._request("GET", f"/rest/v1/{table}", params=params))
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
        resolved = json.loads(
            self._request(
                "POST",
                "/rest/v1/rpc/resolve_task_version",
                body={"p_org": task.org, "p_name": task.name, "p_ref": f"sha256:{task.content_hash}"},
            )
        )
        if not resolved or resolved.get("content_hash", "").removeprefix("sha256:") != task.content_hash:
            raise PackageStoreError(f"The store returned a different identity for {task.package}")
        return str(resolved["archive_path"])

    # -- download --------------------------------------------------------------------------

    def download_task(self, task: PackageTask, target: Path) -> None:
        """Download and unpack one task package into ``target``, checking its digest.

        ``target`` is ``<dataset folder>/<task name>``; whatever is there is replaced, but only
        after checking that it really resolves inside the dataset folder.
        """
        archive = self._request(
            "GET", "/storage/v1/object/packages/" + urllib.parse.quote(self.archive_path(task), safe="/")
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".nemo-gym-package-", dir=target.parent) as tmp:
            staged = Path(tmp) / "package"
            staged.mkdir()
            with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
                tar.extractall(staged, filter="data")
            actual = content_hash(staged)
            if actual != task.content_hash:
                raise PackageStoreError(
                    f"{task.package} content hash mismatch: expected sha256:{task.content_hash}, got sha256:{actual}"
                )
            contained_path(target, target.parent, error=PackageStoreError)
            if target.exists():
                shutil.rmtree(target)
            shutil.move(str(staged), str(target))


def _write_manifest(folder: Path, dataset: ResolvedDataset) -> None:
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
            "",
        ]
    (folder / MANIFEST_FILE).write_text("\n".join(lines))


def fetch_package_dataset(
    ref: PackageRef, root: Path, store: PackageStore | None = None, *, force: bool = False
) -> Path:
    """Fetch every task of a package-store dataset into ``root/<folder>/<task>/``.

    A task folder whose digest already matches is kept, so a rerun after a partial
    fetch completes it. A task folder whose digest differs (it was edited locally, or the
    store republished it) stops the fetch with both digests named, so edits are never
    discarded silently; with ``force`` it is replaced with the store's copy, logged at
    warning level. Returns the dataset folder.
    """
    store = store or PackageStore()
    dataset = store.resolve_dataset(ref)
    if not dataset.tasks:
        raise PackageStoreError(f"{ref.package}@{ref.ref} has no tasks")
    folder = Path(root) / ref.folder_name(dataset.content_hash)
    folder.mkdir(parents=True, exist_ok=True)
    for task in dataset.tasks:
        target = folder / task.name
        if (target / "task.toml").is_file():
            actual = content_hash(target)
            if actual == task.content_hash:
                continue
            if not force:
                raise PackageStoreError(
                    f"{target} differs from the store: its content hash is sha256:{actual} but the store's "
                    f"{task.package} (in {ref.package}@{ref.ref}) is sha256:{task.content_hash}. Keep your local "
                    "copy, or pass --force to replace it with the store's copy"
                )
            logger.warning(
                "Replacing %s (sha256:%s) with %s from the store (sha256:%s) because --force was passed",
                target,
                actual[:12],
                task.package,
                task.content_hash[:12],
            )
        logger.info("Fetching %s (sha256:%s)", task.package, task.content_hash[:12])
        store.download_task(task, target)
    _write_manifest(folder, dataset)
    return folder
