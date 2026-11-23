# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reference deliverables read from S3, one task at a time.

A reference model's ``deliverables_dir`` may be an ``s3://bucket/prefix`` URI laid out like a local one
(``<prefix>/task_<task_id>/repeat_<n>/``). /verify then downloads only the task it judges into a local cache,
so a server whose credentials can read the prefix needs neither the whole set on disk nor a shared filesystem.
"""

from __future__ import annotations

import asyncio
import hashlib
import shutil
import uuid
from pathlib import Path
from typing import Any, Callable, Dict

from resources_servers.gdpval import persisted_layout


# The agent transcripts are large and the judges never read them.
SKIPPED_NAMES = frozenset({"history.json", "history.pkl"})


def is_s3_uri(path: str) -> bool:
    return path.startswith("s3://")


def _split(uri: str) -> tuple[str, str]:
    bucket, _, prefix = uri.removeprefix("s3://").partition("/")
    return bucket, prefix.strip("/")


class S3ReferenceCache:
    """Downloads each ``(reference URI, task)`` once into ``cache_root`` and returns its local root.

    The returned directory stands in for the reference's ``deliverables_dir``: it holds ``task_<task_id>/``.
    """

    def __init__(self, cache_root: Path, client_factory: Callable[[], Any] | None = None):
        self._cache_root = cache_root
        self._client_factory = client_factory
        self._client: Any = None
        self._locks: Dict[tuple[str, str], asyncio.Lock] = {}
        # Tasks the reference has no files for, so /verify does not list them again.
        self._missing: set[tuple[str, str]] = set()

    def _s3(self) -> Any:
        if self._client is None:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                import boto3

                self._client = boto3.client("s3")
        return self._client

    async def local_root(self, uri: str, task_id: str) -> Path:
        root = self._cache_root / hashlib.sha256(uri.encode()).hexdigest()[:16]
        target = persisted_layout.task_dir(root, task_id)
        lock = self._locks.setdefault((uri, task_id), asyncio.Lock())
        async with lock:
            if not target.is_dir() and (uri, task_id) not in self._missing:
                if not await asyncio.to_thread(self._download_task, uri, task_id, target):
                    self._missing.add((uri, task_id))
        return root

    def _download_task(self, uri: str, task_id: str, target: Path) -> int:
        """Download the task into ``target`` and return the number of files; zero leaves ``target`` absent."""
        bucket, prefix = _split(uri)
        task_prefix = f"{prefix}/task_{task_id}/" if prefix else f"task_{task_id}/"
        staging = target.with_name(f".{target.name}.{uuid.uuid4().hex}")
        staging.mkdir(parents=True)
        count = 0
        try:
            paginator = self._s3().get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=task_prefix):
                for item in page.get("Contents", []):
                    relative = item["Key"][len(task_prefix) :]
                    if not relative or relative.endswith("/") or Path(relative).name in SKIPPED_NAMES:
                        continue
                    local = staging / relative
                    if staging.resolve() not in local.resolve().parents:
                        raise ValueError(f"S3 key {item['Key']!r} escapes the task directory")
                    local.parent.mkdir(parents=True, exist_ok=True)
                    self._s3().download_file(bucket, item["Key"], str(local))
                    count += 1
            if count:
                staging.rename(target)
            else:
                # No task directory means no reference deliverable, which /verify reports as reference_missing.
                shutil.rmtree(staging)
            return count
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
