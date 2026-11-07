# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Content-addressed files for large ``data:`` URLs, shared by the delta capture format and rollout rows.

A blob is the URL's UTF-8 text in a file named by its SHA-256, so one image is stored once however many places
name it. The JSON that held it keeps ``null`` there and lists the place, as a path of keys and indexes, beside it.
"""

import hashlib
import os
import re
import threading
from pathlib import Path
from typing import Any, Optional


# Shorter data URLs (icons, tiny crops) are cheaper inline than as a file.
BLOB_MIN_CHARS = 1024
DATA_URL_PATTERN = re.compile(r"data:[\w.+-]+/[\w.+-]+[;,]")
SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")


def extract_data_urls(
    value: Any,
    path: list[Any],
    blobs: list[list[Any]],
    *,
    min_chars: int = BLOB_MIN_CHARS,
    digests: Optional[dict[int, str]] = None,
) -> Any:
    """``value`` with data URLs of at least ``min_chars`` replaced by ``None``; appends ``[path, sha256, text]``.

    Containers are copied only where something inside them changed, so the caller's value is never modified.
    ``digests`` caches each string object's hash by identity for one walk, so a URL object named in several
    places is hashed once; the caller keeps the walked value alive while it is in use.
    """
    if isinstance(value, str):
        if len(value) >= min_chars and DATA_URL_PATTERN.match(value):
            digest = digests.get(id(value)) if digests is not None else None
            if digest is None:
                digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
                if digests is not None:
                    digests[id(value)] = digest
            blobs.append([list(path), digest, value])
            return None
        return value
    if isinstance(value, dict):
        changed: Optional[dict[str, Any]] = None
        for key, item in value.items():
            if not isinstance(item, (str, dict, list)):
                continue
            path.append(key)
            replaced = extract_data_urls(item, path, blobs, min_chars=min_chars, digests=digests)
            path.pop()
            if replaced is not item:
                changed = changed if changed is not None else dict(value)
                changed[key] = replaced
        return changed if changed is not None else value
    if isinstance(value, list):
        changed_list: Optional[list[Any]] = None
        for index, item in enumerate(value):
            if not isinstance(item, (str, dict, list)):
                continue
            path.append(index)
            replaced = extract_data_urls(item, path, blobs, min_chars=min_chars, digests=digests)
            path.pop()
            if replaced is not item:
                changed_list = changed_list if changed_list is not None else list(value)
                changed_list[index] = replaced
        return changed_list if changed_list is not None else value
    return value


def write_blob(directory: Path, digest: str, text: str) -> bool:
    """Write a blob once, atomically and durably, so a reference written after it never names a partial file.

    Returns whether this call wrote it (``False`` when it was already there).
    """
    target = directory / digest
    if target.exists():
        return False
    if not directory.is_dir():
        directory.mkdir(parents=True, exist_ok=True)
        fsync_directory(directory.parent.parent)
        fsync_directory(directory.parent)
    temporary = directory / f".{digest}.{os.getpid()}.{threading.get_ident()}.tmp"
    try:
        with temporary.open("wb") as handle:
            handle.write(text.encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    fsync_directory(directory)
    return True


def fsync_directory(directory: Path) -> None:
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
