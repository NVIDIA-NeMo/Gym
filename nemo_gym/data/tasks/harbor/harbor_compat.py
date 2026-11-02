# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Environment content hashing and image naming vendored from Harbor v0.22.0 (Apache-2.0).

Gym builds the images of Harbor tasks with its own tools (buildah, podman or docker; see
``nemo_gym.data.tasks.harbor.image_build``) but hashes the task's ``environment/`` folder and names the image
the way Harbor does, so a task prepared by either framework resolves to the same image name. Kept
byte-for-byte so Gym and Harbor compute the same image names for the same build context.

Vendored from https://github.com/harbor-framework/harbor (Apache-2.0, see ``ATTRIBUTIONS.md``):

- :func:`environment_content_hash` and :data:`_CONTENT_HASH_IGNORE_NAMES`:
  https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/definition.py
- :func:`sanitize_docker_image_name` (``_sanitize_docker_image_name`` upstream):
  https://github.com/harbor-framework/harbor/blob/v0.22.0/src/harbor/environments/docker/docker.py

Harbor's environment id is ``environment_content_hash(environment_dir)`` and its task image is
``sanitize_docker_image_name(f"hb__{environment_id}")``: ``hb__<32 hex characters>``.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path


# -- Vendored from Harbor v0.22.0, unmodified (bodies byte-for-byte; the private sanitiser is exposed
# -- without its leading underscore). Do not edit: any change breaks image-name parity with Harbor.

_CONTENT_HASH_IGNORE_NAMES = frozenset({".DS_Store", ".git", "__pycache__"})


def environment_content_hash(
    environment_dir: Path,
    *,
    docker_image: str | None = None,
    truncate: int = 32,
) -> str:
    """Return a stable SHA-256 hash of the environment contents.

    The result is consistent across Linux and macOS. For an empty environment,
    the Docker image reference is hashed instead.
    """
    candidates: list[tuple[str, Path]] = []
    for path in environment_dir.rglob("*"):
        if not path.is_file() or path.is_symlink():
            continue
        rel = path.relative_to(environment_dir)
        if _CONTENT_HASH_IGNORE_NAMES & set(rel.parts):
            continue
        candidates.append((rel.as_posix(), path))

    if not candidates:
        seed = (docker_image or environment_dir.name).encode("utf-8")
        return hashlib.sha256(seed).hexdigest()[:truncate]

    candidates.sort(key=lambda item: item[0])
    digest = hashlib.sha256()
    for rel_posix, path in candidates:
        rel_bytes = rel_posix.encode("utf-8")
        data = path.read_bytes()
        digest.update(len(rel_bytes).to_bytes(4, "big"))
        digest.update(rel_bytes)
        digest.update(len(data).to_bytes(4, "big"))
        digest.update(data)
    return digest.hexdigest()[:truncate]


def sanitize_docker_image_name(name: str) -> str:
    """
    Sanitize a name to be a valid Docker image name.

    See: https://github.com/opencontainers/distribution-spec/blob/5e57cc0a07ea002e507a65d4757e823f133fcb52/spec.md#pulling-manifests
    """
    # Convert to lowercase
    name = name.lower()
    # If the first character is not alphanumeric, prepend '0'
    if not re.match(r"^[a-z0-9]", name):
        name = "0" + name
    # Replace any character that is not a-z, 0-9, ., _, - with -
    # Note: / is not allowed here because we want only one directory hierarchy.
    name = re.sub(r"[^a-z0-9._-]", "-", name)
    return name


# -- End of the vendored section.
