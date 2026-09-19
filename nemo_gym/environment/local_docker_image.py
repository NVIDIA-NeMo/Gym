# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build an environment's OCI image with a local Docker daemon."""

from __future__ import annotations

import hashlib
import logging
import re
import shlex
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Protocol

from nemo_gym.config_types import ConfigError
from nemo_gym.environment.authoring import LoadedEnvironment


LOGGER = logging.getLogger(__name__)
DOCKER_INSPECT_TIMEOUT_SECONDS = 30
DOCKER_BUILD_TIMEOUT_SECONDS = 1800


class _Digest(Protocol):
    def update(self, data: bytes) -> None: ...


def build_local_docker_image(loaded: LoadedEnvironment) -> str:
    """Build or reuse the environment's OCI image with the local Docker CLI."""

    docker = shutil.which("docker")
    if docker is None:
        raise ConfigError("The local Docker image builder requires the `docker` CLI on PATH.")

    dockerfile = loaded.resolve_file(loaded.definition.runtime.dockerfile, description="runtime.dockerfile")
    context_dir = dockerfile.parent
    digest = _runtime_digest(context_dir)
    slug = re.sub(r"[^a-z0-9_.-]+", "-", loaded.definition.name.lower()).strip("-.") or "environment"
    image = f"nemo-gym-{slug}:{digest[:16]}"
    inspect_result = _run_docker(
        docker,
        "image",
        "inspect",
        image,
        timeout_s=DOCKER_INSPECT_TIMEOUT_SECONDS,
    )
    if inspect_result.returncode == 0:
        return image

    LOGGER.info("Building runtime image %s from %s", image, dockerfile)
    build_result = _run_docker(
        docker,
        "build",
        "--tag",
        image,
        "--file",
        str(dockerfile),
        str(context_dir),
        timeout_s=DOCKER_BUILD_TIMEOUT_SECONDS,
    )
    if build_result.returncode != 0:
        detail = build_result.stderr.strip() or build_result.stdout.strip() or "docker build failed"
        raise ConfigError(f"Could not build runtime image from {dockerfile}: {detail}")
    return image


def _runtime_digest(context_dir: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(context_dir.rglob("*"), key=lambda item: item.relative_to(context_dir).as_posix()):
        relative = path.relative_to(context_dir)
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            _update_digest_record(
                digest,
                "symlink",
                relative.as_posix(),
                str(mode),
                str(metadata.st_mtime_ns),
                str(path.readlink()),
            )
        elif path.is_dir():
            _update_digest_record(digest, "directory", relative.as_posix(), str(mode), str(metadata.st_mtime_ns))
        elif path.is_file():
            content_digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    content_digest.update(chunk)
            _update_digest_record(
                digest,
                "file",
                relative.as_posix(),
                str(mode),
                str(metadata.st_mtime_ns),
                content_digest.hexdigest(),
            )
    return digest.hexdigest()


def _update_digest_record(digest: _Digest, *fields: str) -> None:
    for field in fields:
        value = field.encode()
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)


def _run_docker(docker: str, *args: str, timeout_s: int) -> subprocess.CompletedProcess[str]:
    command = [docker, *args]
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        raise ConfigError(f"Docker command timed out after {timeout_s} seconds: {shlex.join(command)}") from error


__all__ = ["build_local_docker_image"]
