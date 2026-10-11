# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted helper operations; invoked only by the Resources runtime provisioner."""

import hashlib
import json
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def verify(path: Path, expected: str) -> bool:
    return path.is_file() and digest(path) == expected


def publish(target: Path, staging: Path, count: int, expected: str):
    parts = [Path(str(staging) + "." + str(i)) for i in range(count)]
    with staging.open("wb") as output:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output)
    if digest(staging) != expected:
        raise ValueError("Uploaded runtime SHA-256 mismatch")
    staging.chmod(0o755)
    os.replace(staging, target)
    for part in parts:
        part.unlink()


def python_install(archive_path: Path) -> Path:
    """Unpack the complete install once; role trees reuse immutable hardlinks."""
    installed = archive_path.with_name(archive_path.name + ".python")
    if (installed / ".ready").is_file():
        return installed
    temporary = Path(tempfile.mkdtemp(prefix=archive_path.name + ".unpack-", dir=archive_path.parent))
    try:
        with tarfile.open(archive_path, "r:gz") as archive:
            archive.extractall(temporary, filter="data")
        if not (temporary / "python/bin/python3").is_file():
            raise ValueError("Runtime archive is missing the full Python install")
        (temporary / ".ready").write_text(archive_path.name)
        try:
            os.rename(temporary, installed)
        except FileExistsError:
            if not (installed / ".ready").is_file():
                raise
        return installed
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def prepare(root: Path, roles: dict, owner: str):
    root.mkdir(parents=True, exist_ok=True)
    for role, artifacts in roles.items():
        target = root / role
        target.mkdir(exist_ok=True)
        for name, source in artifacts.items():
            source = Path(source)
            if name == "python":
                shutil.copytree(python_install(source) / "python", target / "python", copy_function=os.link)
            else:
                os.link(source, target / name)
        (target / "owner").write_text(owner)


def release(root: Path, parent_name: str):
    if root.parent.name != parent_name:
        raise ValueError("Runtime cleanup path is outside owned staging directory")
    if root.exists():
        shutil.rmtree(root)
    if root.parent.exists() and not any(root.parent.iterdir()):
        root.parent.rmdir()


def main():
    action, *args = sys.argv[1:]
    if action == "init":
        Path(args[0]).mkdir(parents=True, exist_ok=True)
    elif action == "verify":
        print(int(verify(Path(args[0]), args[1])))
    elif action == "publish":
        publish(Path(args[0]), Path(args[1]), int(args[2]), args[3])
    elif action == "prepare":
        prepare(Path(args[0]), json.loads(args[1]), args[2])
    elif action == "release":
        release(Path(args[0]), args[1])
    else:
        raise ValueError("Unknown trusted runtime operation")


if __name__ == "__main__":
    main()
