# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare OpenCode's native ripgrep cache without requiring release-host egress.

This script runs in the sandbox using the adapter's Python prerequisite.
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath


def version_of(binary):
    return subprocess.check_output([str(binary), "--version"], text=True, timeout=10).splitlines()[0]


def install(url, archive_sha256, version, destination):
    # Match native resolution order: an existing PATH binary takes precedence.
    system = shutil.which("rg")
    if system:
        return {"path": system, "version": version_of(system), "source": "system"}
    target = Path(destination)
    expected = "ripgrep " + version
    if target.is_file():
        if version_of(target) != expected:
            raise RuntimeError("Cached ripgrep version mismatch")
        return {"path": str(target), "version": expected, "source": "cache"}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ripgrep-", dir=target.parent) as temporary:
        archive = Path(temporary) / "archive.tar.gz"
        with urllib.request.urlopen(url, timeout=60) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with archive.open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != archive_sha256:
            raise RuntimeError("Ripgrep archive SHA-256 mismatch")
        binary = Path(temporary) / "rg"
        with tarfile.open(archive, "r:gz") as package:
            candidates = [
                member
                for member in package.getmembers()
                if len(PurePosixPath(member.name).parts) == 2
                and PurePosixPath(member.name).parts[0].startswith("ripgrep-" + version + "-")
                and PurePosixPath(member.name).name == "rg"
            ]
            if len(candidates) != 1 or not candidates[0].isfile():
                raise RuntimeError("Ripgrep archive must contain one regular release executable")
            # Read only the verified executable; never extract archive paths or links.
            with package.extractfile(candidates[0]) as source, binary.open("wb") as output:
                shutil.copyfileobj(source, output)
        binary.chmod(0o755)
        if version_of(binary) != expected:
            raise RuntimeError("Prefetched ripgrep version mismatch")
        os.replace(binary, target)
    return {"path": str(target), "version": expected, "source": "prefetched", "archive_sha256": archive_sha256}


if __name__ == "__main__":
    print(json.dumps(install(*sys.argv[1:])))
