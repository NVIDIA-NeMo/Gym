# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Install a checksum-pinned Pool archive using Python 3.8+ and curl inside the sandbox."""

from __future__ import annotations

import hashlib
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


# Release checksums from https://downloads.poolside.ai/pool/v1.0.16/checksums.txt.
# Updating the runtime requires reviewing and updating these pins together.
RELEASE_SHA256: dict[str, dict[str, str]] = {
    "v1.0.16": {
        "linux-amd64": "e86aa8c9feef003540673ab494e91bfadc273218d531c43d662cafb69e464146",
        "linux-arm64": "466343b66b03ee4e66476fcc69be1eb5bf8e9155a4ab73e0e72a232d1f8d2a12",
        "darwin-amd64": "6d75fae2d7de6c35b6b467b5f682935e3ecde8ff611cb620c56b7bd607e0afde",
        "darwin-arm64": "0932af3eb2b57a863acacb664ec8b2b1d3a76c2570a788b086001608cc585f74",
    }
}


def install(root: Path, *, version: str, accept_eula: bool) -> None:
    """Verify the complete archive before extracting or executing its single Pool binary."""
    if not accept_eula:
        raise ValueError("Accept https://poolside.ai/eula with accept_pool_eula=true before installing Pool")
    os_name = platform.system().lower()
    machine = platform.machine().lower()
    arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}.get(machine)
    target = f"{os_name}-{arch}"
    checksum = RELEASE_SHA256.get(version, {}).get(target)
    if checksum is None:
        raise ValueError(f"No pinned Pool archive for version={version}, OS={os_name}, architecture={machine}")
    binary_name = f"pool-{target}"
    url = f"https://downloads.poolside.ai/pool/{version}/{binary_name}.tar.gz"
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pool-download-", dir=root) as temporary:
        archive = Path(temporary) / "pool.tar.gz"
        subprocess.run(
            ["curl", "--retry", "3", "--connect-timeout", "30", "--max-time", "120", "-fsSL", url, "-o", str(archive)],
            check=True,
        )
        digest = hashlib.sha256()
        with archive.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError(f"Pool archive SHA-256 mismatch for {version}/{binary_name}.tar.gz")
        binary = Path(temporary) / "pool"
        with tarfile.open(archive, "r:gz") as release:
            member = release.getmember(binary_name)
            if not member.isfile():
                raise ValueError("Pool archive binary must be a regular file")
            # Copy only this member; never extract archive paths, links, or permissions.
            with release.extractfile(member) as source, binary.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        binary.chmod(0o755)
        result = subprocess.run([str(binary), "--version"], capture_output=True, timeout=30, check=True)
        installed = result.stdout.decode(errors="replace").strip()
        if installed != version[1:]:
            raise ValueError(f"Pool version mismatch: expected {version}, got {installed}")
        (root / "bin").mkdir(exist_ok=True)
        binary.replace(root / "bin/pool")


if __name__ == "__main__":
    install(Path(sys.argv[1]), version=sys.argv[2], accept_eula=sys.argv[3] == "true")
